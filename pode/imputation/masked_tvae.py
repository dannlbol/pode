"""Engine ``tvae-masked``: a mask-aware TVAE trained for imputation.

Differences from the SDV/ctgan TVAE used for synthesis:

* **Trains on incomplete data.** Per-column normalisers are fitted on observed
  values only (no mean-fill), and the reconstruction loss covers observed
  cells only.
* **Mask-aware encoder.** The encoder sees ``[x with missing spans zeroed,
  column mask]``, so it learns q(z | x_obs, m) for arbitrary patterns.
* **Denoising objective.** Each batch hides extra observed cells (MCAR at a
  per-row rate drawn from U(0, p_max), plus random whole block groups), and
  those hidden cells are up-weighted in the loss. The model is thus trained on
  the task it performs: reconstructing held-out fields from context.
* **Keeps its encoder.** At inference the encoder proposes latents which are
  importance-weighted against the decoder likelihood (MIWAE-style SIR),
  optionally refined with pCN, before missing cells are sampled.

Architecture hyperparameters come from :class:`~pode.config.SynthesizerConfig`,
so values tuned by ``pode synthesis tune`` carry over.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.tvae import Decoder, Encoder
from torch import nn

from ..config import SynthesizerConfig
from ..schema.schema import DatasetSchema
from .base import Imputer, category_key, resolve_columns
from .config import LatentConfig, MaskedTVAEConfig
from .spans import column_loglik, seed_transformer, spans_from_transformer
from .tvae_common import draw_completions, summarise_diagnostics


class MaskedDataTransformer(DataTransformer):
    """ctgan ``DataTransformer`` whose column normalisers ignore missing values."""

    def _fit_continuous(self, data):
        return super()._fit_continuous(data.dropna())

    def _fit_discrete(self, data):
        return super()._fit_discrete(data.dropna())

    def fit(self, raw_data: pd.DataFrame, discrete_columns=()):
        empty = [c for c in raw_data.columns if raw_data[c].isna().all()]
        if empty:
            raise ValueError(f"Columns with no observed values cannot be modelled: {empty}")
        super().fit(raw_data, discrete_columns)
        self._discrete = set(discrete_columns)
        self._ranges = {}
        self._fill = {}
        for info in self._column_transform_info_list:
            col = info.column_name
            if info.column_type == "continuous":
                x = pd.to_numeric(raw_data[col], errors="coerce")
                self._ranges[col] = (float(x.min()), float(x.max()))
                self._fill[col] = float(x.median())
            else:
                self._fill[col] = list(info.transform.dummies)[0]

    def encode(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cols = [i.column_name for i in self._column_transform_info_list]
        raw = df.reindex(columns=cols).copy()
        observed = raw.notna().to_numpy()
        for j, info in enumerate(self._column_transform_info_list):
            col = info.column_name
            if info.column_type == "discrete":
                dummies = list(info.transform.dummies)
                lookup = {k: d for k, d in zip(category_key(dummies), dummies)}
                keys = category_key(raw[col].to_numpy())
                unknown = np.array([k is not None and k not in lookup for k in keys])
                observed[:, j] &= ~unknown
                vals = [lookup.get(k, self._fill[col]) if k is not None else self._fill[col] for k in keys]
                raw[col] = pd.Series(vals, index=raw.index, dtype=object).infer_objects()
            else:
                raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(self._fill[col])
        return self.transform(raw).astype(np.float32), observed

    def decode(self, samples: np.ndarray) -> pd.DataFrame:
        out = self.inverse_transform(samples.astype(np.float64), sigmas=None)
        for col, (lo, hi) in self._ranges.items():
            out[col] = pd.to_numeric(out[col], errors="coerce").clip(lo, hi)
        return out


class _MaskedNet(nn.Module):
    def __init__(self, data_dim: int, n_cols: int, cfg: SynthesizerConfig):
        super().__init__()
        self.encoder = Encoder(data_dim + n_cols, cfg.compress_dims, cfg.embedding_dim)
        self.decoder = Decoder(cfg.embedding_dim, cfg.decompress_dims, data_dim)


class MaskedTVAEImputer(Imputer):
    name = "tvae-masked"

    def __init__(
        self,
        synth_config: SynthesizerConfig | None = None,
        masked: MaskedTVAEConfig | None = None,
        latent: LatentConfig | None = None,
        block_groups: dict[str, list[str]] | None = None,
        epochs: int | None = None,
        fit_seed: int | None = None,
        device: str | None = None,
        atomic_blocks: bool = True,
    ):
        super().__init__()
        self.synth_config = synth_config or SynthesizerConfig()
        self.masked = masked or MaskedTVAEConfig()
        self.latent = latent or LatentConfig()
        self.block_groups = block_groups or {}
        self.atomic_blocks = atomic_blocks
        self.epochs = epochs or self.masked.epochs or self.synth_config.epochs
        self.fit_seed = fit_seed
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.net: _MaskedNet | None = None
        self.loss_history: list[float] = []

    # ---- helpers -------------------------------------------------------------
    def _span_cols(self) -> torch.Tensor:
        """Index of the raw column owning each transformed dimension."""
        owner = np.zeros(self.spans[-1].end, dtype=np.int64)
        for j, sp in enumerate(self.spans):
            owner[sp.start : sp.end] = j
        return torch.as_tensor(owner, device=self.device)

    def _encode_latent(self, x: torch.Tensor, vis: torch.Tensor):
        x_in = x * vis[:, self._owner]
        mu, std, logvar = self.net.encoder(torch.cat([x_in, vis], dim=1))
        return mu, std, logvar

    def _training_drop(self, ob: torch.Tensor, groups: torch.Tensor | None, cell_ok: torch.Tensor) -> torch.Tensor:
        """Artificial mask for one batch (True = hide an observed cell).

        Cells in ``cell_ok`` columns are hidden at a per-row rate drawn from
        U(0, p_max); each block group is independently hidden whole with
        probability ``block_prob`` (so a row can lose several modalities).
        """
        b = ob.shape[0]
        rate = torch.rand(b, 1, device=ob.device) * self.masked.p_max
        drop = (torch.rand(ob.shape, device=ob.device) < rate) & ob & cell_ok
        if groups is not None and self.masked.block_prob > 0:
            chosen = torch.rand(b, groups.shape[0], device=ob.device) < self.masked.block_prob
            drop |= (chosen.float() @ groups.float()).bool() & ob
        return drop

    # ---- training ------------------------------------------------------------
    def _fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "MaskedTVAEImputer":
        from ..reproducibility.seeds import set_all_seeds

        if self.fit_seed is not None:
            set_all_seeds(self.fit_seed)
        self.schema = schema
        self.columns = [c for c in schema.columns if c in df.columns]
        data = df[self.columns].reset_index(drop=True).copy()
        for c in self.columns:
            if schema.types[c].is_numeric:
                data[c] = pd.to_numeric(data[c], errors="coerce")
        discrete = [c for c in self.columns if schema.types[c].is_discrete]

        self.transformer = MaskedDataTransformer()
        self.transformer.fit(data, discrete)
        self.spans = spans_from_transformer(self.transformer)
        x_np, obs_np = self.transformer.encode(data)
        data_dim, n_cols = x_np.shape[1], len(self.spans)

        cfg = self.synth_config
        self.net = _MaskedNet(data_dim, n_cols, cfg).to(self.device)
        self._owner = self._span_cols()
        # (n_groups, n_cols) membership of configured block groups.
        members = [
            [c in set(resolve_columns(g, self.columns)) for c in self.columns]
            for g in self.block_groups.values()
        ]
        members = [row for row in members if any(row)]
        groups = torch.as_tensor(members, dtype=torch.bool, device=self.device) if members else None
        # Columns that may be dropped one at a time: everything, or only
        # non-block columns when blocks are all-or-nothing.
        cell_ok = torch.ones(n_cols, dtype=torch.bool, device=self.device)
        if self.atomic_blocks and groups is not None:
            cell_ok = ~groups.any(dim=0)

        x_all = torch.as_tensor(x_np, device=self.device)
        obs_all = torch.as_tensor(obs_np, device=self.device)
        opt = torch.optim.Adam(self.net.parameters(), weight_decay=cfg.l2scale)
        mcfg = self.masked
        n = x_all.shape[0]
        bs = min(cfg.batch_size, n)
        t0 = time.perf_counter()
        self.loss_history = []
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            for st in range(0, n, bs):
                idx = perm[st : st + bs]
                xb, ob = x_all[idx], obs_all[idx]
                b = xb.shape[0]
                drop = self._training_drop(ob, groups, cell_ok)
                vis = (ob & ~drop).float()
                mu, std, logvar = self._encode_latent(xb, vis)
                z = mu + std * torch.randn_like(std)
                rec, sig = self.net.decoder(z)
                ll = column_loglik(rec, sig, xb, self.spans)
                weight = ob.float() * (1.0 + (mcfg.hidden_weight - 1.0) * drop.float())
                recon = -(ll * weight).sum() / b
                kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()) / b
                loss = cfg.loss_factor * recon + mcfg.beta * kl
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.net.decoder.sigma.data.clamp_(0.01, 1.0)
                epoch_loss += float(loss.detach()) * b
            self.loss_history.append(epoch_loss / n)
        self._diagnostics["fit_rows"] = int(n)
        self._diagnostics["fit_seconds"] = time.perf_counter() - t0
        self._diagnostics["final_loss"] = self.loss_history[-1] if self.loss_history else float("nan")
        return self

    # ---- imputation ----------------------------------------------------------
    def _impute(self, df: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
        if self.net is None:
            raise RuntimeError("Imputer is not fitted. Call fit() or load() first.")
        self.net.eval()
        seed_transformer(self.transformer, seed)
        x, obs = self.transformer.encode(df)
        todo = np.flatnonzero(~obs.all(axis=1))
        draws = [df.copy() for _ in range(m)]
        if todo.size == 0:
            self._diagnostics.update(summarise_diagnostics({}, m))
            return draws

        def decoder(z: torch.Tensor):
            return self.net.decoder(z)

        def encoder(xc: torch.Tensor, oc: torch.Tensor):
            mu, std, _ = self._encode_latent(xc, oc)
            return mu, std

        power = self.latent.likelihood_power
        power = float(self.synth_config.loss_factor) if power is None else power
        t0 = time.perf_counter()
        samples, diag = draw_completions(
            decoder, x[todo], obs[todo], self.spans, self.synth_config.embedding_dim,
            m, self.latent, power, self.device, seed, encoder=encoder,
        )
        for i in range(m):
            filled = self.transformer.decode(samples[:, i, :])
            for col in self.columns:
                draws[i][col] = draws[i][col].astype(object)
                draws[i].loc[todo, col] = filled[col].to_numpy()
        self._diagnostics.update(summarise_diagnostics(diag, m))
        self._diagnostics["impute_seconds"] = time.perf_counter() - t0
        return [d.infer_objects() for d in draws]

    # ---- persistence ---------------------------------------------------------
    def save(self, out_dir: str | Path) -> Path:
        if self.net is None:
            raise RuntimeError("Nothing to save; imputer is not fitted.")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        device = self.device
        self.net.to("cpu")
        self._owner = self._owner.cpu()
        self.device = torch.device("cpu")
        with open(out_dir / "model.pkl", "wb") as f:
            pickle.dump(self, f)
        self.device = device
        self.net.to(device)
        self._owner = self._owner.to(device)
        self.schema.to_yaml(out_dir / "schema.yaml")
        meta = {
            "engine": self.name,
            "columns": self.columns,
            "epochs": self.epochs,
            "synthesizer": dataclasses.asdict(self.synth_config),
            "masked_tvae": dataclasses.asdict(self.masked),
            "block_groups": self.block_groups,
            "final_loss": self._diagnostics.get("final_loss"),
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        pd.DataFrame({"epoch": range(len(self.loss_history)), "loss": self.loss_history}).to_csv(
            out_dir / "loss_history.csv", index=False
        )
        return out_dir

    @classmethod
    def load(cls, model_dir: str | Path, latent: LatentConfig | None = None, device: str | None = None) -> "MaskedTVAEImputer":
        with open(Path(model_dir) / "model.pkl", "rb") as f:
            obj: MaskedTVAEImputer = pickle.load(f)
        want = device if device not in (None, "auto") else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            obj.net.to(want)
        except RuntimeError as exc:
            if device == "cuda" or "CUDA" not in str(exc):
                raise
            warnings.warn(f"Could not move the model to the GPU ({str(exc).splitlines()[0]}); using CPU.",
                          stacklevel=2)
            want = "cpu"
            obj.net.to(want)
        obj.device = torch.device(want)
        obj._owner = obj._owner.to(obj.device)
        if latent is not None:
            obj.latent = latent
        return obj


def is_masked_model_dir(path: str | Path) -> bool:
    meta = Path(path) / "meta.json"
    if not meta.is_file():
        return False
    with open(meta) as f:
        return json.load(f).get("engine") == MaskedTVAEImputer.name
