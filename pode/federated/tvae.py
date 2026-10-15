"""Federated TVAE: ctgan's encoder/decoder trained by FedAvg/FedProx across sites.

Architecture and loss are ctgan's TVAE (as used by SDV and PODE synthesis):
mode-specific normalised continuous columns with a Gaussian likelihood, one-hot
categorical columns with a softmax likelihood, and a standard-normal prior.
What changes is *who trains it*:

1. The sites agree a shared column encoder from summaries only
   (:func:`~pode.federated.transformer.federated_encoder`).
2. Each round the coordinator broadcasts the current weights; every site runs
   ``local_epochs`` epochs of Adam on its own rows (``site_local_update``),
   optionally with the FedProx term ``mu/2 ||w - w_global||^2`` that keeps
   non-IID sites from drifting apart, optionally under DP-SGD.
3. Sites return their weights, pre-scaled by their aggregation weight and,
   with ``secure_aggregation``, pairwise-masked, so the coordinator learns only
   the weighted average (:mod:`~pode.federated.aggregate`).
4. Optionally each site then fine-tunes its own copy (``personalise_epochs``):
   a personalised generator that stays at the site and samples there.

Each site keeps its Adam state and random stream across rounds, so a one-site
federation with ``rounds x local_epochs`` epochs is exactly a plain TVAE fit.
That is how the ``local`` and ``central`` baselines are produced: the same code
on one site's rows or on pooled rows, holding the architecture constant.

The global generator is shared with all parties (that is the point of a
federated generator); synthetic rows sampled from it are subject to the same
privacy gate as any other release.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ctgan.synthesizers.tvae import Decoder, Encoder
from torch import nn
from torch.nn.functional import cross_entropy

from ..config import SynthesizerConfig
from .aggregate import SecureSum, flatten_state, secure_weighted_average, site_weights, unflatten_state, \
    weighted_average
from .config import FederatedConfig
from .ledger import DisclosureLedger
from .sites import Site, stateful
from .transformer import SITE_COLUMN, FederatedEncoder, federated_encoder


# ---- network and loss ------------------------------------------------------------
class TVAENet(nn.Module):
    def __init__(self, data_dim: int, compress_dims, decompress_dims, embedding_dim: int):
        super().__init__()
        self.encoder = Encoder(data_dim, compress_dims, embedding_dim)
        self.decoder = Decoder(embedding_dim, decompress_dims, data_dim)

    def forward(self, x: torch.Tensor, eps: torch.Tensor):
        mu, std, logvar = self.encoder(x)
        rec, sigma = self.decoder(mu + std * eps)
        return rec, sigma, mu, logvar


class LossSpec:
    """ctgan's TVAE loss, per row (its batch mean equals ctgan's ``_loss_function``)."""

    def __init__(self, output_info_list, factor: float):
        cont, soft, st = [], [], 0
        for col in output_info_list:
            for span in col:
                if span.activation_fn != "softmax":
                    cont.append(st)
                else:
                    soft.append((st, st + span.dim))
                st += span.dim
        self.cont = torch.as_tensor(cont, dtype=torch.long)
        self.soft = soft
        self.factor = float(factor)

    def to(self, device) -> "LossSpec":
        self.cont = self.cont.to(device)
        return self

    def per_row(self, out, x: torch.Tensor) -> torch.Tensor:
        rec, sigma, mu, logvar = out
        recon = torch.zeros(x.shape[0], device=x.device, dtype=rec.dtype)
        if len(self.cont):
            sd = sigma[self.cont]
            eq = x[:, self.cont] - torch.tanh(rec[:, self.cont])
            recon = recon + (eq ** 2 / 2 / sd ** 2).sum(1) + torch.log(sd).sum()
        for st, ed in self.soft:
            recon = recon + cross_entropy(rec[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction="none")
        kld = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)
        return recon * self.factor + kld


@dataclass
class Arch:
    data_dim: int
    compress_dims: tuple
    decompress_dims: tuple
    embedding_dim: int

    def build(self) -> TVAENet:
        return TVAENet(self.data_dim, self.compress_dims, self.decompress_dims, self.embedding_dim)


@dataclass
class TrainSpec:
    """What a site needs to run its local update (all public)."""

    epochs: int
    batch_size: int
    lr: float
    l2scale: float
    loss_factor: float
    prox_mu: float = 0.0
    dp_noise: float | None = None  # None -> no DP
    dp_max_norm: float = 1.0
    dp_batch: int = 64


@dataclass
class SecureSpec:
    n_sites: int
    secret: int
    index: int
    round: int
    weight: float


# ---- site operations ---------------------------------------------------------------
@stateful
def site_setup(data: pd.DataFrame, local: dict, encoder: FederatedEncoder, arch: Arch, device: str,
               seed: int, site_name: str) -> int:
    """Encode the site's rows once and create its private model, optimiser and RNG streams."""
    # ctgan's mode assignment draws from numpy's global RNG: seed it so the
    # encoding depends only on the rows (not on which site encoded first).
    np_state = np.random.get_state()
    np.random.seed(int(seed) % (2 ** 32))
    try:
        x = encoder.encode(data, site=site_name)
    finally:
        np.random.set_state(np_state)
    local["x"] = torch.as_tensor(x, device=device)
    local["encoder"] = encoder
    local["device"] = device
    local["net"] = arch.build().to(device)
    local["opt"] = None
    # Same stream at every site (so identical sites give identical updates);
    # DP noise uses a site-specific stream.
    local["gen"] = torch.Generator().manual_seed(int(seed))
    local["dp_gen"] = torch.Generator().manual_seed(int(seed) * 7919 + sum(map(ord, site_name)))
    return int(x.shape[0])


def _train(net: TVAENet, x: torch.Tensor, spec: TrainSpec, loss: LossSpec, gen: torch.Generator,
           dp_gen: torch.Generator, opt: torch.optim.Optimizer, anchor: dict | None) -> tuple[float, int]:
    """``spec.epochs`` epochs of (optionally DP, optionally proximal) Adam. Returns (mean loss, steps)."""
    from .dp import clip_and_noise, per_sample_grads

    n, device = x.shape[0], x.device
    emb = net.decoder.seq[0].in_features
    total, rows, steps = 0.0, 0, 0
    params = dict(net.named_parameters())
    net.train()
    for _ in range(spec.epochs):
        if spec.dp_noise is None:
            perm = torch.randperm(n, generator=gen).to(device)
            batches = [perm[st:st + spec.batch_size] for st in range(0, n, spec.batch_size)]
        else:
            q = min(1.0, spec.dp_batch / n)
            batches = [torch.nonzero(torch.rand(n, generator=gen) < q).flatten().to(device)
                       for _ in range(max(1, round(1 / q)))]
        for idx in batches:
            xb = x[idx]
            eps = torch.randn(len(idx), emb, generator=gen).to(device)
            opt.zero_grad()
            if spec.dp_noise is None:
                rows_loss = loss.per_row(net(xb, eps), xb)
                obj = rows_loss.mean()
                if anchor is not None and spec.prox_mu > 0:
                    obj = obj + 0.5 * spec.prox_mu * sum(((p - anchor[k]) ** 2).sum() for k, p in params.items())
                obj.backward()
            else:
                q = min(1.0, spec.dp_batch / n)
                if len(idx):
                    grads = per_sample_grads(net, loss.per_row, xb, eps)
                else:  # an empty Poisson batch still takes a (pure-noise) step
                    grads = {k: torch.zeros((0, *p.shape), device=device) for k, p in params.items()}
                noisy, _ = clip_and_noise(grads, spec.dp_max_norm, spec.dp_noise, q * n, dp_gen)
                for k, p in params.items():
                    g = noisy[k]
                    if anchor is not None and spec.prox_mu > 0:  # data-independent: no privacy cost
                        g = g + spec.prox_mu * (p.detach() - anchor[k])
                    p.grad = g.to(p.dtype)
                with torch.no_grad():
                    rows_loss = loss.per_row(net(xb, eps), xb) if len(idx) else torch.zeros(0)
            opt.step()
            net.decoder.sigma.data.clamp_(0.01, 1.0)
            steps += 1
            total += float(rows_loss.detach().sum())
            rows += len(idx)
    return total / max(rows, 1), steps


def _make_opt(net: TVAENet, spec: TrainSpec) -> torch.optim.Optimizer:
    return torch.optim.Adam(net.parameters(), lr=spec.lr, weight_decay=spec.l2scale)


@stateful
def site_local_update(data: pd.DataFrame, local: dict, global_state: dict, spec: TrainSpec,
                      secure: SecureSpec | None = None) -> dict:
    """One round at a site: load global weights, train locally, return (masked, weighted) weights."""
    net: TVAENet = local["net"]
    device = local["device"]
    net.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    if local["opt"] is None:
        local["opt"] = _make_opt(net, spec)
    loss = LossSpec(local["encoder"].output_info_list, spec.loss_factor).to(device)
    anchor = {k: v.detach().clone() for k, v in net.named_parameters()} if spec.prox_mu > 0 else None
    mean_loss, steps = _train(net, local["x"], spec, loss, local["gen"], local["dp_gen"], local["opt"], anchor)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    out = {"loss": mean_loss, "steps": steps}
    if secure is None:
        out["state"] = state
    else:
        vec, _ = flatten_state(state)
        out["masked"] = SecureSum(secure.n_sites, secure.secret).mask(secure.index, secure.round,
                                                                       secure.weight * vec)
    return out


@stateful
def site_personalise(data: pd.DataFrame, local: dict, global_state: dict, spec: TrainSpec) -> dict:
    """Fine-tune a private copy of the global model on the site's rows; it stays at the site."""
    device = local["device"]
    net: TVAENet = local["net"]
    net.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    opt = _make_opt(net, spec)
    loss = LossSpec(local["encoder"].output_info_list, spec.loss_factor).to(device)
    mean_loss, steps = _train(net, local["x"], spec, loss, local["gen"], local["dp_gen"], opt, None)
    local["personal_state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    return {"loss": mean_loss, "steps": steps}


@stateful
def site_sample(data: pd.DataFrame, local: dict, n: int, seed: int, arch: Arch) -> pd.DataFrame:
    """Synthetic rows from the site's personalised generator (sampled at the site)."""
    if "personal_state" not in local:
        raise RuntimeError("This site has no personalised generator; set generator.personalise_epochs.")
    net = arch.build()
    net.load_state_dict(local["personal_state"])
    return _generate(net, local["encoder"], n, seed)


def _generate(net: TVAENet, encoder: FederatedEncoder, n: int, seed: int | None) -> pd.DataFrame:
    gen = torch.Generator().manual_seed(int(seed or 0))
    net = net.cpu().eval()
    emb = net.decoder.seq[0].in_features
    with torch.no_grad():
        z = torch.randn(n, emb, generator=gen)
        rec, sigma = net.decoder(z)
        fake = torch.tanh(rec).numpy()
    state = np.random.get_state()
    np.random.seed(int(seed or 0) % (2 ** 32))
    try:
        return encoder.decode(fake, sigma.detach().numpy())
    finally:
        np.random.set_state(state)


# ---- coordinator -------------------------------------------------------------------
def resolve_device(device: str) -> str:
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class FitRecord:
    rounds: list[dict] = field(default_factory=list)
    privacy: dict = field(default_factory=dict)
    fit_seconds: float = 0.0
    bytes_per_round: float = 0.0


class FederatedTVAE:
    """A TVAE trained across :class:`~pode.federated.sites.Site` objects."""

    def __init__(self, synth_config: SynthesizerConfig | None = None, config: FederatedConfig | None = None,
                 schema=None, censor_threshold: float | None = None):
        self.synth = synth_config or SynthesizerConfig()
        self.cfg = (config or FederatedConfig()).validate()
        self.schema = schema
        self.censor_threshold = censor_threshold
        self.encoder: FederatedEncoder | None = None
        self.arch: Arch | None = None
        self.state: dict | None = None
        self.record = FitRecord()
        self.sites: list[Site] = []
        self.columns: list[str] = []

    # ---- fitting ---------------------------------------------------------------
    def _discrete(self, columns: list[str], sites: list[Site]) -> list[str]:
        if self.schema is not None:
            return [c for c in columns if c in self.schema.types and not self.schema.types[c].is_numeric]
        sample = sites[0]._data  # dtype only; no values leave the site in a real deployment
        return [c for c in columns if not pd.api.types.is_numeric_dtype(sample[c])]

    def _train_spec(self, epochs: int, noise: float | None) -> TrainSpec:
        g, d = self.cfg.generator, self.cfg.dp
        return TrainSpec(epochs=epochs, batch_size=self.synth.batch_size, lr=g.learning_rate,
                         l2scale=self.synth.l2scale, loss_factor=self.synth.loss_factor,
                         prox_mu=g.prox_mu if g.algorithm == "fedprox" else 0.0,
                         dp_noise=noise, dp_max_norm=d.max_grad_norm, dp_batch=d.batch_size)

    def _plan_dp(self, sites: list[Site], ns: list[int]) -> dict[str, dict]:
        """Per-site noise multiplier and (expected) budget, fixed before training."""
        from .dp import default_delta, noise_for_target, require_opacus

        g, d = self.cfg.generator, self.cfg.dp
        require_opacus()
        plan = {}
        for s, n in zip(sites, ns):
            q = min(1.0, d.batch_size / n)
            spe = max(1, round(1 / q))
            steps = (g.rounds * g.local_epochs + g.personalise_epochs) * spe
            delta = d.delta if d.delta is not None else default_delta(n)
            noise = d.noise_multiplier if d.noise_multiplier is not None else \
                noise_for_target(d.target_epsilon, delta, q, steps, d.accountant)
            plan[s.name] = {"noise_multiplier": float(noise), "sample_rate": q, "planned_steps": steps,
                            "delta": delta}
        return plan

    def fit(self, sites: list[Site], columns: list[str] | None = None, log=print,
            discrete: list[str] | None = None) -> "FederatedTVAE":
        """Train across ``sites`` on ``columns``. Discrete columns come from the schema,
        else ``discrete``, else non-numeric dtypes."""
        from ..reproducibility.seeds import set_all_seeds

        t0 = time.perf_counter()
        cfg, g = self.cfg, self.cfg.generator
        device = resolve_device(cfg.device)
        self.sites = sites
        if columns is None:
            columns = [c for c in sites[0]._data.columns]
        self.columns = list(columns)
        discrete = list(discrete) if discrete is not None else self._discrete(self.columns, sites)
        self.encoder = federated_encoder(sites, self.columns, discrete, cfg.transformer, cfg.dp,
                                         g.site_in_model, cfg.seed)
        self.arch = Arch(self.encoder.output_dimensions, tuple(self.synth.compress_dims),
                         tuple(self.synth.decompress_dims), int(self.synth.embedding_dim))
        ns = [s.run(site_setup, self.encoder, self.arch, device, cfg.seed, s.name, label="site_count")
              for s in sites]
        dp_plan = self._plan_dp(sites, ns) if cfg.dp.enabled else {}
        set_all_seeds(cfg.seed)
        net = self.arch.build()
        state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        weights = site_weights(ns, g.weighting, g.cap_fraction)
        secure = g.secure_aggregation and len(sites) > 1
        ledger = sites[0].ledger
        steps = {s.name: 0 for s in sites}
        for r in range(g.rounds):
            results = []
            for i, s in enumerate(sites):
                spec = self._train_spec(g.local_epochs, dp_plan.get(s.name, {}).get("noise_multiplier"))
                sec = SecureSpec(len(sites), cfg.seed, i, r, float(weights[i])) if secure else None
                results.append(s.run(site_local_update, state, spec, sec, label="model_update"))
            if secure:
                _, layout = flatten_state(state)
                state = unflatten_state(SecureSum(len(sites), cfg.seed).aggregate([x["masked"] for x in results]),
                                        layout)
            else:
                state = weighted_average([x["state"] for x in results], weights)
            for s, x in zip(sites, results):
                steps[s.name] += x["steps"]
            loss = float(np.sum([w * x["loss"] for w, x in zip(weights, results)]))
            self.record.rounds.append({"round": r, "loss": loss, **{f"loss_{s.name}": x["loss"]
                                                                    for s, x in zip(sites, results)}})
            if ledger is not None:
                ledger.next_round()
            if log and (r == 0 or (r + 1) % max(1, g.rounds // 5) == 0):
                log(f"  round {r + 1}/{g.rounds}: loss {loss:.3f}")
        self.state = state
        if g.personalise_epochs > 0:
            for s in sites:
                spec = self._train_spec(g.personalise_epochs, dp_plan.get(s.name, {}).get("noise_multiplier"))
                res = s.run(site_personalise, state, spec, label="personalise_ack")
                steps[s.name] += res["steps"]
        self.record.privacy = self._account(sites, ns, dp_plan, steps)
        self.record.fit_seconds = time.perf_counter() - t0
        if ledger is not None:
            upd = ledger.frame
            upd = upd[upd["op"] == "model_update"]
            self.record.bytes_per_round = float(upd["bytes"].sum() / max(g.rounds, 1)) if len(upd) else 0.0
        return self

    def _account(self, sites, ns, plan, steps) -> dict:
        from .dp import epsilon_spent

        d = self.cfg.dp
        out = {}
        for s, n in zip(sites, ns):
            if not d.enabled:
                out[s.name] = {"dp": False, "epsilon_total": float("inf"), "n": n}
                continue
            p = plan[s.name]
            eps = epsilon_spent(p["noise_multiplier"], p["sample_rate"], steps[s.name], p["delta"], d.accountant)
            out[s.name] = {"dp": True, "n": n, **p, "steps": steps[s.name], "epsilon_sgd": eps,
                           "epsilon_transformer": d.transformer_epsilon,
                           "epsilon_total": eps + d.transformer_epsilon,
                           "transformer_notes": list(self.encoder.dp_notes)}
        return out

    # ---- sampling --------------------------------------------------------------
    def _net(self) -> TVAENet:
        if self.state is None:
            raise RuntimeError("FederatedTVAE is not fitted. Call fit() or load() first.")
        net = self.arch.build()
        net.load_state_dict(self.state)
        return net

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.schema is None:
            return df
        from ..augmentation.generator import apply_constraints, with_survival_rules
        from ..generation.postprocess import correct_survival_boundaries
        from ..imputation.base import subset_schema

        sub = subset_schema(self.schema, [c for c in self.schema.columns if c in df.columns])
        df, _ = apply_constraints(df, with_survival_rules(sub, self.cfg.generator.constraints))
        return correct_survival_boundaries(df, sub, self.censor_threshold)

    def sample(self, n: int, seed: int | None = None, site: str | None = None, keep_site: bool = False,
               max_tries: int = 20) -> pd.DataFrame:
        """``n`` rows from the global generator (optionally only rows labelled ``site``)."""
        if site is not None and not self.encoder.site_names:
            raise ValueError("Sampling for a site needs generator.site_in_model: true.")
        net, seed = self._net(), int(self.cfg.seed if seed is None else seed)
        parts, have = [], 0
        for t in range(max_tries):
            batch = self._postprocess(_generate(net, self.encoder, max(n - have, 1) * (2 if t else 1), seed + t))
            if site is not None:
                batch = batch[batch[SITE_COLUMN] == str(site)]
            parts.append(batch)
            have += len(batch)
            if have >= n:
                break
        out = pd.concat(parts, ignore_index=True).iloc[:n].reset_index(drop=True)
        if SITE_COLUMN in out.columns and not keep_site:
            out = out.drop(columns=[SITE_COLUMN])
        return out

    def sample_personalised(self, site: Site, n: int, seed: int | None = None, max_tries: int = 20) -> pd.DataFrame:
        """``n`` rows from ``site``'s personalised generator (sampled at the site)."""
        seed = int(self.cfg.seed if seed is None else seed)
        parts, have = [], 0
        for t in range(max_tries):
            raw = site.run(site_sample, max(n - have, 1) * (2 if t else 1), seed + t, self.arch,
                           label="synthetic_rows")
            batch = self._postprocess(raw)
            parts.append(batch)
            have += len(batch)
            if have >= n:
                break
        out = pd.concat(parts, ignore_index=True).iloc[:n].reset_index(drop=True)
        return out.drop(columns=[SITE_COLUMN]) if SITE_COLUMN in out.columns else out

    # ---- persistence -----------------------------------------------------------
    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state, out / "model.pt")
        with open(out / "encoder.pkl", "wb") as f:
            pickle.dump({"encoder": self.encoder, "arch": self.arch, "schema": self.schema,
                         "columns": self.columns, "censor_threshold": self.censor_threshold}, f)
        self.cfg.to_yaml(out / "federated_config_used.yaml")
        from ..privacy.assess import jsonable

        (out / "fit.json").write_text(json.dumps(jsonable({
            "rounds": self.record.rounds, "privacy": self.record.privacy, "fit_seconds": self.record.fit_seconds,
            "bytes_per_round": self.record.bytes_per_round, "synthesizer": vars(self.synth)}), indent=2))
        return out

    @classmethod
    def load(cls, model_dir: str | Path, synth_config: SynthesizerConfig | None = None) -> "FederatedTVAE":
        from .config import FederatedConfig

        d = Path(model_dir)
        cfg = FederatedConfig.from_yaml(d / "federated_config_used.yaml")
        with open(d / "encoder.pkl", "rb") as f:
            meta = pickle.load(f)
        obj = cls(synth_config, cfg, meta["schema"], meta["censor_threshold"])
        obj.encoder, obj.arch, obj.columns = meta["encoder"], meta["arch"], meta["columns"]
        obj.state = torch.load(d / "model.pt", weights_only=False)
        fit = json.loads((d / "fit.json").read_text())
        obj.record = FitRecord(fit.get("rounds", []), fit.get("privacy", {}), fit.get("fit_seconds", 0.0),
                               fit.get("bytes_per_round", 0.0))
        return obj


def fit_on_frame(df: pd.DataFrame, synth: SynthesizerConfig, cfg: FederatedConfig, schema=None,
                 columns: list[str] | None = None, name: str = "pooled", censor_threshold=None,
                 log=None, discrete: list[str] | None = None) -> FederatedTVAE:
    """The same trainer on one table (one site): the ``local`` / ``central`` baselines."""
    one = cfg.copy()
    one.generator.secure_aggregation = False
    one.generator.site_in_model = False
    site = Site(name, df, DisclosureLedger())
    return FederatedTVAE(synth, one, schema, censor_threshold).fit([site], columns, log=log, discrete=discrete)
