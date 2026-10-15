"""GAIN baseline (Yoon, Jordon & van der Schaar, 2018).

A generator fills missing components given the observed ones and the mask; a
discriminator, helped by a partial "hint" of the mask, tries to tell imputed
from observed components. Mixed types are handled by min-max scaling
continuous columns and one-hot encoding categoricals, with masks applied per
raw column. Observed training cells are additionally hidden at random so GAIN
can learn even when its training rows are complete.

Multiple draws come from different generator noise, and from sampling
categorical levels in proportion to the generator's outputs. GAIN is known to
under-represent uncertainty, which the benchmark's coverage metrics expose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..schema.schema import DatasetSchema
from .baselines import TabularCodec
from .base import Imputer
from .config import GainConfig


def _mlp(d_in: int, h: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d_out), nn.Sigmoid())


class GAINEngine(Imputer):
    name = "gain"
    multiple = True

    def __init__(self, cfg: GainConfig | None = None, fit_seed: int | None = None, device: str | None = None):
        super().__init__()
        self.cfg = cfg or GainConfig()
        self.fit_seed = fit_seed
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    # ---- encoding ------------------------------------------------------------
    def _encode(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.codec.encode(df)
        obs = ~np.isnan(raw)
        scaled = (raw - self.lo) / self.span
        x = torch.as_tensor(np.nan_to_num(scaled, nan=0.0), dtype=torch.float32, device=self.device)
        return x, torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def _col_level(self, b: int, rate: float, gen: torch.Generator | None = None) -> torch.Tensor:
        """Per-raw-column Bernoulli(rate) draws expanded to the encoded width."""
        draw = torch.rand((b, self.n_cols), generator=gen, device=self.device) < rate
        return draw[:, self.owner].float()

    # ---- training ------------------------------------------------------------
    def _fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "GAINEngine":
        if self.fit_seed is not None:
            torch.manual_seed(self.fit_seed)
            np.random.seed(self.fit_seed)
        self.schema = schema
        self.columns = [c for c in schema.columns if c in df.columns]
        ref = df[self.columns].reset_index(drop=True)
        self.codec = TabularCodec(schema, ref, onehot=True, standardise=False)
        raw = self.codec.encode(ref)
        self.lo = np.nan_to_num(np.nanmin(raw, axis=0), nan=0.0)
        hi = np.nan_to_num(np.nanmax(raw, axis=0), nan=1.0)
        self.span = np.where(hi - self.lo > 0, hi - self.lo, 1.0)

        owner = []
        for j, c in enumerate(self.codec.columns):
            owner += [j] * (1 if self.codec.numeric[c] else len(self.codec.categories[c]))
        self.owner = torch.as_tensor(owner, device=self.device)
        self.n_cols = len(self.codec.columns)
        self.cat_slices = []
        st = 0
        for c in self.codec.columns:
            w = 1 if self.codec.numeric[c] else len(self.codec.categories[c])
            if not self.codec.numeric[c]:
                self.cat_slices.append(slice(st, st + w))
            st += w

        x_all, m_all = self._encode(ref)
        n, d = x_all.shape
        h = self.cfg.hidden_dim or d
        self.G = _mlp(2 * d, h, d).to(self.device)
        self.D = _mlp(2 * d, h, d).to(self.device)
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.cfg.lr)
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.cfg.lr)
        bs = min(self.cfg.batch_size, n)
        eps = 1e-8
        for _ in range(self.cfg.epochs):
            idx = torch.randint(n, (bs,), device=self.device)
            xb, mb = x_all[idx], m_all[idx]
            mb = mb * (1 - self._col_level(bs, self.cfg.train_mask_rate))
            z = torch.rand_like(xb) * 0.01
            x_in = mb * xb + (1 - mb) * z
            b = self._col_level(bs, self.cfg.hint_rate)
            hint = mb * b + 0.5 * (1 - b)

            g = self.G(torch.cat([x_in, mb], 1))
            x_hat = mb * x_in + (1 - mb) * g
            d_prob = self.D(torch.cat([x_hat.detach(), hint], 1))
            loss_d = -torch.mean(mb * torch.log(d_prob + eps) + (1 - mb) * torch.log(1 - d_prob + eps))
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            d_prob = self.D(torch.cat([x_hat, hint], 1))
            loss_adv = -torch.mean((1 - mb) * torch.log(d_prob + eps))
            loss_rec = torch.sum((mb * xb - mb * g) ** 2) / torch.clamp(mb.sum(), min=1.0)
            loss_g = loss_adv + self.cfg.alpha * loss_rec
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
        return self

    # ---- imputation ----------------------------------------------------------
    @torch.no_grad()
    def _impute(self, df: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
        x, mask = self._encode(df)
        rng = np.random.default_rng(seed)
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        draws = []
        for _ in range(m):
            z = torch.rand(x.shape, generator=gen, device=self.device) * 0.01
            x_in = mask * x + (1 - mask) * z
            g = self.G(torch.cat([x_in, mask], 1))
            filled = (mask * x + (1 - mask) * g).cpu().numpy()
            for sl in self.cat_slices:
                p = np.clip(filled[:, sl], 1e-6, None)
                p = p / p.sum(1, keepdims=True)
                pick = (rng.random((p.shape[0], 1)) > np.cumsum(p, 1)).sum(1)
                pick = np.minimum(pick, p.shape[1] - 1)
                filled[:, sl] = 0.0
                filled[np.arange(p.shape[0]), sl.start + pick] = 1.0
            unscaled = filled * self.span + self.lo
            draws.append(self.codec.decode(unscaled, index=df.index))
        return draws
