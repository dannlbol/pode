"""Gaussian-mixture fallback synthesizer (plumbing stand-in only).

A dependency-light synthesizer used so the *pipeline* runs end to end on machines
without SDV/torch, and so the evaluation harness has something to consume in CI.
It fits a diagonal-ish full-covariance GMM on the numeric encoding of the table
and samples from it; categorical/boolean columns are label-encoded, sampled as
numeric, then rounded and clipped back to their observed support.

This is NOT a substitute for TVAE for real work -- it ignores mode structure in
continuous columns and models categoricals crudely. Use :class:`SdvTvaeSynthesizer`
for anything that matters. It exists purely to exercise the plumbing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .base import Synthesizer


class GaussianMixtureFallback(Synthesizer):
    def __init__(self, n_components: int = 10, random_state: int = 0):
        self.n_components = n_components
        self.random_state = random_state
        self._gmm = None
        self._columns = None
        self._discrete = {}   # col -> sorted unique values
        self._means = None
        self._stds = None

    def fit(self, table: pd.DataFrame, metadata: dict):
        self._columns = list(table.columns)
        M = np.zeros((len(table), len(self._columns)), dtype=float)
        for j, col in enumerate(self._columns):
            spec = metadata["columns"].get(col, {"sdtype": "numerical"})
            if spec["sdtype"] in ("categorical", "boolean"):
                vals = np.sort(pd.unique(table[col]))
                self._discrete[col] = vals
                lut = {v: i for i, v in enumerate(vals)}
                M[:, j] = table[col].map(lut).to_numpy(dtype=float)
            else:
                M[:, j] = table[col].to_numpy(dtype=float)
        self._means = M.mean(axis=0)
        self._stds = M.std(axis=0) + 1e-8
        Mn = (M - self._means) / self._stds
        k = min(self.n_components, max(1, len(table) // 10))
        self._gmm = GaussianMixture(
            n_components=k, covariance_type="full", random_state=self.random_state
        ).fit(Mn)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._gmm is None:
            raise RuntimeError("GaussianMixtureFallback is not fitted.")
        Sn, _ = self._gmm.sample(n)
        S = Sn * self._stds + self._means
        out = {}
        for j, col in enumerate(self._columns):
            if col in self._discrete:
                vals = self._discrete[col]
                idx = np.clip(np.round(S[:, j]).astype(int), 0, len(vals) - 1)
                out[col] = vals[idx]
            else:
                out[col] = S[:, j]
        df = pd.DataFrame(out, columns=self._columns)
        # survival time must be non-negative
        if "time" in df.columns:
            df["time"] = np.clip(df["time"], a_min=1e-6, a_max=None)
        return df
