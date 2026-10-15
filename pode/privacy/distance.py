"""Mixed-type distances and record keys for the privacy checks.

* :class:`GowerSpace` encodes tables once (numeric columns scaled by their
  *training* range, categories mapped to shared integer codes) and computes a
  weighted, NaN-aware Gower distance: per column |a-b|/range (capped at 1) for
  numbers, 0/1 mismatch for categories, averaged with the configured weights
  over the columns both rows observe. A missing category is its own level.
* :meth:`GowerSpace.nearest` returns the k nearest rows of B for each row of A,
  computing the distance matrix one chunk of A at a time, so memory stays
  bounded however large the tables are.
* :func:`row_keys` / :class:`Binner` turn rows into hashable keys: exact
  (canonicalised whole records, for copy detection) or coarse (quantile-binned
  quasi-identifiers, for linkage and rare-combination checks).
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..imputation.base import category_key, n_workers

MISSING = "<NA>"


def _cat_strings(values) -> np.ndarray:
    keys = category_key(values)
    return np.array([MISSING if k is None else k for k in keys], dtype=object)


def column_weights(columns: list[str], weights: dict[str, float] | None) -> np.ndarray:
    """Per-column weights from a {glob: weight} map (first matching glob wins; default 1)."""
    out = np.ones(len(columns))
    for i, c in enumerate(columns):
        for pattern, w in (weights or {}).items():
            if fnmatch.fnmatchcase(c, pattern):
                out[i] = float(w)
                break
    return out


@dataclass
class Encoded:
    num: np.ndarray  # (n, p_num) range-scaled floats, NaN preserved
    cat: np.ndarray  # (n, p_cat) integer codes

    def __len__(self) -> int:
        return len(self.num)


class GowerSpace:
    """Gower distance over fixed numeric/categorical columns, ranged on a reference table."""

    def __init__(self, reference: pd.DataFrame, numeric: list[str], categorical: list[str],
                 weights: dict[str, float] | None = None):
        self.numeric, self.categorical = list(numeric), list(categorical)
        ref = reference[self.numeric].apply(pd.to_numeric, errors="coerce")
        rng = (ref.max() - ref.min()).to_numpy(float) if self.numeric else np.zeros(0)
        self.ranges = np.where(np.isfinite(rng) & (rng > 0), rng, 1.0)
        self.w_num = column_weights(self.numeric, weights)
        self.w_cat = column_weights(self.categorical, weights)
        self._vocab: list[dict[str, int]] = [{} for _ in self.categorical]

    def encode(self, df: pd.DataFrame) -> Encoded:
        num = np.empty((len(df), len(self.numeric)))
        for j, c in enumerate(self.numeric):
            num[:, j] = pd.to_numeric(df[c], errors="coerce").to_numpy(float) / self.ranges[j]
        cat = np.empty((len(df), len(self.categorical)), dtype=np.int64)
        for j, c in enumerate(self.categorical):
            vocab = self._vocab[j]
            cat[:, j] = [vocab.setdefault(v, len(vocab)) for v in _cat_strings(df[c].to_numpy())]
        return Encoded(num, cat)

    def block(self, a: Encoded, b: Encoded) -> np.ndarray:
        """Dense distance matrix between all rows of ``a`` and ``b``."""
        total = np.zeros((len(a), len(b)))
        weight = np.zeros((len(a), len(b)))
        for j, w in enumerate(self.w_num):
            if w == 0:
                continue
            d = np.minimum(np.abs(a.num[:, j, None] - b.num[None, :, j]), 1.0)
            ok = np.isfinite(d)
            total += w * np.where(ok, d, 0.0)
            weight += w * ok
        for j, w in enumerate(self.w_cat):
            if w == 0:
                continue
            total += w * (a.cat[:, j, None] != b.cat[None, :, j])
            weight += w
        return total / np.maximum(weight, 1e-12)

    def paired(self, a: Encoded, ia: np.ndarray, b: Encoded, ib: np.ndarray) -> np.ndarray:
        """Distance between row ``ia[i]`` of ``a`` and row ``ib[i]`` of ``b`` for each i."""
        total = np.zeros(len(ia))
        weight = np.zeros(len(ia))
        for j, w in enumerate(self.w_num):
            d = np.minimum(np.abs(a.num[ia, j] - b.num[ib, j]), 1.0)
            ok = np.isfinite(d)
            total += w * np.where(ok, d, 0.0)
            weight += w * ok
        for j, w in enumerate(self.w_cat):
            total += w * (a.cat[ia, j] != b.cat[ib, j])
            weight += w
        return total / np.maximum(weight, 1e-12)

    def nearest(self, a: Encoded, b: Encoded, k: int = 1, exclude_self: bool = False,
                n_jobs: int = 4, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Distances and indices of the ``k`` nearest rows of ``b`` for each row of ``a``.

        ``exclude_self`` (``a`` and ``b`` the same table) skips each row's own
        match: the leave-one-out nearest real neighbour. Missing neighbours (``b``
        smaller than ``k``) come back as distance inf, index -1.
        """
        from joblib import Parallel, delayed

        n_a, n_b = len(a), len(b)
        dist = np.full((n_a, k), np.inf)
        idx = np.full((n_a, k), -1, dtype=np.int64)
        if n_a == 0 or n_b == 0:
            return dist, idx

        def run(start: int):
            stop = min(start + chunk, n_a)
            sub = Encoded(a.num[start:stop], a.cat[start:stop])
            d = self.block(sub, b)
            if exclude_self:
                rows = np.arange(stop - start)
                d[rows, rows + start] = np.inf
            kk = min(k, n_b)
            part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
            pd_ = np.take_along_axis(d, part, axis=1)
            order = np.argsort(pd_, axis=1, kind="stable")
            return start, np.take_along_axis(pd_, order, axis=1), np.take_along_axis(part, order, axis=1)

        starts = range(0, n_a, chunk)
        workers = min(n_workers(n_jobs), len(starts))
        if workers > 1:
            results = Parallel(n_jobs=workers, prefer="threads")(delayed(run)(s) for s in starts)
        else:
            results = [run(s) for s in starts]
        for start, d, i in results:
            dist[start:start + len(d), :d.shape[1]] = d
            idx[start:start + len(d), :i.shape[1]] = i
        if exclude_self:  # a 1-row table has no other neighbour
            idx[~np.isfinite(dist)] = -1
        return dist, idx


def row_keys(df: pd.DataFrame, numeric: list[str], categorical: list[str], decimals: int = 6) -> np.ndarray:
    """Canonical whole-record keys: rounded numbers, normalised categories, one NaN code."""
    parts = []
    for c in numeric:
        v = pd.to_numeric(df[c], errors="coerce").round(decimals).to_numpy(float)
        parts.append(np.array([MISSING if not np.isfinite(x) else repr(float(x) + 0.0) for x in v], dtype=object))
    for c in categorical:
        parts.append(_cat_strings(df[c].to_numpy()))
    return _join(parts, len(df))


def _join(parts: list[np.ndarray], n: int) -> np.ndarray:
    if not parts:
        return np.full(n, "", dtype=object)
    return np.array(["\x1f".join(row) for row in zip(*parts)], dtype=object)


class Binner:
    """Coarse keys over quasi-identifier columns: numeric values in training-quantile bins."""

    def __init__(self, reference: pd.DataFrame, columns: list[str], numeric: set[str], bins: int = 5):
        self.columns = list(columns)
        self.edges: dict[str, np.ndarray] = {}
        for c in self.columns:
            if c in numeric:
                v = pd.to_numeric(reference[c], errors="coerce").dropna().to_numpy(float)
                q = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1))) if len(v) else np.zeros(0)
                self.edges[c] = q[1:-1]

    def labels(self, df: pd.DataFrame, column: str) -> np.ndarray:
        if column in self.edges:
            v = pd.to_numeric(df[column], errors="coerce").to_numpy(float)
            b = np.digitize(v, self.edges[column])
            return np.array([MISSING if not np.isfinite(x) else f"bin{int(i)}" for x, i in zip(v, b)], dtype=object)
        return _cat_strings(df[column].to_numpy())

    def keys(self, df: pd.DataFrame) -> np.ndarray:
        return _join([self.labels(df, c) for c in self.columns], len(df))

    def describe(self, column: str, label: str) -> str:
        """Human-readable range for a numeric bin label (``bin2`` -> ``[3.1, 7.4)``)."""
        if column not in self.edges or not label.startswith("bin"):
            return label
        e, i = self.edges[column], int(label[3:])
        lo = f"{e[i - 1]:.4g}" if i > 0 else "-inf"
        hi = f"{e[i]:.4g}" if i < len(e) else "inf"
        return f"[{lo}, {hi})"
