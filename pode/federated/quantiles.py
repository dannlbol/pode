"""Exact federated order statistics from counting queries.

MetaCPG scales every risk score by the median and interquartile range of the
training patients' raw scores, so a federated MetaCPG needs those quantiles
over patients spread across sites, without pooling the scores.

Protocol, for many targets at once (e.g. one per candidate feature count):

1. Each site sends the minimum and maximum of its values per target, which
   brackets every order statistic.
2. The coordinator places a grid of thresholds inside each bracket. Sites
   return only **how many** of their values are <= each threshold, and the
   bracket shrinks to the grid cell holding the wanted rank. A few rounds bring
   each bracket down to a handful of patients.
3. Sites return the distinct values (with multiplicities) inside the final
   tiny bracket. The coordinator merges them and reads off the exact k-th
   smallest value.

Steps 1 and 3 reveal individual values, but only order statistics (the
extremes and the values at the requested ranks, the same numbers the final
model publishes as its centre and scale). The ledger records them as such.
``np.median`` and ``np.percentile`` arithmetic is then applied to the exact
order statistics, so the results equal the pooled computation bit for bit.

Site values live in the site's private ``local[key]`` as a ``[targets, n_site]`` array.
"""

from __future__ import annotations

import numpy as np

from .sites import Site, stateful

GRID = 32
MAX_ROUNDS = 60
SMALL = 16


@stateful
def site_value_bounds(data, local: dict, key: str) -> np.ndarray:
    """[targets, 2] min and max of the site's values (NaN when the site has none)."""
    v = local[key]
    if v.shape[1] == 0:
        return np.full((v.shape[0], 2), np.nan)
    return np.stack([v.min(axis=1), v.max(axis=1)], axis=1)


@stateful
def site_count_le(data, local: dict, key: str, thresholds: np.ndarray) -> np.ndarray:
    """Counts of the site's values <= each threshold; thresholds are [targets, ranks, grid]."""
    v = local[key]
    if v.shape[1] == 0:
        return np.zeros(thresholds.shape, dtype=np.int64)
    s = np.sort(v, axis=1)
    out = np.empty(thresholds.shape, dtype=np.int64)
    for t in range(v.shape[0]):
        out[t] = np.searchsorted(s[t], thresholds[t], side="right")
    return out


@stateful
def site_values_in(data, local: dict, key: str, lo: np.ndarray, hi: np.ndarray) -> list:
    """Per (target, rank): distinct values in (lo, hi] with their multiplicities."""
    v = local[key]
    out = []
    for t in range(lo.shape[0]):
        row = []
        for k in range(lo.shape[1]):
            x = v[t][(v[t] > lo[t, k]) & (v[t] <= hi[t, k])]
            u, c = np.unique(x, return_counts=True)
            row.append((u, c))
        out.append(row)
    return out


def federated_order_stats(sites: list[Site], key: str, ranks: np.ndarray) -> np.ndarray:
    """Exact ``ranks``-th smallest values (0-based, [targets, k]) of the union of the sites' values."""
    ranks = np.asarray(ranks, dtype=np.int64)
    bounds = np.stack([s.run(site_value_bounds, key, label="quantile_bounds") for s in sites])  # [S, T, 2]
    lo_all = np.nanmin(bounds[:, :, 0], axis=0)
    hi_all = np.nanmax(bounds[:, :, 1], axis=0)
    T, K = ranks.shape
    # Bracket (lo, hi]: count(<= lo) <= rank < count(<= hi). lo starts just below the minimum.
    lo = np.repeat(np.nextafter(lo_all, -np.inf)[:, None], K, axis=1)
    hi = np.repeat(hi_all[:, None], K, axis=1)
    c_lo = np.zeros((T, K), dtype=np.int64)
    c_hi = None
    for _ in range(MAX_ROUNDS):
        if c_hi is not None and np.all(c_hi - c_lo <= SMALL):
            break
        frac = np.linspace(0, 1, GRID + 1)[1:-1]
        grid = lo[:, :, None] + (hi - lo)[:, :, None] * frac[None, None, :]
        grid = np.concatenate([grid, hi[:, :, None]], axis=2)
        counts = sum(s.run(site_count_le, key, grid, label="quantile_counts") for s in sites)  # [T, K, G]
        # First grid point whose count exceeds the rank: the value lies in (prev, that point].
        idx = np.argmax(counts > ranks[:, :, None], axis=2)
        new_hi = np.take_along_axis(grid, idx[:, :, None], 2)[:, :, 0]
        new_c_hi = np.take_along_axis(counts, idx[:, :, None], 2)[:, :, 0]
        prev = np.where(idx > 0, idx - 1, 0)
        new_lo = np.where(idx > 0, np.take_along_axis(grid, prev[:, :, None], 2)[:, :, 0], lo)
        new_c_lo = np.where(idx > 0, np.take_along_axis(counts, prev[:, :, None], 2)[:, :, 0], c_lo)
        stuck = (new_hi == hi) & (new_lo == lo)
        lo, hi, c_lo, c_hi = new_lo, new_hi, new_c_lo, new_c_hi
        if np.all(stuck | (c_hi - c_lo <= SMALL)):
            break
    parts = [s.run(site_values_in, key, lo, hi, label="quantile_order_values") for s in sites]
    out = np.empty((T, K))
    for t in range(T):
        for k in range(K):
            vals = np.concatenate([p[t][k][0] for p in parts])
            cnts = np.concatenate([p[t][k][1] for p in parts])
            order = np.argsort(vals, kind="stable")
            cum = np.cumsum(cnts[order])
            pos = int(ranks[t, k] - c_lo[t, k])
            out[t, k] = vals[order][int(np.searchsorted(cum, pos, side="right"))]
    return out


def _lerp(a: float, b: float, t: float) -> float:
    """numpy's percentile interpolation, operation for operation."""
    diff = b - a
    return float(b - diff * (1 - t)) if t >= 0.5 else float(a + diff * t)


def median_iqr_ranks(n: int) -> tuple[np.ndarray, list]:
    """Ranks needed for np.median and np.percentile(25/75) on ``n`` values, plus how to combine them."""
    ranks, recipe = [], []

    def need(r):
        r = int(min(max(r, 0), n - 1))
        if r not in ranks:
            ranks.append(r)
        return ranks.index(r)

    if n % 2:
        recipe.append(("median_odd", need((n - 1) // 2)))
    else:
        recipe.append(("median_even", need(n // 2 - 1), need(n // 2)))
    for p in (25, 75):
        virtual = (n - 1) * (np.true_divide(p, 100))
        k = int(np.floor(virtual))
        gamma = float(virtual - k)
        recipe.append(("pct", need(k), need(k + 1), gamma))
    return np.array(ranks), recipe


def median_iqr_from(values: np.ndarray, recipe: list) -> tuple[float, float]:
    """(median, p75 - p25) exactly as ``metacpg.core._legacy_scale`` computes them."""
    med = recipe[0]
    if med[0] == "median_odd":
        center = float(values[med[1]])
    else:
        center = float(np.mean([values[med[1]], values[med[2]]]))
    q25 = _lerp(values[recipe[1][1]], values[recipe[1][2]], recipe[1][3])
    q75 = _lerp(values[recipe[2][1]], values[recipe[2][2]], recipe[2][3])
    return center, float(q75 - q25)
