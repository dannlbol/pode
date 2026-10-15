"""Kaplan-Meier estimator and a real-vs-synthetic curve distance (NumPy).

The KM comparison is where vanilla VAEs are documented to drift -- early follow-up
often matches while mid-to-late survival diverges -- so we report an *integrated
absolute difference* over a shared, dense time grid, and also the late-follow-up
slice explicitly.
"""
from __future__ import annotations

import numpy as np


def kaplan_meier(time: np.ndarray, event: np.ndarray):
    """Return (event_times, survival_prob) as right-continuous step values."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(int)
    order = np.argsort(time)
    time, event = time[order], event[order]

    uniq = np.unique(time)
    n = len(time)
    S = 1.0
    surv_times = []
    surv_probs = []
    at_risk = n
    i = 0
    for t in uniq:
        mask = time == t
        d = int(event[mask].sum())       # events at t
        n_t = at_risk
        if n_t > 0 and d > 0:
            S *= (1.0 - d / n_t)
        surv_times.append(t)
        surv_probs.append(S)
        at_risk -= int(mask.sum())
    return np.asarray(surv_times), np.asarray(surv_probs)


def _step_eval(surv_times, surv_probs, grid):
    """Evaluate a right-continuous KM step function on ``grid``."""
    idx = np.searchsorted(surv_times, grid, side="right") - 1
    out = np.ones_like(grid, dtype=float)
    valid = idx >= 0
    out[valid] = surv_probs[idx[valid]]
    return out


def km_distance(time_real, event_real, time_syn, event_syn, n_grid: int = 200,
                late_quantile: float = 0.5) -> dict:
    """Integrated absolute KM difference over a shared grid.

    Returns overall integrated difference and the late-follow-up integrated
    difference (over times beyond ``late_quantile`` of the real time range).
    """
    tr_t, tr_s = kaplan_meier(time_real, event_real)
    sy_t, sy_s = kaplan_meier(time_syn, event_syn)

    hi = min(tr_t.max(), sy_t.max()) if len(tr_t) and len(sy_t) else 0.0
    grid = np.linspace(0.0, hi, n_grid)
    fr = _step_eval(tr_t, tr_s, grid)
    fs = _step_eval(sy_t, sy_s, grid)
    diff = np.abs(fr - fs)

    overall = float(np.trapezoid(diff, grid) / max(hi, 1e-8))
    late_cut = late_quantile * hi
    late_mask = grid >= late_cut
    if late_mask.sum() > 1:
        late = float(
            np.trapezoid(diff[late_mask], grid[late_mask])
            / max(hi - late_cut, 1e-8)
        )
    else:
        late = float("nan")
    return {"km_iad": overall, "km_iad_late": late}
