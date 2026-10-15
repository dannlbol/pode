"""Data-driven cutoff optimisation (penalised grid search).
Python mirror of the R metrics_cutoffs.R.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from ..config import Config
from .survival import assign_risk_groups


def _km_surv_at(durations, events, at) -> float:
    if len(durations) < 2 or np.sum(events) == 0:
        return np.nan
    kmf = KaplanMeierFitter().fit(durations, events)
    return float(kmf.predict(at))


def _score_cutoffs(cuts, pred, dur, ev, at, fav_min_surv, min_group_frac,
                   hi_max_surv, labels):
    groups = assign_risk_groups(pred, thresholds=list(cuts), labels=labels)
    counts = pd.Series(groups).value_counts()
    fracs = counts / len(pred)
    if (fracs < min_group_frac).any():
        return -np.inf

    surv_by = {}
    g = np.asarray(groups)
    for lab in labels:
        m = g == lab
        surv_by[lab] = _km_surv_at(dur[m], ev[m], at) if m.sum() else np.nan

    fav = surv_by[labels[0]]
    worst = surv_by[labels[-1]]
    if np.isnan(fav) or fav < fav_min_surv:
        return -np.inf
    score = float(fracs.get(labels[0], 0.0)) * 100
    if hi_max_surv is not None and not np.isnan(worst) and worst > hi_max_surv:
        score -= (worst - hi_max_surv) * 300
    return score


def optimise_cutoffs(data: pd.DataFrame, pred: Union[np.ndarray, str], cfg: Config,
                     labels: Optional[Sequence[str]] = None,
                     fav_min_surv: float = 0.90, hi_max_surv: Optional[float] = 0.60,
                     min_group_frac: float = 0.02,
                     grid: Optional[Sequence[float]] = None) -> dict:
    """Penalised grid search for descending risk thresholds.

    Maximises the proportion in the top tier subject to that tier's KM survival at
    the horizon being >= ``fav_min_surv``, each group meeting ``min_group_frac``,
    and (softly) the worst tier being genuinely high-risk.
    """
    if isinstance(pred, str):
        pred = data[pred].to_numpy()
    pred = np.asarray(pred, float)
    labels = list(labels) if labels is not None else list(cfg["risk_groups"]["labels"])
    n_cuts = len(labels) - 1
    if n_cuts < 1:
        raise ValueError("optimise_cutoffs: need at least 2 tiers.")
    if grid is None:
        grid = np.round(np.arange(0.15, 0.96, 0.02), 4)

    dur = data[cfg.col("time")].to_numpy(float)
    ev = data[cfg.col("status")].astype(float).to_numpy()
    at = cfg.horizon
    grid_desc = sorted(set(np.round(grid, 6)), reverse=True)

    rows = []
    for cuts in combinations(grid_desc, n_cuts):  # already strictly descending
        sc = _score_cutoffs(cuts, pred, dur, ev, at, fav_min_surv,
                            min_group_frac, hi_max_surv, labels)
        if np.isfinite(sc):
            rows.append(tuple(cuts) + (sc,))
    if not rows:
        raise ValueError(
            "optimise_cutoffs: no feasible cutoff set found. Relax constraints "
            "(fav_min_surv / min_group_frac) or widen the grid.")

    cols = [f"cut{i+1}" for i in range(n_cuts)] + ["score"]
    results = pd.DataFrame(rows, columns=cols).sort_values(
        "score", ascending=False).reset_index(drop=True)
    best = results.loc[0, [f"cut{i+1}" for i in range(n_cuts)]].to_numpy(float)
    return {
        "best": best,
        "best_score": float(results.loc[0, "score"]),
        "groups": assign_risk_groups(pred, thresholds=list(best), labels=labels),
        "results": results,
    }
