"""Calibration and external-validation metrics.
Python mirror of the R metrics_calibration.R.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

from ..config import Config
from .survival import CoxModel, _as_model, predict_survival_at_time
from .cutoffs import _km_surv_at


def calibration_table(data: pd.DataFrame, pred: Union[np.ndarray, str], cfg: Config,
                      n_bins: int = 10) -> dict:
    """Predicted vs observed (KM) survival by risk bin, plus the E/O ratio."""
    if isinstance(pred, str):
        pred = data[pred].to_numpy()
    pred = np.asarray(pred, float)
    time_col, status_col = cfg.col("time"), cfg.col("status")
    at = cfg.horizon

    dur = data[time_col].to_numpy(float)
    ev = data[status_col].astype(float).to_numpy()
    ok = np.isfinite(pred) & ~np.isnan(dur) & ~np.isnan(ev)
    pred, dur, ev = pred[ok], dur[ok], ev[ok]

    n_bins = max(2, min(n_bins, len(pred) // 15))
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)))
    binid = np.clip(np.digitize(pred, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for b in sorted(np.unique(binid)):
        m = binid == b
        kmf = KaplanMeierFitter().fit(dur[m], ev[m])
        obs = float(kmf.predict(at))
        ci = kmf.confidence_interval_survival_function_
        # nearest row at/after `at`
        try:
            lo = float(ci.iloc[(ci.index >= at).argmax(), 0])
            hi = float(ci.iloc[(ci.index >= at).argmax(), 1])
        except Exception:
            lo = hi = np.nan
        rows.append({"bin": int(b), "n": int(m.sum()),
                     "mean_pred": float(pred[m].mean()), "obs_km": obs,
                     "obs_lower": lo, "obs_upper": hi})
    tab = pd.DataFrame(rows)
    overall_obs = _km_surv_at(dur, ev, at)
    return {"table": tab, "eo_ratio": float(pred.mean() / overall_obs)}


def external_validate(valid_data: pd.DataFrame, frozen_fit: CoxModel, cfg: Config) -> dict:
    """External validation of a frozen model: C-index (with CI), calibration slope
    and E/O ratio at the horizon."""
    model = _as_model(frozen_fit)
    time_col, status_col = cfg.col("time"), cfg.col("status")
    off = cfg["survival"].get("landmark_offset")
    d = valid_data
    if off:
        d = d[d[time_col] > off].copy()
        d[time_col] = d[time_col] - off

    dur = d[time_col].to_numpy(float)
    ev = d[status_col].astype(float).to_numpy()
    lp = model.fitter.predict_log_partial_hazard(d).to_numpy()
    ok = np.isfinite(lp) & ~np.isnan(dur) & ~np.isnan(ev)
    dur, ev, lp = dur[ok], ev[ok], lp[ok]

    # Discrimination: higher lp = higher risk -> use -lp for concordance.
    cidx = concordance_index(dur, -lp, ev)
    # Approx SE via Noether formula on concordant/discordant pairs is complex;
    # use a simple bootstrap for the CI.
    rng = np.random.default_rng(cfg["survival"]["bootstrap"]["seed"])
    boot = []
    for _ in range(200):
        idx = rng.integers(0, len(dur), len(dur))
        try:
            boot.append(concordance_index(dur[idx], -lp[idx], ev[idx]))
        except Exception:
            pass
    conf = cfg["survival"]["bootstrap"]["conf"]
    lo, hi = np.percentile(boot, [100 * (1 - conf) / 2, 100 * (1 - (1 - conf) / 2)])

    # Calibration slope: Cox of outcome on lp (ideal slope = 1).
    from lifelines import CoxPHFitter
    cal = pd.DataFrame({"t": dur, "e": ev, "lp": lp})
    try:
        cs = CoxPHFitter().fit(cal, duration_col="t", event_col="e", formula="lp")
        cal_slope = float(cs.params_["lp"])
    except Exception:
        cal_slope = np.nan

    pred = predict_survival_at_time(model, d.iloc[np.where(ok)[0]], time=cfg.horizon)
    eo = float(np.nanmean(pred) / _km_surv_at(dur, ev, cfg.horizon))

    return {
        "n": int(len(dur)), "events": int(ev.sum()),
        "c_index": float(cidx), "c_index_lower": float(lo), "c_index_upper": float(hi),
        "calibration_slope": cal_slope, "eo_ratio": eo,
    }
