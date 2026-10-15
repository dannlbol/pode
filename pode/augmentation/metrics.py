"""Held-out utility metrics, computed on real patients only.

Reference points (for readers who do not use these daily):

* **C-index** (Harrell; Uno = censoring-weighted): probability the model ranks
  a random pair of patients in the right order. 0.5 = coin toss, 0.7 = useful
  prognostic score, 1 = perfect. A gain of 0.01-0.02 is small but real.
* **td-AUC@h**: like C, but only "dies before h" vs "survives past h".
* **Brier@h / IBS**: mean squared error of predicted survival (lower is better).
  **IPA** (= Brier skill) rescales it like R²: 0 = no better than the
  Kaplan-Meier average, 0.1-0.2 is typical for a good clinical model.
* **Calibration slope** (ideal 1): < 1 means predictions are too extreme
  (overfitting), > 1 too timid. **O/E** (ideal 1): observed / expected events
  by the horizon; > 1 means risk is under-predicted.
* **Net benefit**: decision-curve value of acting on the model at a risk
  threshold, in "true positives per patient" (higher is better).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def km(time: np.ndarray, event: np.ndarray, at) -> np.ndarray:
    """Kaplan-Meier survival at times ``at``."""
    time, event = np.asarray(time, float), np.asarray(event, float)
    uniq = np.unique(time[event == 1])
    if not len(uniq):  # no events: survival stays at 1
        return np.ones(len(np.atleast_1d(at)))
    at_risk = np.array([(time >= u).sum() for u in uniq], float)
    deaths = np.array([((time == u) & (event == 1)).sum() for u in uniq], float)
    s = np.cumprod(1 - deaths / np.maximum(at_risk, 1))
    idx = np.searchsorted(uniq, np.atleast_1d(at), side="right") - 1
    return np.where(idx >= 0, s[np.clip(idx, 0, None)], 1.0)


def harrell_c(time, event, risk) -> float:
    from sksurv.metrics import concordance_index_censored

    event = np.asarray(event).astype(bool)
    if event.sum() == 0:
        return np.nan
    return float(concordance_index_censored(event, np.asarray(time, float), np.asarray(risk, float))[0])


def _surv(time, event):
    from sksurv.util import Surv

    return Surv.from_arrays(np.asarray(event).astype(bool), np.asarray(time, float))


def _safe(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except (ValueError, ZeroDivisionError, FloatingPointError, IndexError):
        return np.nan


def _ibs_grid(test_time, horizon: float, n: int = 20) -> np.ndarray:
    hi = min(horizon, np.max(test_time) * 0.999)
    lo = max(np.min(test_time), 1e-6)
    return np.linspace(lo, hi, n) if hi > lo else np.array([])


def calibration_at(time, event, surv_h, horizon: float, groups: int = 5) -> dict:
    """Calibration of predicted event risk by ``horizon`` against KM observed."""
    from .models import BreslowCalibrator

    time, event, surv_h = (np.asarray(a, float) for a in (time, event, surv_h))
    risk = np.clip(1 - surv_h, 1e-6, 1 - 1e-6)
    observed = 1 - float(km(time, event, horizon)[0])
    out = {"oe_ratio": observed / risk.mean() if risk.mean() > 0 else np.nan,
           "mean_pred_risk": float(risk.mean()), "observed_risk": observed}
    # Slope: Cox coefficient of the complementary log-log of predicted risk.
    cll = np.log(-np.log(1 - risk))
    out["cal_slope"] = BreslowCalibrator().fit(cll, time, event).coef if event.sum() >= 3 else np.nan
    out["cal_slope_err"] = abs(out["cal_slope"] - 1)
    out["oe_log_err"] = abs(np.log(out["oe_ratio"])) if out["oe_ratio"] > 0 else np.nan
    # Mean |predicted - observed| over risk quantile groups (an ICI-style error).
    edges = np.unique(np.quantile(risk, np.linspace(0, 1, groups + 1)))
    gid = np.clip(np.digitize(risk, edges[1:-1]), 0, len(edges) - 2)
    errs = [abs(risk[gid == g].mean() - (1 - km(time[gid == g], event[gid == g], horizon)[0]))
            for g in np.unique(gid) if (gid == g).sum() >= 5]
    out["cal_abs_err"] = float(np.mean(errs)) if errs else np.nan
    return out


def net_benefit(time, event, risk_h, horizon: float, thresholds) -> dict:
    """Survival decision-curve net benefit (KM-based, Vickers et al. 2008)."""
    time, event, risk_h = (np.asarray(a, float) for a in (time, event, risk_h))
    out = {}
    for pt in thresholds:
        pos = risk_h >= pt
        if not pos.any():
            nb = 0.0
        else:
            event_rate = 1 - float(km(time[pos], event[pos], horizon)[0])
            frac = pos.mean()
            nb = frac * event_rate - frac * (1 - event_rate) * pt / (1 - pt)
        out[f"nb_{pt:g}"] = nb
    out["nb_mean"] = float(np.mean([out[f"nb_{pt:g}"] for pt in thresholds])) if thresholds else np.nan
    return out


def auprc_minority(time, event, risk_h, horizon: float) -> tuple[float, str]:
    """AUPRC for whichever landmark class (death by h / survival past h) is rarer."""
    from sklearn.metrics import average_precision_score

    time, event, risk_h = (np.asarray(a, float) for a in (time, event, risk_h))
    ok = (time >= horizon) | (event == 1)
    died = ((time <= horizon) & (event == 1))[ok]
    if died.all() or not died.any():
        return np.nan, "-"
    if died.mean() <= 0.5:
        return float(average_precision_score(died, risk_h[ok])), "death"
    return float(average_precision_score(~died, -risk_h[ok])), "survival"


def utility_metrics(
    time, event, risk, surv_h, surv_grid, grid, train_time, train_event, horizon: float, thresholds,
) -> dict:
    """All held-out metrics for one fitted model on one test set."""
    from sksurv.metrics import (brier_score, concordance_index_ipcw, cumulative_dynamic_auc,
                                integrated_brier_score)

    time, event, risk = (np.asarray(a, float) for a in (time, event, risk))
    tr, te = _surv(train_time, train_event), _surv(time, event)
    out = {"n_test": int(len(time)), "events_test": int(event.sum()), "c_harrell": harrell_c(time, event, risk)}
    tau = min(np.max(train_time), np.max(time)) * 0.999
    out["c_uno"] = _safe(lambda: concordance_index_ipcw(tr, te, risk, tau=tau)[0])
    out["auc_h"] = _safe(lambda: cumulative_dynamic_auc(tr, te, risk, [horizon])[0][0])
    if surv_h is not None:
        out["brier_h"] = _safe(lambda: brier_score(tr, te, surv_h[:, None], [horizon])[1][0])
        km_null = np.full((len(time), 1), km(train_time, train_event, horizon)[0])
        null = _safe(lambda: brier_score(tr, te, km_null, [horizon])[1][0])
        out["ipa_h"] = 1 - out["brier_h"] / null if null and np.isfinite(null) else np.nan
        out.update(calibration_at(time, event, surv_h, horizon))
        out.update(net_benefit(time, event, 1 - surv_h, horizon, thresholds))
        out["auprc_minority"], out["minority_class"] = auprc_minority(time, event, 1 - surv_h, horizon)
    else:
        out["auprc_minority"], out["minority_class"] = auprc_minority(time, event, risk, horizon)
    if surv_grid is not None and len(grid) >= 2:
        out["ibs"] = _safe(lambda: integrated_brier_score(tr, te, surv_grid, grid))
    return out


def subgroup_metrics(test: pd.DataFrame, time, event, risk, surv_h, columns: list[str],
                     horizon: float, min_n: int, min_events: int) -> list[dict]:
    """C-index and O/E within each level of each subgroup column (size-guarded)."""
    rows = []
    time, event, risk = (np.asarray(a, float) for a in (time, event, risk))
    for col in columns:
        if col not in test.columns:
            continue
        for level, idx in test.groupby(test[col].astype(str)).indices.items():
            n, ev = len(idx), int(event[idx].sum())
            if n < min_n or ev < min_events:
                continue
            row = {"subgroup": col, "level": level, "n": n, "events": ev,
                   "c_harrell": harrell_c(time[idx], event[idx], risk[idx])}
            if surv_h is not None:
                obs = 1 - km(time[idx], event[idx], horizon)[0]
                pred = float(np.mean(1 - surv_h[idx]))
                row["oe_ratio"] = obs / pred if pred > 0 else np.nan
            rows.append(row)
    return rows


# ---- paired inference ------------------------------------------------------------
def corrected_cv_ci(diffs: np.ndarray, n_train: float, n_test: float, alpha: float = 0.05) -> dict:
    """Mean and CI of per-split paired differences from repeated K-fold CV,
    with the Nadeau-Bengio correction for overlapping training sets."""
    from scipy.stats import t as student_t

    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    k = len(d)
    if k < 2:
        return {"mean": float(d.mean()) if k else np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_better": np.nan, "n_splits": k}
    var = d.var(ddof=1) * (1 / k + n_test / max(n_train, 1))
    se = np.sqrt(var)
    q = student_t.ppf(1 - alpha / 2, k - 1)
    mean = float(d.mean())
    p_better = float(student_t.cdf(mean / se, k - 1)) if se > 0 else float(mean > 0)
    return {"mean": mean, "ci_low": mean - q * se, "ci_high": mean + q * se,
            "p_better": p_better, "n_splits": k}


def paired_bootstrap(time, event, risk_a, risk_b, n_boot: int, seed: int, strata=None) -> dict:
    """Bootstrap of C(a) - C(b) on one test set, resampling patients (stratified)."""
    rng = np.random.default_rng(seed)
    time, event = np.asarray(time, float), np.asarray(event, float)
    groups = [np.arange(len(time))] if strata is None else [
        np.flatnonzero(np.asarray(strata) == s) for s in np.unique(strata)]
    diffs = []
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(g, len(g)) for g in groups])
        ca, cb = harrell_c(time[b], event[b], risk_a[b]), harrell_c(time[b], event[b], risk_b[b])
        if np.isfinite(ca) and np.isfinite(cb):
            diffs.append(ca - cb)
    d = np.asarray(diffs)
    if not len(d):
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_better": np.nan, "n_boot": 0}
    return {"mean": float(d.mean()), "ci_low": float(np.quantile(d, 0.025)),
            "ci_high": float(np.quantile(d, 0.975)), "p_better": float((d > 0).mean()), "n_boot": len(d)}
