"""Prognostic metrics: Cox modelling, survival prediction, bootstrap CIs and
risk-group assignment. Python mirror of the R package's metrics_survival.R,
built on lifelines.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from ..config import Config


# --- formula / landmark helpers ---------------------------------------------

def _split_covariates(cfg, covariates):
    covariates = covariates if covariates is not None else cfg["columns"].get("covariates")
    if not covariates:
        raise ValueError(
            "fit_cox: no covariates supplied (set columns.covariates in the config).")
    model_terms, strata = [], []
    for t in covariates:
        if str(t).startswith("strata:"):
            strata.append(t.split("strata:", 1)[1])
        else:
            model_terms.append(t)
    return model_terms, strata


def _apply_landmark(data: pd.DataFrame, cfg, time_col: str) -> pd.DataFrame:
    off = cfg["survival"].get("landmark_offset")
    if off in (None, False):
        return data
    d = data[data[time_col].notna() & (data[time_col] > off)].copy()
    d[time_col] = d[time_col] - off
    return d


class CoxModel:
    """Thin wrapper around a fitted lifelines CoxPHFitter, plus the fit metadata."""

    def __init__(self, fitter, data, model_terms, strata, time_col, status_col):
        self.fitter = fitter
        self.data = data
        self.model_terms = model_terms
        self.strata = strata
        self.time_col = time_col
        self.status_col = status_col

    @property
    def params(self):
        return self.fitter.params_


def fit_cox(data: pd.DataFrame, cfg: Config, covariates: Optional[Sequence[str]] = None,
            time_col: Optional[str] = None, status_col: Optional[str] = None) -> CoxModel:
    """Fit a Cox proportional-hazards model (lifelines).

    Covariate terms prefixed ``strata:`` are entered as strata. The formula and
    columns default to the config unless overridden.
    """
    time_col = cfg.col("time", time_col)
    status_col = cfg.col("status", status_col)
    model_terms, strata = _split_covariates(cfg, covariates)

    d = _apply_landmark(data, cfg, time_col).copy()
    d[status_col] = d[status_col].astype(float)
    keep = [time_col, status_col] + model_terms + strata
    d = d.dropna(subset=keep)
    if len(d) < 10:
        raise ValueError("fit_cox: fewer than 10 complete cases.")

    fitter = CoxPHFitter()
    fit_kwargs = dict(duration_col=time_col, event_col=status_col,
                      formula=" + ".join(model_terms))
    if strata:
        fit_kwargs["strata"] = strata
    fitter.fit(d[keep], **fit_kwargs)
    return CoxModel(fitter, d, model_terms, strata, time_col, status_col)


def _as_model(fit) -> CoxModel:
    if isinstance(fit, CoxModel):
        return fit
    raise TypeError("Expected a CoxModel from fit_cox().")


def predict_survival_at_time(fit: CoxModel, newdata: pd.DataFrame,
                             time: Optional[float] = None,
                             cfg: Optional[Config] = None) -> np.ndarray:
    """Per-patient survival probability at ``time`` (default: the config horizon)."""
    model = _as_model(fit)
    if time is None:
        time = cfg.horizon if cfg is not None else 5
    sf = model.fitter.predict_survival_function(newdata, times=[time])
    return sf.iloc[0].to_numpy()


def bootstrap_survival_ci(data: pd.DataFrame, cfg: Config,
                          covariates: Optional[Sequence[str]] = None,
                          time_col: Optional[str] = None, status_col: Optional[str] = None,
                          n: Optional[int] = None, conf: Optional[float] = None,
                          seed: Optional[int] = None) -> pd.DataFrame:
    """Bootstrapped survival prediction with bias-corrected point estimates and
    percentile confidence intervals."""
    boot = cfg["survival"]["bootstrap"]
    n = n if n is not None else boot["n"]
    conf = conf if conf is not None else boot["conf"]
    seed = seed if seed is not None else boot["seed"]
    time = cfg.horizon

    model = fit_cox(data, cfg, covariates, time_col, status_col)
    d = model.data.reset_index(drop=True)
    pred0 = predict_survival_at_time(model, d, time=time)

    rng = np.random.default_rng(seed)
    boot_preds = np.full((n, len(d)), np.nan)
    keep = [model.time_col, model.status_col] + model.model_terms + model.strata
    for b in range(n):
        idx = rng.integers(0, len(d), len(d))
        sample = d.iloc[idx]
        try:
            cph = CoxPHFitter()
            kw = dict(duration_col=model.time_col, event_col=model.status_col,
                      formula=" + ".join(model.model_terms))
            if model.strata:
                kw["strata"] = model.strata
            cph.fit(sample[keep], **kw)
            boot_preds[b] = cph.predict_survival_function(d, times=[time]).iloc[0].to_numpy()
        except Exception:
            continue

    lo = np.nanpercentile(boot_preds, 100 * (1 - conf) / 2, axis=0)
    hi = np.nanpercentile(boot_preds, 100 * (1 - (1 - conf) / 2), axis=0)
    bias = np.nanmean(boot_preds, axis=0) - pred0

    id_col = cfg["columns"].get("id")
    ids = d[id_col].to_numpy() if id_col and id_col in d else np.arange(len(d))
    return pd.DataFrame({
        "id": ids,
        "pred_original": pred0,
        "pred_bias_corrected": np.clip(pred0 - bias, 0, 1),
        "ci_lower": np.clip(lo, 0, 1),
        "ci_upper": np.clip(hi, 0, 1),
    })


def _breslow_baseline_survival(durations, events, lp, t):
    """Breslow baseline survival S0(t) for a frozen linear predictor ``lp``."""
    order = np.argsort(durations)
    dur, ev, r = durations[order], events[order], np.exp(lp[order])
    H0 = 0.0
    for i in range(len(dur)):
        if dur[i] > t:
            break
        if ev[i] == 1:
            at_risk = r[i:].sum()
            if at_risk > 0:
                H0 += 1.0 / at_risk
    return np.exp(-H0)


def validate_frozen_model(data: pd.DataFrame, frozen_fit: CoxModel, cfg: Config,
                          conf: Optional[float] = None) -> pd.DataFrame:
    """Apply a frozen (already-fitted) model to a new cohort: point predictions
    with a frozen-coefficient, cohort-baseline bootstrap CI."""
    model = _as_model(frozen_fit)
    conf = conf if conf is not None else cfg["survival"]["bootstrap"]["conf"]
    time = cfg.horizon
    time_col = cfg.col("time")
    data = _apply_landmark(data, cfg, time_col).reset_index(drop=True)

    pred = predict_survival_at_time(model, data, time=time)

    # Frozen linear predictor on the validation cohort.
    lp = model.fitter.predict_log_partial_hazard(data).to_numpy()
    dur = data[time_col].to_numpy(float)
    ev = data[cfg.col("status")].astype(float).to_numpy()

    rng = np.random.default_rng(cfg["survival"]["bootstrap"]["seed"])
    B = 200
    s0_boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(data), len(data))
        s0_boot[b] = _breslow_baseline_survival(dur[idx], ev[idx], lp[idx], time)
    s0_lo = np.percentile(s0_boot, 100 * (1 - conf) / 2)
    s0_hi = np.percentile(s0_boot, 100 * (1 - (1 - conf) / 2))
    r = np.exp(lp)

    id_col = cfg["columns"].get("id")
    ids = data[id_col].to_numpy() if id_col and id_col in data else np.arange(len(data))
    return pd.DataFrame({
        "id": ids,
        "pred": np.clip(pred, 0, 1),
        "ci_lower": np.clip(s0_lo ** r, 0, 1),
        "ci_upper": np.clip(s0_hi ** r, 0, 1),
    })


def assign_risk_groups(pred_surv: Union[np.ndarray, Sequence[float]],
                       cfg: Optional[Config] = None,
                       thresholds: Optional[Sequence[float]] = None,
                       labels: Optional[Sequence[str]] = None) -> pd.Categorical:
    """Cut predicted survival into ordered, named risk tiers (descending thresholds)."""
    if thresholds is None:
        thresholds = cfg["risk_groups"]["thresholds"]
    if labels is None:
        labels = cfg["risk_groups"]["labels"]
    if len(labels) != len(thresholds) + 1:
        raise ValueError("assign_risk_groups: len(labels) must equal len(thresholds)+1.")
    pred = np.asarray(pred_surv, dtype=float)
    th = sorted(thresholds, reverse=True)
    # bins ascending: [-inf, th_last, ..., th_first, inf] -> labels reversed
    edges = [-np.inf] + list(reversed(th)) + [np.inf]
    codes = np.digitize(pred, edges[1:-1], right=False)  # 0..len(th)
    lab_by_code = list(reversed(labels))
    values = [lab_by_code[c] for c in codes]
    return pd.Categorical(values, categories=list(labels), ordered=True)
