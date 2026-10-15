"""Survival hazard-ratio adjustment.

Ported from ``standalone_synthetic_generator_adjusted.py``. Lets synthetic
survival be nudged to match a target population's survival, either uniformly or
stratified by a risk group, using one of three methods.

Relationship used: ``S_target(t) = S_source(t) ** HR`` so
``HR = log(S_target) / log(S_source)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_hazard_adjustment_factor(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    time_col_src: str,
    event_col_src: str,
    time_col_tgt: str,
    event_col_tgt: str,
    timepoint: float = 5.0,
) -> tuple[float, float | None, float | None]:
    """Estimate the hazard ratio needed for ``source`` to match ``target``.

    Returns ``(adjustment_factor, survival_source, survival_target)``.
    ``< 1`` means better survival (fewer events); ``> 1`` means worse.
    """
    from lifelines import KaplanMeierFitter

    kmf_source, kmf_target = KaplanMeierFitter(), KaplanMeierFitter()
    df_src = df_source[[time_col_src, event_col_src]].dropna()
    df_tgt = df_target[[time_col_tgt, event_col_tgt]].dropna()
    kmf_source.fit(df_src[time_col_src], df_src[event_col_src])
    kmf_target.fit(df_tgt[time_col_tgt], df_tgt[event_col_tgt])

    try:
        surv_source = kmf_source.survival_function_at_times(timepoint).values[0]
        surv_target = kmf_target.survival_function_at_times(timepoint).values[0]
    except Exception:
        median_source = kmf_source.median_survival_time_
        median_target = kmf_target.median_survival_time_
        factor = median_source / median_target if median_target > 0 else 1.0
        return factor, None, None

    if 0 < surv_source < 1 and 0 < surv_target < 1:
        factor = np.log(surv_target) / np.log(surv_source)
    else:
        factor = 1.0
    return factor, float(surv_source), float(surv_target)


def calculate_stratified_hazard_adjustments(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    time_col_src: str,
    event_col_src: str,
    time_col_tgt: str,
    event_col_tgt: str,
    stratify_col: str,
    timepoint: float = 5.0,
    min_per_stratum: int = 10,
) -> tuple[dict, dict]:
    """Per-stratum hazard adjustments. Returns ``(hazard_by_stratum, details)``."""
    for col, frame in ((stratify_col, df_source), (stratify_col, df_target)):
        if col not in frame.columns:
            raise ValueError(f"Stratification column '{col}' not found")

    hazard_by_stratum: dict = {}
    details: dict = {}
    for stratum in sorted(df_source[stratify_col].dropna().unique()):
        src = df_source[df_source[stratify_col] == stratum]
        tgt = df_target[df_target[stratify_col] == stratum]
        if len(src) >= min_per_stratum and len(tgt) >= min_per_stratum:
            factor, surv_src, surv_tgt = estimate_hazard_adjustment_factor(
                src, tgt, time_col_src, event_col_src, time_col_tgt, event_col_tgt, timepoint
            )
            hazard_by_stratum[stratum] = factor
            details[stratum] = {
                "n_source": len(src),
                "n_target": len(tgt),
                "survival_source": surv_src,
                "survival_target": surv_tgt,
                "hazard_adjustment": factor,
            }
    return hazard_by_stratum, details


def _apply_event_conversion(df, mask_events, mask_censored, n_ev, n_cens, hazard, rng):
    """Bidirectional event<->censored conversion for one (sub)population."""
    if hazard < 1 and n_ev > 0:
        keep = rng.random(n_ev) < hazard
        return mask_events, keep.astype(int)
    if hazard > 1 and n_cens > 0:
        convert = rng.random(n_cens) < (1 - 1 / hazard)
        return mask_censored, convert.astype(int)
    return None, None


def adjust_survival_hazard(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    hazard_adjustment: float = 1.0,
    method: str = "event_conversion",
    seed: int = 42,
    stratify_col: str | None = None,
    hazard_by_stratum: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Adjust survival in ``df`` by modifying the hazard rate.

    ``method`` is one of ``'event_conversion'`` (recommended, bidirectional),
    ``'time_extension'`` or ``'hybrid'``. When ``stratify_col`` and
    ``hazard_by_stratum`` are given, adjustment is per-stratum; otherwise it is
    uniform using ``hazard_adjustment``. Returns ``(df_adjusted, stats)``.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    original_events = int(out[event_col].sum())

    def _apply_one(sub_mask, hazard):
        events_mask = sub_mask & (out[event_col] == 1)
        censored_mask = sub_mask & (out[event_col] == 0)
        n_ev, n_cens = int(events_mask.sum()), int(censored_mask.sum())

        if method == "event_conversion":
            target_mask, new_vals = _apply_event_conversion(
                out, events_mask, censored_mask, n_ev, n_cens, hazard, rng
            )
            if target_mask is not None:
                out.loc[target_mask, event_col] = new_vals
        elif method == "time_extension":
            out.loc[sub_mask, time_col] = out.loc[sub_mask, time_col] * (1 / hazard)
        elif method == "hybrid":
            out.loc[sub_mask, time_col] = out.loc[sub_mask, time_col] * np.sqrt(1 / hazard)
            if hazard < 1 and n_ev > 0:
                keep = rng.random(n_ev) < np.sqrt(hazard)
                out.loc[events_mask, event_col] = keep.astype(int)
            elif hazard > 1 and n_cens > 0:
                convert = rng.random(n_cens) < (1 - np.sqrt(1 / hazard))
                out.loc[censored_mask, event_col] = convert.astype(int)
        else:
            raise ValueError(f"Unknown adjustment method '{method}'")

    if stratify_col is not None and hazard_by_stratum is not None:
        if stratify_col not in out.columns:
            raise ValueError(f"Stratification column '{stratify_col}' not found")
        for stratum, hazard in hazard_by_stratum.items():
            _apply_one(out[stratify_col] == stratum, hazard)
        stratified = True
    else:
        _apply_one(pd.Series(True, index=out.index), hazard_adjustment)
        stratified = False

    adjusted_events = int(out[event_col].sum())
    stats = {
        "method": method,
        "stratified": stratified,
        "original_events": original_events,
        "adjusted_events": adjusted_events,
        "events_converted": adjusted_events - original_events,
        "n": len(out),
    }
    return out, stats
