"""Generic, data-agnostic fidelity metrics.

These primitives replace the medulloblastoma-specific objective terms from the
grid-search notebooks with cohort-independent measures: total-variation
distance for discrete columns, Cox hazard-ratio preservation across
auto-detected covariates, Kaplan-Meier log-rank / survival-probability
agreement, and the SDV overall quality score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema.schema import DatasetSchema


def total_variation_distance(real: pd.Series, synth: pd.Series) -> float:
    """TVD between two categorical distributions (0 = identical, 1 = disjoint)."""
    real_p = real.dropna().value_counts(normalize=True)
    synth_p = synth.dropna().value_counts(normalize=True)
    categories = real_p.index.union(synth_p.index)
    real_p = real_p.reindex(categories, fill_value=0.0)
    synth_p = synth_p.reindex(categories, fill_value=0.0)
    return float(0.5 * np.abs(real_p - synth_p).sum())


def mean_tvd(real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: DatasetSchema) -> float:
    """Mean TVD across all discrete (non-survival-time) columns."""
    cols = [
        c
        for c in schema.categorical_columns
        if c in real_df.columns and c in synth_df.columns
    ]
    if not cols:
        return np.nan
    return float(np.mean([total_variation_distance(real_df[c], synth_df[c]) for c in cols]))


def compute_hazard_ratios(
    df: pd.DataFrame,
    covariates: list[str],
    schema: DatasetSchema,
    time_col: str,
    event_col: str,
    min_rows: int = 15,
) -> dict[str, float]:
    """Fit a univariate Cox model per covariate; return ``{covariate: HR}``.

    Categorical covariates are wrapped in a ``C(`col`)`` formula to tolerate
    special characters. Covariates that are constant or fail to converge are
    skipped.
    """
    import warnings

    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError

    hazard_ratios: dict[str, float] = {}
    # Degenerate covariates can emit convergence / overflow noise even when the
    # fit is simply discarded; silence it here since failures are handled.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for cov in covariates:
            if cov not in df.columns or df[cov].nunique(dropna=True) < 2:
                continue
            is_categorical = cov in schema.categorical_columns
            formula = f"C(`{cov}`)" if is_categorical else f"`{cov}`"
            sub = df[[cov, time_col, event_col]].dropna()
            if len(sub) < min_rows:
                continue
            try:
                cph = CoxPHFitter()
                cph.fit(sub, time_col, event_col, formula=formula)
                hr_series = cph.summary.loc[
                    cph.summary.index.str.contains(cov, regex=False), "exp(coef)"
                ]
                if not hr_series.empty:
                    hazard_ratios[cov] = float(hr_series.iloc[0])
            except (ConvergenceError, ValueError, np.linalg.LinAlgError, KeyError):
                continue
    return hazard_ratios


def hazard_ratio_mae(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    schema: DatasetSchema,
    time_col: str | None = None,
    event_col: str | None = None,
) -> float:
    """Mean absolute error between real and synthetic hazard ratios.

    Covariates are the schema's feature columns. Time/event default to the
    primary survival pair.
    """
    pair = schema.primary_survival_pair()
    if pair is None and (time_col is None or event_col is None):
        return np.nan
    time_col = time_col or pair.time_col
    event_col = event_col or pair.event_col

    covariates = [c for c in schema.feature_columns if c in real_df.columns]
    real_hr = compute_hazard_ratios(real_df, covariates, schema, time_col, event_col)
    synth_hr = compute_hazard_ratios(synth_df, list(real_hr.keys()), schema, time_col, event_col)

    diffs = [
        abs(real_hr[cov] - synth_hr[cov]) for cov in real_hr if cov in synth_hr
    ]
    return float(np.mean(diffs)) if diffs else np.nan


def survival_agreement(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    time_col: str,
    event_col: str,
    timepoint: float = 5.0,
) -> tuple[float, float]:
    """Return ``(logrank_p, abs_survival_prob_diff)`` at ``timepoint``."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    kmf_r, kmf_s = KaplanMeierFitter(), KaplanMeierFitter()
    try:
        kmf_r.fit(real_df[time_col], real_df[event_col])
        kmf_s.fit(synth_df[time_col], synth_df[event_col])
    except Exception:
        return np.nan, np.nan

    try:
        logrank_p = logrank_test(
            real_df[time_col], synth_df[time_col], real_df[event_col], synth_df[event_col]
        ).p_value
    except Exception:
        logrank_p = np.nan

    try:
        diff = abs(
            kmf_r.predict(timepoint, interpolate=True)
            - kmf_s.predict(timepoint, interpolate=True)
        )
    except Exception:
        diff = np.nan
    return float(logrank_p), float(diff)


def sdv_quality_score(
    real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: DatasetSchema
) -> float:
    """SDV overall quality score in [0, 1] (1 = best). NaN if uncomputable."""
    try:
        from sdmetrics.reports.single_table import QualityReport

        metadata = schema.to_sdv_metadata().to_dict()
        cols = [c for c in schema.columns if c in real_df.columns and c in synth_df.columns]
        report = QualityReport()
        report.generate(real_df[cols], synth_df[cols], metadata, verbose=False)
        return float(report.get_score())
    except Exception:
        return np.nan
