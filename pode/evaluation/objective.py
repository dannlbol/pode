"""Configurable composite fidelity objective with a metric-plugin registry.

The composite score is a weighted sum of named *penalty* metrics (lower is
always better). Metrics are looked up in a registry, so the deferred biomarker
phase can ``register_metric('metagene_hr', ...)`` later and reference it purely
through :class:`~pode.config.ObjectiveConfig` weights - no core code changes.

A metric is a callable ``fn(real_df, synth_df, schema, **ctx) -> float`` where
``ctx`` currently carries ``timepoint``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..config import ObjectiveConfig
from ..schema.schema import DatasetSchema
from . import metrics as M

Metric = Callable[..., float]

_REGISTRY: dict[str, Metric] = {}


def register_metric(name: str, fn: Metric | None = None):
    """Register a penalty metric. Usable as a decorator or a direct call."""
    if fn is not None:
        _REGISTRY[name] = fn
        return fn

    def _decorator(func: Metric) -> Metric:
        _REGISTRY[name] = func
        return func

    return _decorator


def get_metric(name: str) -> Metric:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_metrics() -> list[str]:
    return sorted(_REGISTRY)


# --- built-in penalty metrics (lower = better) ------------------------------
def _survival_pair(schema: DatasetSchema):
    pair = schema.primary_survival_pair()
    return (pair.time_col, pair.event_col) if pair else (None, None)


@register_metric("hr_mae")
def _hr_mae(real_df, synth_df, schema, **ctx) -> float:
    return M.hazard_ratio_mae(real_df, synth_df, schema)


@register_metric("tvd_mean")
def _tvd_mean(real_df, synth_df, schema, **ctx) -> float:
    return M.mean_tvd(real_df, synth_df, schema)


@register_metric("surv_prob_diff_5yr")
def _surv_prob_diff(real_df, synth_df, schema, **ctx) -> float:
    time_col, event_col = _survival_pair(schema)
    if time_col is None:
        return np.nan
    _, diff = M.survival_agreement(
        real_df, synth_df, time_col, event_col, ctx.get("timepoint", 5.0)
    )
    return diff


@register_metric("logrank_dissim")
def _logrank_dissim(real_df, synth_df, schema, **ctx) -> float:
    """1 - log-rank p: small when curves are statistically indistinguishable."""
    time_col, event_col = _survival_pair(schema)
    if time_col is None:
        return np.nan
    p, _ = M.survival_agreement(
        real_df, synth_df, time_col, event_col, ctx.get("timepoint", 5.0)
    )
    return np.nan if np.isnan(p) else 1.0 - p


@register_metric("quality_penalty")
def _quality_penalty(real_df, synth_df, schema, **ctx) -> float:
    """1 - SDV quality score: small when overall fidelity is high."""
    score = M.sdv_quality_score(real_df, synth_df, schema)
    return np.nan if np.isnan(score) else 1.0 - score


def compute_composite_score(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    schema: DatasetSchema,
    config: ObjectiveConfig | None = None,
) -> dict:
    """Evaluate weighted penalty metrics and return scores + breakdown.

    Returns a dict with ``'composite'`` (weighted sum, NaN metrics skipped) and
    ``'metrics'`` (raw per-metric values).
    """
    config = config or ObjectiveConfig()
    ctx = {"timepoint": config.survival_timepoint}

    raw: dict[str, float] = {}
    weighted_sum = 0.0
    for name, weight in config.weights.items():
        if weight == 0:
            continue
        value = get_metric(name)(real_df, synth_df, schema, **ctx)
        raw[name] = value
        if not (value is None or (isinstance(value, float) and np.isnan(value))):
            weighted_sum += weight * value

    return {"composite": weighted_sum, "metrics": raw}
