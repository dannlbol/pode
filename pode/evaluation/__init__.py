"""Evaluation layer: fidelity metrics, configurable objective, and plots."""

from .metrics import (
    compute_hazard_ratios,
    hazard_ratio_mae,
    mean_tvd,
    sdv_quality_score,
    survival_agreement,
    total_variation_distance,
)
from .objective import (
    available_metrics,
    compute_composite_score,
    get_metric,
    register_metric,
)
from .plots import plot_correlation_comparison, plot_marginals
from .survival import plot_kaplan_meier, univariate_cox_table

__all__ = [
    "total_variation_distance",
    "mean_tvd",
    "compute_hazard_ratios",
    "hazard_ratio_mae",
    "survival_agreement",
    "sdv_quality_score",
    "register_metric",
    "get_metric",
    "available_metrics",
    "compute_composite_score",
    "plot_kaplan_meier",
    "univariate_cox_table",
    "plot_marginals",
    "plot_correlation_comparison",
]
