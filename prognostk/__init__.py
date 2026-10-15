"""prognostk: generic, configuration-driven prognostic metrics and figures.

The public API mirrors the R package: the same YAML config drives both.
"""
from .config import Config, load_config, validate_config
from .metrics.survival import (
    CoxModel, fit_cox, predict_survival_at_time, bootstrap_survival_ci,
    validate_frozen_model, assign_risk_groups,
)
from .metrics.cutoffs import optimise_cutoffs
from .metrics.calibration import calibration_table, external_validate
from .utils.theme import format_pvalue, prognostk_palette
from .plots.km import plot_km
from .plots.distribution import plot_distribution
from .plots.forest import plot_forest
from .plots.calibration import plot_calibration, plot_individual_curves
from .plots.oncoplot import plot_oncoplot

__version__ = "0.1.0"

__all__ = [
    "Config", "load_config", "validate_config",
    "CoxModel", "fit_cox", "predict_survival_at_time", "bootstrap_survival_ci",
    "validate_frozen_model", "assign_risk_groups", "optimise_cutoffs",
    "calibration_table", "external_validate",
    "format_pvalue", "prognostk_palette",
    "plot_km", "plot_distribution", "plot_forest", "plot_calibration",
    "plot_individual_curves", "plot_oncoplot",
]
