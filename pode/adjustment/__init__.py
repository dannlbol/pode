"""Adjustment layer: survival hazard-ratio matching to target populations."""

from .survival import (
    adjust_survival_hazard,
    calculate_stratified_hazard_adjustments,
    estimate_hazard_adjustment_factor,
)

__all__ = [
    "estimate_hazard_adjustment_factor",
    "calculate_stratified_hazard_adjustments",
    "adjust_survival_hazard",
]
