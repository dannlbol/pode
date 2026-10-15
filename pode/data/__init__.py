"""Data layer: I/O, wrangling, censoring and imputation."""

from .censoring import apply_administrative_censoring, censor_by_schema
from .imputation import impute_synthetic_nans
from .io import ensure_dir, load_dataset, save_dataset
from .wrangling import (
    analyze_data_quality,
    coerce_numeric,
    collapse_categories,
    drop_columns_by_pattern,
    filter_variables_by_quality,
)

__all__ = [
    "load_dataset",
    "save_dataset",
    "ensure_dir",
    "analyze_data_quality",
    "filter_variables_by_quality",
    "collapse_categories",
    "coerce_numeric",
    "drop_columns_by_pattern",
    "apply_administrative_censoring",
    "censor_by_schema",
    "impute_synthetic_nans",
]
