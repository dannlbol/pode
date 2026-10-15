"""Native Python implementation of the metaCpG workflow."""

from .core import (
    compare_results,
    fit_metacpg,
    read_config,
    run_pipeline,
    search_features,
    select_feature_count,
    submit_slurm,
    validate_inputs,
)

__all__ = [
    "read_config", "validate_inputs", "search_features",
    "select_feature_count", "fit_metacpg", "run_pipeline",
    "submit_slurm", "compare_results",
]

__version__ = "0.2.0"
