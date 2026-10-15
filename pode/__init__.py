"""PODE - Pediatric Oncology Discovery Engine.

First capability: data-agnostic synthetic clinical-data generation built on a
TVAE backend, with schema auto-detection, conditional sampling, survival
hazard-ratio adjustment, a configurable fidelity objective, and hyperparameter
tuning. Designed so later phases (biomarker discovery, feature selection) plug
into the same schema and metric-registry abstractions.
"""

from __future__ import annotations

from .config import PodeConfig, load_config
from .pipeline import build_schema, generate_synthetic, prepare_data
from .schema import ColumnRole, ColumnType, DatasetSchema
from .utils.warnings import configure_warnings

configure_warnings()

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "PodeConfig",
    "load_config",
    "DatasetSchema",
    "ColumnType",
    "ColumnRole",
    "build_schema",
    "prepare_data",
    "generate_synthetic",
]
