"""Configuration objects for PODE.

All tunable constants that used to be scattered through the notebooks (censor
thresholds, TVAE hyperparameters, filter thresholds, objective weights) live
here as dataclasses. A whole run is described by a :class:`PodeConfig`, which
can be loaded from / saved to YAML so that each cohort becomes a config file
rather than a forked notebook.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SchemaConfig:
    """Controls automatic column-type detection.

    None of these are hardcoded column names baked into logic - they are
    defaults a user can override per dataset.
    """

    # A numeric column with <= this many distinct integer-valued entries is
    # treated as ordinal/categorical rather than continuous.
    max_cardinality_for_discrete: int = 20
    # Column-name suffixes used to pair survival time/event columns, e.g.
    # ("OS_Time", "OS_Status") -> survival pair "OS".
    survival_time_suffixes: list[str] = field(default_factory=lambda: ["_Time", "_time"])
    survival_event_suffixes: list[str] = field(
        default_factory=lambda: ["_Status", "_status", "_Event", "_event"]
    )
    # Explicit survival pairs for datasets that do not follow suffix conventions,
    # e.g. [{"time_col": "OS", "event_col": "OS.status"}].
    survival_pairs: list[dict[str, str]] = field(default_factory=list)
    # Name prefixes whose columns should always be treated as continuous floats
    # (e.g. methylation beta values "cg...").
    continuous_prefixes: list[str] = field(default_factory=list)
    # Identifier detection: columns that uniquely tag rows (Sample_ID, Basename,
    # sentrix.id, ...). These get the IDENTIFIER role and are dropped before
    # modelling so they neither bloat the synthesizer nor poison Cox models.
    detect_identifiers: bool = True
    identifier_unique_fraction: float = 0.95
    identifier_patterns: list[str] = field(
        default_factory=lambda: [
            "*_id",
            "*_ID",
            "ID",
            "Sample_ID",
            "*Basename*",
            "sentrix*",
            "Sentrix*",
        ]
    )
    # Explicit per-column type overrides: {column_name: ColumnType value}.
    type_overrides: dict[str, str] = field(default_factory=dict)
    # Explicit per-column role overrides: {column_name: ColumnRole value}.
    role_overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Data loading, cleaning and survival-censoring options."""

    # Glob-style name patterns of columns to drop entirely (e.g. "PFS_*").
    drop_patterns: list[str] = field(default_factory=list)
    # Explicit column names to drop.
    drop_columns: list[str] = field(default_factory=list)
    # Administrative censoring horizon (same units as survival time, e.g. years).
    censor_time_threshold: float | None = None
    # Multiply every detected survival-time column by this factor during data
    # preparation, e.g. 1 / 365.25 to convert days to years.
    survival_time_scale: float = 1.0
    # Rows missing all of these columns are dropped before modelling.
    require_columns: list[str] = field(default_factory=list)
    # Quality-based variable filtering.
    max_na_fraction: float = 1.0
    min_category_count: int = 0


@dataclass
class SynthesizerConfig:
    """TVAE hyperparameters and training options."""

    epochs: int = 1500
    batch_size: int = 500
    embedding_dim: int = 128
    compress_dims: tuple[int, ...] = (256, 256)
    decompress_dims: tuple[int, ...] = (256, 256)
    l2scale: float = 1e-6
    loss_factor: int = 2
    verbose: bool = True

    def tvae_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``TVAESynthesizer`` (excludes epochs/verbose)."""
        return {
            "batch_size": self.batch_size,
            "embedding_dim": self.embedding_dim,
            "compress_dims": tuple(self.compress_dims),
            "decompress_dims": tuple(self.decompress_dims),
            "l2scale": self.l2scale,
            "loss_factor": self.loss_factor,
        }


@dataclass
class ObjectiveConfig:
    """Weights for the configurable composite fidelity objective.

    Keys are metric names registered in :mod:`pode.evaluation.objective`.
    Lower composite score = better. Only metrics present here contribute.
    """

    # Metric names must match those registered in pode.evaluation.objective.
    # All registered metrics are penalties (lower = better); metrics with weight
    # 0 are skipped. ``logrank_dissim`` and ``quality_penalty`` are available too.
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "hr_mae": 1.0,
            "tvd_mean": 1.0,
            "surv_prob_diff_5yr": 1.0,
        }
    )
    # Time-point (in survival-time units) for the survival-probability metric.
    survival_timepoint: float = 5.0


@dataclass
class TuningConfig:
    """Grid-search / hyperparameter-tuning options."""

    epochs: int = 500
    stability_runs: int = 3
    param_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "batch_size": [250, 500],
            "embedding_dim": [64, 128],
            "compress_dims": [[128, 128], [256, 256]],
            "decompress_dims": [[128, 128], [256, 256]],
            "l2scale": [1e-5, 1e-6],
        }
    )


@dataclass
class AdjustmentConfig:
    """Survival hazard-ratio adjustment options."""

    enabled: bool = False
    method: str = "event_conversion"  # event_conversion | time_extension | hybrid
    timepoint: float = 5.0
    stratify_col: str | None = None


@dataclass
class PodeConfig:
    """Top-level configuration aggregating all sub-configs."""

    schema: SchemaConfig = field(default_factory=SchemaConfig)
    data: DataConfig = field(default_factory=DataConfig)
    synthesizer: SynthesizerConfig = field(default_factory=SynthesizerConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    adjustment: AdjustmentConfig = field(default_factory=AdjustmentConfig)

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PodeConfig":
        data = data or {}
        return cls(
            schema=SchemaConfig(**data.get("schema", {})),
            data=DataConfig(**data.get("data", {})),
            synthesizer=SynthesizerConfig(**data.get("synthesizer", {})),
            objective=ObjectiveConfig(**data.get("objective", {})),
            tuning=TuningConfig(**data.get("tuning", {})),
            adjustment=AdjustmentConfig(**data.get("adjustment", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PodeConfig":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def load_config(path: str | Path | None) -> PodeConfig:
    """Load a config from YAML, or return defaults when ``path`` is None."""
    if path is None:
        return PodeConfig()
    return PodeConfig.from_yaml(path)
