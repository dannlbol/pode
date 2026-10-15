"""High-level orchestration tying the PODE stages together.

These functions are the programmatic API the CLI is built on: prepare data,
build a schema, train + sample, and evaluate. Each stage is data-agnostic and
driven by :class:`~pode.config.PodeConfig`.
"""

from __future__ import annotations

import pandas as pd

from .config import PodeConfig
from .data.censoring import censor_by_schema
from .data.imputation import impute_synthetic_nans
from .data.wrangling import drop_columns_by_pattern, filter_variables_by_quality
from .generation.conditional import ConditionGroup
from .generation.synthesizer import TVAEGenerator
from .schema.schema import DatasetSchema


def build_schema(df: pd.DataFrame, config: PodeConfig) -> DatasetSchema:
    """Auto-detect a schema honouring the config's schema overrides."""
    return DatasetSchema.from_dataframe(df, config.schema)


def prepare_data(
    df: pd.DataFrame, config: PodeConfig, schema: DatasetSchema | None = None
) -> tuple[pd.DataFrame, DatasetSchema]:
    """Clean and censor raw data, returning ``(prepared_df, schema)``.

    Steps: drop unwanted columns -> build/refresh schema -> quality filter
    (protecting survival columns) -> administrative censoring -> drop rows
    missing required columns.
    """
    from .schema.column_types import ColumnRole
    from .schema.detect import detect_identifiers

    data_cfg = config.data

    df, _ = drop_columns_by_pattern(df, data_cfg.drop_patterns, data_cfg.drop_columns)

    # Drop identifier columns (unique row tags) before modelling: they neither
    # belong in the synthesizer nor in survival analysis. Union auto-detected
    # identifiers with any IDENTIFIER-role columns from a caller-supplied schema.
    id_cols = set(detect_identifiers(df, config.schema))
    if schema is not None:
        id_cols.update(schema.columns_of_role(ColumnRole.IDENTIFIER))
    id_cols = {c for c in id_cols if c in df.columns}
    if id_cols:
        df = df.drop(columns=list(id_cols))
        schema = None  # rebuild below now that identifiers are gone

    if schema is None:
        schema = build_schema(df, config)

    if data_cfg.survival_time_scale != 1.0:
        from .schema.column_types import ColumnType

        for col in schema.columns_of_type(ColumnType.SURVIVAL_TIME):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * data_cfg.survival_time_scale

    # Protect survival time/event columns from quality filtering.
    from .schema.column_types import ColumnType

    protected = schema.columns_of_type(ColumnType.SURVIVAL_TIME, ColumnType.SURVIVAL_EVENT)
    df, removed = filter_variables_by_quality(
        df,
        max_na_fraction=data_cfg.max_na_fraction,
        min_category_count=data_cfg.min_category_count,
        protected_columns=protected,
        manual_exclude=data_cfg.drop_columns,
    )
    if removed:
        # Rebuild schema to drop references to removed columns.
        schema = DatasetSchema.from_dataframe(df, config.schema)

    if data_cfg.censor_time_threshold is not None:
        df = censor_by_schema(df, schema, data_cfg.censor_time_threshold)

    require = list(data_cfg.require_columns)
    if not require:
        pair = schema.primary_survival_pair()
        if pair is not None:
            require = [pair.time_col, pair.event_col]
    require = [c for c in require if c in df.columns]
    if require:
        df = df.dropna(subset=require).reset_index(drop=True)

    return df, schema


def generate_synthetic(
    real_data: pd.DataFrame,
    config: PodeConfig,
    num_samples: int,
    schema: DatasetSchema | None = None,
    conditions: str | list[ConditionGroup] | None = None,
    seed: int | None = None,
    impute: bool = False,
) -> tuple[pd.DataFrame, TVAEGenerator]:
    """Train a TVAE on ``real_data`` and sample synthetic rows.

    Returns ``(synthetic_df, fitted_generator)``.
    """
    if schema is None:
        schema = build_schema(real_data, config)

    generator = TVAEGenerator(schema, config.synthesizer).fit(real_data, seed=seed)
    synthetic = generator.sample(
        num_samples,
        conditions=conditions,
        seed=seed,
        censor_threshold=config.data.censor_time_threshold,
    )
    if impute:
        synthetic = impute_synthetic_nans(synthetic, real_data, schema)
    return synthetic, generator
