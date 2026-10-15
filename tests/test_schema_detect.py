"""Tests for column-type detection and DatasetSchema."""

import pandas as pd

from pode.config import SchemaConfig
from pode.schema import ColumnRole, ColumnType, DatasetSchema
from pode.schema.detect import (
    detect_column_types,
    detect_identifiers,
    detect_survival_pairs,
)


def test_basic_type_detection(sample_df):
    config = SchemaConfig(continuous_prefixes=["cg"])
    types = detect_column_types(sample_df, config)

    assert types["Age"] == ColumnType.CONTINUOUS
    assert types["biomarker"] == ColumnType.CONTINUOUS
    assert types["Sex"] == ColumnType.BINARY
    assert types["treated"] == ColumnType.BINARY
    assert types["stage"] == ColumnType.ORDINAL
    assert types["subgroup"] == ColumnType.CATEGORICAL


def test_survival_pair_detected(sample_df):
    pairs = detect_survival_pairs(list(sample_df.columns), SchemaConfig())
    assert len(pairs) == 1
    assert pairs[0].time_col == "OS_Time"
    assert pairs[0].event_col == "OS_Status"

    types = detect_column_types(sample_df, SchemaConfig())
    assert types["OS_Time"] == ColumnType.SURVIVAL_TIME
    assert types["OS_Status"] == ColumnType.SURVIVAL_EVENT


def test_explicit_survival_pairs_detected():
    df = pd.DataFrame(
        {
            "OS": [1.0, 2.0],
            "OS.status": [0, 1],
            "Efs": [3.0, 4.0],
            "EFS.status": [1, 0],
            "DFS": [5.0, 6.0],
            "DFS.status": [0, 1],
            "Age": [7, 8],
        }
    )
    config = SchemaConfig(
        survival_pairs=[
            {"time_col": "OS", "event_col": "OS.status"},
            {"time_col": "Efs", "event_col": "EFS.status"},
            {"time_col": "DFS", "event_col": "DFS.status"},
        ]
    )

    pairs = detect_survival_pairs(list(df.columns), config)
    assert [(p.time_col, p.event_col) for p in pairs] == [
        ("OS", "OS.status"),
        ("Efs", "EFS.status"),
        ("DFS", "DFS.status"),
    ]

    types = detect_column_types(df, config)
    assert types["OS"] == ColumnType.SURVIVAL_TIME
    assert types["OS.status"] == ColumnType.SURVIVAL_EVENT
    assert types["Efs"] == ColumnType.SURVIVAL_TIME
    assert types["EFS.status"] == ColumnType.SURVIVAL_EVENT
    assert types["DFS"] == ColumnType.SURVIVAL_TIME
    assert types["DFS.status"] == ColumnType.SURVIVAL_EVENT


def test_continuous_prefix_overrides_cardinality(sample_df):
    """cg_0002 has only 3 integer values; the prefix forces it continuous."""
    no_prefix = detect_column_types(sample_df, SchemaConfig())
    assert no_prefix["cg_0002"] == ColumnType.ORDINAL

    with_prefix = detect_column_types(sample_df, SchemaConfig(continuous_prefixes=["cg"]))
    assert with_prefix["cg_0002"] == ColumnType.CONTINUOUS
    assert with_prefix["cg_0001"] == ColumnType.CONTINUOUS


def test_explicit_type_override(sample_df):
    config = SchemaConfig(type_overrides={"stage": "categorical"})
    types = detect_column_types(sample_df, config)
    assert types["stage"] == ColumnType.CATEGORICAL


def test_schema_roles_and_features(sample_df):
    schema = DatasetSchema.from_dataframe(sample_df, SchemaConfig(continuous_prefixes=["cg"]))
    # Survival columns get the SURVIVAL role and are excluded from features.
    assert schema.roles["OS_Time"] == ColumnRole.SURVIVAL
    assert "OS_Time" not in schema.feature_columns
    assert "OS_Status" not in schema.feature_columns
    assert "Age" in schema.feature_columns
    assert schema.primary_survival_pair().time_col == "OS_Time"


def test_schema_override_method(sample_df):
    schema = DatasetSchema.from_dataframe(sample_df)
    schema.override("subgroup", role=ColumnRole.TARGET)
    assert schema.roles["subgroup"] == ColumnRole.TARGET
    schema.override("stage", type=ColumnType.CATEGORICAL)
    assert schema.types["stage"] == ColumnType.CATEGORICAL


def test_detect_identifiers(sample_df):
    ids = detect_identifiers(sample_df, SchemaConfig())
    # Unique string column is an identifier; continuous Age is not.
    assert "Sample_ID" in ids
    assert "Age" not in ids
    assert "Sex" not in ids


def test_detect_identifiers_can_be_disabled(sample_df):
    ids = detect_identifiers(sample_df, SchemaConfig(detect_identifiers=False))
    assert ids == set()


def test_identifier_role_and_excluded_from_features(sample_df):
    schema = DatasetSchema.from_dataframe(sample_df, SchemaConfig(continuous_prefixes=["cg"]))
    assert schema.roles["Sample_ID"] == ColumnRole.IDENTIFIER
    assert "Sample_ID" not in schema.feature_columns


def test_schema_roundtrip_dict(sample_df):
    schema = DatasetSchema.from_dataframe(sample_df, SchemaConfig(continuous_prefixes=["cg"]))
    restored = DatasetSchema.from_dict(schema.to_dict())
    assert restored.types == schema.types
    assert restored.primary_survival_pair().event_col == "OS_Status"
