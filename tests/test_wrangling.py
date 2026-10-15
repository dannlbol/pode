"""Tests for data wrangling and censoring."""

import numpy as np
import pandas as pd

from pode.config import SchemaConfig
from pode.data.censoring import apply_administrative_censoring
from pode.data.wrangling import (
    analyze_data_quality,
    collapse_categories,
    drop_columns_by_pattern,
    filter_variables_by_quality,
)
from pode.schema import DatasetSchema


def test_drop_columns_by_pattern(sample_df):
    df = sample_df.copy()
    df["PFS_Time"] = 1.0
    df["PFS_Status"] = 0
    out, dropped = drop_columns_by_pattern(df, patterns=["PFS_*"], columns=["biomarker"])
    assert "PFS_Time" in dropped and "PFS_Status" in dropped and "biomarker" in dropped
    assert "PFS_Time" not in out.columns
    assert "Age" in out.columns


def test_analyze_data_quality(df_with_missing):
    report = analyze_data_quality(df_with_missing)
    assert report.loc["biomarker", "na_fraction"] > 0.5
    assert report.loc["Age", "na_fraction"] == 0.0


def test_filter_drops_high_missingness_and_protects(df_with_missing):
    filtered, removed = filter_variables_by_quality(
        df_with_missing, max_na_fraction=0.5, protected_columns=["OS_Time", "OS_Status"]
    )
    assert "biomarker" in removed
    assert "biomarker" not in filtered.columns
    # Protected survival columns survive even if they had missingness.
    assert "OS_Time" in filtered.columns


def test_collapse_categories(sample_df):
    mapping = {"G3": "G3G4", "G4": "G3G4"}
    out = collapse_categories(sample_df, "subgroup", mapping)
    assert set(out["subgroup"].unique()) <= {"G3G4", "SHH", "WNT"}


def test_administrative_censoring():
    df = pd.DataFrame({"OS_Time": [2.0, 8.0, 12.0], "OS_Status": [1, 1, 1]})
    out = apply_administrative_censoring(df, "OS_Time", "OS_Status", threshold=10.0)
    # The event after the horizon is censored at the horizon.
    assert out.loc[2, "OS_Time"] == 10.0
    assert out.loc[2, "OS_Status"] == 0
    # Events before the horizon are untouched.
    assert out.loc[0, "OS_Status"] == 1


def test_censor_by_schema(sample_df):
    schema = DatasetSchema.from_dataframe(sample_df, SchemaConfig())
    from pode.data.censoring import censor_by_schema

    out = censor_by_schema(sample_df, schema, threshold=10.0)
    assert (out["OS_Time"] <= 10.0).all()
