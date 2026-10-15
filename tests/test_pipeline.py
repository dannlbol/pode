"""Tests for pipeline preparation and warning configuration (import-light)."""

import pandas as pd

from pode.config import PodeConfig
from pode.pipeline import prepare_data
from pode.utils.warnings import configure_warnings


def test_prepare_data_drops_identifiers(sample_df):
    config = PodeConfig()
    config.schema.continuous_prefixes = ["cg"]
    real, schema = prepare_data(sample_df, config)
    # The unique-id column is dropped before modelling.
    assert "Sample_ID" not in real.columns
    assert "Sample_ID" not in schema.columns
    # Real covariates survive.
    assert "Age" in real.columns
    assert "OS_Time" in real.columns


def test_prepare_data_keeps_identifiers_when_disabled(sample_df):
    config = PodeConfig()
    config.schema.detect_identifiers = False
    real, _ = prepare_data(sample_df, config)
    assert "Sample_ID" in real.columns


def test_prepare_data_scales_survival_times():
    config = PodeConfig()
    config.schema.survival_pairs = [{"time_col": "OS", "event_col": "OS.status"}]
    config.data.survival_time_scale = 1 / 365.25
    df = pd.DataFrame(
        {
            "USI": ["a", "b"],
            "OS": [365.25, 730.5],
            "OS.status": [1, 0],
            "Age": [10, 12],
        }
    )

    real, schema = prepare_data(df, config)

    assert real["OS"].tolist() == [1.0, 2.0]
    assert schema.primary_survival_pair().time_col == "OS"


def test_configure_warnings_runs():
    # Should not raise in either mode.
    configure_warnings(show=False)
    configure_warnings(show=True)
