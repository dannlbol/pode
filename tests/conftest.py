"""Shared test fixtures.

A small synthetic cohort covering every column type PODE detects, kept free of
SDV/torch so the schema/data/conditional tests run fast and offline.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame(
        {
            # identifier: unique string per row (should get the IDENTIFIER role)
            "Sample_ID": [f"S{i:04d}" for i in range(n)],
            # continuous: many distinct float values
            "Age": rng.uniform(0.5, 18.0, n),
            "biomarker": rng.normal(0, 1, n),
            # binary (object) and binary (numeric)
            "Sex": rng.choice(["M", "F"], n),
            "treated": rng.integers(0, 2, n),
            # ordinal: few distinct integers
            "stage": rng.integers(1, 5, n),
            # categorical: several string levels
            "subgroup": rng.choice(["G3", "G4", "SHH", "WNT"], n),
            # survival pair (OS_Time / OS_Status)
            "OS_Time": rng.uniform(0.1, 12.0, n),
            "OS_Status": rng.integers(0, 2, n),
            # methylation-style column forced continuous via prefix
            "cg_0001": rng.uniform(0, 1, n),
            # low-cardinality integer that becomes continuous only via prefix
            "cg_0002": rng.integers(0, 3, n),
        }
    )


@pytest.fixture
def df_with_missing(sample_df):
    df = sample_df.copy()
    # Make one column almost entirely missing.
    df.loc[df.index[10:], "biomarker"] = np.nan
    return df
