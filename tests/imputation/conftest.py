"""Fixtures for the imputation tests: a small cohort with real structure.

The tests run on CPU: the server's GPU is shared, and another user's job holding
its memory would otherwise make TVAE tests fail with CUDA out-of-memory. Set
PODE_TEST_GPU=1 to exercise the GPU path instead.
"""

import os

if os.environ.get("PODE_TEST_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import pytest

from pode.schema.schema import DatasetSchema


def make_cohort() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    age = rng.uniform(0.5, 18.0, n)
    bio = 0.3 * age + rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "Age": age,
            "bio": bio,
            "Sex": rng.choice(["M", "F"], n),
            "treated": (bio > 2).astype(int),
            "stage": rng.integers(1, 5, n),
            "subgroup": np.where(age > 9, "G4", rng.choice(["G3", "SHH", "WNT"], n)),
            "lat_0": rng.normal(0, 1, n),
            "lat_1": rng.normal(0, 1, n),
            "OS_Time": rng.uniform(0.1, 12.0, n) * np.exp(-0.05 * bio),
            "OS_Status": rng.integers(0, 2, n),
        }
    )
    return df


@pytest.fixture
def cohort():
    return make_cohort()


@pytest.fixture(scope="module")
def module_cohort():
    return make_cohort()


@pytest.fixture
def cohort_schema(cohort):
    return DatasetSchema.from_dataframe(cohort)
