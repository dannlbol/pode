"""Fixtures for the privacy tests: one population, split into training / holdout,
plus a fresh sample from the same population (a synthetic release with no memorisation).

The TVAE audit test runs on CPU: the server's GPU is shared. Set PODE_TEST_GPU=1
to exercise the GPU path instead.
"""

import os

if os.environ.get("PODE_TEST_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import pytest

from pode.privacy import PrivacyConfig


def make_cohort(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(0.5, 18.0, n)
    return pd.DataFrame({
        "Age": age.round(2),
        "bio": (0.3 * age + rng.normal(0, 1, n)).round(3),
        "Sex": rng.choice(["M", "F"], n),
        "subtype": rng.choice(["A", "B", "C", "RARE"], n, p=[0.45, 0.35, 0.18, 0.02]),
        "stage": rng.integers(1, 5, n),
        "OS_Time": rng.uniform(0.1, 12.0, n).round(2),
        "OS_Status": rng.integers(0, 2, n),
    })


@pytest.fixture(scope="module")
def split():
    pop = make_cohort(600, 1)
    train = pop.iloc[:300].reset_index(drop=True)
    holdout = pop.iloc[300:450].reset_index(drop=True)
    fresh = make_cohort(300, 9)
    return train, holdout, fresh


@pytest.fixture
def pcfg():
    return PrivacyConfig(quasi_identifiers={"basic": ["Sex", "Age", "subtype"]}, sensitive=["OS_Status", "stage"],
                         strata=["subtype"], n_bootstrap=50, n_jobs=2)
