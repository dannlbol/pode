"""Fixtures for the federated tests: a small multi-site cohort with unequal sites,
site effects and a rare subtype, plus a tiny TVAE architecture.

Runs on CPU: the server's GPU is shared. Set PODE_TEST_GPU=1 to exercise the GPU path.
"""

import os

if os.environ.get("PODE_TEST_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import pytest

from pode.config import SynthesizerConfig
from pode.federated import FederatedConfig


def make_cohort(n: int = 300, seed: int = 0, sites=("big", "mid", "small"), p=(0.55, 0.3, 0.15)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    site = rng.choice(list(sites), n, p=list(p))
    shift = pd.Series(site).map({s: i * 0.5 for i, s in enumerate(sites)}).to_numpy()
    age = rng.uniform(0.5, 18.0, n)
    bio = 0.3 * age + shift + rng.normal(0, 1, n)
    risk = 0.6 * bio - 0.05 * age
    t = rng.exponential(np.exp(-risk) * 5)
    c = rng.uniform(0, 8, n)
    return pd.DataFrame({
        "Age": age.round(2),
        "bio": np.where(rng.random(n) < 0.1, np.nan, bio.round(3)),
        "Sex": rng.choice(["M", "F"], n),
        "subtype": rng.choice(["A", "B", "RARE"], n, p=[0.6, 0.37, 0.03]),
        "OS_Time": np.minimum(t, c).round(3) + 0.01,
        "OS_Status": (t <= c).astype(int),
        "site": site,
    })


@pytest.fixture
def cohort():
    return make_cohort()


@pytest.fixture
def tiny_synth():
    return SynthesizerConfig(compress_dims=(32, 32), decompress_dims=(32, 32), embedding_dim=16, batch_size=64)


@pytest.fixture
def fcfg():
    cfg = FederatedConfig()
    cfg.device = "cpu"
    cfg.n_jobs = 2
    cfg.generator.rounds = 3
    cfg.generator.local_epochs = 2
    cfg.transformer.pseudo_rows = 500
    return cfg
