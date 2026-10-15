import numpy as np
import pandas as pd
import pytest
import prognostk as pk


def make_toy(n=150, seed=7):
    rng = np.random.default_rng(seed)
    score = np.clip(rng.beta(2, 2, n), 0.01, 0.99)
    metastasis = rng.binomial(1, 0.35, n)
    grp = rng.choice(["A", "B"], n)
    lp = -2.2 * (score - 0.5) + 0.5 * metastasis
    et = -np.log(rng.uniform(size=n)) / (0.06 * np.exp(lp))
    ct = rng.uniform(2, 14, n)
    return pd.DataFrame({
        "sample_id": [f"S{i:03d}" for i in range(n)],
        "OS_Time": np.round(np.minimum(et, ct), 2),
        "OS_Status": (et <= ct).astype(int),
        "metaCpG": np.round(score, 4),
        "metastasis": metastasis,
        "grp": grp,
        "Sex": rng.choice(["M", "F"], n),
    })


def toy_cfg(**overrides):
    base = dict(
        columns=dict(id="sample_id", time="OS_Time", status="OS_Status",
                     score="metaCpG", group="risk",
                     covariates=["metaCpG", "metastasis"],
                     categorical=["Sex", "grp"]))
    base.update(overrides)
    return pk.load_config(None, **base)


@pytest.fixture
def toy():
    return make_toy()


@pytest.fixture
def cfg():
    return toy_cfg()
