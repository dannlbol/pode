"""Fixtures for the augmentation tests: a small prognostic cohort with signal.

Tests run on CPU (the server's GPU is shared); set PODE_TEST_GPU=1 to use it.
"""

import os

if os.environ.get("PODE_TEST_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import pytest
import yaml


def make_cohort(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Survival depends on cg_1, cg_2, Metastases and age; three cohorts + an external one."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(0.2, 10.0, n)
    cg = rng.normal(0, 1, (n, 4))
    met = rng.choice(["No", "Yes"], n, p=[0.8, 0.2])
    lp = 0.8 * cg[:, 0] - 0.5 * cg[:, 1] + 0.9 * (met == "Yes") - 0.08 * age
    t_event = rng.exponential(30 * np.exp(-lp))
    t_cens = rng.uniform(5, 80, n)
    df = pd.DataFrame({
        "sample_id": [f"S{i:04d}" for i in range(n)],
        "Age": age.round(1),
        "AgeGroup": np.where(age < 1, "<1y", ">1y"),
        "Metastases": met,
        "Subgroup": rng.choice(["TYR", "SHH", "MYC"], n),
        **{f"cg_{j + 1}": cg[:, j] for j in range(4)},
        "Cohort": rng.choice(["A", "B", "C", "EXT"], n, p=[0.3, 0.3, 0.2, 0.2]),
        "OS_Time": np.maximum(np.minimum(t_event, t_cens), 0.5).round(1),
        "OS_Status": (t_event <= t_cens).astype(int),
    })
    return df


@pytest.fixture
def cohort():
    return make_cohort()


TINY_TVAE = {"synthesizer": {"epochs": 5, "batch_size": 100, "embedding_dim": 8,
                             "compress_dims": [32], "decompress_dims": [32]}}


def augment_config(**over) -> dict:
    cfg = {
        "n_jobs": 2,
        "device": "cpu",
        "id_column": "sample_id",
        "endpoint": {"horizon": 24},
        "splits": {"folds": 3, "repeats": 1, "stratify": ["Cohort"]},
        "external": {"column": "Cohort", "values": ["EXT"]},
        "generator": {"exclude_columns": ["sample_id", "Cohort"], "epochs": 5},
        "strata": ["Cohort"],
        "subgroups": ["Metastases"],
        "min_subgroup_n": 10,
        "min_subgroup_events": 3,
        "ratios": [0.5, 1.0],
        "draws": 2,
        "strategies": [
            "tvae_natural",
            {"kind": "tvae_targeted", "label": "tvae_mplus", "target": {"equals": {"Metastases": "Yes"}}},
            {"kind": "tvae_subgroup", "params": {"columns": ["Subgroup"]}},
            {"kind": "duplicate", "target": {"equals": {"Metastases": "Yes"}}},
            {"kind": "jitter", "params": {"sd_frac": 0.1}},
            {"kind": "weight", "target": {"equals": {"Metastases": "Yes"}}},
        ],
        "models": [
            {"kind": "metacpg_fixed", "features": ["cg_*"]},
            "coxnet",
            "cox_ph",
            "logistic_landmark",
        ],
        "learning_curve": {"fractions": [0.5, 0.75], "draws": 1},
        "bootstrap": 50,
    }
    cfg.update(over)
    return cfg


@pytest.fixture
def written(tmp_path, cohort):
    data = tmp_path / "cohort.csv"
    cohort.to_csv(data, index=False)
    pode_cfg = tmp_path / "pode.yaml"
    pode_cfg.write_text(yaml.safe_dump(TINY_TVAE))
    aug_cfg = tmp_path / "augment.yaml"
    aug_cfg.write_text(yaml.safe_dump(augment_config()))
    return data, pode_cfg, aug_cfg
