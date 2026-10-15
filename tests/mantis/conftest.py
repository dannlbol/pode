import numpy as np
import pandas as pd
import pytest
import yaml

import pode_mantis as pm

N_SAMPLES = 12
N_CPGS = 40


def make_toy_beta(n_samples=N_SAMPLES, n_cpgs=N_CPGS, seed=7):
    rng = np.random.default_rng(seed)
    samples = [f"S{i:03d}" for i in range(n_samples)]
    cpgs = [f"cg{i:05d}" for i in range(n_cpgs)]
    beta = rng.uniform(0.05, 0.95, size=(n_samples, n_cpgs))
    beta[rng.random((n_samples, n_cpgs)) < 0.2] = np.nan
    df = pd.DataFrame(beta, columns=cpgs)
    df.insert(0, "sample_id", samples)
    return df, samples, cpgs


def make_toy_clinical(samples, seed=7):
    rng = np.random.default_rng(seed)
    n = len(samples)
    m = rng.integers(0, 2, n).astype(float)
    return pd.DataFrame(
        {
            "pid": samples,
            "t": rng.uniform(0.5, 10, n),
            "e": rng.integers(0, 2, n),
            "pfs_t": rng.uniform(0.5, 8, n),
            "pfs_e": rng.integers(0, 2, n),
            "met": m,
        }
    )


def n_cpgs_train(cpgs):
    return int(len(cpgs) * 0.75)


@pytest.fixture
def toy_paths(tmp_path):
    beta_df, samples, cpgs = make_toy_beta()
    clinical_df = make_toy_clinical(samples)

    cpg_order_path = tmp_path / "cpg_order.txt"
    cpg_order_path.write_text("\n".join(cpgs) + "\n")

    beta_path = tmp_path / "beta.csv"
    beta_df.to_csv(beta_path, index=False)

    train_means_path = tmp_path / "training_means.csv"
    pd.DataFrame({"probe_id": cpgs[: n_cpgs_train(cpgs)], "mean_beta": np.linspace(0.2, 0.8, n_cpgs_train(cpgs))}).to_csv(
        train_means_path, index=False
    )

    clinical_path = tmp_path / "clinical.csv"
    clinical_df.to_csv(clinical_path, index=False)

    reference_config = {
        "data": {
            "conn_mat_path": "data/conn_mat_beta.csv",
            "conn_mat_is_one_based": True,
            "time_bins": [float(x) for x in np.linspace(0.1, 20, 15)],
            "toy_train_surv_path": "data/surv_toy_train.csv",
        },
        "run_setup": {
            "training_cohorts": ["toy_train"],
            "feature_types": ["beta", "cnv"],
            "metastasis_recoding": {"from": 0, "to": -1},
        },
        "model": {
            "part1": {"input_dim": 90, "layer_dims": [16, 8, 4, 2], "dropout_rate": 0.1, "c2_input_offset": 40},
            "part2": {"num_time_bins": 15, "dropout_rate": 0.1},
            "combined": {"num_clinical_features": 1, "clinical_feature_weight": -4.4},
        },
    }
    reference_config_path = tmp_path / "reference_config.yaml"
    with open(reference_config_path, "w") as f:
        yaml.safe_dump(reference_config, f)

    return {
        "tmp_path": tmp_path,
        "beta_path": beta_path,
        "cpg_order_path": cpg_order_path,
        "train_means_path": train_means_path,
        "clinical_path": clinical_path,
        "reference_config_path": reference_config_path,
        "samples": samples,
        "out_dir": tmp_path / "out",
    }


def toy_cfg(toy_paths, **overrides):
    base = dict(
        beta={
            "cohort_path": str(toy_paths["beta_path"]),
            "orientation": "samples_by_features",
            "cpg_order_path": str(toy_paths["cpg_order_path"]),
            "training_means_path": str(toy_paths["train_means_path"]),
            "output_dir": str(toy_paths["out_dir"] / "beta"),
        },
        survival={
            "clinical_path": str(toy_paths["clinical_path"]),
            "id_column": "pid",
            "time_column": "t",
            "event_column": "e",
            "pfs_time_column": "pfs_t",
            "pfs_event_column": "pfs_e",
            "metastasis_column": "met",
            "output_dir": str(toy_paths["out_dir"] / "survival"),
        },
        reference_config={
            "endpoint": "os",
            "fixed_sections_from": str(toy_paths["reference_config_path"]),
            "output_path": str(toy_paths["out_dir"] / "mantis_config.yaml"),
        },
        inference={
            "model_weights_path": str(toy_paths["tmp_path"] / "fake_weights.pt"),
            "output_predictions_path": str(toy_paths["out_dir"] / "predictions.csv"),
        },
    )
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key].update(val)
        else:
            base[key] = val
    return pm.load_config(None, **base)


@pytest.fixture
def cfg(toy_paths):
    return toy_cfg(toy_paths)
