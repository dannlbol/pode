import numpy as np
import pandas as pd
import pytest

import pode_foundation as pf


def make_toy_matrix(n_samples=24, n_cpgs=300, seed=7):
    """A small synthetic samples x probes beta matrix, chr/pos annotation
    (with intentional correlated clusters for LD-pruning to find), and a
    survival phenotype sidecar."""
    rng = np.random.default_rng(seed)
    samples = np.array([f"S{i:03d}" for i in range(n_samples)])
    cpgs = np.array([f"cg{i:07d}" for i in range(n_cpgs)])

    beta = rng.uniform(0.05, 0.95, size=(n_samples, n_cpgs)).astype(np.float32)
    chrom = np.resize(["chr1", "chr2", "chr3"], n_cpgs)
    pos = (np.arange(n_cpgs) * 2000 + rng.integers(0, 500, n_cpgs)).astype(np.int64)
    for i in range(4, n_cpgs, 5):
        pos[i] = pos[i - 1] + rng.integers(50, 400)
        beta[:, i] = np.clip(beta[:, i - 1] + rng.normal(0, 0.02, n_samples), 0.01, 0.99)

    beta_df = pd.DataFrame(beta, columns=cpgs)
    beta_df.insert(0, "sample_id", samples)
    anno_df = pd.DataFrame({"Name": cpgs, "chr": chrom, "pos": pos})

    risk = 2.0 * (beta[:, 0] - 0.5)
    event_time = np.maximum(0.1, rng.exponential(np.exp(-risk / 2) * 5))
    censor_time = rng.uniform(1, 10, n_samples)
    phenotype_df = pd.DataFrame(
        {
            "sample_id": samples,
            "time": np.round(np.minimum(event_time, censor_time), 2),
            "event": (event_time <= censor_time).astype(int),
            "fold": rng.integers(0, 4, n_samples),
        }
    )
    return beta_df, anno_df, phenotype_df


@pytest.fixture
def toy_paths(tmp_path):
    beta_df, anno_df, phenotype_df = make_toy_matrix()
    paths = {
        "beta": tmp_path / "beta.csv",
        "anno": tmp_path / "anno.csv",
        "phenotype": tmp_path / "phenotype.csv",
        "out_dir": tmp_path / "out",
    }
    beta_df.to_csv(paths["beta"], index=False)
    anno_df.to_csv(paths["anno"], index=False)
    phenotype_df.to_csv(paths["phenotype"], index=False)
    return paths


def toy_cfg(toy_paths, **overrides):
    base = dict(
        matrix={"path": str(toy_paths["beta"]), "orientation": "samples_by_features", "scale": "beta"},
        preprocess={
            "annotation_path": str(toy_paths["anno"]),
            "n_keep": 50,
            "n_candidates": 150,
            "output_dir": str(toy_paths["out_dir"] / "preprocess"),
        },
        model={"encoder_dims": [16, 8], "latent_dim": 4, "dropout": 0.0, "output_activation": "sigmoid"},
        training={
            "mask_ratio": 0.2,
            "batch_size": 8,
            "epochs": 2,
            "patience": 2,
            "val_frac": 0.25,
            "val_mask_draws": 1,
            "seed": 42,
            "device": "cpu",
            "output_dir": str(toy_paths["out_dir"] / "pretrain"),
        },
        finetune_stage2={"epochs": 2, "patience": 2, "output_dir": str(toy_paths["out_dir"] / "stage2")},
        survival={
            "clinical_path": str(toy_paths["phenotype"]),
            "time_column": "time",
            "event_column": "event",
            "fold_column": "fold",
            "output_dir": str(toy_paths["out_dir"] / "survival"),
        },
        embed={"output_path": str(toy_paths["out_dir"] / "embeddings.csv")},
    )
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key].update(val)
        else:
            base[key] = val
    return pf.load_config(None, **base)


@pytest.fixture
def cfg(toy_paths):
    return toy_cfg(toy_paths)
