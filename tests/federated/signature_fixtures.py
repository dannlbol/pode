"""A small multi-site methylation cohort and MetaCPG configs for the federated signature tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def write_cohort(tmp: Path, n: int = 66, features: int = 40, seed: int = 0) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    site = np.repeat(["A", "B", "C"], [30, 20, n - 50])
    age = rng.choice(["young", "old"], n)
    signal = rng.normal(0, 1, n)
    beta = 1 / (1 + np.exp(-(rng.normal(0, 1, (features, n)) + 0.8 * signal[None, :] * (np.arange(features) < 6)[:, None])))
    hazard = np.exp(0.9 * signal + 0.3 * (site == "B"))
    t = rng.exponential(30 / hazard)
    c = rng.uniform(5, 60, n)
    ids = [f"S{i:03d}" for i in range(n)]
    matrix = pd.DataFrame(beta.clip(0.01, 0.99), index=[f"cg{i:05d}" for i in range(features)], columns=ids)
    pheno = pd.DataFrame({"sample_id": ids, "time": np.minimum(t, c).round(2) + 0.1,
                          "status": (t <= c).astype(int), "site": site, "age": age})
    mp, pp = tmp / "matrix.csv", tmp / "pheno.csv"
    matrix.to_csv(mp)
    pheno.to_csv(pp, index=False)
    return mp, pp


def config(tmp: Path, design: str = "pooled", engine: str = "batched", count: int | None = None) -> dict:
    mp, pp = write_cohort(tmp)
    endpoint = {"name": "os", "time": "time", "status": "status", "prediction_time": 24}
    covariates = []
    if design == "pooled":
        covariates = [{"column": "site", "type": "categorical", "mode": "strata"},
                      {"column": "age", "type": "categorical", "mode": "strata"}]
    else:
        endpoint["group_column"] = "site"
    cfg = {"schema_version": 1, "methodology": "legacy_exact", "analysis_id": f"fed_{design}",
           "matrix": {"path": str(mp), "orientation": "features_by_samples", "scale": "beta",
                      "feature_id_column": "first"},
           "phenotype": {"path": str(pp), "sample_id_column": "sample_id"},
           "endpoints": [endpoint], "covariates": covariates,
           "final_model": {"weight_refit": "covariate_aware"},
           "search": {"min_features": 3, "max_features": 8, "step": 1, "cross_validation": "loo"},
           "selection": {"mode": "manual", "feature_count": count},
           "execution": {"workers": 1, "engine": engine, "progress": False},
           "output_dir": str(tmp / "out")}
    (tmp / "config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg
