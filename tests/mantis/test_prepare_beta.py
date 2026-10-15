import numpy as np
import pandas as pd

from pode_mantis.core.prepare_beta import run_prepare_beta

from conftest import toy_cfg


def test_prepare_beta_shapes_and_no_nans(toy_paths):
    cfg = toy_cfg(toy_paths)
    result = run_prepare_beta(cfg)

    assert result.n_samples == len(toy_paths["samples"])
    assert result.n_cpgs == 40

    numeric = pd.read_csv(result.numeric_path, header=None)
    assert numeric.shape == (result.n_cpgs, result.n_samples)  # probes x samples on disk
    assert not numeric.isna().values.any()
    assert ((numeric.to_numpy() >= 0) & (numeric.to_numpy() <= 1)).all()

    sample_order = [l.strip() for l in open(result.sample_order_path)]
    assert sample_order == toy_paths["samples"]


def test_prepare_beta_imputes_missing_probes(toy_paths):
    cfg = toy_cfg(toy_paths)
    result = run_prepare_beta(cfg)
    coverage = pd.read_csv(result.coverage_path)
    assert (coverage["n_imputed"] > 0).any()  # toy beta has ~20% missingness by construction
    assert set(coverage.columns) >= {"sample_id", "n_present", "n_imputed", "pct_imputed", "primary_analysis_flag"}


def test_prepare_beta_reindexes_probe_not_in_cohort(toy_paths):
    """A CpG in the canonical order but entirely absent from the cohort's own
    array should still appear as a row (imputed), never silently dropped --
    dropping would shift every other value for a per-sample-normalized model."""
    cpgs_in_order = [l.strip() for l in open(toy_paths["cpg_order_path"])]
    extra_cpg = "cgNOTINDATA"
    with open(toy_paths["cpg_order_path"], "a") as f:
        f.write(extra_cpg + "\n")

    cfg = toy_cfg(toy_paths)
    result = run_prepare_beta(cfg)
    assert result.n_cpgs == len(cpgs_in_order) + 1
    numeric = pd.read_csv(result.numeric_path, header=None)
    assert numeric.shape[0] == len(cpgs_in_order) + 1
    assert not numeric.isna().values.any()  # the extra probe row was imputed (own-median fallback), not dropped


def test_prepare_beta_requires_beta_path():
    import pytest
    import pode_mantis as pm

    cfg = pm.load_config(None, beta={"cpg_order_path": "somewhere.txt"})
    with pytest.raises(ValueError, match="beta matrix"):
        run_prepare_beta(cfg)


def test_prepare_beta_requires_cpg_order():
    import pytest
    import pode_mantis as pm

    cfg = pm.load_config(None, beta={"cohort_path": "somewhere.csv"})
    with pytest.raises(ValueError, match="CpG order"):
        run_prepare_beta(cfg)
