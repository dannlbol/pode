"""Core limits: parallel scoring matches serial, and limits are applied/restored."""

import pandas as pd
import pytest
import torch

from pode.config import PodeConfig
from pode.imputation.base import limit_cpu, n_workers
from pode.imputation.benchmark import run_benchmark
from pode.imputation.config import ImputationConfig


def _icfg(n_jobs):
    return ImputationConfig.from_dict({
        "m": 3, "folds": 2, "n_jobs": n_jobs, "plots": False,
        "scenarios": [{"name": "a", "kind": "mcar", "rate": 0.2}, {"name": "b", "kind": "mcar", "rate": 0.4}],
        "engines": ["mean_mode", "mice"],
        "cox": {"covariates": ["bio", "treated"]},
        "delta": {"deltas": [-1.0, 0.0, 1.0]},
    })


def test_parallel_scoring_matches_serial(tmp_path, cohort, cohort_schema):
    serial = run_benchmark(cohort, cohort_schema, PodeConfig(), _icfg(1), tmp_path / "s", log=lambda _: None)
    par = run_benchmark(cohort, cohort_schema, PodeConfig(), _icfg(2), tmp_path / "p", log=lambda _: None)
    for name in ("summary", "rubin_cox", "delta_sensitivity"):
        a = serial["tables"][name].sort_values(list(serial["tables"][name].columns[:3])).reset_index(drop=True)
        b = par["tables"][name].sort_values(list(par["tables"][name].columns[:3])).reset_index(drop=True)
        drop = [c for c in a.columns if "seconds" in c]
        pd.testing.assert_frame_equal(a.drop(columns=drop), b.drop(columns=drop))


def test_limit_cpu_caps_and_restores_torch_threads():
    before = torch.get_num_threads()
    with limit_cpu(1) as cores:
        assert cores == 1 and torch.get_num_threads() == 1
    assert torch.get_num_threads() == before


@pytest.mark.parametrize("n_jobs,expected", [(1, 1), (10**6, None), (-1, None)])
def test_n_workers(n_jobs, expected):
    import os

    assert n_workers(n_jobs) == (expected or os.cpu_count())
