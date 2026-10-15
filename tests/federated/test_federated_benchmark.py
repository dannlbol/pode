from pathlib import Path

import pandas as pd
import pytest

from pode.federated import FederatedConfig, load_federated_config

from .conftest import make_cohort

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["benchmark.example.yaml", "toy.example.yaml"])
def test_example_configs_validate(name):
    load_federated_config(ROOT / "configs" / "federated" / name).validate()


def test_benchmark_quick_end_to_end(tmp_path):
    from pode.cli.main import main

    path = tmp_path / "cohort.csv"
    make_cohort(300, seed=4).to_csv(path, index=False)
    cfg = FederatedConfig()
    cfg.sites.loso = ["small"]
    cfg.dp.batch_size = 32
    cfg.evaluation.n_bootstrap = 20
    cfg.evaluation.conditions = ["local", "exchange", "fedavg", "fed_dp_eps3", "central_fed_tvae", "central_copula"]
    cfg.to_yaml(tmp_path / "fed.yaml")
    out = tmp_path / "bench"
    args = ["federated", "benchmark", "--data", str(path), "--site-column", "site", "--federated-config",
            str(tmp_path / "fed.yaml"), "--quick", "--device", "cpu", "--n-jobs", "2", "--output-dir", str(out)]
    assert main([*args, "--dry-run"]) == 0
    assert main(args) == 0
    sc = pd.read_csv(out / "scorecard.csv")
    assert set(sc["condition"]) == set(cfg.evaluation.conditions)
    assert set(sc["verdict"]) <= {"UNSAFE", "RECOMMENDED", "NON-INFERIOR", "NOT RECOMMENDED", "BASELINE"}
    res = pd.read_csv(out / "results.csv")
    assert {"real_local", "real_pooled", "real_federated_cox"} <= set(res["condition"])
    # Utility is only ever measured on real test patients of each site.
    assert set(res.loc[res["arm"] == "within_site", "site"]) == {"big", "mid", "small"}
    assert set(pd.read_csv(out / "loso.csv")["heldout_site"]) == {"small"}
    assert pd.read_csv(out / "generators.csv").query("condition == 'fed_dp_eps3'")["epsilon_max"].notna().all()
    text = (out / "report.md").read_text()
    assert "## Scorecard" in text and "Leave-one-site-out" in text
    # Re-running resumes from checkpoints; report re-renders.
    assert main(args) == 0
    assert main(["federated", "report", "--output-dir", str(out)]) == 0
