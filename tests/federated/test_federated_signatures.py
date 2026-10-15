import copy
import json

import numpy as np
import pandas as pd
import pytest

from metacpg.core import fit_metacpg
from pode.federated import DisclosureLedger
from pode.federated.quantiles import median_iqr_from, median_iqr_ranks
from pode.federated.signatures import (federated_loso, federated_metacpg, merge_moments, resolve_design,
                                       site_local_signature_transfer)

from .signature_fixtures import config


@pytest.mark.parametrize("design", ["pooled", "meta"])
def test_federated_metacpg_reproduces_central(tmp_path, design):
    cfg = config(tmp_path, design, count=5)
    central = fit_metacpg(copy.deepcopy(cfg))
    led = DisclosureLedger()
    fed = federated_metacpg(copy.deepcopy(cfg), "site" if design == "pooled" else None, led)
    assert central["ranking"]["feature"].tolist() == fed["ranking"]["feature"].tolist()
    s1 = central["scores"].set_index(["sample_id", "feature_count"])["risk_score"]
    s2 = fed["scores"].set_index(["sample_id", "feature_count"])["risk_score"].reindex(s1.index)
    assert float((s1 - s2).abs().max()) < 1e-10
    assert central["selected_features"]["feature"].tolist() == fed["selected_features"]["feature"].tolist()
    assert np.allclose(central["selected_features"]["standardized_coef"],
                       fed["selected_features"]["standardized_coef"], atol=1e-12)
    for k in ("center", "scale"):
        assert central["scaling"][k] == pytest.approx(fed["scaling"][k], abs=1e-12)
    for k, v in central["performance"].items():
        assert fed["performance"][k] == pytest.approx(v, abs=1e-8), k
    # Nothing per-patient crossed a boundary except the held-out LOO scores and the sites' own outputs.
    s = led.summary().set_index("op")
    assert set(s.index[s["per_patient"]]) == {"loo_heldout_score", "final_risk_scores_site_output"}


def test_spanning_strata_are_refused(tmp_path):
    cfg = config(tmp_path, "pooled", count=5)
    cfg["covariates"] = [{"column": "age", "type": "categorical", "mode": "strata"}]
    with pytest.raises(ValueError, match="spans"):
        federated_metacpg(cfg, "site")
    cfg["covariates"] = []
    with pytest.raises(ValueError, match="does not decompose"):
        federated_metacpg(cfg, "site")


def test_within_site_metrics_share_only_aggregates(tmp_path):
    led = DisclosureLedger()
    res = federated_metacpg(config(tmp_path, "pooled", count=5), "site", led, metrics="within_site")
    assert not led.frame.loc[led.frame["op"] == "loo_heldout_score", "per_patient"].any()
    assert res["metrics"]["c_index"].between(0, 1).all()
    assert res["selected_features"].shape[0] == 5


def test_loso_and_single_site_transfer(tmp_path):
    res = federated_loso(config(tmp_path, "pooled", count=5), "site", counts=[3, 5])
    assert set(res["per_site"]["heldout_site"]) == {"A", "B", "C"}
    assert {"c_index_stratified", "mean_jaccard"} <= set(res["curve"].columns)
    tr = site_local_signature_transfer(res["fed"], [3, 5])
    assert len(tr) == 3 * 2 * 2 and tr["c_index"].between(0, 1).all()


def test_moments_and_quantiles_exact():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(57, 4))
    parts = [(len(p), p.mean(0), ((p - p.mean(0)) ** 2).sum(0)) for p in (x[:20], x[20:21], x[21:])]
    n, sd = merge_moments(parts)
    assert n == 57 and np.allclose(sd, x.std(axis=0, ddof=1), atol=1e-14)
    for n in (57, 58, 2, 3):
        v = np.sort(rng.normal(size=n))
        ranks, recipe = median_iqr_ranks(n)
        c, s = median_iqr_from(v[ranks], recipe)
        assert c == np.median(v) and s == np.percentile(v, 75) - np.percentile(v, 25)


def test_federated_order_stats_protocol():
    from pode.federated.quantiles import federated_order_stats
    from pode.federated.sites import Site

    rng = np.random.default_rng(5)
    vals = [rng.normal(size=(3, k)).round(2) for k in (40, 1, 25)]  # ties via rounding
    sites = []
    for i, v in enumerate(vals):
        s = Site(str(i), pd.DataFrame({"x": range(v.shape[1])}))
        s.local["v"] = v
        sites.append(s)
    allv = np.sort(np.concatenate(vals, axis=1), axis=1)
    ranks = np.array([[0, 10, 32, 65]] * 3)
    got = federated_order_stats(sites, "v", ranks)
    assert np.array_equal(got, np.take_along_axis(allv, ranks, 1))


def test_signature_cli(tmp_path):
    from pode.cli.main import main

    config(tmp_path, "pooled", count=4)
    cfgp = str(tmp_path / "config.yml")
    assert main(["federated", "signature", "verify", "--config", cfgp, "--site-column", "site",
                 "--feature-count", "4", "--extraction-only", "--output-dir", str(tmp_path / "v")]) == 0
    rep = json.loads((tmp_path / "v" / "verify_report.json").read_text())
    assert rep["agreement"]["selected_identical"]
    assert (tmp_path / "v" / "federated" / "python" / "model.json").is_file()
    assert main(["federated", "signature", "run", "--config", cfgp, "--site-column", "site", "--cv", "loso",
                 "--output-dir", str(tmp_path / "r")]) == 0
    assert (tmp_path / "r" / "python" / "model.json").is_file()
    assert main(["federated", "signature", "benchmark", "--config", cfgp, "--site-column", "site",
                 "--counts", "3,4", "--output-dir", str(tmp_path / "b")]) == 0
    assert "leave-one-site-out" in (tmp_path / "b" / "report.md").read_text()
