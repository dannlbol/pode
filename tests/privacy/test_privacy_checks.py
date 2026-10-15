"""Unit tests: each check fires on memorised data and stays quiet on a fresh sample."""

import numpy as np
import pandas as pd
import pytest

from pode.privacy import PrivacyConfig, assess_privacy
from pode.privacy.distance import Binner, GowerSpace, row_keys


def _status(report, check, metric_prefix=""):
    g = report.gate
    hit = g[(g["check"] == check) & g["metric"].str.startswith(metric_prefix)]
    return list(hit["status"])


def _quiet(_msg):
    pass


def test_copies_fail(split, pcfg):
    train, holdout, _ = split
    rep = assess_privacy(train, train.sample(frac=1.0, random_state=0), holdout, config=pcfg, log=_quiet)
    s = rep.summary
    assert rep.verdict == "FAIL"
    assert s["exact_copies"] == len(train)
    assert s["membership_auc"] > 0.8 and s["dcr_p5_ratio"] < 0.1
    assert s["attribute_member_gap_max"] > 0.3
    assert s["disclosive_links"] >= 1
    for check in ("exact_copies", "membership", "attribute_inference", "rare_combinations"):
        assert "FAIL" in _status(rep, check), check
    assert "FAIL" in _status(rep, "linkage", "disclosive")


def test_fresh_sample_passes(split, pcfg):
    train, holdout, fresh = split
    rep = assess_privacy(train, fresh, holdout, config=pcfg, log=_quiet)
    assert rep.verdict == "PASS", rep.gate[rep.gate["status"] != "PASS"]
    assert 0.35 < rep.summary["membership_auc"] < 0.65
    assert rep.summary["exact_copies"] == 0
    assert 0.6 < rep.summary["dcr_p5_ratio"] < 1.6


def test_a_few_planted_rare_copies_are_caught(split, pcfg):
    train, holdout, fresh = split
    rare = train[train["subtype"] == "RARE"]
    assert 2 <= len(rare) <= 10
    pcfg.rare_k = 10  # treat the RARE subtype as a small stratum
    synth = pd.concat([fresh, rare], ignore_index=True)
    rep = assess_privacy(train, synth, holdout, config=pcfg, log=_quiet)
    assert rep.summary["exact_copies"] == len(rare)
    assert rep.summary["exact_copies_small_strata"] == len(rare)
    assert "FAIL" in _status(rep, "exact_copies")
    assert rep.verdict == "FAIL"
    # The copied children are listed for case review, by training row number.
    flagged = rep.tables["outliers"]
    assert set(rare.index) <= set(flagged.loc[flagged["replicated"], "train_row"])
    assert (rep.tables["closest_pairs"]["distance"].head(len(rare)) == 0).all()


def test_no_holdout_is_never_a_pass(split, pcfg):
    train, _, fresh = split
    rep = assess_privacy(train, fresh, None, config=pcfg, log=_quiet)
    assert rep.threat_model["reference"] == "leave-one-out"
    assert _status(rep, "membership") == ["NOT_ASSESSED"]
    assert rep.verdict != "PASS"


def test_tiny_holdout_is_ignored(split, pcfg):
    train, holdout, fresh = split
    rep = assess_privacy(train, fresh, holdout.head(5), config=pcfg, log=_quiet)
    assert rep.threat_model["n_holdout"] == 0
    assert any("not used" in n for n in rep.threat_model["notes"])


def test_identifier_column_and_values_fail(split):
    train, holdout, fresh = split
    t = train.assign(Sample_ID=[f"S{i:04d}" for i in range(len(train))])
    s = fresh.assign(Sample_ID=t["Sample_ID"].to_numpy())
    rep = assess_privacy(t, s, holdout, config=PrivacyConfig(id_column="Sample_ID", n_bootstrap=20), log=_quiet)
    issues = set(rep.tables["identifiers"]["issue"])
    assert {"identifier column", "real identifier value"} <= issues
    assert _status(rep, "identifiers") == ["FAIL"]
    assert "Sample_ID" not in rep.threat_model["columns"]


def test_date_and_free_text_columns_flagged(split):
    train, holdout, fresh = split
    s = fresh.assign(dx_date=np.resize(pd.date_range("2010-01-01", periods=40).astype(str), len(fresh)),
                     notes=["patient presented with headache and vomiting " + str(i % 3) for i in range(len(fresh))])
    rep = assess_privacy(train, s, holdout, config=PrivacyConfig(n_bootstrap=20), log=_quiet)
    issues = dict(zip(rep.tables["identifiers"]["column"], rep.tables["identifiers"]["issue"]))
    assert issues == {"dx_date": "date column", "notes": "free text"}
    assert _status(rep, "identifiers") == ["REVIEW"]


def test_default_threat_model_uses_schema(split):
    train, holdout, fresh = split
    msgs = []
    rep = assess_privacy(train, fresh, holdout, config=PrivacyConfig(n_bootstrap=20), log=msgs.append)
    tm = rep.threat_model
    assert tm["sensitive"] == ["OS_Status"]  # survival event from the inferred schema
    assert "categorical" in tm["quasi_identifier_tiers"]
    assert "OS_Status" not in tm["quasi_identifier_tiers"]["categorical"]
    assert any("quasi_identifiers" in m for m in msgs)


def test_report_saves_all_outputs(tmp_path, split, pcfg):
    train, holdout, fresh = split
    rep = assess_privacy(train, fresh, holdout, config=pcfg, log=_quiet)
    rep.save(tmp_path)
    for name in ("report.md", "gate.csv", "privacy_summary.json", "privacy_config_used.yaml", "membership.csv",
                 "attribute_inference.csv", "linkage.csv", "rare_combinations.csv", "strata.csv", "outliers.csv",
                 "closest_pairs.csv"):
        assert (tmp_path / name).is_file(), name
    text = (tmp_path / "report.md").read_text()
    assert text.startswith("# Synthetic-data privacy assessment") and "Verdict: PASS" in text
    # Case-review tables hold row numbers and distances, never real values.
    assert not set(pd.read_csv(tmp_path / "outliers.csv").columns) & set(train.columns)


# ---- distance primitives ----------------------------------------------------

def test_gower_categories_nan_and_weights():
    a = pd.DataFrame({"x": [0.0, np.nan], "c": [1, "B"], "d": ["u", "u"]})
    b = pd.DataFrame({"x": [10.0], "c": ["1"], "d": ["v"]})
    space = GowerSpace(pd.DataFrame({"x": [0.0, 10.0]}).assign(c=1, d="u"), ["x"], ["c", "d"])
    d = space.block(space.encode(a), space.encode(b))
    # Row 0: x differs by the full range (1), c equal (1 == "1"), d differs -> 2/3.
    # Row 1: x missing (ignored), c differs, d differs -> 2/2.
    assert d[:, 0] == pytest.approx([2 / 3, 1.0])
    w = GowerSpace(pd.DataFrame({"x": [0.0, 10.0]}).assign(c=1, d="u"), ["x"], ["c", "d"], weights={"d": 0})
    assert w.block(w.encode(a), w.encode(b))[0, 0] == pytest.approx(0.5)


def test_chunked_nearest_matches_dense():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=60), "y": rng.normal(size=60), "c": rng.choice(list("ab"), 60)})
    other = df.sample(40, random_state=1).assign(x=lambda d: d["x"] + 0.01)
    space = GowerSpace(df, ["x", "y"], ["c"])
    ea, eb = space.encode(other), space.encode(df)
    dense = space.block(ea, eb)
    dist, idx = space.nearest(ea, eb, k=2, chunk=7, n_jobs=2)
    assert np.allclose(dist[:, 0], dense.min(axis=1))
    assert np.allclose(dist[:, 1], np.sort(dense, axis=1)[:, 1])
    loo, _ = space.nearest(eb, eb, k=1, exclude_self=True, chunk=11)
    self_d = space.block(eb, eb)
    np.fill_diagonal(self_d, np.inf)
    assert np.allclose(loo[:, 0], self_d.min(axis=1))


def test_row_keys_and_binner():
    df = pd.DataFrame({"x": [1.0000001, 1.0, np.nan], "c": [1, 1.0, None]})
    keys = row_keys(df, ["x"], ["c"], decimals=6)
    assert keys[0] == keys[1] and keys[2] != keys[0]
    b = Binner(pd.DataFrame({"x": np.arange(100.0)}), ["x"], {"x"}, bins=4)
    labels = b.labels(pd.DataFrame({"x": [0.0, 99.0, np.nan]}), "x")
    assert list(labels) == ["bin0", "bin3", "<NA>"]
    assert b.describe("x", "bin0").startswith("[-inf")


def test_config_round_trip_and_validation(tmp_path):
    cfg = PrivacyConfig.from_dict({"quasi_identifiers": {"t": ["Age"]}, "gates": {"max_membership_auc": 0.55},
                                   "audit": {"repeats": 3}})
    path = tmp_path / "p.yaml"
    cfg.to_yaml(path)
    back = PrivacyConfig.from_yaml(path)
    assert back.gates.max_membership_auc == 0.55 and back.audit.repeats == 3
    assert back.quasi_identifiers == {"t": ["Age"]}
    with pytest.raises(ValueError):
        PrivacyConfig.from_dict({"audit": {"holdout_frac": 1.5}}).validate()
    with pytest.raises(ValueError):
        PrivacyConfig.from_dict({"near_copy_quantiles": [0.0]}).validate()
