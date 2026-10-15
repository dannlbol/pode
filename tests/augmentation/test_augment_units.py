"""Unit tests: config, splits, strategies, constraints, models, metrics, privacy, efficiency."""

import numpy as np
import pandas as pd
import pytest

from conftest import augment_config, make_cohort
from pode.augmentation.config import AugmentationConfig, StrategySpec
from pode.augmentation.efficiency import real_data_equivalent
from pode.augmentation.generator import apply_constraints, with_survival_rules
from pode.augmentation.metrics import calibration_at, corrected_cv_ci, harrell_c, km
from pode.augmentation.models import BreslowCalibrator, Preprocessor, build_model
from pode.augmentation.privacy import privacy_metrics
from pode.augmentation.scorecard import _utility
from pode.augmentation.splits import make_splits, split_external
from pode.augmentation.strategies import SOURCE, build_pool, draw, subgroup_groups, target_mask
from pode.schema.schema import DatasetSchema


# ---- config ----------------------------------------------------------------------
def test_config_roundtrip_and_validation(tmp_path):
    cfg = AugmentationConfig.from_dict(augment_config())
    assert [s.key for s in cfg.strategies][:2] == ["tvae_natural", "tvae_mplus"]
    path = tmp_path / "a.yaml"
    cfg.to_yaml(path)
    assert AugmentationConfig.from_yaml(path).to_dict() == cfg.to_dict()
    with pytest.raises(ValueError, match="Unknown strategy"):
        AugmentationConfig.from_dict({"strategies": ["smote"]})
    with pytest.raises(ValueError, match="needs a 'target'"):
        AugmentationConfig.from_dict({"strategies": ["tvae_targeted"]})
    with pytest.raises(ValueError, match="Duplicate strategy"):
        AugmentationConfig.from_dict({"strategies": ["duplicate", "duplicate"]})
    q = cfg.quick()
    assert q.splits.repeats == 1 and q.draws <= 2 and len(q.ratios) <= 2


# ---- splits ----------------------------------------------------------------------
def test_external_rows_removed_and_folds_partition():
    df = make_cohort()
    cfg = AugmentationConfig.from_dict(augment_config(splits={"folds": 4, "repeats": 2, "stratify": ["Cohort"]}))
    dev, ext = split_external(df, cfg)
    assert set(ext["Cohort"]) == {"EXT"} and "EXT" not in set(dev["Cohort"])
    splits = make_splits(dev, cfg, "OS_Status")
    assert len(splits) == 8
    for rep in (0, 1):
        tests = np.concatenate([s.test for s in splits if s.repeat == rep])
        assert sorted(tests) == list(range(len(dev)))  # every row tested exactly once per repeat
    for s in splits:
        assert not set(s.train) & set(s.test)


def test_group_column_keeps_clusters_together():
    df = make_cohort()
    df["family"] = np.arange(len(df)) // 3
    cfg = AugmentationConfig.from_dict(augment_config(splits={"folds": 3, "repeats": 1, "group_column": "family"}))
    dev, _ = split_external(df, cfg)
    for s in make_splits(dev, cfg, "OS_Status"):
        assert not set(dev.iloc[s.train]["family"]) & set(dev.iloc[s.test]["family"])


# ---- strategies -------------------------------------------------------------------
def test_target_mask_matches_numeric_categories():
    df = pd.DataFrame({"OS_Status": [1.0, 0.0, 1.0], "Met": ["Yes", "No", "No"]})
    assert target_mask(df, {"equals": {"OS_Status": 1}}).tolist() == [True, False, True]
    groups = {"groups": [{"proportion": 0.5, "equals": {"Met": "Yes"}},
                         {"proportion": 0.5, "equals": {"OS_Status": 0}}]}
    assert target_mask(df, groups).tolist() == [True, True, False]
    assert target_mask(df, {}).all()


def test_subgroup_groups_favour_rare_levels():
    df = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 4 + ["c"] * 1})
    groups = subgroup_groups(df, ["g"])
    props = {g["equals"]["g"]: g["proportion"] for g in groups}
    assert "a" not in props  # the largest level has no deficit
    assert props["c"] > props["b"] and np.isclose(sum(props.values()), 1.0)


def _schema(df):
    return DatasetSchema.from_dataframe(df)


def test_duplicate_and_jitter_keep_outcomes():
    df = make_cohort().drop(columns=["sample_id"])
    schema = _schema(df)
    outcomes = ["OS_Time", "OS_Status"]
    target = {"equals": {"Metastases": "Yes"}}
    dup = build_pool(StrategySpec("duplicate", target=target), df, schema, None, 50, 0, {}, outcomes)
    assert (dup.rows["Metastases"] == "Yes").all() and len(dup.rows) == 50
    jit = build_pool(StrategySpec("jitter", target=target, params={"sd_frac": 0.2}), df, schema, None, 50, 0, {},
                     outcomes)
    # Same resampled rows as `dup` (same seed): outcomes and categories copied, continuous features moved.
    assert (jit.rows[outcomes].to_numpy() == dup.rows[outcomes].to_numpy()).all()
    assert (jit.rows["Subgroup"] == dup.rows["Subgroup"]).all()
    assert not np.allclose(jit.rows["cg_1"], dup.rows["cg_1"])


def test_weight_pool_and_draw():
    df = make_cohort().drop(columns=["sample_id"])
    target = {"equals": {"Metastases": "Yes"}}
    pool = build_pool(StrategySpec("weight", target=target), df, _schema(df), None, 0, 0, {}, [])
    out, w = draw(pool, df, 40, np.random.default_rng(0))
    assert len(out) == len(df) and (out[SOURCE] == "real").all()
    n_t = (df["Metastases"] == "Yes").sum()
    assert np.isclose(w.sum(), len(df) + 40) and np.allclose(w[(df["Metastases"] == "Yes").to_numpy()], 1 + 40 / n_t)
    dup = build_pool(StrategySpec("duplicate"), df, _schema(df), None, 30, 0, {}, [])
    out, w = draw(dup, df, 20, np.random.default_rng(0))
    assert len(out) == len(df) + 20 and (out[SOURCE] != "real").sum() == 20


def test_constraints_drop_and_count():
    df = pd.DataFrame({"OS_Time": [5, -1, 3, 8], "PFS_Time": [4, 1, 9, 2], "Age": [1, 2, 50, 3],
                       "Met": ["Yes", "No", "No", "Yes"], "RT": ["Yes", "No", "No", "No"]})
    schema = DatasetSchema.from_dict({"types": {"OS_Time": "survival_time", "OS_Status": "survival_event"},
                                      "survival_pairs": [{"base": "OS", "time_col": "OS_Time",
                                                          "event_col": "OS_Status"}]})
    rules = with_survival_rules(schema, {"ranges": {"Age": {"min": 0, "max": 21}}, "ordered": [["PFS_Time", "OS_Time"]],
                                         "forbid": [{"Met": "Yes", "RT": "Yes"}]})
    kept, counts = apply_constraints(df, rules)
    assert kept["OS_Time"].tolist() == [8]  # the only row breaking no rule
    # Row 2 breaks both the Age range and PFS<=OS ordering; a row is counted once, under its first rule.
    assert counts == {"range:Age": 1, "positive:OS_Time": 1, "forbid:0": 1}


# ---- models ----------------------------------------------------------------------
def _toy_survival(n=300, seed=1, beta=0.7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.exponential(10 * np.exp(-beta * x))
    c = rng.uniform(2, 30, n)
    return x, np.minimum(t, c), (t <= c).astype(float)


def test_breslow_calibrator_matches_lifelines():
    from lifelines import CoxPHFitter

    x, t, e = _toy_survival()
    cal = BreslowCalibrator().fit(x, t, e)
    ref = CoxPHFitter(baseline_estimation_method="breslow").fit(pd.DataFrame({"x": x, "t": t, "e": e}), "t", "e")
    assert abs(cal.coef - ref.params_["x"]) < 1e-3
    s_ref = ref.predict_survival_function(pd.DataFrame({"x": x[:5]}), times=[8.0]).to_numpy()[0]
    assert np.allclose(cal.surv(x[:5], [8.0])[:, 0], s_ref, atol=0.01)


def test_metacpg_fixed_matches_metacpg_weights():
    from metacpg.core import _endpoint_stats, _legacy_full_scores

    df = make_cohort()
    feats = ["cg_1", "cg_2", "cg_3", "cg_4"]
    spec = AugmentationConfig.from_dict(augment_config()).models[0]
    strata = df["Cohort"].to_numpy()
    m = build_model(spec, feats, [], 24, 0).fit(df.assign(**{SOURCE: "real"}), df["OS_Time"], df["OS_Status"],
                                                strata, np.ones(len(df)), np.ones(len(df), bool))
    matrix = df[feats].T
    pheno = pd.DataFrame({"t": df["OS_Time"], "s": df["OS_Status"], "k": strata})
    stats = _endpoint_stats(matrix, pheno, {"name": "x", "time": "t", "status": "s"},
                            [{"column": "k", "type": "categorical", "mode": "strata"}])
    scores, _, scaling = _legacy_full_scores(matrix, stats, feats)
    assert np.allclose(m.weights, stats.set_index("feature").loc[feats, "standardized_coef"], atol=1e-5)
    assert np.allclose(m.risk(df), scores["risk_score"], atol=1e-5)


def test_preprocessing_ignores_synthetic_rows():
    df = make_cohort().iloc[:100]
    synth = df.assign(cg_1=df["cg_1"] + 50.0, Subgroup="NEW")
    joint = pd.concat([df, synth], ignore_index=True)
    real = np.r_[np.ones(len(df), bool), np.zeros(len(synth), bool)]
    pre = Preprocessor(["cg_1"], ["Subgroup"]).fit(joint[real])
    assert abs(pre.mean[0] - df["cg_1"].mean()) < 1e-12 and "NEW" not in pre.levels["Subgroup"]
    X = pre.transform(df)
    assert np.allclose(X.mean(axis=0), 0) and np.allclose(X.std(axis=0, ddof=1), 1)  # dummies included


@pytest.mark.parametrize("kind", ["coxnet", "cox_ph", "logistic_landmark", "metacpg_fixed"])
def test_models_rank_risk(kind):
    df = make_cohort(n=300, seed=3)
    cfg = AugmentationConfig.from_dict({"models": [{"kind": kind}]})
    num = ["cg_1", "cg_2", "Age"]
    m = build_model(cfg.models[0], num, [] if kind == "metacpg_fixed" else ["Metastases"], 24, 0)
    m.fit(df.assign(**{SOURCE: "real"}), df["OS_Time"], df["OS_Status"], None, np.ones(len(df)),
          np.ones(len(df), bool))
    assert harrell_c(df["OS_Time"], df["OS_Status"], m.risk(df)) > 0.6
    s = m.surv(df, [24.0])
    assert s is not None and ((s >= 0) & (s <= 1)).all()


# ---- metrics ---------------------------------------------------------------------
def test_harrell_c_matches_lifelines_and_km():
    from lifelines import KaplanMeierFitter
    from lifelines.utils import concordance_index

    x, t, e = _toy_survival()
    assert abs(harrell_c(t, e, x) - concordance_index(t, -x, e)) < 1e-6
    ref = KaplanMeierFitter().fit(t, e).predict([5.0, 12.0]).to_numpy()
    assert np.allclose(km(t, e, [5.0, 12.0]), ref)


def test_calibration_slope_near_one_when_correct():
    x, t, e = _toy_survival(n=3000, seed=4, beta=0.8)
    cal = BreslowCalibrator().fit(x, t, e)  # the true model, fitted in-sample
    out = calibration_at(t, e, cal.surv(x, [8.0])[:, 0], 8.0)
    assert abs(out["cal_slope"] - 1) < 0.1 and abs(out["oe_ratio"] - 1) < 0.1


def test_corrected_cv_ci_widens_with_overlap():
    d = np.random.default_rng(0).normal(0.02, 0.02, 25)
    naive = corrected_cv_ci(d, n_train=160, n_test=0)  # no overlap correction
    corrected = corrected_cv_ci(d, n_train=160, n_test=40)
    assert corrected["ci_high"] - corrected["ci_low"] > naive["ci_high"] - naive["ci_low"]
    assert _utility(0.01, 0.05, 0.01) == "improves" and _utility(-0.005, 0.03, 0.01) == "non-inferior"
    assert _utility(-0.05, -0.01, 0.01) == "worse" and _utility(-0.05, 0.03, 0.01) == "inconclusive"


# ---- privacy ---------------------------------------------------------------------
def test_privacy_flags_copies_and_passes_fresh_samples():
    df = make_cohort(n=400, seed=5)
    train, hold, fresh = df.iloc[:200], df.iloc[200:300], df.iloc[300:]
    num, cat = ["Age", "cg_1", "cg_2", "cg_3"], ["Metastases", "Subgroup"]
    copies = privacy_metrics(train, hold, train.sample(150, random_state=0), num, cat)
    assert copies["exact_duplicates"] == 150 and copies["membership_auc"] > 0.8 and copies["dcr_p5_ratio"] < 0.1
    clean = privacy_metrics(train, hold, fresh, num, cat)
    assert clean["exact_duplicates"] == 0 and 0.35 < clean["membership_auc"] < 0.65


# ---- efficiency --------------------------------------------------------------------
def test_real_data_equivalent_inverts_known_curve():
    # C = 0.5 + 0.05 log(n): +0.05 at n=100 is worth n_equiv = 100 e = 272 real patients.
    rows = []
    for split in range(3):
        for frac in (0.5, 0.75, 1.0):
            n = 100 * frac
            rows.append({"evaluation": "internal", "split": split, "model": "m",
                         "arm": "real_only" if frac == 1 else "learning_curve", "strategy": "real_only",
                         "ratio": 0.0, "fraction": frac, "n_real": n, "c_harrell": 0.5 + 0.05 * np.log(n)})
    paired = pd.DataFrame([{"model": "m", "strategy": "s", "ratio": 2.0, "metric": "c_harrell", "delta": 0.05,
                            "ci_low": 0.0, "ci_high": 0.1, "n_real": 100.0, "n_synthetic": 200.0}])
    eff = real_data_equivalent(paired, pd.DataFrame(rows), ("c_harrell",)).iloc[0]
    assert abs(eff["n_equiv"] - 100 * np.e) < 1e-6
    assert abs(eff["efficiency"] - (100 * np.e - 100) / 200) < 1e-6
    assert eff["efficiency_ci_low"] == 0.0 and eff["flag"] == "extrapolated >2x"


def test_flat_learning_curve_leaves_equivalent_undefined():
    rng = np.random.default_rng(0)
    rows = [{"evaluation": "internal", "split": split, "model": "m",
             "arm": "real_only" if frac == 1 else "learning_curve", "strategy": "real_only", "ratio": 0.0,
             "fraction": frac, "n_real": 100 * frac, "c_harrell": 0.63 + rng.normal(0, 0.02)}
            for split in range(10) for frac in (0.5, 0.75, 1.0)]
    paired = pd.DataFrame([{"model": "m", "strategy": "s", "ratio": 0.5, "metric": "c_harrell", "delta": 0.01,
                            "ci_low": -0.01, "ci_high": 0.03, "n_real": 100.0, "n_synthetic": 50.0}])
    eff = real_data_equivalent(paired, pd.DataFrame(rows), ("c_harrell",)).iloc[0]
    assert np.isnan(eff["efficiency"]) and eff["flag"].startswith("flat learning curve")


def test_top_variance_and_m_values_use_real_rows_only():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"cg_a": rng.uniform(0.1, 0.9, 100), "cg_b": rng.uniform(0.45, 0.55, 100),
                         "cg_c": rng.uniform(0.3, 0.7, 100), "Age": rng.uniform(0, 10, 100)})
    # Synthetic rows make cg_b look hugely variable; the filter must ignore them.
    synth = real.assign(cg_b=rng.choice([0.01, 0.99], 100))
    joint = pd.concat([real, synth], ignore_index=True)
    is_real = np.r_[np.ones(100, bool), np.zeros(100, bool)]
    pre = Preprocessor(["cg_a", "cg_b", "cg_c", "Age"], [], m_value_columns=["cg_a", "cg_b", "cg_c"],
                       top_variance=2, top_variance_columns=["cg_a", "cg_b", "cg_c"]).fit(joint[is_real])
    assert pre.numeric == ["cg_a", "cg_c", "Age"]  # cg_b dropped; Age always kept
    raw = pre._raw(real.iloc[:3])
    assert np.allclose(raw[:, 0], np.log2(real["cg_a"].iloc[:3] / (1 - real["cg_a"].iloc[:3])))
    assert np.allclose(raw[:, 2], real["Age"].iloc[:3])


def test_calibration_gate_only_penalises_moving_away_from_one():
    from pode.augmentation.scorecard import build_scorecard

    rows = []
    for strategy, slope_delta, err_delta in (("towards_one", 0.2, -0.2), ("away_from_one", -0.2, 0.2)):
        for metric, d in (("c_harrell", 0.02), ("cal_slope", slope_delta), ("cal_slope_err", err_delta)):
            rows.append({"model": "m", "strategy": strategy, "ratio": 1.0, "metric": metric, "delta": d,
                         "ci_low": d - 0.01, "ci_high": d + 0.01})
    card = build_scorecard(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                           AugmentationConfig(), set()).set_index("strategy")
    assert card.loc["towards_one", "verdict"] == "RECOMMENDED"
    assert card.loc["away_from_one", "verdict"] == "NOT RECOMMENDED"
    assert "further from 1" in card.loc["away_from_one", "reasons"]
