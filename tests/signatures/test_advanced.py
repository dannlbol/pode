import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from metacpg.advanced import (load_advanced_data, method_catalog, predict_signature,
                              read_advanced_config, run_advanced_pipeline)
from metacpg.advanced.pipeline import _endpoint_rows, _folds, _prepare, benchmark_methods
from metacpg.advanced.base import cox_loss_gradient
from metacpg.advanced.models import LinearCoxModel
from metacpg.advanced.registry import REGISTRY, build_estimator
from metacpg.advanced.slurm import (aggregate_advanced_tasks, run_advanced_worker,
                                    submit_advanced_slurm)
from pode.cli.main import main as pode_main


def advanced_fixture(tmp_path: Path, methods=None, row_order=False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12)
    n = 18
    samples = [f"S{i:02}" for i in range(n)]
    expression = rng.normal(size=(n, 6))
    methylation = rng.uniform(.05, .95, size=(n, 5))
    risk = expression[:, 0] - expression[:, 1] + methylation[:, 0]
    phenotype = pd.DataFrame({
        "sample_id": samples, "cohort": np.repeat(["A", "B", "C"], 6),
        "age": np.arange(n) + 3, "group": np.resize(["III", "IV"], n),
        "os_time": np.maximum(.2, 8 - risk + rng.normal(0, .2, n)),
        "os_event": np.resize([1, 0, 1], n),
        "pfs_time": np.maximum(.2, 5 - .5 * risk + rng.normal(0, .2, n)),
        "pfs_event": np.resize([0, 1], n),
    })
    phenotype.loc[n - 1, "pfs_time"] = np.nan
    phenotype.loc[n - 2, "pfs_event"] = np.nan
    exp = pd.DataFrame(expression, columns=[f"g{i}" for i in range(6)])
    meth = pd.DataFrame(methylation[2:] if not row_order else methylation,
                        columns=[f"cg{i}" for i in range(5)])
    if not row_order:
        exp.insert(0, "sample_id", samples)
        meth.insert(0, "sample_id", samples[2:])
    phenotype.to_csv(tmp_path / "phenotype.csv", index=False)
    exp.to_csv(tmp_path / "expression.csv", index=False)
    meth.to_csv(tmp_path / "methylation.csv", index=False)
    pd.DataFrame({"source": ["g0", "g1"], "target": ["g1", "g2"], "weight": [1, 1]}).to_csv(
        tmp_path / "graph.csv", index=False
    )
    pd.DataFrame({"feature_index": range(6), "pathway_index": np.arange(6) % 4}).to_csv(
        tmp_path / "pathways.csv", index=False
    )
    cfg = {
        "schema_version": 2, "analysis_id": "advanced-test",
        "alignment": {"mode": "row_order" if row_order else "sample_id", "sample_policy": "union"},
        "phenotype": {"path": "phenotype.csv", "cohort_column": "cohort"},
        "omics": [
            {"name": "expression", "path": "expression.csv", "orientation": "samples_by_features",
             "transform": "none", "standardize": True},
            {"name": "methylation", "path": "methylation.csv", "orientation": "samples_by_features",
             "transform": "beta_to_m", "standardize": True},
        ],
        "endpoints": [
            {"name": "os", "time": "os_time", "status": "os_event", "prediction_time": 5},
            {"name": "pfs", "time": "pfs_time", "status": "pfs_event", "prediction_time": 3},
        ],
        "covariates": [{"column": "age", "type": "numeric", "penalized": False},
                       {"column": "group", "type": "categorical", "penalized": False}],
        "methods": {"enabled": methods or ["elastic_net_cox"], "config": {
            "elastic_net_cox": {"max_iter": 80},
            "coxboost": {"steps": 8},
            "stability_coxboost": {"steps": 5, "repetitions": 3},
            "net_cox": {"graph_path": "graph.csv", "max_iter": 30},
            "cox_nnet": {"pathway_path": "pathways.csv", "epochs": 2, "hidden": 4},
        }},
        "selection": {"feature_counts": [2, 3]},
        "evaluation": {"outer_folds": 3, "inner_folds": 2, "primary_metric": "harrell_c"},
        "execution": {"seed": 3, "slurm": {"cpus": 2, "memory": "2G", "time": "00:10:00"}},
        "output_dir": "out",
    }
    if not row_order:
        cfg["phenotype"]["sample_id_column"] = "sample_id"
        cfg["omics"][0]["sample_id_column"] = "sample_id"
        cfg["omics"][1]["sample_id_column"] = "sample_id"
    path = tmp_path / "advanced.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def test_v2_masked_union_and_row_order(tmp_path):
    path = advanced_fixture(tmp_path / "union")
    data = load_advanced_data(path)
    assert len(data.sample_ids) == 18
    assert data.blocks["methylation"].present.sum() == 16
    row_path = advanced_fixture(tmp_path / "rows", row_order=True)
    row_data = load_advanced_data(row_path)
    assert all(block.present.all() for block in row_data.blocks.values())


def test_samples_with_missing_cohort_are_excluded(tmp_path):
    path = advanced_fixture(tmp_path)
    phenotype_path = tmp_path / "phenotype.csv"
    phenotype = pd.read_csv(phenotype_path)
    removed_sample = phenotype.loc[0, "sample_id"]
    phenotype.loc[0, "cohort"] = "  "
    phenotype.to_csv(phenotype_path, index=False)

    data = load_advanced_data(path)

    assert len(data.sample_ids) == 17
    assert removed_sample not in set(data.sample_ids)
    for block in data.blocks.values():
        assert removed_sample not in set(block.values.index)


def test_pipeline_outputs_and_portable_prediction(tmp_path):
    path = advanced_fixture(tmp_path)
    result = run_advanced_pipeline(path, "elastic_net_cox")
    out = Path(result["output_dir"])
    assert (out / "fold_predictions.csv").is_file()
    assert (out / "selected_features.csv").is_file()
    assert (out / "model" / "manifest.json").is_file()
    assert set(result["fold_predictions"]["model_type"]) == {"native", "reduced_signature"}
    scores = predict_signature(out / "model", path, tmp_path / "predicted")
    assert len(scores) == 72
    assert set(scores["model_type"]) == {"native", "reduced_signature"}
    assert np.isfinite(scores["risk_score"]).all()
    expression = pd.read_csv(tmp_path / "expression.csv")
    expression = expression[["sample_id", *reversed([c for c in expression if c not in {"sample_id", "g5"}])]]
    expression.to_csv(tmp_path / "expression.csv", index=False)
    phenotype = pd.read_csv(tmp_path / "phenotype.csv")
    phenotype.loc[0, "group"] = "unseen-level"
    phenotype.to_csv(tmp_path / "phenotype.csv", index=False)
    reordered = predict_signature(out / "model", path, tmp_path / "predicted-reordered")
    assert np.isfinite(reordered["risk_score"]).all()


def test_read_advanced_config_accepts_strata_covariate(tmp_path):
    path = advanced_fixture(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["covariates"][1]["mode"] = "strata"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    loaded = read_advanced_config(path)
    assert loaded["covariates"][1]["mode"] == "strata"


def test_validate_rejects_bad_covariate_mode(tmp_path):
    path = advanced_fixture(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["covariates"][1]["mode"] = "bogus"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(ValueError, match="covariate.mode must be adjust or strata"):
        read_advanced_config(path)


def test_strata_with_sksurv_method_fails_only_for_that_method(tmp_path):
    # A strata covariate and an incompatible scikit-survival method can coexist in
    # methods.enabled (e.g. a shared config used by several `--method` invocations);
    # only actually building/running the incompatible method should fail, and it
    # should fail regardless of whether the optional scikit-survival dependency is
    # installed.
    path = advanced_fixture(tmp_path, methods=["elastic_net_cox", "random_survival_forest"])
    cfg = yaml.safe_load(path.read_text())
    cfg["covariates"][1]["mode"] = "strata"
    cfg["endpoints"] = cfg["endpoints"][:1]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    read_advanced_config(path)  # does not raise: no method has been selected yet
    run_advanced_pipeline(path, "elastic_net_cox")  # unaffected by the other, unused method
    with pytest.raises(ValueError, match="random_survival_forest does not support covariate mode: strata"):
        run_advanced_pipeline(path, "random_survival_forest")


def test_strata_covariate_gets_no_coefficient(tmp_path):
    path = advanced_fixture(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["covariates"][1]["mode"] = "strata"
    cfg["endpoints"] = cfg["endpoints"][:1]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = run_advanced_pipeline(path, "elastic_net_cox")
    terms = set(result["signature_coefficients"]["term"])
    assert not any(term.startswith("clinical::group") for term in terms)
    assert any(term.startswith("clinical::age") for term in terms)


def test_deep_method_strata_covariate_smoke(tmp_path):
    path = advanced_fixture(tmp_path, methods=["deepsurv"])
    cfg = yaml.safe_load(path.read_text())
    cfg["covariates"][1]["mode"] = "strata"
    cfg["methods"]["config"]["deepsurv"] = {"epochs": 1, "hidden": 4}
    cfg["endpoints"] = cfg["endpoints"][:1]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = run_advanced_pipeline(path, "deepsurv")
    assert set(result["fold_metrics"]["endpoint"]) == {"os"}


def test_cox_loss_gradient_stratified_matches_per_group_sum():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(12, 3))
    beta = rng.normal(scale=.1, size=3)
    time = np.array([1., 2., 3., 4., 5., 6., 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    status = np.resize([1, 0, 1], 12).astype(float)
    strata = np.array(["A"] * 6 + ["B"] * 6)

    stratified_loss, stratified_grad = cox_loss_gradient(beta, x, time, status, strata)

    total_loss = 0.0
    total_grad = np.zeros_like(beta)
    total_events = 0
    for group in ("A", "B"):
        mask = strata == group
        group_loss, group_grad = cox_loss_gradient(beta, x[mask], time[mask], status[mask])
        events = int(status[mask].sum())
        total_loss += group_loss * events
        total_grad += group_grad * events
        total_events += events
    np.testing.assert_allclose(stratified_loss, total_loss / total_events, rtol=1e-10)
    np.testing.assert_allclose(stratified_grad, total_grad / total_events, rtol=1e-10)


def test_schema_v2_cli_run_validate_methods_and_predict(tmp_path, capsys):
    path = advanced_fixture(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["endpoints"] = cfg["endpoints"][:1]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    assert pode_main(["signatures", "methods"]) == 0
    assert "elastic_net_cox" in capsys.readouterr().out
    assert pode_main(["signatures", "validate", "--config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert pode_main(["signatures", "submit", "--config", str(path), "--dry-run",
                      "--method", "elastic_net_cox"]) == 0
    assert len(json.loads(capsys.readouterr().out)["scripts"]) == 2
    assert pode_main(["signatures", "benchmark", "--config", str(path), "--preset", "quick"]) == 0
    benchmark_capture = capsys.readouterr()
    assert json.loads(benchmark_capture.out)["leaderboard"][0]["status"] == "completed"
    assert "Benchmark preset 'quick'" in benchmark_capture.err
    assert "fold" in benchmark_capture.err
    assert pode_main(["signatures", "run", "--config", str(path),
                      "--method", "elastic_net_cox"]) == 0
    run_capture = capsys.readouterr()
    run_output = json.loads(run_capture.out)
    assert "elastic_net_cox: endpoint os" in run_capture.err
    assert "elastic_net_cox: complete" in run_capture.err
    model = Path(run_output["output_dir"]) / "model"
    assert pode_main(["signatures", "predict", "--config", str(path),
                      "--model-dir", str(model), "--output-dir", str(tmp_path / "cli-predict")]) == 0
    predict_capture = capsys.readouterr()
    assert json.loads(predict_capture.out)["rows"] == 36
    assert "Prediction complete" in predict_capture.err


def test_method_catalog_and_optional_dependency_status():
    catalog = {row["name"]: row for row in method_catalog()}
    assert set(REGISTRY) == set(catalog)
    assert catalog["hierarchical_cox"]["maturity"] == "experimental"
    assert catalog["random_survival_forest"]["available"] == (importlib.util.find_spec("sksurv") is not None)


def test_quick_benchmark_records_experimental_skip(tmp_path):
    path = advanced_fixture(tmp_path, methods=["elastic_net_cox", "hierarchical_cox"])
    cfg = yaml.safe_load(path.read_text())
    cfg["endpoints"] = cfg["endpoints"][:1]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    messages = []
    result = benchmark_methods(path, "quick", progress=messages.append)
    statuses = result["leaderboard"].set_index("method")["status"].to_dict()
    assert statuses == {"elastic_net_cox": "completed", "hierarchical_cox": "skipped"}
    assert any("starting elastic_net_cox" in message for message in messages)
    assert any("fold" in message for message in messages)
    assert any("leaderboard written" in message for message in messages)


def test_benchmark_writes_full_comparison_artifacts(tmp_path):
    path = advanced_fixture(tmp_path, methods=["elastic_net_cox", "coxboost"])
    cfg = yaml.safe_load(path.read_text())
    cfg["endpoints"] = cfg["endpoints"][:1]
    # os_time is centred near 8, so a horizon of 5 leaves no cases and the
    # time-dependent AUC is undefined. Move it inside the observed range.
    cfg["endpoints"][0]["prediction_time"] = 8
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    result = benchmark_methods(path, "quick", progress=None)

    out = Path(result["output_dir"])
    for name in ("leaderboard", "benchmark_metrics", "fold_metrics",
                 "method_comparison", "signature_overlap"):
        assert (out / f"{name}.csv").is_file()
        assert not pd.read_csv(out / f"{name}.csv").empty
    assert (out / "benchmark_report.md").is_file()
    assert (out / "result.json").is_file()

    leaderboard = pd.read_csv(out / "leaderboard.csv")
    assert len(leaderboard) == 2
    assert sorted(leaderboard["rank"].tolist()) == [1, 2]
    for column in ("mean_time_dependent_auc", "mean_integrated_brier_score",
                   "mean_incremental_c_index", "sd_c_index", "ci_low_c_index",
                   "ci_high_c_index", "native_mean_c_index", "n_folds",
                   "median_feature_count", "n_metric_warnings"):
        assert column in leaderboard.columns
    assert leaderboard["mean_time_dependent_auc"].notna().all()
    assert leaderboard["mean_incremental_c_index"].notna().all()
    assert leaderboard["n_folds"].gt(0).all()
    assert leaderboard["ci_low_c_index"].le(leaderboard["mean_c_index"]).all()
    assert leaderboard["ci_high_c_index"].ge(leaderboard["mean_c_index"]).all()

    tidy = pd.read_csv(out / "benchmark_metrics.csv")
    assert set(tidy["model_type"]) == {"native", "reduced_signature"}
    assert {"c_index", "integrated_brier_score"} <= set(tidy["metric"])

    pooled = pd.read_csv(out / "fold_metrics.csv")
    assert "inner_scores" not in pooled.columns
    assert len(pooled) == sum(len(value["fold_metrics"]) for value in result["results"].values())

    comparison = pd.read_csv(out / "method_comparison.csv")
    forward = comparison[(comparison["method_a"] == "elastic_net_cox")
                         & (comparison["method_b"] == "coxboost") & (comparison["metric"] == "c_index")]
    reverse = comparison[(comparison["method_a"] == "coxboost")
                         & (comparison["method_b"] == "elastic_net_cox") & (comparison["metric"] == "c_index")]
    assert len(forward) == len(reverse) == 1
    assert forward["mean_diff"].iloc[0] == pytest.approx(-reverse["mean_diff"].iloc[0])
    assert forward["n_pairs"].iloc[0] == int(leaderboard.set_index("method").loc["coxboost", "n_folds"])

    overlap = pd.read_csv(out / "signature_overlap.csv")
    assert overlap["jaccard"].between(0, 1).all()

    manifest = json.loads((out / "result.json").read_text())
    assert manifest["primary_metric"] == "c_index"
    assert manifest["ranking"] == leaderboard.sort_values("rank")["method"].tolist()
    assert "benchmark_report.md" in manifest["artifacts"]
    report = (out / "benchmark_report.md").read_text()
    assert "## Ranking" in report and "elastic_net_cox" in report


def test_benchmark_with_no_completed_methods_still_writes_leaderboard(tmp_path):
    path = advanced_fixture(tmp_path, methods=["elastic_net_cox"])
    cfg = yaml.safe_load(path.read_text())
    cfg["endpoints"] = cfg["endpoints"][:1]
    cfg["methods"]["config"]["elastic_net_cox"] = {"max_iter": "not-a-number"}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    result = benchmark_methods(path, "quick", progress=None)

    out = Path(result["output_dir"])
    assert result["leaderboard"]["status"].tolist() == ["failed"]
    assert result["ranking"] == []
    assert (out / "leaderboard.csv").is_file()
    assert (out / "result.json").is_file()
    assert (out / "benchmark_report.md").is_file()
    assert not (out / "method_comparison.csv").exists()


def test_deep_method_uses_multitask_endpoint_heads(tmp_path):
    path = advanced_fixture(tmp_path, methods=["deepsurv"])
    cfg = yaml.safe_load(path.read_text())
    cfg["methods"]["config"]["deepsurv"] = {"epochs": 1, "hidden": 4}
    cfg["selection"]["feature_counts"] = [2]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = run_advanced_pipeline(path, "deepsurv")
    assert set(result["fold_metrics"]["endpoint"]) == {"os", "pfs"}
    for endpoint in ("os", "pfs"):
        manifest = json.loads((Path(result["output_dir"]) / "model" / endpoint / "manifest.json").read_text())
        assert manifest["native_state"] == "native_state.pt"


def test_unavailable_sksurv_method_has_actionable_error():
    if importlib.util.find_spec("sksurv") is not None:
        pytest.skip("optional dependency is installed")
    context = {"n_omics": 2, "feature_names": ["a", "b"], "design_names": ["a", "b"],
               "block_slices": {"a": (0, 2)}, "seed": 1, "cohort": None}
    model = build_estimator("random_survival_forest", context, {})
    with pytest.raises(RuntimeError, match="signatures-advanced"):
        model.fit(np.ones((5, 2)), np.arange(1, 6), np.array([1, 0, 1, 0, 1]))


def test_cox_and_graph_gradients_match_finite_differences():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(9, 4))
    time = np.arange(1, 10, dtype=float)
    status = np.resize([1, 0, 1], 9).astype(float)
    beta = rng.normal(scale=.1, size=4)
    _, analytic = cox_loss_gradient(beta, x, time, status)
    epsilon = 1e-6
    numeric = np.zeros(4)
    for i in range(4):
        step = np.zeros(4); step[i] = epsilon
        plus = cox_loss_gradient(beta + step, x, time, status)[0]
        minus = cox_loss_gradient(beta - step, x, time, status)[0]
        numeric[i] = (plus - minus) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)

    graph = LinearCoxModel(l2=.03, n_omics=3, graph_edges=[(0, 1, 2.0)], graph_strength=.2)
    _, analytic = graph._objective_grad(beta, x, time, status)
    for i in range(4):
        step = np.zeros(4); step[i] = epsilon
        plus = graph._objective_grad(beta + step, x, time, status)[0]
        minus = graph._objective_grad(beta - step, x, time, status)[0]
        numeric[i] = (plus - minus) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", [
    "elastic_net_cox", "coxboost", "stability_coxboost", "pls_cox_multiblock",
    "deepsurv", "autoencoder_cox", "net_cox", "cox_nnet", "surv_transformer",
    "prototype_contrastive_survival", "moli_survival", "hierarchical_cox", "maml_cox", "frailty_cox",
])
def test_all_native_method_factories_smoke(tmp_path, name):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 9))
    time = np.arange(1, 13, dtype=float)
    status = np.resize([1, 0, 1], 12).astype(float)
    masks = np.ones((12, 2), dtype=float)
    cohort = np.repeat(["A", "B", "C"], 4)
    graph = tmp_path / "graph.csv"
    pd.DataFrame({"source": ["a::f0"], "target": ["a::f1"]}).to_csv(graph, index=False)
    pathways = tmp_path / "pathways.csv"
    pd.DataFrame({"feature_index": range(6), "pathway_index": np.arange(6) % 4}).to_csv(pathways, index=False)
    cfg = {"epochs": 2, "hidden": 4, "steps": 5, "repetitions": 3,
           "max_iter": 30, "graph_path": str(graph), "pathway_path": str(pathways)}
    context = {"n_omics": 6, "feature_names": [f"a::f{i}" for i in range(3)] + [f"b::f{i}" for i in range(3)],
               "design_names": [f"x{i}" for i in range(9)], "block_slices": {"a": (0, 3), "b": (3, 6)},
               "seed": 2, "cohort": cohort}
    estimator = build_estimator(name, context, cfg)
    estimator.fit(x, time, status, masks=masks, cohort=cohort)
    risk = estimator.predict(x, masks=masks, cohort=cohort)
    assert len(estimator.importance_) == 6
    assert np.isfinite(risk).all()


def test_advanced_slurm_manifest(tmp_path):
    path = advanced_fixture(tmp_path, methods=["elastic_net_cox"])
    cfg = yaml.safe_load(path.read_text())
    cfg["endpoints"] = cfg["endpoints"][:1]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = submit_advanced_slurm(path, dry_run=True)
    assert len(result["tasks"]) == 3
    assert {"method", "endpoint", "fold", "stage"}.issubset(result["tasks"][0])
    script = Path(result["scripts"][0]).read_text()
    assert "pode signatures worker" in script
    assert "--task-manifest" in script
    assert "pode signatures aggregate" in Path(result["scripts"][1]).read_text()
    for task in result["tasks"]:
        run_advanced_worker(path, result["task_manifest"], task["task_index"])
    aggregated = aggregate_advanced_tasks(path, result["task_manifest"], result["run_id"])
    assert Path(aggregated["methods"]["elastic_net_cox"]).is_dir()


# --- cohort_column semantics -------------------------------------------------------------
# cohort_column marks samples drawn from several training sources whose underlying survival
# rates may differ. It must influence how models are FIT, never how data is split.


def _two_cohort_survival(seed=5, n=16):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    # Interleaved cohorts, so rescaling one cohort's clock genuinely reorders the pooled
    # risk sets while leaving the within-cohort ordering untouched.
    cohort = np.resize(["A", "B"], n)
    time = np.linspace(1, 9, n) + rng.uniform(0, .1, n)
    status = np.resize([1, 1, 0], n).astype(float)
    return x, time, status, cohort


def _rescale_one_cohort(time, cohort, level="B", factor=3.0):
    """Monotone, cohort-specific time change: within-cohort order kept, pooled order altered."""
    return np.where(cohort == level, time * factor, time)


def test_cohort_strata_make_cox_invariant_to_per_cohort_time_rescaling():
    # A Cox model stratified on cohort depends only on the WITHIN-cohort ordering of event
    # times, so rescaling one cohort's clock must not move the coefficients. Pooling the risk
    # sets (no strata) does move them - that difference is exactly the bias cohort_column is
    # meant to remove.
    x, time, status, cohort = _two_cohort_survival()
    shifted = _rescale_one_cohort(time, cohort)
    beta = np.array([.3, -.2, .1])

    stratified = cox_loss_gradient(beta, x, time, status, strata=cohort)
    stratified_shifted = cox_loss_gradient(beta, x, shifted, status, strata=cohort)
    np.testing.assert_allclose(stratified[0], stratified_shifted[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stratified[1], stratified_shifted[1], rtol=1e-12, atol=1e-12)

    pooled = cox_loss_gradient(beta, x, time, status)
    pooled_shifted = cox_loss_gradient(beta, x, shifted, status)
    assert not np.allclose(pooled[1], pooled_shifted[1], atol=1e-6)

    fitted = LinearCoxModel(l1=0, l2=1e-3, n_omics=3).fit(x, time, status, strata=cohort).coef_
    fitted_shifted = LinearCoxModel(l1=0, l2=1e-3, n_omics=3).fit(
        x, shifted, status, strata=cohort).coef_
    np.testing.assert_allclose(fitted, fitted_shifted, rtol=1e-6, atol=1e-8)

    unstratified = LinearCoxModel(l1=0, l2=1e-3, n_omics=3).fit(x, time, status).coef_
    unstratified_shifted = LinearCoxModel(l1=0, l2=1e-3, n_omics=3).fit(x, shifted, status).coef_
    assert not np.allclose(unstratified, unstratified_shifted, atol=1e-4)


def test_prepare_crosses_cohort_into_the_cox_strata(tmp_path):
    path = advanced_fixture(tmp_path)
    data = load_advanced_data(path)
    rows = np.arange(len(data.sample_ids))
    prepared = _prepare(data, rows, rows)

    # No covariate is in strata mode in the fixture, so the covariate strata stay empty and
    # the cohort labels become the strata on their own.
    assert prepared.strata is None
    assert set(prepared.cohort_strata) == {"A", "B", "C"}
    # elastic_net_cox stratifies on cohort; frailty_cox puts cohort on the linear predictor
    # instead, so it must NOT also receive a stratified baseline.
    assert prepared.method_strata(REGISTRY["elastic_net_cox"].cohort_stratifies) is not None
    assert prepared.method_strata(REGISTRY["frailty_cox"].cohort_stratifies) is None
    # The reduced-signature refit is a plain Cox fit and always stratifies on cohort.
    assert set(prepared.refit_strata()) == {"A", "B", "C"}


def test_cohort_strata_cross_with_covariate_strata(tmp_path):
    path = advanced_fixture(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["covariates"][1]["mode"] = "strata"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    data = load_advanced_data(path)
    rows = np.arange(len(data.sample_ids))
    prepared = _prepare(data, rows, rows)

    assert set(prepared.strata) == {"III", "IV"}
    assert set(prepared.cohort_strata) <= {f"{g}|{c}" for g in ("III", "IV") for c in "ABC"}
    assert len(set(prepared.cohort_strata)) == 6


def test_cohort_column_no_longer_defines_the_outer_split(tmp_path):
    path = advanced_fixture(tmp_path)
    data = load_advanced_data(path)
    endpoint = data.endpoints[0]
    rows = _endpoint_rows(data, endpoint)

    folds = _folds(data, endpoint, rows, 3, seed=1)

    assert [name for _, _, name in folds] == ["fold:0", "fold:1", "fold:2"]
    for train_rows, test_rows, _ in folds:
        # Every cohort stays represented in training, so cohort-specific terms remain
        # estimable and no cohort is silently held out.
        assert set(data.cohort.iloc[train_rows]) == {"A", "B", "C"}
        assert not set(train_rows) & set(test_rows)
    assert sorted(np.concatenate([test for _, test, _ in folds])) == sorted(rows)


def test_leave_one_cohort_out_is_opt_in(tmp_path):
    path = advanced_fixture(tmp_path)
    data = load_advanced_data(path)
    endpoint = data.endpoints[0]
    rows = _endpoint_rows(data, endpoint)

    folds = _folds(data, endpoint, rows, 3, seed=1, mode="leave_one_cohort_out")

    assert [name for _, _, name in folds] == ["cohort:A", "cohort:B", "cohort:C"]
    for train_rows, test_rows, name in folds:
        held_out = name.split(":")[1]
        assert set(data.cohort.iloc[test_rows]) == {held_out}
        assert held_out not in set(data.cohort.iloc[train_rows])


def test_leave_one_cohort_out_flows_through_the_pipeline(tmp_path):
    path = advanced_fixture(tmp_path, methods=["elastic_net_cox"])
    cfg = yaml.safe_load(path.read_text())
    cfg["endpoints"] = cfg["endpoints"][:1]
    cfg["selection"]["feature_counts"] = [2]
    cfg["evaluation"]["cohort_validation"] = "leave_one_cohort_out"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    result = run_advanced_pipeline(path, "elastic_net_cox")

    assert sorted(set(result["fold_metrics"]["fold"])) == ["cohort:A", "cohort:B", "cohort:C"]
    assert result["cohort_validation"] == "leave_one_cohort_out"
    # The model still stratifies on cohort while being validated across cohorts, and the
    # harsher validation question is recorded as a warning.
    assert result["cohort_handling"] == "stratified"
    assert "cross-cohort transfer" in result["cohort_warning"]


def test_leave_one_cohort_out_requires_a_cohort_column(tmp_path):
    path = advanced_fixture(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    del cfg["phenotype"]["cohort_column"]
    cfg["evaluation"]["cohort_validation"] = "leave_one_cohort_out"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(ValueError, match="requires phenotype.cohort_column"):
        read_advanced_config(path)


def test_missing_phenotype_columns_are_caught_during_validation(tmp_path):
    # A cohort_column naming a column that does not exist used to surface once per method as
    # an identical "failed" row in the benchmark leaderboard. It must be caught up front.
    path = advanced_fixture(tmp_path, methods=["elastic_net_cox", "coxboost"])
    cfg = yaml.safe_load(path.read_text())
    cfg["phenotype"]["cohort_column"] = "trial"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    with pytest.raises(ValueError) as excinfo:
        read_advanced_config(path)
    message = str(excinfo.value)
    assert "phenotype.cohort_column" in message
    assert "'trial' is not in phenotype.csv" in message
    # The available columns are listed so the fix is obvious without opening the file.
    assert "cohort" in message and "sample_id" in message
    # And the benchmark refuses before running any method, rather than reporting N failures.
    with pytest.raises(ValueError, match="is not in phenotype.csv"):
        benchmark_methods(path, "quick")


def test_missing_endpoint_and_covariate_columns_are_reported(tmp_path):
    path = advanced_fixture(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["endpoints"][0]["time"] = "survival_time"
    cfg["covariates"][0]["column"] = "years"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    with pytest.raises(ValueError) as excinfo:
        read_advanced_config(path)
    message = str(excinfo.value)
    assert "'survival_time' is not in phenotype.csv" in message
    assert "'years' is not in phenotype.csv" in message


def test_invalid_cohort_keys_are_rejected(tmp_path):
    path = advanced_fixture(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["evaluation"]["cohort_validation"] = "leave_one_out"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(ValueError, match="cohort_validation must be"):
        read_advanced_config(path)


@pytest.mark.parametrize("method", ["hierarchical_cox", "frailty_cox"])
def test_cohort_aware_methods_run_with_exactly_two_cohorts(tmp_path, method):
    # Previously these needed three cohorts, because leave-one-cohort-out demanded at least
    # two TRAINING cohorts after holding one out. With cohort as a fitting term, two is enough.
    path = advanced_fixture(tmp_path, methods=[method])
    phenotype = pd.read_csv(tmp_path / "phenotype.csv")
    phenotype["cohort"] = np.repeat(["A", "B"], 9)
    phenotype.to_csv(tmp_path / "phenotype.csv", index=False)
    cfg = yaml.safe_load(path.read_text())
    cfg["endpoints"] = cfg["endpoints"][:1]
    cfg["selection"]["feature_counts"] = [2]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    result = run_advanced_pipeline(path, method)

    assert set(result["fold_metrics"]["fold"]) == {"fold:0", "fold:1", "fold:2"}
    assert result["cohort_handling"] == "explicit"
    assert result["cohort_warning"] is None


def test_methods_that_cannot_honour_cohort_are_flagged(tmp_path):
    if importlib.util.find_spec("sksurv") is None:
        pytest.skip("scikit-survival is not installed")
    path = advanced_fixture(tmp_path, methods=["elastic_net_cox", "random_survival_forest"])
    cfg = yaml.safe_load(path.read_text())
    cfg["endpoints"] = cfg["endpoints"][:1]
    cfg["methods"]["config"]["random_survival_forest"] = {"params": {"n_estimators": 5}}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    messages = []

    leaderboard = benchmark_methods(path, "quick", progress=messages.append)["leaderboard"]

    handling = leaderboard.set_index("method")["cohort_handling"].to_dict()
    assert handling["elastic_net_cox"] == "stratified"
    assert handling["random_survival_forest"] == "ignored"
    warnings = leaderboard.set_index("method")["cohort_warning"].to_dict()
    assert warnings["elastic_net_cox"] is None
    assert "ignores phenotype.cohort_column" in warnings["random_survival_forest"]
    assert any("WARNING" in message and "random_survival_forest" in message for message in messages)


def test_cohort_handling_skip_excludes_incompatible_methods(tmp_path):
    if importlib.util.find_spec("sksurv") is None:
        pytest.skip("scikit-survival is not installed")
    path = advanced_fixture(tmp_path, methods=["elastic_net_cox", "random_survival_forest"])
    cfg = yaml.safe_load(path.read_text())
    cfg["endpoints"] = cfg["endpoints"][:1]
    cfg["phenotype"]["cohort_handling"] = "skip"
    cfg["methods"]["config"]["random_survival_forest"] = {"params": {"n_estimators": 5}}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    leaderboard = benchmark_methods(path, "quick", progress=None)["leaderboard"]

    statuses = leaderboard.set_index("method")["status"].to_dict()
    assert statuses == {"elastic_net_cox": "completed", "random_survival_forest": "skipped"}


def test_within_cohort_c_index_is_reported(tmp_path):
    path = advanced_fixture(tmp_path, methods=["elastic_net_cox"])
    cfg = yaml.safe_load(path.read_text())
    cfg["endpoints"] = cfg["endpoints"][:1]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    result = run_advanced_pipeline(path, "elastic_net_cox")

    assert "c_index_within_cohort" in result["fold_metrics"]
    assert result["fold_metrics"]["c_index_within_cohort"].notna().any()
    manifest = json.loads((Path(result["output_dir"]) / "model" / "os" / "manifest.json").read_text())
    assert manifest["cohort"]["cohort_handling"] == "stratified"


def test_without_a_cohort_column_nothing_changes(tmp_path):
    path = advanced_fixture(tmp_path, methods=["elastic_net_cox"])
    cfg = yaml.safe_load(path.read_text())
    del cfg["phenotype"]["cohort_column"]
    cfg["endpoints"] = cfg["endpoints"][:1]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    data = load_advanced_data(path)
    rows = _endpoint_rows(data, data.endpoints[0])

    assert data.cohort is None
    assert _prepare(data, rows, rows).cohort_strata is None
    result = run_advanced_pipeline(path, "elastic_net_cox")
    assert result["cohort_handling"] == "none"
    assert result["fold_metrics"]["c_index_within_cohort"].isna().all()
