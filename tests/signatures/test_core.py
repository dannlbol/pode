import json
import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from metacpg import read_config, run_pipeline, select_feature_count, submit_slurm, validate_inputs
from metacpg.cli import build_parser, main
from metacpg.core import (ProgressReporter, _CoxBackend, _cox_fit, _design, _endpoint_stats,
                          _fingerprint, _legacy_full_scores, _legacy_performance,
                          _legacy_performance_adjusted, _process_memory,
                          _process_worker_plan, _read_matrix, _survival_roc_nne_auc, aggregate_run,
                          apply_model, beta_to_m, extract_signature, heartbeat_seconds_from_config,
                          parse_memory_budget, preflight_resources, progress_enabled_from_config,
                          run_pipeline as _run_pipeline, run_worker, search_features, search_only,
                          verify_engine_parity)


def _manual_no_count_config(tmp_path: Path) -> Path:
    """A fixture config in manual mode with no configured feature count (survey-then-extract)."""
    path = fixture(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["selection"] = {"mode": "manual", "feature_count": None}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def test_run_without_count_writes_search_summary_and_does_not_crash(tmp_path):
    path = _manual_no_count_config(tmp_path)
    result = run_pipeline(path)  # no feature_count, manual mode -> must not raise
    assert result["phase"] == "search"
    assert "extract --config" in result["message"]
    out = tmp_path / "out" / "python"
    assert (out / "metrics.csv").is_file()
    assert (out / "ranking.csv").is_file()
    assert (out / "search_metrics.png").is_file()
    assert not (out / "selected_features.csv").exists()
    assert not (out / "plot_warning.txt").exists()
    metrics = pd.read_csv(out / "metrics.csv")
    assert "neglog10_p" in metrics.columns
    assert not any(c.endswith("_adjusted") for c in metrics.columns)  # no covariates configured


def test_extract_only_matches_coupled_extraction(tmp_path):
    # Extraction-only must be identical to the extraction inside a one-shot run.
    full = run_pipeline(read_config(fixture(tmp_path / "a")), feature_count=2)
    extracted = extract_signature(read_config(fixture(tmp_path / "b")), feature_count=2)
    assert (full["selected_features"]["feature"].tolist()
            == extracted["selected_features"]["feature"].tolist())
    pd.testing.assert_frame_equal(full["risk_scores"].reset_index(drop=True),
                                  extracted["risk_scores"].reset_index(drop=True))
    assert full["performance"]["feature_count"] == extracted["performance"]["feature_count"] == 2
    assert "neglog10_p" in full["performance"] and "neglog10_p" in extracted["performance"]


def test_extract_after_search_preserves_summary(tmp_path):
    path = _manual_no_count_config(tmp_path)
    run_pipeline(path)  # writes the search summary
    result = extract_signature(read_config(path), feature_count=2)
    out = tmp_path / "out" / "python"
    assert (out / "selected_features.csv").is_file()
    assert (out / "risk_scores.csv").is_file()
    assert (out / "metrics.csv").is_file()  # search summary left intact
    assert result["performance"]["feature_count"] == 2
    assert "neglog10_p" in result["performance"]


def test_extract_rejects_out_of_range_and_missing_count(tmp_path):
    path = _manual_no_count_config(tmp_path)
    with pytest.raises(ValueError, match="outside the search range"):
        extract_signature(read_config(path), feature_count=99)
    with pytest.raises(ValueError, match="requires an explicit --feature-count"):
        extract_signature(read_config(path))


def _train_and_load_model(tmp_path):
    path = fixture(tmp_path)
    result = run_pipeline(read_config(path), feature_count=2)
    out = tmp_path / "out" / "python"
    model = json.loads((out / "model.json").read_text())
    matrix = _read_matrix(str(tmp_path / "matrix.csv"), model["input"]["orientation"],
                          model["input"]["feature_id_column"])
    return path, out, result, model, matrix


def test_extract_writes_portable_model(tmp_path):
    _, out, result, model, _ = _train_and_load_model(tmp_path)
    assert (out / "model.json").is_file() and (out / "signature_model.csv").is_file()
    sel = pd.read_csv(out / "selected_features.csv")
    assert [entry["feature"] for entry in model["signature"]] == sel["feature"].astype(str).tolist()
    np.testing.assert_allclose([entry["weight"] for entry in model["signature"]],
                               sel["standardized_coef"].to_numpy(float))
    assert model["feature_count"] == 2
    assert model["input"]["scale"] == "beta"
    assert model["weight_refit"] == "covariate_aware"
    assert json.loads((out / "result.json").read_text())["weight_refit"] == "covariate_aware"
    assert model["scaling"]["center"] == pytest.approx(result["scaling"]["center"])
    assert model["scaling"]["scale"] == pytest.approx(result["scaling"]["scale"])


def test_apply_model_reproduces_training_scores(tmp_path):
    # The saved model applied to its own training matrix must reproduce risk_scores.csv exactly.
    _, out, _, model, matrix = _train_and_load_model(tmp_path)
    applied = apply_model(model, matrix)
    train = pd.read_csv(out / "risk_scores.csv")
    merged = train.merge(applied, on="sample_id", suffixes=("_train", "_applied"))
    assert len(merged) == len(train)
    np.testing.assert_allclose(merged["risk_score_train"], merged["risk_score_applied"], atol=1e-12)
    np.testing.assert_allclose(merged["raw_score_train"], merged["raw_score_applied"], atol=1e-12)


def test_final_scaling_excludes_samples_ineligible_for_training(tmp_path):
    """Incomplete endpoint rows may be scored, but cannot set model centre/IQR."""
    path = fixture(tmp_path)
    pheno_path = tmp_path / "pheno.csv"
    pheno = pd.read_csv(pheno_path)
    pheno.loc[pheno.index[-2:], ["time1", "status1"]] = np.nan
    pheno.to_csv(pheno_path, index=False)

    result = extract_signature(read_config(path), feature_count=2)
    scores = result["risk_scores"]
    eligible_ids = pheno.loc[pheno["time1"].notna() & pheno["status1"].notna(), "sample_id"]
    eligible_raw = scores.loc[scores["sample_id"].isin(eligible_ids), "raw_score"].to_numpy(float)
    all_raw = scores["raw_score"].to_numpy(float)

    assert len(scores) == len(pheno)  # non-eligible rows still receive a portable model score
    assert result["scaling"]["center"] == pytest.approx(np.median(eligible_raw))
    assert result["scaling"]["scale"] == pytest.approx(
        np.percentile(eligible_raw, 75) - np.percentile(eligible_raw, 25))
    assert result["scaling"]["center"] != pytest.approx(np.median(all_raw))


def test_apply_model_missing_features(tmp_path):
    _, _, _, model, matrix = _train_and_load_model(tmp_path)
    reduced = matrix.drop(index=model["signature"][0]["feature"])
    with pytest.raises(ValueError, match="missing from the"):
        apply_model(model, reduced)
    scored = apply_model(model, reduced, allow_missing=True)
    assert scored["features_used"].iloc[0] == 1
    assert len(scored) == matrix.shape[1] and scored["risk_score"].notna().all()


def test_cli_apply_writes_scores(tmp_path, capsys):
    path = fixture(tmp_path)
    run_pipeline(read_config(path), feature_count=2)
    out = tmp_path / "out" / "python"
    dest = tmp_path / "applied.csv"
    assert main(["apply", "--model", str(out / "model.json"),
                 "--matrix", str(tmp_path / "matrix.csv"), "--output", str(dest)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["samples"] == 10 and payload["features_used"] == 2
    scored = pd.read_csv(dest)
    assert list(scored.columns[:3]) == ["sample_id", "risk_score", "raw_score"] and len(scored) == 10


def test_cli_search_then_extract_workflow(tmp_path, capsys):
    path = _manual_no_count_config(tmp_path)
    assert main(["run", "--config", str(path)]) == 0  # run without a count must not crash
    assert json.loads(capsys.readouterr().out)["phase"] == "search"
    assert main(["search", "--config", str(path)]) == 0
    assert (tmp_path / "out" / "python" / "metrics.csv").is_file()
    assert json.loads(capsys.readouterr().out)["phase"] == "search"
    assert main(["extract", "--config", str(path), "--feature-count", "2"]) == 0
    assert (tmp_path / "out" / "python" / "selected_features.csv").is_file()
    assert json.loads(capsys.readouterr().out)["performance"]["feature_count"] == 2


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _batch_stats_fixture():
    """Feature matrix with ties and censoring (as real methylation data would have) for kernel tests."""
    rng = np.random.default_rng(11)
    n_samples, n_features = 40, 50
    time = rng.integers(1, 12, n_samples).astype(float)  # integer times guarantee ties
    status = rng.integers(0, 2, n_samples).astype(float)
    status[:10] = 1.0  # ensure enough events
    values = rng.normal(size=(n_features, n_samples))
    values[1, :] = time * -0.25 + rng.normal(scale=0.2, size=n_samples)  # a strongly informative feature
    matrix = pd.DataFrame(values, index=[f"cg{i:04}" for i in range(n_features)],
                          columns=[f"S{j}" for j in range(n_samples)])
    pheno = pd.DataFrame({"sample_id": matrix.columns, "time": time, "event": status}).set_index("sample_id", drop=False)
    endpoint = {"name": "d", "time": "time", "status": "event"}
    return matrix, pheno, endpoint


def _stats_max_abs_diff(reference, candidate, cols):
    a = reference[cols].to_numpy(float)
    b = candidate.loc[reference.index][cols].to_numpy(float)
    finite = np.isfinite(a) & np.isfinite(b)
    assert np.array_equal(~np.isfinite(a), ~np.isfinite(b)), "non-finite pattern differs"
    return float(np.abs(a[finite] - b[finite]).max())


def test_batched_cox_matches_exact_cpu():
    matrix, pheno, endpoint = _batch_stats_fixture()
    cols = ["coef", "se", "ordinary_z", "standardized_coef", "standardized_z", "p_value"]
    exact = _endpoint_stats(matrix, pheno, endpoint, [], engine="exact").set_index("feature")
    for kind in ("numpy", "torch_cpu"):
        got = _endpoint_stats(matrix, pheno, endpoint, [], engine="batched",
                              backend=_CoxBackend(kind)).set_index("feature")
        assert _stats_max_abs_diff(exact, got, cols) < 1e-5


@pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU not available")
def test_batched_cox_gpu_matches_cpu_to_machine_precision():
    matrix, pheno, endpoint = _batch_stats_fixture()
    cols = ["coef", "se", "ordinary_z", "standardized_coef", "standardized_z", "p_value"]
    cpu = _endpoint_stats(matrix, pheno, endpoint, [], engine="batched",
                          backend=_CoxBackend("numpy")).set_index("feature")
    gpu = _endpoint_stats(matrix, pheno, endpoint, [], engine="gpu",
                          backend=_CoxBackend("torch_cuda")).set_index("feature")
    # CPU and GPU run the identical float64 kernel: they must agree far below the reference tolerance.
    assert _stats_max_abs_diff(cpu, gpu, cols) < 1e-9


def test_batched_constant_feature_is_frozen():
    # A perfectly constant covariate carries no survival signal; the batched engine must
    # freeze it (coef=0, se=0) rather than diverge, so it sorts to the bottom of the ranking.
    matrix, pheno, endpoint = _batch_stats_fixture()
    matrix.iloc[0, :] = 0.5
    got = _endpoint_stats(matrix, pheno, endpoint, [], engine="batched",
                          backend=_CoxBackend("numpy")).set_index("feature")
    assert got.loc["cg0000", "coef"] == 0.0
    assert got.loc["cg0000", "se"] == 0.0
    assert np.isfinite(got.loc["cg0001", "ordinary_z"])  # informative feature still fit


def _covariate_fixture():
    rng = np.random.default_rng(23)
    n_samples, n_features = 50, 40
    time = rng.integers(1, 14, n_samples).astype(float)
    status = rng.integers(0, 2, n_samples).astype(float)
    status[:12] = 1.0
    grp = rng.choice(["A", "B", "C"], size=n_samples)          # strata (3 levels)
    age = rng.normal(60, 10, n_samples)                        # numeric adjustment
    values = rng.normal(size=(n_features, n_samples))
    values[1, :] = time * -0.2 + rng.normal(scale=0.3, size=n_samples)
    matrix = pd.DataFrame(values, index=[f"cg{i:04}" for i in range(n_features)],
                          columns=[f"S{j}" for j in range(n_samples)])
    pheno = pd.DataFrame({"sample_id": matrix.columns, "time": time, "event": status,
                          "grp": grp, "age": age}).set_index("sample_id", drop=False)
    return matrix, pheno, {"name": "d", "time": "time", "status": "event"}


_COV_CASES = {
    "strata": [{"column": "grp", "type": "categorical", "mode": "strata"}],
    "adjust_numeric": [{"column": "age", "type": "numeric", "mode": "adjust"}],
    "adjust_categorical": [{"column": "grp", "type": "categorical", "mode": "adjust"}],
    "strata_and_adjust": [{"column": "grp", "type": "categorical", "mode": "strata"},
                          {"column": "age", "type": "numeric", "mode": "adjust"}],
}


@pytest.mark.parametrize("case", list(_COV_CASES))
def test_batched_covariates_match_exact(case):
    matrix, pheno, endpoint = _covariate_fixture()
    covariates = _COV_CASES[case]
    cols = ["coef", "se", "ordinary_z", "standardized_coef", "standardized_z", "p_value"]
    exact = _endpoint_stats(matrix, pheno, endpoint, covariates, engine="exact").set_index("feature")
    got = _endpoint_stats(matrix, pheno, endpoint, covariates, engine="batched",
                          backend=_CoxBackend("numpy")).set_index("feature")
    assert _stats_max_abs_diff(exact, got, cols) < 1e-5


@pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU not available")
def test_batched_covariates_gpu_matches_cpu():
    matrix, pheno, endpoint = _covariate_fixture()
    covariates = _COV_CASES["strata_and_adjust"]
    cols = ["coef", "se", "ordinary_z", "standardized_coef", "standardized_z", "p_value"]
    cpu = _endpoint_stats(matrix, pheno, endpoint, covariates, engine="batched",
                          backend=_CoxBackend("numpy")).set_index("feature")
    gpu = _endpoint_stats(matrix, pheno, endpoint, covariates, engine="gpu",
                          backend=_CoxBackend("torch_cuda")).set_index("feature")
    assert _stats_max_abs_diff(cpu, gpu, cols) < 1e-9


def _stacked_meta_configs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    n, p = 30, 10
    samples = [f"S{i:02}" for i in range(n)]
    beta = rng.uniform(.05, .95, (p, n))
    matrix = pd.DataFrame(beta, columns=samples)
    matrix.insert(0, "feature", [f"cg{i:03}" for i in range(p)])
    matrix.to_csv(tmp_path / "matrix.csv", index=False)
    cohort = rng.choice(["A", "B"], size=n)
    time = np.maximum(1, 300 - 100 * (beta[0] - beta[1]) + rng.normal(0, 20, n))
    status = rng.integers(0, 2, n).astype(float)
    ph = pd.DataFrame({"sample_id": samples, "time": time, "event": status, "cohort": cohort})
    ph["time_A"] = np.where(cohort == "A", time, np.nan); ph["status_A"] = np.where(cohort == "A", status, np.nan)
    ph["time_B"] = np.where(cohort == "B", time, np.nan); ph["status_B"] = np.where(cohort == "B", status, np.nan)
    ph.to_csv(tmp_path / "pheno.csv", index=False)
    base = dict(schema_version=1, methodology="legacy_exact", analysis_id="gm",
                matrix={"path": "matrix.csv", "orientation": "features_by_samples", "scale": "beta", "feature_id_column": "feature"},
                phenotype={"path": "pheno.csv", "sample_id_column": "sample_id"},
                search={"min_features": 1, "max_features": 3, "step": 1, "cross_validation": "loo"},
                selection={"mode": "manual", "feature_count": 2}, execution={"workers": 1},
                validation={"allow_unmatched_samples": False}, output_dir="out")
    grouped = {**base, "endpoints": [{"name": "pooled", "time": "time", "status": "event",
                                      "prediction_time": 200, "group_column": "cohort"}]}
    wide = {**base, "endpoints": [{"name": "pooled:A", "time": "time_A", "status": "status_A", "prediction_time": 200},
                                  {"name": "pooled:B", "time": "time_B", "status": "status_B", "prediction_time": 200}]}
    (tmp_path / "grouped.yml").write_text(yaml.safe_dump(grouped))
    (tmp_path / "wide.yml").write_text(yaml.safe_dump(wide))
    return tmp_path / "grouped.yml", tmp_path / "wide.yml"


@pytest.mark.parametrize("engine", ["exact", "batched"])
def test_group_column_meta_ranking_matches_wide_two_endpoints(tmp_path, engine):
    # The random-effects pooling (hence the ranking and selected features) is independent of the
    # per-fold score normalization, so group_column matches the hand-written wide two-endpoint form
    # on the ranking. (The per-fold risk-score normalization differs: group_column uses a uniform
    # all-cohort scheme, while the wide form preserves the legacy two-endpoint quirk.)
    grouped, wide = _stacked_meta_configs(tmp_path)
    gcfg = read_config(grouped); gcfg.setdefault("execution", {})["engine"] = engine
    wcfg = read_config(wide); wcfg.setdefault("execution", {})["engine"] = engine
    data = validate_inputs(gcfg)
    assert [ep["name"] for ep in data.endpoints] == ["pooled:A", "pooled:B"]  # expanded internally
    g = search_features(gcfg); w = search_features(wcfg)
    cols = ["coef", "se", "ordinary_z", "standardized_coef"]
    gr, wr = g["ranking"].set_index("feature"), w["ranking"].set_index("feature")
    assert gr.index.tolist() == wr.index.tolist()
    np.testing.assert_allclose(gr[cols].to_numpy(float), wr.loc[gr.index, cols].to_numpy(float),
                               rtol=0, atol=1e-10, equal_nan=True)


def test_group_column_supports_more_than_two_cohorts(tmp_path):
    grouped, _ = _stacked_meta_configs(tmp_path)
    rng = np.random.default_rng(9)
    ph = pd.read_csv(tmp_path / "pheno.csv")
    ph["cohort"] = rng.choice(["A", "B", "C"], size=len(ph))
    ph["event"] = np.where(np.arange(len(ph)) % 2 == 0, 1.0, 0.0)  # events spread across all cohorts
    ph.to_csv(tmp_path / "pheno.csv", index=False)
    data = validate_inputs(read_config(grouped))
    assert [ep["name"] for ep in data.endpoints] == ["pooled:A", "pooled:B", "pooled:C"]  # three cohorts
    result = search_features(read_config(grouped))
    assert not result["ranking"].empty and result["ranking"]["ordinary_z"].notna().any()
    # a single-level cohort column is rejected
    ph["cohort"] = "A"; ph.to_csv(tmp_path / "pheno.csv", index=False)
    with pytest.raises(ValueError, match="at least two"):
        validate_inputs(read_config(grouped))


def test_verify_engine_parity_passes(tmp_path):
    path = fixture(tmp_path)
    engines = ["exact", "batched"] + (["gpu"] if _cuda_available() else [])
    report = verify_engine_parity(path, feature_count=2, engines=engines, tolerance=1e-5)
    assert report["overall_pass"], report
    assert report["comparisons"]["exact_vs_batched"]["selected_features_identical"]
    if _cuda_available():
        assert report["comparisons"]["batched_vs_gpu"]["pass"]


@pytest.mark.parametrize("weight_refit", ["covariate_aware", "legacy_unadjusted"])
def test_batched_engine_pipeline_matches_exact_for_both_weight_refits(tmp_path, weight_refit):
    path = fixture(tmp_path, covariate={"column":"group", "type":"categorical", "mode":"strata"})
    exact_cfg = read_config(path)
    exact_cfg["final_model"] = {"weight_refit":weight_refit}
    exact = _run_pipeline(exact_cfg, feature_count=2)
    cfg = read_config(path)
    cfg["final_model"] = {"weight_refit":weight_refit}
    cfg.setdefault("execution", {})["engine"] = "batched"
    batched = _run_pipeline(cfg, feature_count=2)
    assert (exact["selected_features"]["feature"].tolist()
            == batched["selected_features"]["feature"].tolist())
    np.testing.assert_allclose(exact["risk_scores"]["risk_score"].to_numpy(float),
                               batched["risk_scores"]["risk_score"].to_numpy(float), atol=1e-5)


def fixture(tmp_path: Path, paired=False, covariate=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    samples = [f"S{i:02}" for i in range(10)]
    features = [f"cg{i:05}" for i in range(5)]
    beta = rng.uniform(.05, .95, (5, 10))
    matrix = pd.DataFrame(beta, columns=samples)
    matrix.insert(0, "feature", features)
    matrix.to_csv(tmp_path / "matrix.csv", index=False)
    risk = beta[0] - beta[1]
    pheno = pd.DataFrame({"sample_id": samples, "time1": np.maximum(1, 300 - 100*risk + rng.normal(0, 20, 10)),
                          "status1": [1, 0] * 5, "time2": np.maximum(1, 250 - 80*risk + rng.normal(0, 20, 10)),
                          "status2": [0, 1, 1, 0, 1] * 2, "group": ["A", "B"] * 5, "age": np.arange(10) + 2})
    pheno.to_csv(tmp_path / "pheno.csv", index=False)
    cfg = {"schema_version": 1, "methodology": "legacy_exact", "analysis_id": "synthetic", "matrix": {"path": "matrix.csv",
           "orientation": "features_by_samples", "scale": "beta", "feature_id_column": "feature"},
           "phenotype": {"path": "pheno.csv", "sample_id_column": "sample_id"},
           "endpoints": [{"name": "one", "time": "time1", "status": "status1", "prediction_time": 200}],
           "covariates": [], "search": {"min_features": 1, "max_features": 2, "step": 1,
           "cross_validation": "loo"}, "selection": {"mode": "manual", "feature_count": 2}, "execution": {"workers": 1,
           "seed": 1, "slurm": {"cpus": 2, "memory": "2G", "time": "00:10:00"}},
           "annotation": None, "validation": {"allow_unmatched_samples": False}, "output_dir": "out"}
    if paired:
        cfg["endpoints"].append({"name": "two", "time": "time2", "status": "status2", "prediction_time": 200})
    if covariate:
        cfg["covariates"] = [covariate]
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def test_beta_to_m():
    np.testing.assert_allclose(beta_to_m(np.array([.2, .5, .8])), [-2, 0, 2])


def test_survival_roc_nne_matches_r_reference():
    if not shutil.which("Rscript"):
        return
    time = np.array([2,3,4,5,7,8,9,11,12,14],float); status = np.array([1,0,1,1,0,1,0,1,0,1])
    marker = np.array([.1,.4,.2,.8,.6,.9,.3,.7,.5,1.0])
    expression = ("library(survivalROC);cat(sprintf('%.17g',survivalROC(Stime=c(2,3,4,5,7,8,9,11,12,14),"
                  "status=c(1,0,1,1,0,1,0,1,0,1),marker=c(.1,.4,.2,.8,.6,.9,.3,.7,.5,1),"
                  "predict.time=10,span=.25*10^(-.20))$AUC))")
    expected = float(subprocess.check_output(["Rscript","-e",expression],text=True))
    assert _survival_roc_nne_auc(time,status,marker,10) == pytest.approx(expected,abs=1e-12)


def test_native_efron_cox_matches_r_reference():
    if not shutil.which("Rscript"):
        return
    time = np.array([1,2,2,3,4,4,5,6],float); status = np.array([1,1,1,0,1,1,0,1],float)
    feature = np.array([-.8,-.3,.1,.4,.7,1.0,1.2,1.5])
    expression = ("library(survival);x=c(-.8,-.3,.1,.4,.7,1,1.2,1.5);"
                  "f=coxph(Surv(c(1,2,2,3,4,4,5,6),c(1,1,1,0,1,1,0,1))~x,ties='efron');"
                  "s=summary(f)$coefficients[1,];cat(sprintf('%.17g %.17g',s[['coef']],s[['se(coef)']]))")
    expected = np.fromstring(subprocess.check_output(["Rscript","-e",expression],text=True),sep=" ")
    beta,se = _cox_fit(time,status,feature)
    np.testing.assert_allclose([beta[0],se[0]],expected,rtol=1e-6,atol=1e-7)


def test_selection_tie_uses_smallest():
    metrics = pd.DataFrame({"feature_count": [3, 4, 5], "auc": [.7, .8, .8]})
    assert select_feature_count(metrics, mode="max_auc") == 4
    assert select_feature_count(metrics, 5) == 5


def test_validation_and_local_pipeline(tmp_path):
    path = fixture(tmp_path)
    cfg = read_config(path)
    data = validate_inputs(cfg)
    assert data.matrix.shape == (5, 10)
    result = run_pipeline(cfg, feature_count=2)
    assert result["performance"]["feature_count"] == 2
    assert (tmp_path / "out" / "python" / "selected_features.csv").is_file()
    assert json.loads((tmp_path / "out" / "python" / "result.json").read_text())["engine"] == "python"


def test_local_progress_uses_stderr_style_stream(tmp_path):
    stream = io.StringIO()
    progress = ProgressReporter(stream=stream, heartbeat_seconds=3600)
    try:
        run_pipeline(fixture(tmp_path), feature_count=2, progress=progress)
    finally:
        progress.close()
    output = stream.getvalue()
    assert "Validating inputs" in output
    assert "Search:" in output
    assert "Extraction:" in output
    assert "Results written" in output


def test_worker_cli_aliases_override_only_runtime_config(tmp_path, monkeypatch, capsys):
    path = fixture(tmp_path)
    original = path.read_text()
    captured = []

    def fake_run(cfg, backend, dry_run, feature_count, progress):
        captured.append((cfg["execution"]["workers"], backend))
        return {"output_dir": "out", "performance": {}}

    monkeypatch.setattr("metacpg.cli.run_pipeline", fake_run)
    assert main(["run", "--config", str(path), "--workers", "3"]) == 0
    assert main(["run", "--config", str(path), "--cores", "4"]) == 0
    assert captured == [(3, "local"), (4, "local")]
    assert path.read_text() == original
    stdout = capsys.readouterr().out
    assert stdout.count('"output_dir": "out"') == 2


def test_worker_cli_validation_and_slurm_rejection(tmp_path):
    parser = build_parser()
    assert parser.parse_args(["run", "--config", "analysis.yml", "--cores", "2"]).workers == 2
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--config", "analysis.yml", "--workers", "0"])
    with pytest.raises(ValueError, match="local-only"):
        main(["run", "--config", str(fixture(tmp_path)), "--backend", "slurm", "--workers", "2"])


def test_heartbeat_options_and_config(tmp_path, monkeypatch):
    parser = build_parser()
    assert parser.parse_args(["run", "--config", "analysis.yml", "--no-heartbeat"]).heartbeat_seconds == 0
    assert parser.parse_args(["run", "--config", "analysis.yml", "--heartbeat-seconds", "900"]).heartbeat_seconds == 900
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--config", "analysis.yml", "--no-heartbeat", "--heartbeat-seconds", "5"])
    assert heartbeat_seconds_from_config({"execution": {}}) == 300
    assert heartbeat_seconds_from_config({"execution": {"heartbeat_seconds": 0}}) == 0
    with pytest.raises(ValueError, match="non-negative"):
        heartbeat_seconds_from_config({"execution": {"heartbeat_seconds": -1}})
    assert parser.parse_args(["run", "--config", "analysis.yml", "--no-progress"]).show_progress is False
    assert progress_enabled_from_config({"execution": {}})
    assert not progress_enabled_from_config({"execution": {"progress": False}})
    with pytest.raises(ValueError, match="true or false"):
        progress_enabled_from_config({"execution": {"progress": "no"}})

    observed = {}

    def fake_run(cfg, backend, dry_run, feature_count, progress):
        observed["heartbeat_seconds"] = cfg["execution"]["heartbeat_seconds"]
        observed["progress"] = cfg["execution"]["progress"]
        return {"output_dir": "out", "performance": {}}

    monkeypatch.setattr("metacpg.cli.run_pipeline", fake_run)
    assert main(["run", "--config", str(fixture(tmp_path)), "--no-heartbeat", "--no-progress"]) == 0
    assert observed["heartbeat_seconds"] == 0
    assert observed["progress"] is False


def test_memory_budget_cli_and_parser(tmp_path, monkeypatch):
    assert parse_memory_budget("8G") == 8 * 1024 ** 3
    assert parse_memory_budget("1.5MiB") == int(1.5 * 1024 ** 2)
    with pytest.raises(ValueError, match="positive"):
        parse_memory_budget("0G")
    with pytest.raises(ValueError, match="units"):
        parse_memory_budget("eight gigs")
    captured = []

    def fake_run(cfg, backend, dry_run, feature_count, progress):
        captured.append(cfg["execution"]["memory_budget"])
        return {"output_dir": "out", "performance": {}}

    monkeypatch.setattr("metacpg.cli.run_pipeline", fake_run)
    assert main(["run", "--config", str(fixture(tmp_path)), "--memory-budget", "2G"]) == 0
    assert captured == [2 * 1024 ** 3]


def test_process_memory_reads_pss_then_rss(tmp_path):
    proc = tmp_path / "proc" / "42"
    proc.mkdir(parents=True)
    (proc / "smaps_rollup").write_text("Rss: 90 kB\nPss: 40 kB\n")
    assert _process_memory(42, tmp_path / "proc") == (40 * 1024, "PSS")
    (proc / "smaps_rollup").unlink()
    (proc / "status").write_text("Name:\tpython\nVmRSS:\t64 kB\n")
    assert _process_memory(42, tmp_path / "proc") == (64 * 1024, "RSS")


def test_memory_budget_caps_process_workers(tmp_path, monkeypatch):
    data = validate_inputs(read_config(fixture(tmp_path)))
    monkeypatch.setattr("metacpg.core._process_memory", lambda pid: (256 * 1024 ** 2, "PSS"))
    monkeypatch.setattr("metacpg.core.os.cpu_count", lambda: 8)
    plan = _process_worker_plan(data, requested=8, task_count=100, memory_budget=768 * 1024 ** 2)
    assert plan["permitted_workers"] == 2
    with pytest.raises(ValueError, match="cannot run one process worker"):
        _process_worker_plan(data, requested=8, task_count=100, memory_budget=500 * 1024 ** 2)


def test_preflight_estimates_resources_without_writing_output(tmp_path):
    path = fixture(tmp_path)
    result = preflight_resources(path)
    assert result["valid"]
    assert result["data"] == {"features": 5, "samples": 10, "endpoints": 1}
    # One LOO task per held-out fold; the ranking is count-independent so the candidate
    # count grid no longer multiplies the task or fit totals (search: 5 full-data + 5*10 LOO).
    assert result["search"]["loo_tasks"] == 10
    assert result["search"]["cox_feature_fits"] == 55
    assert result["extraction"] == {"feature_count": 2, "loo_tasks": 10, "cox_feature_fits": 55}
    assert result["engine"] == "exact"
    assert not (tmp_path / "out").exists()


def test_preflight_cli_uses_runtime_resource_overrides(tmp_path, monkeypatch, capsys):
    observed = {}

    def fake_preflight(cfg, feature_count):
        observed.update(cfg["execution"])
        return {"valid": True, "memory": {}}

    monkeypatch.setattr("metacpg.cli.preflight_resources", fake_preflight)
    assert main(["run", "--config", str(fixture(tmp_path)), "--cores", "3",
                 "--memory-budget", "2G", "--preflight"]) == 0
    assert observed["workers"] == 3
    assert observed["memory_budget"] == 2 * 1024 ** 3
    assert json.loads(capsys.readouterr().out)["valid"]


def test_cli_progress_stays_off_stdout(tmp_path, capsys):
    assert main(["run", "--config", str(fixture(tmp_path)), "--workers", "1"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["output_dir"].endswith("out/python")
    assert "Search:" in captured.err
    assert "Extraction:" in captured.err


def test_worker_count_preserves_legacy_results(tmp_path):
    path = fixture(tmp_path)
    single = run_pipeline(path, feature_count=2)
    cfg = yaml.safe_load(path.read_text())
    cfg["execution"]["workers"] = 2
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    multiple = run_pipeline(path, feature_count=2)
    pd.testing.assert_frame_equal(single["scores"].reset_index(drop=True),
                                  multiple["scores"].reset_index(drop=True))
    pd.testing.assert_frame_equal(single["extraction_scores"].reset_index(drop=True),
                                  multiple["extraction_scores"].reset_index(drop=True))


def test_all_covariate_and_meta_modes(tmp_path):
    for i, cov in enumerate((
        {"column": "group", "type": "categorical", "mode": "strata"},
        {"column": "group", "type": "categorical", "mode": "adjust"},
        {"column": "age", "type": "numeric", "mode": "adjust"},
    )):
        path = fixture(tmp_path / str(i), paired=True, covariate=cov)
        result = run_pipeline(path, feature_count=2)
        assert len(result["selected_features"]) == 2
        metrics = result["metrics"]
        # The tiny synthetic fixture can make survivalROC's NNE AUC estimator degenerate (NaN)
        # regardless of covariates — check column presence and the more robust C-index/p-value.
        for col in ("auc_adjusted", "c_index_adjusted", "neglog10_p_adjusted"):
            assert col in metrics.columns
        assert metrics["c_index_adjusted"].notna().all()
        assert "auc_adjusted" in result["performance"]


def test_design_handles_missing_categorical_strata_labels():
    pheno = pd.DataFrame({
        "cohort": pd.Series(["A", "B", None], dtype="category"),
        "age_group": pd.Series(["younger", "older", None], dtype="category"),
    })
    design, strata = _design(pheno, [
        {"column": "cohort", "type": "categorical", "mode": "strata"},
        {"column": "age_group", "type": "categorical", "mode": "strata"},
    ])
    assert design.shape == (3, 0)
    assert strata.tolist() == ["A|younger", "B|older", "__MISSING__|__MISSING__"]


def test_slurm_dry_run(tmp_path):
    cfg = read_config(fixture(tmp_path))
    result = submit_slurm(cfg, dry_run=True)
    script = Path(result["scripts"][0]).read_text()
    assert "#SBATCH --cpus-per-task=2" in script
    assert "#SBATCH --array=0-1" in script
    assert "metacpg-py worker" in script
    assert "--phase search" in script
    assert "--phase extraction" in Path(result["scripts"][1]).read_text()
    assert "metacpg-py aggregate" in Path(result["scripts"][2]).read_text()
    assert not result["submitted"]


def test_leave_one_out_search_smoke(tmp_path):
    path = fixture(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["search"].update({"cross_validation": "loo", "min_features": 1, "max_features": 2})
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = run_pipeline(path, feature_count=2)
    assert not result["inclusion_frequency"].empty
    assert set(result["scores"]["feature_count"]) == {1, 2}
    assert not list((tmp_path / "out" / "cache").rglob("*.json"))
    rerun = run_pipeline(path, feature_count=2)
    pd.testing.assert_frame_equal(result["scores"].reset_index(drop=True), rerun["scores"].reset_index(drop=True))


def test_literal_refit_counts_and_fresh_extraction(tmp_path):
    result = run_pipeline(fixture(tmp_path), feature_count=2)
    features, folds, counts = 5, 10, 2
    assert result["search_cox_fit_count"] == features + features * folds * counts
    assert result["extraction_cox_fit_count"] == features * folds + features
    assert (result["scores"]["fold_scale"] > 0).all()
    assert set(result["task_provenance"]["resumed"]) == {False}
    assert set(result["extraction_task_provenance"]["phase"]) == {"extraction"}


def test_covariate_selection_exports_covariate_aware_weights_by_default(tmp_path, capsys):
    cov = {"column":"group","type":"categorical","mode":"strata"}; path = fixture(tmp_path,covariate=cov)
    cfg = read_config(path)
    cfg["covariates"].append({"column":"age","type":"numeric","mode":"adjust"})
    data = validate_inputs(cfg); result = run_pipeline(cfg,feature_count=2)
    expected = _endpoint_stats(data.matrix,data.phenotype,data.endpoints[0],data.covariates).set_index("feature")
    selected = result["selected_features"].set_index("feature")
    np.testing.assert_allclose(selected["standardized_coef"],expected.loc[selected.index,"standardized_coef"])
    assert result["weight_refit"] == "covariate_aware"
    captured = capsys.readouterr()
    assert "LOO selection is unchanged" in captured.out + captured.err


def test_legacy_unadjusted_weight_refit_is_available(tmp_path):
    cov = {"column":"group","type":"categorical","mode":"strata"}; path = fixture(tmp_path,covariate=cov)
    cfg = read_config(path); cfg["final_model"] = {"weight_refit": "legacy_unadjusted"}
    data = validate_inputs(cfg); result = run_pipeline(cfg,feature_count=2)
    expected = _endpoint_stats(data.matrix,data.phenotype,data.endpoints[0],[]).set_index("feature")
    selected = result["selected_features"].set_index("feature")
    np.testing.assert_allclose(selected["standardized_coef"],expected.loc[selected.index,"standardized_coef"])
    expected_scores, _, expected_scaling = _legacy_full_scores(
        data.matrix, expected.reset_index(), result["selected_features"]["feature"].tolist())
    pd.testing.assert_frame_equal(result["risk_scores"].reset_index(drop=True), expected_scores)
    assert result["scaling"] == expected_scaling
    assert result["weight_refit"] == "legacy_unadjusted"


def test_weight_refit_mode_changes_only_final_model_for_covariate_run(tmp_path):
    cov = {"column":"group","type":"categorical","mode":"strata"}
    path = fixture(tmp_path,covariate=cov)
    aware_cfg = read_config(path); legacy_cfg = read_config(path)
    aware_cfg["final_model"] = {"weight_refit":"covariate_aware"}
    legacy_cfg["final_model"] = {"weight_refit":"legacy_unadjusted"}
    aware = run_pipeline(aware_cfg,feature_count=2); legacy = run_pipeline(legacy_cfg,feature_count=2)
    pd.testing.assert_frame_equal(aware["metrics"].reset_index(drop=True),legacy["metrics"].reset_index(drop=True))
    pd.testing.assert_frame_equal(aware["extraction_scores"].reset_index(drop=True),legacy["extraction_scores"].reset_index(drop=True))
    assert aware["selected_features"]["feature"].tolist() == legacy["selected_features"]["feature"].tolist()
    assert not np.allclose(aware["selected_features"]["standardized_coef"],
                           legacy["selected_features"]["standardized_coef"])
    assert _fingerprint(aware_cfg) != _fingerprint(legacy_cfg)


def test_omitted_and_explicit_covariate_aware_have_same_fingerprint(tmp_path):
    omitted = read_config(fixture(tmp_path))
    explicit = read_config(fixture(tmp_path))
    explicit["final_model"] = {"weight_refit":"covariate_aware"}
    assert _fingerprint(omitted) == _fingerprint(explicit)


def test_weight_refit_modes_are_identical_without_covariates(tmp_path):
    aware_cfg = read_config(fixture(tmp_path / "aware")); legacy_cfg = read_config(fixture(tmp_path / "legacy"))
    aware_cfg["final_model"] = {"weight_refit":"covariate_aware"}
    legacy_cfg["final_model"] = {"weight_refit":"legacy_unadjusted"}
    aware = run_pipeline(aware_cfg,feature_count=2); legacy = run_pipeline(legacy_cfg,feature_count=2)
    pd.testing.assert_frame_equal(aware["selected_features"].reset_index(drop=True),
                                  legacy["selected_features"].reset_index(drop=True))
    pd.testing.assert_frame_equal(aware["risk_scores"].reset_index(drop=True),
                                  legacy["risk_scores"].reset_index(drop=True))
    assert aware["scaling"] == legacy["scaling"]


def test_invalid_weight_refit_mode_is_rejected(tmp_path):
    cfg = read_config(fixture(tmp_path)); cfg["final_model"] = {"weight_refit":"invalid"}
    with pytest.raises(ValueError,match="final_model.weight_refit"):
        validate_inputs(cfg)


def test_legacy_performance_reports_wald_pvalue():
    rng = np.random.default_rng(3)
    n = 60
    score = rng.normal(size=n)
    time_v = np.maximum(1, 200 - 40 * score + rng.normal(0, 20, n))
    status = rng.integers(0, 2, n).astype(float)
    auc, c_index, p = _legacy_performance(time_v, status, score, 150)
    assert 0 <= auc <= 1 and 0 <= c_index <= 1
    assert 0 < p <= 1


def test_legacy_performance_adjusted_uses_joint_linear_predictor():
    rng = np.random.default_rng(4)
    n = 60
    score = rng.normal(size=n)
    zcov = rng.normal(size=(n, 1))
    time_v = np.maximum(1, 200 - 40 * score - 20 * zcov[:, 0] + rng.normal(0, 20, n))
    status = rng.integers(0, 2, n).astype(float)
    auc, c_index, p = _legacy_performance_adjusted(time_v, status, score, zcov, None, 150)
    assert 0 <= auc <= 1 and 0 <= c_index <= 1
    assert 0 < p <= 1
    # Zero covariate columns (strata-only covariates) must still work, falling back to score alone.
    auc_no_cov, c_index_no_cov, p_no_cov = _legacy_performance_adjusted(
        time_v, status, score, np.empty((n, 0)), None, 150)
    assert 0 <= auc_no_cov <= 1 and 0 <= c_index_no_cov <= 1 and 0 < p_no_cov <= 1


def test_search_metrics_plot_ok_with_covariates(tmp_path):
    cov = {"column": "group", "type": "categorical", "mode": "strata"}
    path = _manual_no_count_config(tmp_path)
    cfg = yaml.safe_load(path.read_text()); cfg["covariates"] = [cov]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    run_pipeline(path)
    out = tmp_path / "out" / "python"
    assert (out / "search_metrics.png").is_file()
    assert not (out / "plot_warning.txt").exists()
    metrics = pd.read_csv(out / "metrics.csv")
    for col in ("auc_adjusted", "c_index_adjusted", "neglog10_p_adjusted"):
        assert col in metrics.columns
    assert metrics["c_index_adjusted"].notna().all()


def test_explicit_resume_never_reuses_search_for_extraction(tmp_path):
    path = fixture(tmp_path); cfg = yaml.safe_load(path.read_text())
    cfg["execution"]["resume_run_id"] = "resume-test"; path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    first = run_pipeline(path,feature_count=2); second = run_pipeline(path,feature_count=2)
    assert set(first["task_provenance"]["resumed"]) == {False}
    assert set(second["task_provenance"]["resumed"]) == {True}
    assert set(first["extraction_task_provenance"]["resumed"]) == {False}
    assert set(second["extraction_task_provenance"]["resumed"]) == {False}
    assert second["extraction_cox_fits_executed"] == first["extraction_cox_fits_executed"]


def test_distributed_workers_and_aggregation_preserve_phase_boundaries(tmp_path):
    path = fixture(tmp_path); run_id = "distributed-test"
    for n in (1,2):
        assert run_worker(path,"search",n,run_id)["cox_fits_executed"] == 50
    assert run_worker(path,"extraction",2,run_id)["cox_fits_executed"] == 50
    result = aggregate_run(path,run_id,2)
    assert set(result["task_provenance"]["resumed"]) == {True}
    assert set(result["extraction_task_provenance"]["resumed"]) == {True}
    assert result["search_cox_fit_count"] == 105
    assert result["extraction_cox_fit_count"] == 55
