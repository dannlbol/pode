"""End-to-end CLI: benchmark --quick, train (masked), and apply (both engines)."""

import json

import numpy as np
import pandas as pd
import yaml

from pode.cli.main import build_parser, main


def _write(tmp_path, cohort):
    data = tmp_path / "cohort.csv"
    cohort.to_csv(data, index=False)
    icfg = tmp_path / "impute.yaml"
    icfg.write_text(yaml.safe_dump({
        "m": 3,
        "folds": 2,
        "block_groups": {"lat": ["lat_*"]},
        "scenarios": [
            {"name": "mcar_20", "kind": "mcar", "rate": 0.2},
            {"name": "lat_block", "kind": "block", "group": "lat", "row_frac": 0.3},
        ],
        "engines": ["mean_mode", "mice", "tvae-decoder", "tvae-masked"],
        "epochs": 5,
        "latent": {"k": 100, "pcn_steps": 5},
        "cox": {"covariates": ["bio", "treated", "lat_0"]},
        "delta": {"deltas": [-1.0, 0.0, 1.0]},
    }))
    cfg = tmp_path / "pode.yaml"
    cfg.write_text(yaml.safe_dump({"synthesizer": {"epochs": 5, "batch_size": 100, "embedding_dim": 8,
                                                   "compress_dims": [32], "decompress_dims": [32]}}))
    return data, icfg, cfg


def test_impute_parser_registered():
    args = build_parser().parse_args(["synthesis", "impute", "apply", "--model", "m.pkl", "--data", "x.csv", "--m", "5"])
    assert args.m == 5 and args.func.__name__ == "cmd_apply"
    # Existing synthesis commands still parse.
    assert build_parser().parse_args(["synthesis", "train", "--data", "x.csv"]).func.__name__ == "cmd_train"


def test_benchmark_quick_end_to_end(tmp_path, cohort):
    data, icfg, cfg = _write(tmp_path, cohort)
    out = tmp_path / "bench"
    rc = main(["synthesis", "impute", "benchmark", "--data", str(data), "--config", str(cfg),
               "--impute-config", str(icfg), "--output-dir", str(out), "--quick", "--n-jobs", "2"])
    assert rc == 0
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["engine"]) == {"mean_mode", "mice", "tvae-decoder", "tvae-masked"}
    assert set(summary["scenario"]) == {"mcar_20", "lat_block"}
    assert summary["cont_crps_norm"].notna().all()
    for name in ["cell_metrics", "rubin_cox", "delta_sensitivity", "diagnostics", "dependence"]:
        assert (out / f"{name}.csv").is_file(), name
    assert (out / "report.md").read_text().startswith("# Imputation benchmark")
    assert (out / "plots" / "metrics_by_scenario.png").is_file()
    assert not (out / "failures.csv").exists()
    # Mean/mode is a single imputation: its 90% intervals have zero width.
    mm = summary[summary["engine"] == "mean_mode"]
    assert (mm["cont_width90_norm"] == 0).all()


def test_train_then_apply_masked_and_decoder(tmp_path, cohort):
    data, icfg, cfg = _write(tmp_path, cohort)
    incomplete = cohort.copy()
    rng = np.random.default_rng(0)
    for c in ["bio", "subgroup", "lat_0"]:
        incomplete.loc[rng.random(len(cohort)) < 0.25, c] = np.nan
    inc_path = tmp_path / "incomplete.csv"
    incomplete.to_csv(inc_path, index=False)

    model_dir = tmp_path / "masked"
    assert main(["synthesis", "impute", "train", "--data", str(inc_path), "--config", str(cfg),
                 "--impute-config", str(icfg), "--output-dir", str(model_dir), "--seed", "1"]) == 0
    assert (model_dir / "meta.json").is_file()

    trained = tmp_path / "sdv"
    assert main(["synthesis", "train", "--data", str(data), "--config", str(cfg),
                 "--output-dir", str(trained), "--seed", "1"]) == 0

    for model in (model_dir, trained / "model.pkl"):
        out = tmp_path / f"imputed_{model.name}"
        assert main(["synthesis", "impute", "apply", "--model", str(model), "--data", str(inc_path),
                     "--impute-config", str(icfg), "--m", "2", "--output-dir", str(out), "--seed", "3"]) == 0
        first = pd.read_csv(out / "imputed_001.csv")
        assert not first[["bio", "subgroup", "lat_0"]].isna().any().any()
        observed = incomplete["bio"].notna()
        assert np.allclose(first.loc[observed, "bio"], incomplete.loc[observed, "bio"])
        long = pd.read_csv(out / "imputed_long.csv")
        assert sorted(long[".imp"].unique()) == [0, 1, 2]
        diag = json.loads((out / "diagnostics.json").read_text())
        assert diag["m"] == 2 and diag["rows_conditioned"] > 0


def _split_files(tmp_path, cohort):
    train, test = cohort.iloc[:150].copy(), cohort.iloc[150:].copy()
    rng = np.random.default_rng(1)
    test.loc[rng.random(len(test)) < 0.3, "bio"] = np.nan
    test.loc[test.index[:12], ["lat_0", "lat_1"]] = np.nan
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path, test


def test_apply_engine_mice_and_hybrid_with_reference(tmp_path, cohort):
    _, icfg, cfg = _write(tmp_path, cohort)
    train_path, test_path, test = _split_files(tmp_path, cohort)

    out = tmp_path / "mice"
    assert main(["synthesis", "impute", "apply", "--engine", "mice", "--data", str(test_path),
                 "--config", str(cfg), "--impute-config", str(icfg), "--m", "2",
                 "--output-dir", str(out), "--seed", "1"]) == 0
    first = pd.read_csv(out / "imputed_001.csv")
    assert not first.isna().any().any()
    diag = json.loads((out / "diagnostics.json").read_text())
    assert diag["engine"] == "mice" and diag["outcome_policy"] == "use" and diag["reference"] is None

    out = tmp_path / "hybrid"
    assert main(["synthesis", "impute", "apply", "--engine", "hybrid", "--reference", str(train_path),
                 "--data", str(test_path), "--config", str(cfg), "--impute-config", str(icfg),
                 "--outcome-policy", "hide", "--fit-scope", "reference", "--m", "2",
                 "--output-dir", str(out), "--seed", "1"]) == 0
    first = pd.read_csv(out / "imputed_001.csv")
    assert not first.isna().any().any()
    assert np.allclose(first["OS_Time"], test["OS_Time"])
    diag = json.loads((out / "diagnostics.json").read_text())
    assert diag["outcome_policy"] == "hide" and diag["fit_scope"] == "reference"
    assert diag["rows_to_block_engine"] == 12 and diag["reference"].endswith("train.csv")


def test_apply_hide_without_reference_becomes_exclude(tmp_path, cohort):
    _, icfg, cfg = _write(tmp_path, cohort)
    _, test_path, _ = _split_files(tmp_path, cohort)
    out = tmp_path / "x"
    assert main(["synthesis", "impute", "apply", "--engine", "mice", "--data", str(test_path),
                 "--impute-config", str(icfg), "--outcome-policy", "hide", "--m", "2",
                 "--output-dir", str(out)]) == 0
    diag = json.loads((out / "diagnostics.json").read_text())
    assert diag["outcome_policy"] == "exclude"
    assert set(diag["excluded_columns"]) == {"OS_Time", "OS_Status"}


def test_apply_argument_errors(tmp_path, cohort):
    _, test_path, _ = _split_files(tmp_path, cohort)
    common = ["synthesis", "impute", "apply", "--data", str(test_path), "--output-dir", str(tmp_path / "e")]
    assert main(common) == 2  # neither --model nor --engine
    assert main(common + ["--model", "m.pkl", "--engine", "mice"]) == 2
    assert main(common + ["--model", "m.pkl", "--reference", "r.csv"]) == 2


def test_benchmark_hybrid_hide_and_union(tmp_path, cohort):
    data, _, cfg = _write(tmp_path, cohort)
    icfg = tmp_path / "impute2.yaml"
    icfg.write_text(yaml.safe_dump({
        "m": 3, "folds": 2, "epochs": 5, "plots": False,
        "block_groups": {"lat": ["lat_*"]},
        "scenarios": [{"name": "mixed", "kind": "union", "parts": [
            {"kind": "mcar", "rate": 0.15}, {"kind": "block", "group": "lat", "row_frac": 0.3}]}],
        "engines": ["mice",
                    {"name": "mice", "label": "mice-hide", "params": {"outcome_policy": "hide"}},
                    {"name": "hybrid", "params": {"gap_engine": "mice", "block_engine": "tvae-masked"}}],
        "latent": {"k": 100, "pcn_steps": 5},
        "cox": {"covariates": ["bio", "treated", "lat_0"]},
        "delta": {"enabled": False},
    }))
    out = tmp_path / "bench2"
    assert main(["synthesis", "impute", "benchmark", "--data", str(data), "--config", str(cfg),
                 "--impute-config", str(icfg), "--output-dir", str(out), "--n-jobs", "2"]) == 0
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["engine"]) == {"mice", "mice-hide", "hybrid"}
    report = json.loads((out / "report.json").read_text())
    assert report["leakage_settings"]["mice-hide"]["outcome_policy"] == "hide"
    assert report["leakage_settings"]["mice"]["outcome_policy"] == "use"
    assert "## Leakage settings" in (out / "report.md").read_text()
    diag = pd.read_csv(out / "diagnostics.csv")
    assert (diag.loc[diag["engine"] == "hybrid", "rows_to_block_engine"] > 0).all()


def test_plots_split_when_engines_exceed_palette(tmp_path):
    from pode.imputation.plots import SERIES, write_plots

    engines = [f"e{i}" for i in range(len(SERIES) + 1)]
    summary = pd.DataFrame([
        {"scenario": s, "engine": e, "cont_crps_norm": 0.5, "cont_coverage50": 0.5, "cont_coverage90": 0.9}
        for s in ("a", "b") for e in engines
    ])
    paths = write_plots({"summary": summary}, tmp_path, None, engines)
    names = {p.name for p in paths}
    assert {"metrics_by_scenario_part1.png", "metrics_by_scenario_part2.png"} <= names
