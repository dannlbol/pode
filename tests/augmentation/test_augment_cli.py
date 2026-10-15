"""End-to-end: benchmark --quick (with leakage checks), generate, report, parser registration."""

import json

import pandas as pd
import pytest
import yaml

from conftest import augment_config
from pode.augmentation import generator as G
from pode.cli.main import build_parser, main


def test_augment_parser_registered():
    args = build_parser().parse_args(["synthesis", "augment", "generate", "--data", "x.csv", "--strategy", "s",
                                      "--ratio", "0.5"])
    assert args.ratio == 0.5 and args.func.__name__ == "cmd_generate"
    # Existing synthesis commands still parse.
    assert build_parser().parse_args(["synthesis", "train", "--data", "x.csv"]).func.__name__ == "cmd_train"
    assert build_parser().parse_args(["synthesis", "impute", "benchmark", "--data", "x.csv"]).func.__name__ \
        == "cmd_benchmark"


@pytest.fixture
def fit_log(monkeypatch):
    """Record the sample ids every generator is trained on."""
    seen = []
    original = G.SplitGenerator.fit.__func__

    def spy(cls, train, *args, **kwargs):
        seen.append(set(train["sample_id"]))
        return original(cls, train, *args, **kwargs)

    monkeypatch.setattr(G.SplitGenerator, "fit", classmethod(spy))
    return seen


def test_benchmark_quick_end_to_end_without_leakage(tmp_path, written, cohort, fit_log):
    data, pode_cfg, aug_cfg = written
    out = tmp_path / "bench"
    rc = main(["synthesis", "augment", "benchmark", "--data", str(data), "--config", str(pode_cfg),
               "--augment-config", str(aug_cfg), "--output-dir", str(out), "--quick", "--n-jobs", "2"])
    assert rc == 0
    for name in ["results_by_fold", "paired_differences", "efficiency", "dose_response", "fidelity", "privacy",
                 "subgroups", "timings", "scorecard", "external_paired", "external_results", "compute_cost",
                 "learning_curve"]:
        assert (out / f"{name}.csv").is_file(), name
    assert not (out / "failures.csv").exists()
    assert (out / "report.md").read_text().startswith("# Augmentation benchmark")
    assert (out / "plots" / "dose_response.png").is_file()
    splits = pd.read_csv(out / "splits.csv")
    assert set(splits["role"]) == {"train", "test"} and splits["sample_id"].notna().all()
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["config"]["splits"]["repeats"] == 1

    # Leakage: 3 internal generators + 1 external; none saw an external patient,
    # and each internal one saw exactly its split's training rows (never its test rows).
    external = set(cohort.loc[cohort["Cohort"] == "EXT", "sample_id"])
    assert len(fit_log) == 4
    assert all(not (ids & external) for ids in fit_log)
    dev = set(cohort["sample_id"]) - external
    internal = fit_log[:3]
    assert set.union(*[dev - ids for ids in internal]) == dev  # each patient was held out once
    assert all(len(dev - ids) > 0 for ids in internal)

    res = pd.read_csv(out / "results_by_fold.csv")
    assert set(res["evaluation"]) == {"internal", "external"}
    assert set(res["arm"]) == {"real_only", "learning_curve", "augmented"}
    # Weights only reach models that accept them.
    w = res[res["strategy"] == "weight"]
    assert set(w["model"]) == {"cox_ph", "logistic_landmark"}
    card = pd.read_csv(out / "scorecard.csv")
    assert set(card["verdict"]) <= {"RECOMMENDED", "NON-INFERIOR", "NOT RECOMMENDED", "UNSAFE"}
    # Real-copy baselines are never privacy-gated.
    assert not card[card["strategy"].isin(["duplicate", "jitter", "weight"])]["verdict"].eq("UNSAFE").any()

    # A re-run resumes every finished split from its checkpoint: same results, nothing refitted.
    first = pd.read_csv(out / "results_by_fold.csv")
    assert len(list((out / "checkpoints").glob("*.pkl"))) == 4
    fit_log.clear()
    assert main(["synthesis", "augment", "benchmark", "--data", str(data), "--config", str(pode_cfg),
                 "--augment-config", str(aug_cfg), "--output-dir", str(out), "--quick", "--n-jobs", "1"]) == 0
    assert fit_log == []
    pd.testing.assert_frame_equal(pd.read_csv(out / "results_by_fold.csv"), first)

    # Without checkpoints, the per-split generator cache still avoids refitting the TVAEs.
    for f in (out / "checkpoints").glob("*.pkl"):
        f.unlink()
    assert main(["synthesis", "augment", "benchmark", "--data", str(data), "--config", str(pode_cfg),
                 "--augment-config", str(aug_cfg), "--output-dir", str(out), "--quick", "--n-jobs", "1"]) == 0
    timings = pd.read_csv(out / "timings.csv")
    assert set(timings.loc[timings["stage"] == "generator_fit", "source"]) == {"cache"}

    # report re-renders from saved tables.
    (out / "report.md").unlink()
    assert main(["synthesis", "augment", "report", "--output-dir", str(out)]) == 0
    assert (out / "report.md").is_file()


def test_pretrained_generator_only_used_for_external(tmp_path, written, fit_log):
    data, pode_cfg, aug_cfg = written
    cfg = augment_config(strategies=["tvae_natural"], models=["cox_ph"], learning_curve={"enabled": False})
    aug_cfg.write_text(yaml.safe_dump(cfg))
    # A pre-trained model (here: trained on everything).
    model_dir = tmp_path / "pretrained"
    assert main(["synthesis", "train", "--data", str(data), "--config", str(pode_cfg),
                 "--output-dir", str(model_dir), "--seed", "1"]) == 0
    out = tmp_path / "bench"
    assert main(["synthesis", "augment", "benchmark", "--data", str(data), "--config", str(pode_cfg),
                 "--augment-config", str(aug_cfg), "--output-dir", str(out), "--quick", "--n-jobs", "1",
                 "--generator-model", str(model_dir / "model.pkl")]) == 0
    assert len(fit_log) == 3  # internal splits still fit their own generators
    timings = pd.read_csv(out / "timings.csv")
    src = timings[timings["stage"] == "generator_fit"].set_index("split")["source"]
    assert src["external"].endswith("model.pkl") and (src.drop("external") == "fitted").all()


def test_generate_writes_tagged_table(tmp_path, written, cohort):
    data, pode_cfg, aug_cfg = written
    out = tmp_path / "gen"
    rc = main(["synthesis", "augment", "generate", "--data", str(data), "--config", str(pode_cfg),
               "--augment-config", str(aug_cfg), "--strategy", "tvae_mplus", "--ratio", "0.5",
               "--output-dir", str(out)])
    assert rc == 0
    aug = pd.read_csv(out / "augmented_training_table.csv")
    n_dev = (cohort["Cohort"] != "EXT").sum()
    assert (aug["_source"] == "real").sum() == n_dev
    added = aug[aug["_source"] != "real"]
    assert len(added) == round(0.5 * n_dev) and (added["Metastases"] == "Yes").all()
    assert (added["_stratum"] == "__synthetic__").all()
    diag = json.loads((out / "diagnostics.json").read_text())
    assert diag["strategy"] == "tvae_mplus" and "privacy_vs_training" in diag
    assert main(["synthesis", "augment", "generate", "--data", str(data), "--config", str(pode_cfg),
                 "--augment-config", str(aug_cfg), "--strategy", "nope", "--ratio", "1",
                 "--output-dir", str(out)]) == 2


def test_empty_tvae_pool_is_skipped_not_scored(tmp_path, written, monkeypatch):
    data, pode_cfg, aug_cfg = written
    aug_cfg.write_text(yaml.safe_dump(augment_config(strategies=["tvae_natural", "duplicate"], models=["cox_ph"],
                                                     learning_curve={"enabled": False})))
    monkeypatch.setattr(G.SplitGenerator, "sample",
                        lambda self, n, conditions=None, seed=None: (pd.DataFrame(columns=self.columns), n))
    out = tmp_path / "bench"
    assert main(["synthesis", "augment", "benchmark", "--data", str(data), "--config", str(pode_cfg),
                 "--augment-config", str(aug_cfg), "--output-dir", str(out), "--quick", "--n-jobs", "1"]) == 0
    res = pd.read_csv(out / "results_by_fold.csv")
    assert "tvae_natural" not in set(res["strategy"]) and "duplicate" in set(res["strategy"])
    fails = pd.read_csv(out / "failures.csv")
    assert (fails["strategy"] == "tvae_natural").all() and fails["error"].str.contains("no usable rows").all()
    assert "**Warning:** no usable rows for `tvae_natural`" in (out / "report.md").read_text()


def test_external_table_option(tmp_path, written, cohort):
    data, pode_cfg, aug_cfg = written
    dev, ext = cohort[cohort["Cohort"] != "EXT"], cohort[cohort["Cohort"] == "EXT"]
    dev.to_csv(data, index=False)
    ext_path = tmp_path / "external.csv"
    ext.to_csv(ext_path, index=False)
    cfg = augment_config(external={}, strategies=["tvae_natural"], models=["cox_ph"],
                         learning_curve={"enabled": False})
    aug_cfg.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "bench"
    assert main(["synthesis", "augment", "benchmark", "--data", str(data), "--external", str(ext_path),
                 "--config", str(pode_cfg), "--augment-config", str(aug_cfg), "--output-dir", str(out),
                 "--quick", "--n-jobs", "1"]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["data"]["external_rows"] == len(ext)
    assert "_external" not in report["columns"]["generated"] and "sample_id" not in report["columns"]["generated"]


def test_dry_run_and_overrides(tmp_path, written, capsys):
    data, pode_cfg, aug_cfg = written
    out = tmp_path / "bench"
    base = ["synthesis", "augment", "benchmark", "--data", str(data), "--config", str(pode_cfg),
            "--augment-config", str(aug_cfg), "--output-dir", str(out)]
    assert main([*base, "--dry-run", "--strategies", "tvae_mplus,duplicate", "--models", "cox_ph",
                 "--ratios", "0.5", "--draws", "1", "--folds", "2", "--repeats", "1"]) == 0
    text = capsys.readouterr().out
    assert "Strategies (2): tvae_mplus [tvae_targeted], duplicate [duplicate]" in text
    assert "Model cox_ph [cox_ph]" in text and "coxnet" not in text
    assert "Splits: 2-fold x 1 repeats = 2" in text and "Ratios: [0.5], draws: 1" in text
    assert not out.exists()  # nothing fitted or written
    with pytest.raises(SystemExit, match="unknown strategies"):
        main([*base, "--dry-run", "--strategies", "nope"])
