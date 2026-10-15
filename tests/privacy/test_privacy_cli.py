"""End-to-end CLI: ``privacy assess`` on CSVs and ``privacy audit`` with a tiny TVAE."""

import json

import pandas as pd
import yaml

from pode.cli.main import build_parser, main


def _write(tmp_path, split):
    train, holdout, fresh = split
    ids = [f"PAT{i:04d}" for i in range(len(train) + len(holdout))]
    paths = {}
    for name, df in (("train", train.assign(Sample_ID=ids[:len(train)])),
                     ("holdout", holdout.assign(Sample_ID=ids[len(train):])), ("synth", fresh)):
        paths[name] = tmp_path / f"{name}.csv"
        df.to_csv(paths[name], index=False)
    paths["privacy"] = tmp_path / "privacy.yaml"
    paths["privacy"].write_text(yaml.safe_dump({
        "id_column": "Sample_ID",
        "quasi_identifiers": {"basic": ["Sex", "Age", "subtype"]},
        "sensitive": ["OS_Status"],
        "strata": ["subtype"],
        "n_bootstrap": 30,
    }))
    paths["config"] = tmp_path / "pode.yaml"
    paths["config"].write_text(yaml.safe_dump({"synthesizer": {"epochs": 5, "batch_size": 100, "embedding_dim": 8,
                                                               "compress_dims": [32], "decompress_dims": [32],
                                                               "verbose": False}}))
    return paths


def test_privacy_parser_registered():
    args = build_parser().parse_args(["synthesis", "privacy", "assess", "--train", "t.csv", "--synthetic", "s.csv"])
    assert args.func.__name__ == "cmd_assess"
    args = build_parser().parse_args(["synthesis", "privacy", "audit", "--data", "d.csv", "--repeats", "2"])
    assert args.func.__name__ == "cmd_audit" and args.repeats == 2
    # Existing synthesis commands still parse.
    assert build_parser().parse_args(["synthesis", "train", "--data", "x.csv"]).func.__name__ == "cmd_train"


def test_assess_end_to_end(tmp_path, split):
    p = _write(tmp_path, split)
    out = tmp_path / "assess"
    rc = main(["synthesis", "privacy", "assess", "--train", str(p["train"]), "--holdout", str(p["holdout"]),
               "--synthetic", str(p["synth"]), "--privacy-config", str(p["privacy"]), "--output-dir", str(out),
               "--n-jobs", "2"])
    assert rc == 0
    summary = json.loads((out / "privacy_summary.json").read_text())
    assert summary["verdict"] == "PASS"
    assert summary["threat_model"]["n_train"] == 300 and summary["threat_model"]["n_holdout"] == 150
    assert "Sample_ID" not in summary["threat_model"]["columns"]
    assert (out / "report.md").is_file() and (out / "gate.csv").is_file()


def test_assess_copies_ids_and_fail_exit_code(tmp_path, split):
    p = _write(tmp_path, split)
    train, _, _ = split
    train.sample(frac=1.0, random_state=0).to_csv(p["synth"], index=False)
    out = tmp_path / "copies"
    rc = main(["synthesis", "privacy", "assess", "--train", str(p["train"]), "--holdout", str(p["holdout"]),
               "--synthetic", str(p["synth"]), "--privacy-config", str(p["privacy"]), "--output-dir", str(out),
               "--fail-on-fail"])
    assert rc == 3
    # Review tables label training rows with their patient ID.
    closest = pd.read_csv(out / "closest_pairs.csv")
    assert closest["id"].str.startswith("PAT").all()
    assert (closest["distance"] == 0).all()


def test_audit_tiny_tvae(tmp_path, split):
    p = _write(tmp_path, split)
    out = tmp_path / "audit"
    rc = main(["synthesis", "privacy", "audit", "--data", str(p["train"]), "--config", str(p["config"]),
               "--privacy-config", str(p["privacy"]), "--output-dir", str(out), "--repeats", "2",
               "--num-samples", "200", "--device", "cpu", "--n-jobs", "2"])
    assert rc == 0
    for name in ("report.md", "repeats.csv", "repeats_summary.csv", "gate_by_repeat.csv", "audit_summary.json",
                 "repeat_0/report.md", "repeat_1/synthetic.csv", "repeat_1/split.csv"):
        assert (out / name).is_file(), name
    reps = pd.read_csv(out / "repeats.csv")
    assert len(reps) == 2 and reps["membership_auc"].between(0, 1).all()
    split0 = pd.read_csv(out / "repeat_0" / "split.csv")
    assert (split0["role"] == "holdout").sum() == 90 and split0["id"].str.startswith("PAT").all()
    assert len(pd.read_csv(out / "repeat_0" / "synthetic.csv")) == 200
    assert (out / "report.md").read_text().startswith("# TVAE privacy audit")
