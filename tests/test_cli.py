"""Tests for CLI parsing and lightweight command orchestration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from pode.cli.main import build_parser
from pode.schema import DatasetSchema


def _write_csv(tmp_path, sample_df):
    path = tmp_path / "cohort.csv"
    sample_df.to_csv(path, index=False)
    return path


def test_train_parser_accepts_model_and_schema_outputs():
    args = build_parser().parse_args(
        [
            "synthesis",
            "train",
            "--data",
            "cohort.csv",
            "--config",
            "config.yaml",
            "--output-dir",
            "trained",
            "--seed",
            "42",
            "--model-out",
            "trained/model.pkl",
            "--schema-out",
            "trained/schema.yaml",
        ]
    )
    assert args.synthesis_command == "train"
    assert args.seed == 42
    assert args.model_out == "trained/model.pkl"
    assert args.schema_out == "trained/schema.yaml"


def test_generate_model_mode_parser_accepts_no_data():
    args = build_parser().parse_args(
        [
            "synthesis",
            "generate",
            "--model",
            "trained/model.pkl",
            "--schema",
            "trained/schema.yaml",
            "--num-samples",
            "10",
        ]
    )
    assert args.synthesis_command == "generate"
    assert args.data is None
    assert args.model == "trained/model.pkl"


def test_generate_model_mode_requires_schema(tmp_path):
    args = build_parser().parse_args(
        ["synthesis",
"generate", "--model", str(tmp_path / "model.pkl"), "--output-dir", str(tmp_path)]
    )
    assert args.func(args) == 2


def test_conditions_and_conditions_config_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "synthesis",
                "generate",
                "--data",
                "cohort.csv",
                "--conditions",
                "1.0:X=1",
                "--conditions-config",
                "conditions.yaml",
            ]
        )


def test_schema_prepared_writes_prepared_schema(tmp_path, sample_df):
    data_path = _write_csv(tmp_path, sample_df)
    out_path = tmp_path / "prepared_schema.yaml"
    args = build_parser().parse_args(
        ["synthesis",
"schema", "--prepared", "--data", str(data_path), "--out", str(out_path)]
    )

    assert args.func(args) == 0
    schema = DatasetSchema.from_yaml(out_path)
    assert "Sample_ID" not in schema.columns
    assert "OS_Time" in schema.columns


def test_train_command_saves_model_schema_and_seed_log(tmp_path, sample_df, monkeypatch):
    import pode.generation.synthesizer as synth_mod

    class FakeGenerator:
        def __init__(self, schema, config):
            self.schema = schema

        def fit(self, real, seed=None):
            assert "Sample_ID" not in real.columns
            assert seed == 123
            return self

        def save(self, path):
            path.write_text("model")

    monkeypatch.setattr(synth_mod, "TVAEGenerator", FakeGenerator)

    data_path = _write_csv(tmp_path, sample_df)
    out_dir = tmp_path / "trained"
    args = build_parser().parse_args(
        [
            "synthesis",
            "train",
            "--data",
            str(data_path),
            "--output-dir",
            str(out_dir),
            "--seed",
            "123",
        ]
    )

    assert args.func(args) == 0
    assert (out_dir / "model.pkl").read_text() == "model"
    schema = DatasetSchema.from_yaml(out_dir / "schema.yaml")
    assert "Sample_ID" not in schema.columns
    assert list(out_dir.glob("seed_log_*.json"))


def test_generate_model_mode_loads_model_without_training(tmp_path, sample_df, monkeypatch):
    import pode.generation.synthesizer as synth_mod

    schema = DatasetSchema.from_dataframe(sample_df)
    schema_path = tmp_path / "schema.yaml"
    schema.to_yaml(schema_path)
    model_path = tmp_path / "model.pkl"
    model_path.write_text("model")

    calls = {}

    class FakeGenerator:
        @classmethod
        def load(cls, path, schema=None):
            calls["loaded"] = (path, schema)
            return cls()

        def sample(self, num_rows, conditions=None, seed=None, censor_threshold=None):
            calls["sampled"] = {
                "num_rows": num_rows,
                "conditions": conditions,
                "seed": seed,
                "censor_threshold": censor_threshold,
            }
            return pd.DataFrame({"Age": [1.0, 2.0, 3.0]})

    monkeypatch.setattr(synth_mod, "TVAEGenerator", FakeGenerator)

    out_dir = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "synthesis",
            "generate",
            "--model",
            str(model_path),
            "--schema",
            str(schema_path),
            "--num-samples",
            "3",
            "--seed",
            "7",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert args.func(args) == 0
    assert calls["loaded"][0] == str(model_path)
    assert calls["sampled"]["num_rows"] == 3
    assert calls["sampled"]["seed"] == 7
    assert pd.read_csv(out_dir / "synthetic_data.csv").shape == (3, 1)


def test_prepare_hybrid_bundle_reads_methylation_latent_prefix(tmp_path):
    from synthvae.cli import _bundle_impl as mod

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    pd.DataFrame({"Age": [1.0, 2.0]}).to_csv(bundle / "clinical_full.csv", index=False)
    pd.DataFrame({
        "methylation_latent_0": [0.1, 0.2],
        "methylation_latent_1": [0.3, 0.4],
    }).to_csv(bundle / "latents_full.csv", index=False)
    pd.DataFrame({"time": [1.0, 2.0], "event": [0, 1]}).to_csv(
        bundle / "survival_full.csv", index=False
    )
    (bundle / "handoff_metadata.json").write_text(json.dumps({
        "matrix_kind": "methylation",
        "latent_prefix": "methylation_latent_",
    }))

    combined, clinical, metadata = mod.load_bundle(bundle, "time", "event")
    latent_prefix = metadata["latent_prefix"]
    latent_columns = [c for c in combined.columns if c.startswith(latent_prefix)]

    assert latent_columns == ["methylation_latent_0", "methylation_latent_1"]
    assert clinical.columns.tolist() == ["Age"]
