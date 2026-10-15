"""Phase 2: hybrid (matrix -> latents -> TVAE -> decode) generator and metacpg_search."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import TINY_TVAE, augment_config
from pode.augmentation.config import AugmentationConfig, ModelSpec
from pode.augmentation.generator import HybridSplitGenerator
from pode.augmentation.models import build_model
from pode.augmentation.selection import selection_stability
from pode.augmentation.strategies import SOURCE
from pode.cli.main import main
from pode.config import PodeConfig, SynthesizerConfig
from pode.schema.schema import DatasetSchema

N_CPG = 40


def make_meth_cohort(n: int = 220, seed: int = 0) -> pd.DataFrame:
    """Beta-value matrix (cg000..cg039); cg000-cg002 drive survival."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    loading = rng.normal(size=(2, N_CPG))
    m = z @ loading + rng.normal(0, 0.8, (n, N_CPG))
    beta = 1 / (1 + np.exp(-m))
    met = rng.choice(["No", "Yes"], n, p=[0.8, 0.2])
    lp = 0.9 * m[:, 0] - 0.7 * m[:, 1] + 0.5 * m[:, 2] + 0.8 * (met == "Yes")
    t_event = rng.exponential(30 * np.exp(-0.5 * lp))
    t_cens = rng.uniform(5, 80, n)
    df = pd.DataFrame({
        "sample_id": [f"S{i:04d}" for i in range(n)],
        "Age": rng.uniform(0.2, 10, n).round(1),
        "Metastases": met,
        "Cohort": rng.choice(["A", "B", "EXT"], n, p=[0.45, 0.35, 0.2]),
        "OS_Time": np.maximum(np.minimum(t_event, t_cens), 0.5).round(1),
        "OS_Status": (t_event <= t_cens).astype(int),
    })
    return pd.concat([df, pd.DataFrame(beta, columns=[f"cg{j:03d}" for j in range(N_CPG)])], axis=1)


HYBRID = {"backend": "hybrid", "exclude_columns": ["sample_id", "Cohort"], "epochs": 5,
          "hybrid": {"matrix_columns": ["cg*"], "matrix_kind": "methylation",
                     "compressor": {"compressor": "survival_vae", "n_latent": 3, "epochs": 5}}}


def _cfg(**over) -> AugmentationConfig:
    base = augment_config(
        generator=HYBRID, features=["cg*", "Age", "Metastases"], subgroups=["Metastases"],
        strategies=["tvae_natural", {"kind": "tvae_targeted", "label": "tvae_mplus",
                                     "target": {"equals": {"Metastases": "Yes"}}}, "duplicate"],
        models=[{"kind": "metacpg_search", "features": ["cg*"], "params": {"n": 5}},
                {"kind": "metacpg_fixed", "features": ["cg000", "cg001", "cg002"]}],
        learning_curve={"enabled": False}, ratios=[0.5], draws=2)
    base.update(over)
    return AugmentationConfig.from_dict(base)


def test_hybrid_generator_decodes_valid_betas():
    df = make_meth_cohort()
    cfg = _cfg()
    schema = DatasetSchema.from_dataframe(df)
    pc = PodeConfig(synthesizer=SynthesizerConfig(**TINY_TVAE["synthesizer"]))
    gen = HybridSplitGenerator.fit(df, schema, pc, cfg, 1, None, endpoint=("OS_Time", "OS_Status"))
    assert gen.latent_cols == ["methylation_latent_0", "methylation_latent_1", "methylation_latent_2"]
    assert not any(c.startswith("cg") for c in gen.columns)  # the TVAE never sees the matrix
    rows, short = gen.sample(60, seed=3)
    assert short == 0 and len(rows) == 60
    M = rows[gen.matrix_cols].to_numpy()
    assert M.shape == (60, N_CPG) and ((M > 0) & (M < 1)).all()
    view = gen.view(df)
    assert list(view.columns[-3:]) == gen.latent_cols and len(view) == len(df)
    # Residual noise restores spread that a decoder mean alone would lose.
    smooth = gen.pre.inverse_transform(gen.compressor.inverse_transform(rows[gen.latent_cols].to_numpy()))
    assert M.std(axis=0).mean() > smooth.std(axis=0).mean()


def test_metacpg_search_matches_metacpg_ranking():
    from metacpg.core import _endpoint_stats, _rank, beta_to_m

    df = make_meth_cohort()
    cg = [f"cg{j:03d}" for j in range(N_CPG)]
    m = build_model(ModelSpec("metacpg_search", params={"n": 5}), cg, [], 24, 0)
    strata = df["Cohort"].to_numpy()
    m.fit(df.assign(**{SOURCE: "real"}), df["OS_Time"], df["OS_Status"], strata, np.ones(len(df)),
          np.ones(len(df), bool))
    M = pd.DataFrame(beta_to_m(df[cg].to_numpy()).T, index=cg)
    stats = _endpoint_stats(M, pd.DataFrame({"t": df["OS_Time"], "s": df["OS_Status"], "k": strata}),
                            {"name": "x", "time": "t", "status": "s"},
                            [{"column": "k", "type": "categorical", "mode": "strata"}])
    top = _rank(stats).head(5)
    assert m.selected_ == top["feature"].tolist()
    assert np.allclose(m.weights, top["standardized_coef"], atol=1e-6)
    assert set(m.selected_) & {"cg000", "cg001", "cg002"}  # planted signal found (CpGs share factors)
    assert m.diagnostics()["selected_features"].count(";") == 4


def test_selection_stability_summary():
    rows = []
    for split in range(3):
        rows.append({"evaluation": "internal", "split": split, "model": "s", "arm": "real_only",
                     "strategy": "real_only", "ratio": 0.0, "draw": 0, "selected_features": "a;b;c"})
        rows.append({"evaluation": "internal", "split": split, "model": "s", "arm": "augmented",
                     "strategy": "x", "ratio": 1.0, "draw": 0, "selected_features": f"a;b;z{split}"})
    summary, freq = selection_stability(pd.DataFrame(rows), ["a", "b", "c"])
    real = summary[summary["arm"] == "real_only"].iloc[0]
    aug = summary[summary["arm"] == "augmented"].iloc[0]
    assert real["jaccard_vs_real_only"] == 1.0 and real["across_split_jaccard"] == 1.0
    assert aug["jaccard_vs_real_only"] == pytest.approx(0.5) and aug["across_split_jaccard"] == pytest.approx(0.5)
    assert aug["reference_overlap"] == pytest.approx(2 / 3)
    assert freq[(freq["arm"] == "augmented") & (freq["feature"] == "a")]["frequency"].iloc[0] == 1.0


def test_hybrid_benchmark_end_to_end(tmp_path):
    df = make_meth_cohort()
    data = tmp_path / "meth.csv"
    df.to_csv(data, index=False)
    pode_cfg = tmp_path / "pode.yaml"
    pode_cfg.write_text(yaml.safe_dump(TINY_TVAE))
    aug = tmp_path / "aug.yaml"
    aug.write_text(yaml.safe_dump(_cfg(reference_signature=["cg000", "cg001", "cg002"]).to_dict()))
    out = tmp_path / "bench"
    assert main(["synthesis", "augment", "benchmark", "--data", str(data), "--config", str(pode_cfg),
                 "--augment-config", str(aug), "--output-dir", str(out), "--quick", "--n-jobs", "2"]) == 0
    assert not (out / "failures.csv").exists()
    res = pd.read_csv(out / "results_by_fold.csv")
    search = res[res["model"] == "metacpg_search"]
    assert search["selected_features"].notna().all() and set(search["evaluation"]) == {"internal", "external"}
    fid = pd.read_csv(out / "fidelity.csv")
    assert {"matrix_discriminator_auc", "matrix_sd_ratio", "discriminator_auc"} <= set(fid.columns)
    sel = pd.read_csv(out / "selection_stability.csv")
    assert set(sel["arm"]) == {"real_only", "augmented"} and sel["reference_overlap"].notna().all()
    assert "## Feature selection" in (out / "report.md").read_text()
    report = json.loads((out / "report.json").read_text())
    assert not any(c.startswith("methylation_latent") for c in report["columns"]["features"])
