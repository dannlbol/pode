"""Smoke + correctness tests runnable with only numpy/sklearn/pandas.

Run: ``python -m pytest tests/ -q`` or ``python tests/test_smoke.py``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_discrete_time_targets_and_nll():
    from synthvae.survival.discretize import TimeGrid
    from synthvae.survival import losses as L

    rng = np.random.default_rng(0)
    time = rng.exponential(1.0, 200)
    event = (rng.random(200) < 0.7).astype(int)
    grid = TimeGrid.from_times(time, event, n_bins=10)
    idx = grid.index(time)
    assert idx.min() >= 0 and idx.max() < grid.n_bins

    logits = rng.standard_normal((200, grid.n_bins))
    nll = L.nll_numpy(logits, idx, event)
    assert np.isfinite(nll) and nll > 0

    S = L.survival_from_logits_numpy(logits)
    # survival is non-increasing across intervals and in [0,1]
    assert np.all(S <= 1.0 + 1e-9) and np.all(S >= -1e-9)
    assert np.all(np.diff(S, axis=1) <= 1e-9)


def test_cindex_sanity():
    from synthvae.evaluation.cindex import concordance_index
    # risk perfectly anti-correlated with survival -> c-index ~ 1
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1, 1, 1, 1, 1])
    risk = -time  # higher risk = shorter time
    c = concordance_index(time, event, risk)
    assert c > 0.99


def test_km_monotone():
    from synthvae.evaluation.km import kaplan_meier
    rng = np.random.default_rng(1)
    time = rng.exponential(1.0, 300)
    event = (rng.random(300) < 0.6).astype(int)
    t, s = kaplan_meier(time, event)
    assert np.all(np.diff(s) <= 1e-9)
    assert s[0] <= 1.0 and s[-1] >= 0.0


def test_numpy_cox_recovers_sign():
    from synthvae.evaluation.cox import _NumpyCoxPH
    rng = np.random.default_rng(2)
    n = 500
    x = rng.standard_normal((n, 2))
    lp = 1.0 * x[:, 0] - 0.8 * x[:, 1]
    u = rng.uniform(1e-6, 1 - 1e-6, n)
    t = (-np.log(u)) * np.exp(-lp)  # exponential baseline
    c = rng.exponential(np.quantile(t, 0.7), n)
    time = np.minimum(t, c)
    event = (t <= c).astype(int)
    cox = _NumpyCoxPH().fit(x, time, event)
    # signs should match the data-generating coefficients
    assert cox.coef_[0] > 0
    assert cox.coef_[1] < 0


def test_numpy_cox_handles_separating_risk_scores_without_nan():
    """The fallback must not underflow an early risk-set denominator."""
    from synthvae.evaluation.cox import _NumpyCoxPH
    # Near-separation drives Newton steps to very large risk-score differences.
    x = np.array([[-1.0], [-0.8], [-0.6], [0.6], [0.8], [1.0]])
    time = np.array([6., 5., 4., 3., 2., 1.])
    event = np.ones(6, dtype=int)
    cox = _NumpyCoxPH(max_iter=80).fit(x, time, event)
    assert np.isfinite(cox.coef_).all()


def test_coxmodel_drops_constant_columns_and_falls_back():
    from synthvae.evaluation.cox import CoxModel

    rng = np.random.default_rng(5)
    n = 80
    x0 = rng.standard_normal(n)
    x1 = np.ones(n)  # constant column that can trigger lifelines failure
    X = np.column_stack([x0, x1])
    time = rng.exponential(1.0, n)
    event = (rng.random(n) < 0.7).astype(int)

    model = CoxModel(prefer_lifelines=True).fit(X, time, event, feature_names=["x0", "x1"])
    risk = model.predict_risk(X)

    assert np.isfinite(risk).all()
    assert model.coef_.shape == (2,)
    assert model.coef_[1] == 0.0


def test_pca_compressor_roundtrip():
    from synthvae.compressors.pca import PCACompressor
    rng = np.random.default_rng(3)
    X = rng.standard_normal((200, 50))
    comp = PCACompressor(n_latent=10).fit(X)
    Z = comp.transform(X)
    assert Z.shape == (200, 10)
    err = comp.reconstruction_error(X)
    assert np.isfinite(err)


def test_methylation_preprocessor_logit_roundtrip():
    from synthvae.data.preprocessing import ExpressionPreprocessor

    X = np.array([
        [0.0, 0.2, 0.8, 1.0],
        [1e-8, 0.3, 0.7, 1.0 - 1e-8],
        [0.1, 0.4, 0.6, 0.9],
    ])
    pre = ExpressionPreprocessor(
        n_top_genes=None,
        matrix_kind="methylation",
        beta_epsilon=1e-6,
    )
    Xt = pre.fit_transform(X)
    assert np.isfinite(Xt).all()

    Xr = pre.inverse_transform(Xt)
    assert np.all(Xr >= 0.0)
    assert np.all(Xr <= 1.0)
    assert np.allclose(Xr[:, 1:3], X[:, 1:3])


def test_end_to_end_fallback():
    from synthvae.data.toy import make_cohort, ToyConfig
    from synthvae.data.preprocessing import ExpressionPreprocessor
    from synthvae.compressors.pca import PCACompressor
    from synthvae.synthesizers.gmm_fallback import GaussianMixtureFallback
    from synthvae.pipeline.hybrid import HybridPipeline
    from synthvae.evaluation.report import evaluate

    clinical, expression, survival, _ = make_cohort(ToyConfig(n=300, n_genes=120, seed=0))
    pipe = HybridPipeline(
        PCACompressor(n_latent=12),
        GaussianMixtureFallback(),
        preprocessor=ExpressionPreprocessor(n_top_genes=None),
        categorical=["stage", "sex"],
    )
    pipe.fit(clinical, expression, survival)
    res = pipe.sample(150)
    assert res.expression.shape[0] == 150
    assert res.latents.shape == (150, 12)

    real = {"clinical": clinical, "latents": pipe.encode_real(expression),
            "time": survival["time"].to_numpy(float), "event": survival["event"].to_numpy(float)}
    synth = {"clinical": res.clinical, "latents": res.latents,
             "time": res.survival["time"].to_numpy(float),
             "event": res.survival["event"].to_numpy(float)}
    m = evaluate(real, synth, prefer_lifelines=False)
    assert np.isfinite(m["tstr_concordance"])
    assert np.isfinite(m["km_iad"])


def test_evaluate_uses_train_fixed_cox_mask():
    from synthvae.evaluation.report import evaluate

    train_real = {
        "clinical": pd.DataFrame({"ConsensusMYC": [0, 0, 0, 0], "Age": [5.0, 6.0, 7.0, 8.0]}),
        "latents": np.array([[0.1, 0.2], [0.2, 0.1], [0.3, 0.4], [0.4, 0.3]]),
        "time": np.array([1.0, 2.0, 3.0, 4.0]),
        "event": np.array([1, 1, 0, 1]),
    }
    real = {
        "clinical": pd.DataFrame({"ConsensusMYC": [0, 1], "Age": [6.5, 7.5]}),
        "latents": np.array([[0.15, 0.25], [0.35, 0.45]]),
        "time": np.array([1.5, 2.5]),
        "event": np.array([1, 0]),
    }
    synth = {
        "clinical": pd.DataFrame({"ConsensusMYC": [1, 1], "Age": [5.5, 8.5]}),
        "latents": np.array([[0.12, 0.22], [0.32, 0.42]]),
        "time": np.array([1.2, 2.2]),
        "event": np.array([1, 1]),
    }
    m = evaluate(real, synth, train_real=train_real, prefer_lifelines=False)
    assert m["cox_features_dropped"] >= 1
    assert m["cox_features_used"] >= 1
    assert np.isfinite(m["tstr_concordance"])


def test_build_table_normalizes_event_to_bool():
    import pandas as pd
    from synthvae.pipeline.latents import build_table

    clinical = pd.DataFrame({"age": [60, 70]})
    latents = np.array([[0.1, 0.2], [0.3, 0.4]])
    survival = pd.DataFrame({"time": [1.0, 2.0], "event": [0, 1]})

    table = build_table(clinical, latents, survival)

    assert table["event"].dtype == bool
    assert table["event"].tolist() == [False, True]


def test_build_metadata_infers_string_clinical_as_categorical():
    from synthvae.pipeline.latents import build_metadata

    clinical = pd.DataFrame({
        "Sex": ["F", "M"],
        "Age": [5.1, 7.2],
        "ConsensusMYC": [0, 1],
    })
    survival = pd.DataFrame({"time": [1.0, 2.0], "event": [0, 1]})

    metadata = build_metadata(clinical, n_latent=2, survival=survival, categorical=["sex"])

    assert metadata["columns"]["Sex"]["sdtype"] == "categorical"
    assert metadata["columns"]["Age"]["sdtype"] == "numerical"
    assert metadata["columns"]["ConsensusMYC"]["sdtype"] == "categorical"


def test_handoff_bundle_roundtrip_decodes_latents():
    from synthvae.compressors.pca import PCACompressor
    from synthvae.data.preprocessing import ExpressionPreprocessor
    from synthvae.data.toy import ToyConfig, make_cohort
    from synthvae.pipeline.handoff import decode_latent_csv, save_handoff_bundle

    _, expression, _, _ = make_cohort(ToyConfig(n=80, n_genes=30, seed=4))
    pre = ExpressionPreprocessor(n_top_genes=12)
    X = pre.fit_transform(expression.to_numpy(dtype=float))
    comp = PCACompressor(n_latent=6).fit(X)

    with tempfile.TemporaryDirectory() as tmp:
        bundle = save_handoff_bundle(
            compressor=comp,
            compressor_kind="pca",
            preprocessor=pre,
            expression=expression,
            out_dir=tmp,
            config={"n_latent": 6},
            metrics={"full_reconstruction_mse": comp.reconstruction_error(X)},
        )
        assert bundle.latents_path.exists()
        assert bundle.checkpoint_path.exists()
        assert bundle.metadata_path.exists()

        latents = pd.read_csv(bundle.latents_path)
        assert latents.shape == (len(expression), 6)
        assert list(latents.columns) == [f"expr_latent_{i}" for i in range(6)]

        out_expression = Path(tmp) / "decoded.csv"
        decoded = decode_latent_csv(
            bundle_dir=tmp,
            latents_csv=bundle.latents_path,
            out_expression_csv=out_expression,
        )
        expected = pre.inverse_transform(comp.inverse_transform(latents.to_numpy(dtype=float)))
        assert out_expression.exists()
        assert decoded.shape == expression.shape
        assert list(decoded.columns) == list(expression.columns)
        assert np.allclose(decoded.to_numpy(dtype=float), expected)


def test_methylation_handoff_uses_methylation_names_and_beta_output():
    from synthvae.compressors.pca import PCACompressor
    from synthvae.data.preprocessing import ExpressionPreprocessor
    from synthvae.pipeline.handoff import decode_latent_csv, save_handoff_bundle

    rng = np.random.default_rng(11)
    methylation = pd.DataFrame(
        rng.beta(2.0, 5.0, size=(80, 24)),
        columns=[f"cg{i:04d}" for i in range(24)],
    )
    methylation.iloc[0, 0] = 0.0
    methylation.iloc[1, 1] = 1.0

    pre = ExpressionPreprocessor(
        n_top_genes=10,
        matrix_kind="methylation",
        beta_epsilon=1e-6,
    )
    X = pre.fit_transform(methylation.to_numpy(dtype=float))
    comp = PCACompressor(n_latent=5).fit(X)

    with tempfile.TemporaryDirectory() as tmp:
        bundle = save_handoff_bundle(
            compressor=comp,
            compressor_kind="pca",
            preprocessor=pre,
            matrix=methylation,
            matrix_kind="methylation",
            out_dir=tmp,
            config={"n_latent": 5},
            include_input=True,
        )

        metadata = json.loads((Path(tmp) / "handoff_metadata.json").read_text())
        assert metadata["matrix_kind"] == "methylation"
        assert metadata["latent_prefix"] == "methylation_latent_"
        assert (Path(tmp) / "methylation_input.csv").exists()

        latents = pd.read_csv(bundle.latents_path)
        assert list(latents.columns) == [f"methylation_latent_{i}" for i in range(5)]

        decoded = decode_latent_csv(
            bundle_dir=tmp,
            latents_csv=bundle.latents_path,
            out_matrix_csv=Path(tmp) / "synthetic_methylation.csv",
        )
        values = decoded.to_numpy(dtype=float)
        assert decoded.shape == methylation.shape
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)


def test_run_hybrid_sweep_helpers():
    from synthvae.cli import hybrid as mod

    grid = mod.expand_grid({
        "fixed": {"epochs": 3},
        "grid": {"n_latent": [2, 4], "beta_max": [0.5, 1.0]},
    })
    assert len(grid) == 4
    assert {"epochs": 3, "n_latent": 2, "beta_max": 0.5} in grid

    score = mod.metric_score({
        "reconstruction_mse": 0.1,
        "concordance_gap": 0.2,
        "km_iad": 0.3,
        "latent_cov_relgap": 0.4,
        "hr_logcorr": 0.8,
    })
    assert np.isfinite(score)
    assert score > 0

    latent_score = mod.metric_score({
        "has_clinical": True,
        "reconstruction_mse": 0.1,
        "concordance_gap": 9.9,
        "km_iad": 9.9,
        "latent_cov_relgap": 9.9,
        "hr_logcorr": 0.1,
        "latent_only_concordance_gap": 0.2,
        "latent_only_km_iad": 0.3,
        "latent_only_latent_cov_relgap": 0.4,
        "latent_only_hr_logcorr": 0.8,
    }, objective="latents_survival")
    assert np.isfinite(latent_score)
    assert latent_score < score + 1.0

    recon_only = mod.metric_score({"reconstruction_mse": 0.25}, objective="reconstruction_only")
    assert recon_only == 0.25

    raw = pd.DataFrame([
        {
            "compressor": "survival_vae",
            "synthesizer": "gmm",
            "n_latent": 32,
            "n_top_genes": 0,
            "epochs": 100,
            "w_surv": 1.0,
            "beta_max": 0.5,
            "free_bits": 0.02,
            "kl_warmup_epochs": 20,
            "seed": 0,
            "candidate": 0,
            "rank_score": 1.1,
            "reconstruction_mse": 0.2,
            "has_clinical": True,
            "tuning_objective": "latents_survival",
        },
        {
            "compressor": "survival_vae",
            "synthesizer": "gmm",
            "n_latent": 32,
            "n_top_genes": 0,
            "epochs": 100,
            "w_surv": 1.0,
            "beta_max": 0.5,
            "free_bits": 0.02,
            "kl_warmup_epochs": 20,
            "seed": 1,
            "candidate": 1,
            "rank_score": 1.3,
            "reconstruction_mse": 0.3,
            "has_clinical": True,
            "tuning_objective": "latents_survival",
        },
    ])
    aggregated = mod.aggregate_seed_results(raw, [
        "compressor", "synthesizer", "n_latent", "n_top_genes", "epochs",
        "w_surv", "beta_max", "free_bits", "kl_warmup_epochs", "seed",
    ])
    assert len(aggregated) == 1
    assert aggregated.loc[0, "n_seeds"] == 2
    assert np.isclose(aggregated.loc[0, "rank_score_mean"], 1.2)


def test_load_data_without_clinical():
    from synthvae.cli import hybrid as mod

    from synthvae.data.toy import ToyConfig, make_cohort

    _, expression, survival, _ = make_cohort(ToyConfig(n=20, n_genes=8, seed=9))

    with tempfile.TemporaryDirectory() as tmp:
        expression_path = Path(tmp) / "expression.csv"
        survival_path = Path(tmp) / "survival.csv"
        expression.to_csv(expression_path, index=False)
        survival.to_csv(survival_path, index=False)

        class Args:
            clinical = None
            expression = str(expression_path)
            methylation = None
            survival = str(survival_path)
            n = 0
            n_genes = 0
            seed = 0

        clinical, expression_loaded, survival_loaded, matrix_kind = mod.load_data(Args())
        assert clinical.shape == (20, 0)
        assert expression_loaded.shape == expression.shape
        assert survival_loaded.shape == survival.shape
        assert matrix_kind == "expression"


def test_run_once_without_clinical():
    from synthvae.cli import hybrid as mod

    from synthvae.data.toy import ToyConfig, make_cohort

    _, expression, survival, _ = make_cohort(ToyConfig(n=60, n_genes=20, seed=10))

    with tempfile.TemporaryDirectory() as tmp:
        expression_path = Path(tmp) / "expression.csv"
        survival_path = Path(tmp) / "survival.csv"
        expression.to_csv(expression_path, index=False)
        survival.to_csv(survival_path, index=False)

        args = SimpleNamespace(
            clinical=None,
            expression=str(expression_path),
            methylation=None,
            survival=str(survival_path),
            n=0,
            n_genes=0,
            seed=0,
            compressor="pca",
            synthesizer="gmm",
            n_latent=4,
            epochs=3,
            n_top_genes=0,
            categorical=["stage", "sex"],
            test_fraction=0.25,
            no_lifelines=True,
            out_dir=None,
            tuning_objective="full_joint",
        )
        metrics = mod.run_once(args, write_outputs=False)
        assert metrics["has_clinical"] is False
        assert np.isfinite(metrics["reconstruction_mse"])
        assert np.isfinite(metrics["concordance_gap"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print("all smoke tests passed")
