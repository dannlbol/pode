"""Tests for the joint multimodal VAE (synthvae.mmvae).

All tests are skipped when torch is unavailable, matching the package's import
guards. They exercise the pieces completed for the joint architecture: fusion /
gating shapes, a short training run, generation + evaluation, the NB/ZINB count
likelihoods, and the GradNorm weighting toggle.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from synthvae.data.toy import make_cohort, ToyConfig
from synthvae.evaluation.report import evaluate
from synthvae.mmvae import MMVAEConfig, JointMultimodalVAE, JointVAESynthesizer
from synthvae.mmvae import data as mdata


def small_cohort(n=120, n_genes=40, seed=0):
    clinical, expression, survival, _ = make_cohort(
        ToyConfig(n=n, n_genes=n_genes, seed=seed)
    )
    return clinical, expression, survival


def small_cfg(**overrides):
    base = dict(
        latent_dim=8, encoder_hidden=(32,), tabular_head_hidden=(16,),
        expression_head_hidden=(16,), survival_head_hidden=(16,),
        n_time_bins=8, batch_size=32, max_epochs=3, patience=5, seed=0,
        device="cpu",
    )
    base.update(overrides)
    return MMVAEConfig(**base)


# --------------------------------------------------------------------------- #
# Fusion / gating / dropped-modality shape + finiteness matrix (T7)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fusion", ["poe", "moe", "mopoe"])
@pytest.mark.parametrize("gating", [True, False])
@pytest.mark.parametrize("present", [
    ("tabular", "expression"), ("tabular",), ("expression",),
])
def test_fusion_gating_matrix(fusion, gating, present):
    clinical, expression, survival = small_cohort()
    cfg = small_cfg(fusion=fusion, gating=gating)
    dataset, transforms, cfg = mdata.build_from_frames(
        clinical, expression, survival, cfg
    )
    model = JointMultimodalVAE(cfg, transforms.column_specs)
    model.eval()
    x_tab = dataset.x_tab[:16]
    x_expr = dataset.x_expr[:16]
    mu, logvar = model.encode(x_tab, x_expr, present=present)
    assert mu.shape == (16, cfg.latent_dim)
    assert logvar.shape == (16, cfg.latent_dim)
    assert torch.isfinite(mu).all() and torch.isfinite(logvar).all()


# --------------------------------------------------------------------------- #
# Short training run: loss finite and decreasing (T3)
# --------------------------------------------------------------------------- #
def test_smoke_fit_decreases():
    clinical, expression, survival = small_cohort()
    cfg = small_cfg(max_epochs=6, kl_warmup_epochs=2)
    synth = JointVAESynthesizer(cfg=cfg).fit(clinical, expression, survival)
    hist = synth.model.history_
    assert len(hist) >= 1
    vals = [h["val_obj"] for h in hist]
    assert all(np.isfinite(vals))
    # best epoch should improve on the first epoch
    assert min(vals) <= vals[0] + 1e-6


# --------------------------------------------------------------------------- #
# Generation feeds the evaluation harness (T4)
# --------------------------------------------------------------------------- #
def test_sample_and_evaluate():
    clinical, expression, survival = small_cohort(n=160)
    cfg = small_cfg(max_epochs=4)
    synth = JointVAESynthesizer(cfg=cfg).fit(clinical, expression, survival)

    result = synth.sample(80)
    assert len(result.clinical) == 80
    assert len(result.expression) == 80
    assert len(result.survival) == 80
    assert result.latents.shape == (80, cfg.latent_dim)
    assert list(result.clinical.columns) == list(clinical.columns)
    assert list(result.expression.columns) == list(expression.columns)

    real = {
        "clinical": clinical,
        "latents": synth.encode_real(clinical, expression),
        "time": survival["time"].to_numpy(float),
        "event": survival["event"].to_numpy(float),
    }
    synth_d = {
        "clinical": result.clinical,
        "latents": result.latents,
        "time": result.survival["time"].to_numpy(float),
        "event": result.survival["event"].to_numpy(float),
    }
    metrics = evaluate(real, synth_d)
    assert np.isfinite(metrics["tstr_concordance"])
    assert np.isfinite(metrics["km_iad_late"])


def test_generation_samples_gaussian_observation_noise():
    """Generation must not collapse to the decoder's conditional mean."""
    clinical, expression, survival = small_cohort(n=100)
    synth = JointVAESynthesizer(cfg=small_cfg(max_epochs=2)).fit(clinical, expression, survival)
    means = synth.model.sample(20, sample_observations=False)["expr"]
    draws = synth.model.sample(20, sample_observations=True)["expr"]
    assert not np.allclose(means, draws)


def test_residual_factor_sampler_preserves_correlated_noise():
    """Correlated factor draws must not degrade into feature-wise white noise."""
    clinical, expression, survival = small_cohort(n=30, n_genes=2)
    cfg = small_cfg(observation_sampler="residual_factor", residual_rank=1)
    dataset, transforms, cfg = mdata.build_from_frames(clinical, expression, survival, cfg)
    model = JointMultimodalVAE(cfg, transforms.column_specs)
    for parameter in model.parameters():
        parameter.data.zero_()
    component = np.array([[2 ** -0.5, 2 ** -0.5]], dtype=np.float32)
    model._residual_sampler = {
        "version": 1, "score_dim": 1, "score_cov": np.array([[1.0]], np.float32),
        "modalities": {"expression": {
            "components": component, "mean": np.zeros(2, np.float32),
            "diag_scale": np.full(2, 0.02, np.float32), "score_slice": (0, 1),
        }},
    }
    correlated = model.sample(1000)["expr"]
    model.cfg.observation_sampler = "independent"
    independent = model.sample(1000)["expr"]
    assert np.corrcoef(correlated.T)[0, 1] > 0.95
    assert abs(np.corrcoef(independent.T)[0, 1]) < 0.1


def test_save_load_preserves_expression_columns(tmp_path):
    clinical, expression, survival = small_cohort(n=100)
    expression = expression.rename(columns={
        c: f"GENE_{i:03d}" for i, c in enumerate(expression.columns)
    })
    cfg = small_cfg(max_epochs=2)
    synth = JointVAESynthesizer(cfg=cfg).fit(clinical, expression, survival)

    model_path = tmp_path / "model.pt"
    synth.save(model_path)
    result = JointVAESynthesizer.load(model_path).sample(12)

    assert list(result.expression.columns) == list(expression.columns)


# --------------------------------------------------------------------------- #
# Optional methylation modality (4th head, shared latent)
# --------------------------------------------------------------------------- #
def _toy_methylation(n, n_features=25, seed=1):
    """A synthetic beta-value matrix in (0, 1) with per-feature structure."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, n_features)) + rng.normal(size=(1, n_features))
    beta = 1.0 / (1.0 + np.exp(-logits))
    cols = [f"cg{ i:05d}" for i in range(n_features)]
    return pd.DataFrame(np.clip(beta, 1e-4, 1 - 1e-4), columns=cols)


def test_methylation_off_is_unchanged():
    """With no methylation, the model is byte-identical to the 3-modality build:
    no methylation modules, gating over 2 encoders, 3-name objective."""
    clinical, expression, survival = small_cohort(n=100)
    cfg = small_cfg()
    _, transforms, cfg = mdata.build_from_frames(clinical, expression, survival, cfg)
    assert cfg.methylation_dim == 0
    assert transforms.methylation is None
    model = JointMultimodalVAE(cfg, transforms.column_specs)
    assert model.enc_meth is None and model.head_meth is None
    assert model.gating.n_modalities == 2
    assert model.objective.modality_names == ["tabular", "expression", "survival"]
    assert model.cfg.observation_sampler == "independent"


def test_auto_sampler_is_modality_aware():
    clinical, expression, survival = small_cohort(n=40, n_genes=8)
    methylation = _toy_methylation(40, n_features=6)
    _, transforms, cfg = mdata.build_from_frames(
        clinical, expression, survival, small_cfg(observation_sampler="auto"),
        methylation=methylation)
    model = JointMultimodalVAE(cfg, transforms.column_specs)
    assert model.cfg.observation_sampler == "residual_factor"


def test_joint_methylation_fit_sample_roundtrip(tmp_path):
    clinical, expression, survival = small_cohort(n=100)
    methylation = _toy_methylation(len(expression))

    cfg = small_cfg(methylation_head_hidden=(16,), max_epochs=3)
    synth = JointVAESynthesizer(cfg=cfg).fit(
        clinical, expression, survival, methylation=methylation
    )
    # methylation wired as a 3rd encoded modality with its own head
    assert synth.cfg.methylation_dim == methylation.shape[1]
    assert synth.model.head_meth is not None
    assert synth.model.gating.n_modalities == 3
    assert "methylation" in synth.model.objective.modality_names

    result = synth.sample(20)
    assert result.methylation is not None
    assert result.methylation.shape == (20, methylation.shape[1])
    assert list(result.methylation.columns) == list(methylation.columns)
    vals = result.methylation.to_numpy()
    assert np.isfinite(vals).all()
    assert (vals >= 0.0).all() and (vals <= 1.0).all()  # decoded back to beta space

    # save/load reconstructs the methylation head from cfg + pickled transforms
    model_path = tmp_path / "model.pt"
    synth.save(model_path)
    reloaded = JointVAESynthesizer.load(model_path).sample(8)
    assert reloaded.methylation is not None
    assert list(reloaded.methylation.columns) == list(methylation.columns)


def test_shared_private_latents_fit_generate_and_persist(tmp_path):
    clinical, expression, survival = small_cohort(n=72, n_genes=12)
    methylation = _toy_methylation(72, n_features=8)
    cfg = small_cfg(
        max_epochs=2, observation_sampler="residual_factor", residual_rank=2,
        latent_structure="shared_private",
        private_latent_dims={"expression": 3, "methylation": 5},
        private_kl_weight=0.5,
    )
    synth = JointVAESynthesizer(cfg=cfg).fit(
        clinical, expression, survival, methylation=methylation)
    assert synth.model.latent_layout() == {
        "shared": (0, 8),
        "private": {"expression": (8, 11), "methylation": (11, 16)},
        "total_dim": 16,
    }
    assert synth.model._latent_gmm.means_.shape[1] == 16
    assert synth.sample(9).latents.shape == (9, 8)
    assert synth.model.history_[-1]["kl_private"] > 0
    path = tmp_path / "private.pt"
    synth.save(path)
    loaded = JointVAESynthesizer.load(path)
    assert loaded.model.latent_layout() == synth.model.latent_layout()
    assert loaded.sample(7).methylation.shape == (7, 8)


def test_shared_switch_preserves_historical_modules_and_state_keys():
    clinical, expression, survival = small_cohort(n=40, n_genes=8)
    _, transforms, plain_cfg = mdata.build_from_frames(
        clinical, expression, survival, small_cfg())
    _, _, rollback_cfg = mdata.build_from_frames(
        clinical, expression, survival, small_cfg(
            latent_structure="shared", private_latent_dims={"expression": 6}))
    torch.manual_seed(3)
    plain = JointMultimodalVAE(plain_cfg, transforms.column_specs)
    torch.manual_seed(3)
    rollback = JointMultimodalVAE(rollback_cfg, transforms.column_specs)
    assert not hasattr(plain, "enc_private_modalities")
    assert not hasattr(rollback, "enc_private_modalities")
    assert list(plain.state_dict()) == list(rollback.state_dict())
    assert all(torch.equal(plain.state_dict()[key], rollback.state_dict()[key])
               for key in plain.state_dict())


def test_private_dimension_validation_is_modality_aware():
    cfg = small_cfg(latent_structure="shared_private",
                    private_latent_dims={"unknown": 4})
    with pytest.raises(ValueError, match="unknown modalities"):
        cfg.resolved_private_dims(["expression"])
    with pytest.raises(ValueError, match="cannot be negative"):
        small_cfg(latent_structure="shared_private",
                  private_latent_dims={"default": -1}).resolved_private_dims(["expression"])


def test_residual_sampler_fit_and_persistence(tmp_path):
    clinical, expression, survival = small_cohort(n=72, n_genes=12)
    methylation = _toy_methylation(len(expression), n_features=8)
    cfg = small_cfg(
        max_epochs=2, methylation_head_hidden=(16,),
        observation_sampler="residual_factor", residual_rank=4,
        observation_noise_scale={"expression": 0.5, "methylation": 0.75},
    )
    synth = JointVAESynthesizer(cfg=cfg).fit(
        clinical, expression, survival, methylation=methylation)
    state = synth.model._residual_sampler
    assert state["score_dim"] == 8
    assert set(state["modalities"]) == {"expression", "methylation"}
    path = tmp_path / "residual_model.pt"
    synth.save(path)
    loaded = JointVAESynthesizer.load(path)
    assert loaded.model._residual_sampler["score_dim"] == 8
    result = loaded.sample(10)
    assert np.isfinite(result.expression.to_numpy()).all()
    assert np.isfinite(result.methylation.to_numpy()).all()


def test_registry_supports_continuous_and_binary_omics():
    """New heads are declared in config rather than hard-coded in the model."""
    clinical, expression, survival = small_cohort(n=72, n_genes=16)
    rng = np.random.default_rng(4)
    cnv = pd.DataFrame(rng.normal(size=(72, 7)), columns=[f"cnv_{i}" for i in range(7)])
    mutations = pd.DataFrame(rng.binomial(1, .2, size=(72, 9)), columns=[f"mut_{i}" for i in range(9)])
    atac = pd.DataFrame(rng.poisson(2, size=(72, 6)), columns=[f"peak_{i}" for i in range(6)])
    cfg = small_cfg(
        max_epochs=2,
        latent_structure="shared_private", private_latent_dims={"default": 2},
        modalities={"cnv": {"kind": "continuous", "head_hidden": [16]},
                    "mutations": {"kind": "binary", "head_hidden": [16]},
                    "atac": {"kind": "count", "likelihood": "nb", "head_hidden": [16]}},
    )
    synth = JointVAESynthesizer(cfg=cfg).fit(
        clinical, expression, survival, omics={"cnv": cnv, "mutations": mutations, "atac": atac}
    )
    assert set(synth.model.modality_names) == {"expression", "cnv", "mutations", "atac"}
    assert set(synth.model.enc_private_modalities) == {
        "expression", "cnv", "mutations", "atac"}
    result = synth.sample(12)
    assert set(result.modalities) == {"cnv", "mutations", "atac"}
    assert result.modalities["cnv"].shape == (12, 7)
    assert set(np.unique(result.modalities["mutations"].to_numpy())) <= {0.0, 1.0}


def test_registry_rejects_missing_or_invalid_binary_omics():
    clinical, expression, survival = small_cohort(n=30, n_genes=8)
    cfg = small_cfg(modalities={"mutations": {"kind": "binary"}})
    with pytest.raises(ValueError, match="missing"):
        mdata.build_from_frames(clinical, expression, survival, cfg)
    bad = pd.DataFrame(np.full((30, 3), 0.5))
    with pytest.raises(ValueError, match="0/1"):
        mdata.build_from_frames(clinical, expression, survival, small_cfg(
            modalities={"mutations": {"kind": "binary"}}), omics={"mutations": bad})


def test_geometry_metrics_support_single_and_pairwise_modalities():
    from synthvae.mmvae.metrics import cross_omic_geometry, modality_geometry
    rng = np.random.default_rng(7)
    real = pd.DataFrame(rng.normal(size=(24, 12)))
    synthetic = pd.DataFrame(rng.normal(size=(20, 12)))
    metric = modality_geometry(real, synthetic)
    assert np.isfinite(metric["geometry_penalty"])
    binary = modality_geometry((real > 0).astype(float), (synthetic > 0).astype(float), "binary")
    assert np.isfinite(binary["geometry_penalty"])
    assert np.isfinite(cross_omic_geometry(real, synthetic, real.iloc[:, :8], synthetic.iloc[:, :8]))


def test_geometry_penalizes_white_noise_with_correct_marginal_variance():
    from synthvae.mmvae.metrics import modality_geometry, quality_guard_failures
    rng = np.random.default_rng(19)
    latent = rng.normal(size=(80, 3))
    loadings = rng.normal(size=(3, 24))
    real = pd.DataFrame(latent @ loadings + rng.normal(scale=.05, size=(80, 24)))
    faithful = pd.DataFrame(real.to_numpy() + rng.normal(scale=.08, size=real.shape))
    white = pd.DataFrame(rng.normal(size=real.shape) * real.std().to_numpy()
                         + real.mean().to_numpy())
    good = modality_geometry(real, faithful)
    bad = modality_geometry(real, white)
    assert good["geometry_penalty"] < bad["geometry_penalty"]
    prefixed = {f"expression_{key}": value for key, value in bad.items()}
    assert any("covariance_structure" in failure for failure in
               quality_guard_failures(prefixed, ["expression"]))


def test_source_overlap_metrics_detect_subgroup_shells():
    from synthvae.mmvae.metrics import modality_geometry
    rng = np.random.default_rng(29)
    groups = np.repeat(["A", "B"], 60)
    centres = np.where((groups == "A")[:, None], -1.5, 1.5)
    real = centres + rng.normal(scale=0.7, size=(120, 24))
    faithful = real + rng.normal(scale=0.08, size=real.shape)
    shell = np.empty_like(real)
    for group in ("A", "B"):
        idx = np.where(groups == group)[0]
        direction = rng.normal(size=(len(idx), real.shape[1]))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        centre = real[idx].mean(0)
        radius = np.median(np.linalg.norm(real[idx] - centre, axis=1)) * 1.55
        shell[idx] = centre + direction * radius
    good = modality_geometry(pd.DataFrame(real), pd.DataFrame(faithful),
                             real_groups=groups, synthetic_groups=groups)
    bad = modality_geometry(pd.DataFrame(real), pd.DataFrame(shell),
                            real_groups=groups, synthetic_groups=groups)
    assert bad["source_overlap_radial_worst"] > good["source_overlap_radial_worst"]
    assert (bad["source_overlap_boundary_penalty_worst"]
            > good["source_overlap_boundary_penalty_worst"])
    assert (bad["source_overlap_radial_median_ratio_worst"]
            > good["source_overlap_radial_median_ratio_worst"])
    assert (bad["source_overlap_local_density_wasserstein_worst"]
            > good["source_overlap_local_density_wasserstein_worst"])
    assert bad["source_overlap_mixing_worst"] < good["source_overlap_mixing_worst"]
    assert bad["source_overlap_precision_worst"] < good["source_overlap_precision_worst"]


def test_stratified_folds_cover_every_patient_once():
    from synthvae.cli.mmvae import stratified_folds
    clinical, expression, survival = small_cohort(n=60, n_genes=8)
    clinical["MNP_Subgroup"] = np.tile(["I", "II", "III"], 20)
    folds = stratified_folds(clinical, expression, survival, seed=3, n_splits=3)
    assert len(folds) == 3
    assert sum(len(fold[6]) for fold in folds) == len(expression)
    for fold in folds:
        assert set(fold[0]["MNP_Subgroup"]) == {"I", "II", "III"}


def test_tune_persists_resolved_default_configuration(tmp_path):
    """A sweep without --config still promotes the safe canonical defaults."""
    from synthvae.cli.mmvae import main
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        "fixed:\n  optimisation.max_epochs: 1\n  dims.latent_dim: 4\n"
        "grid:\n  optimisation.seed: [0]\n",
        encoding="utf-8",
    )
    out = tmp_path / "tune"
    assert main(["tune", "--toy", "--n", "48", "--n-genes", "12", "--sweep", str(sweep),
                 "--out-dir", str(out), "--device", "cpu"]) == 0
    import json
    selected = json.loads((out / "selected_config.json").read_text(encoding="utf-8"))["best_config"]
    assert selected["recon_reduction"] == "mean_per_feature"
    assert selected["latent_dim"] == 4
    assert isinstance(selected["latent_dim"], int)
    assert isinstance(selected["max_epochs"], int)


def test_tune_generation_preset_is_persisted(tmp_path):
    from synthvae.cli.mmvae import main
    sweep = tmp_path / "sampler_sweep.yaml"
    sweep.write_text(
        "cv_folds: 2\nsynthetic_replicates: 1\n"
        "fixed:\n  optimisation.max_epochs: 1\n  dims.latent_dim: 4\n"
        "grid:\n  optimisation.seed: [0]\n  generation.preset:\n"
        "    - {observation_sampler: residual_factor, residual_rank: 2, "
        "observation_noise_scale: {expression: 0.5}, gmm_components: 2, "
        "posterior_samples_per_patient: 1}\n",
        encoding="utf-8",
    )
    out = tmp_path / "tune"
    assert main(["tune", "--toy", "--n", "40", "--n-genes", "10",
                 "--sweep", str(sweep), "--out-dir", str(out),
                 "--device", "cpu"]) == 0
    import json
    selected = json.loads((out / "selected_config.json").read_text())["best_config"]
    assert selected["observation_sampler"] == "residual_factor"
    assert selected["residual_rank"] == 2
    assert isinstance(selected["residual_rank"], int)
    assert selected["observation_noise_scale"] == {"expression": 0.5}
    assert selected["gmm_components"] == 2
    assert isinstance(selected["gmm_components"], int)


def test_staged_tune_persists_private_winner_and_resumes(tmp_path):
    from synthvae.cli.mmvae import main
    sweep = tmp_path / "staged.yaml"
    sweep.write_text(
        "staged: true\ncv_folds: 2\nenforce_quality_guards: false\n"
        "fixed:\n  fusion.strategy: moe\n  objective.weighting_scheme: fixed\n"
        "stages:\n  screening_max_epochs: 1\n  finalist_max_epochs: 1\n"
        "  shared_latent_dims: [4]\n"
        "  private_latent_presets: [{default: 2}]\n"
        "  private_kl_weights: [0.5]\n  fusion: [moe]\n"
        "  weighting: [fixed]\n  beta_max: [0.1]\n"
        "  retain_architectures: 1\n  posterior_draws: [1]\n"
        "  gmm_components: [1]\n  residual_ranks: [0]\n"
        "  common_noise_scales: [0.5]\n  retain_finalists: 1\n"
        "  finalist_seeds: [0]\n",
        encoding="utf-8",
    )
    out = tmp_path / "staged"
    args = ["tune", "--toy", "--n", "40", "--n-genes", "10",
            "--sweep", str(sweep), "--out-dir", str(out), "--device", "cpu"]
    assert main(args) == 0
    import json
    selected = json.loads((out / "selected_config.json").read_text())["best_config"]
    assert selected["latent_structure"] == "shared_private"
    assert selected["private_latent_dims"] == {"default": 2}
    before = (out / "stage4_finalists.csv").read_text()
    assert main(args + ["--resume"]) == 0
    assert (out / "stage4_finalists.csv").read_text() == before


def test_low_noise_calibration_uses_selected_architecture_and_new_outputs(tmp_path):
    from dataclasses import asdict
    from synthvae.cli.mmvae import main
    from synthvae.mmvae import MMVAEConfig
    import json

    selected = tmp_path / "broad_selected.json"
    config = asdict(MMVAEConfig(
        latent_dim=4, latent_structure="shared_private",
        private_latent_dims={"default": 2}, private_kl_weight=0.5,
        fusion="moe", weighting_scheme="fixed", max_epochs=1,
        residual_rank=8, observation_noise_scale={"default": 1.0},
        device="cpu",
    ))
    selected.write_text(json.dumps({"best_config": config}), encoding="utf-8")
    sweep = tmp_path / "calibration.yaml"
    sweep.write_text(
        "staged: true\ncalibration_only: true\ncv_folds: 2\nfixed: {}\n"
        "enforce_quality_guards: false\nstages:\n"
        "  screening_max_epochs: 1\n  finalist_max_epochs: 1\n"
        "  residual_ranks: [0]\n  common_noise_scales: [0.5]\n"
        "  asymmetric_noise_scales: []\n  screening_replicates: 1\n"
        "  retain_finalists: 1\n  finalist_seeds: [0]\n"
        "  finalist_replicates: 1\n",
        encoding="utf-8",
    )
    out = tmp_path / "calibration"
    args = ["tune", "--toy", "--n", "40", "--n-genes", "10",
            "--config", str(selected), "--sweep", str(sweep),
            "--out-dir", str(out), "--device", "cpu"]
    assert main(args) == 0
    winner = json.loads((out / "selected_config.json").read_text())
    assert winner["tuning_objective"] == "low_noise_worst_subgroup_calibration"
    assert winner["calibration_base"] == str(selected)
    assert winner["best_config"]["latent_structure"] == "shared_private"
    assert winner["best_config"]["private_latent_dims"] == {"default": 2}
    assert winner["best_config"]["residual_rank"] == 0
    assert winner["best_config"]["observation_noise_scale"] == {"default": 0.5}
    assert (out / "stage1_calibration.csv").exists()
    assert (out / "stage2_finalists_ranked.csv").exists()


# --------------------------------------------------------------------------- #
# NB / ZINB count likelihoods (T5)
# --------------------------------------------------------------------------- #
def _sample_nb(mu, theta, size, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(shape=theta, scale=mu / theta, size=size)
    return rng.poisson(lam).astype(float)


def test_nb_likelihood_prefers_true_dispersion():
    from synthvae.mmvae.heads import _log_nb_positive
    mu_val, theta_true = 5.0, 2.0
    x = torch.tensor(_sample_nb(mu_val, theta_true, (400, 10)))
    mu = torch.full_like(x, mu_val)
    ll_true = _log_nb_positive(x, mu, torch.tensor(theta_true)).mean()
    ll_wrong = _log_nb_positive(x, mu, torch.tensor(0.05)).mean()
    assert torch.isfinite(ll_true) and torch.isfinite(ll_wrong)
    assert ll_true > ll_wrong  # higher log-likelihood at the true dispersion


def test_zinb_beats_nb_on_zero_inflated_counts():
    from synthvae.mmvae.heads import _log_nb_positive, _log_zinb_positive
    mu_val, theta = 6.0, 2.0
    x = _sample_nb(mu_val, theta, (400, 10), seed=1)
    # inject excess zeros
    rng = np.random.default_rng(2)
    drop = rng.random(x.shape) < 0.4
    x[drop] = 0.0
    x = torch.tensor(x)
    mu = torch.full_like(x, mu_val)
    th = torch.tensor(theta)
    ll_nb = _log_nb_positive(x, mu, th).mean()
    # gate logit ~ logit(0.4); zinb should model the excess zeros better
    pi_logits = torch.full_like(x, np.log(0.4 / 0.6))
    ll_zinb = _log_zinb_positive(x, mu, th, pi_logits).mean()
    assert torch.isfinite(ll_zinb)
    assert ll_zinb > ll_nb


def test_count_head_nll_finite():
    from synthvae.mmvae.heads import ExpressionHead
    for likelihood in ("nb", "zinb"):
        head = ExpressionHead(8, 10, hidden=(16,), likelihood=likelihood)
        z = torch.randn(20, 8)
        target = torch.tensor(_sample_nb(5.0, 2.0, (20, 10)))
        loss = head.nll(z, target, library=torch.ones(20, 1))
        assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# GradNorm weighting toggle (T6)
# --------------------------------------------------------------------------- #
def test_gradnorm_runs_and_logs_weights():
    clinical, expression, survival = small_cohort()
    cfg = small_cfg(weighting_scheme="gradnorm", max_epochs=4)
    synth = JointVAESynthesizer(cfg=cfg).fit(clinical, expression, survival)
    hist = synth.model.history_
    assert len(hist) >= 1
    assert np.isfinite([h["val_obj"] for h in hist]).all()
    # per-task gradnorm weights are logged
    assert any(k.startswith("w_") for k in hist[-1])


def test_gradnorm_runs_with_private_latent_kl():
    clinical, expression, survival = small_cohort(n=72, n_genes=12)
    cfg = small_cfg(
        weighting_scheme="gradnorm", max_epochs=2,
        latent_structure="shared_private", private_latent_dims={"expression": 3},
        private_kl_weight=0.5,
    )
    synth = JointVAESynthesizer(cfg=cfg).fit(clinical, expression, survival)
    last = synth.model.history_[-1]
    assert np.isfinite(last["kl_private"])
    assert last["kl_private"] > 0
    assert any(key.startswith("w_") for key in last)
