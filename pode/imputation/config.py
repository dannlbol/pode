"""Configuration for generative imputation and its benchmarking harness.

Kept in its own YAML (``--impute-config``) rather than as a section of
:class:`~pode.config.PodeConfig`, so existing PODE configs, ``promote`` output
and the train/generate/tune workflow are untouched by this feature.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EngineSpec:
    """One imputer to run: a registered engine name plus constructor params."""

    name: str
    label: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.label or self.name


@dataclass
class LatentConfig:
    """Latent-space conditioning shared by both TVAE engines."""

    # Importance samples per row for sampling-importance-resampling (SIR).
    k: int = 2000
    # 'sir' | 'pcn' | 'smc' | 'auto' (SIR; rows with low ESS fall back to
    # tempered SMC from the prior, or pCN refinement with an encoder proposal).
    method: str = "auto"
    # pCN: steps per chain and initial step size (adapted per row; also the
    # initial step size of SMC mutations).
    pcn_steps: int = 200
    pcn_beta: float = 0.2
    # Tempered SMC: particles (None -> k), pCN moves per stage, ESS fraction
    # kept per tempering increment, and a cap on stages.
    smc_particles: int | None = None
    smc_mutation_steps: int = 5
    smc_target_ess: float = 0.5
    smc_max_stages: int = 100
    # Rows whose SIR effective sample size falls below this are refined under
    # method='auto'. None -> the number of draws M.
    ess_min: float | None = None
    # Exponent on p(x_obs | z). None -> the model's ``loss_factor``, matching
    # the tempered likelihood the TVAE was trained with.
    likelihood_power: float | None = None
    # Upper bound on (rows x samples) decoded in one forward pass.
    max_batch: int = 200_000


@dataclass
class MaskedTVAEConfig:
    """Mask-aware TVAE training options (architecture comes from SynthesizerConfig)."""

    # Per-row artificial masking rate is drawn from U(0, p_max) each batch.
    p_max: float = 0.5
    # Probability a row additionally drops one whole configured block group.
    block_prob: float = 0.25
    # Loss weight on artificially hidden (observed) cells relative to visible ones.
    hidden_weight: float = 2.0
    # KL weight.
    beta: float = 1.0
    # Optional epoch override; None -> SynthesizerConfig.epochs.
    epochs: int | None = None


@dataclass
class GainConfig:
    """GAIN baseline options (Yoon et al., 2018)."""

    epochs: int = 1000
    batch_size: int = 128
    hidden_dim: int | None = None
    hint_rate: float = 0.9
    alpha: float = 100.0
    # Artificial MCAR masking of observed cells during training, so GAIN can
    # learn even when its training rows are complete.
    train_mask_rate: float = 0.2
    lr: float = 1e-3


@dataclass
class CoxConfig:
    """Downstream Cox modelling pooled with Rubin's rules."""

    enabled: bool = True
    # 'univariate' (one model per covariate) or 'multivariable' (one joint model).
    mode: str = "univariate"
    # Covariates (globs allowed). Empty -> the schema's feature columns.
    covariates: list[str] = field(default_factory=list)
    max_covariates: int | None = 25
    penalizer: float = 0.0
    alpha: float = 0.05


@dataclass
class DeltaConfig:
    """MNAR delta-adjustment sensitivity analysis."""

    enabled: bool = True
    # Continuous: shift imputed values by delta * SD(observed).
    # Binary: shift the per-cell log-odds of the second level by delta.
    deltas: list[float] = field(default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0])
    # Columns to shift (globs). Empty -> every imputed continuous/binary column.
    columns: list[str] = field(default_factory=list)
    # Engines to analyse (labels). Empty -> all.
    engines: list[str] = field(default_factory=list)


@dataclass
class ImputationConfig:
    """Top-level imputation / benchmark configuration."""

    # Number of imputations (draws) per incomplete row.
    m: int = 20
    seed: int = 42
    # CPU cores the benchmark may use (worker processes, BLAS and torch CPU
    # threads, MICE chains). -1 = all cores; keep it modest on shared servers.
    n_jobs: int = 4
    # Torch device for TVAE engines and GAIN: auto (GPU if available, falling
    # back to CPU when a saved model cannot be loaded onto it), cpu, or cuda.
    device: str = "auto"
    # ---- what the imputer may see (defaults; engines can override in params)
    # 'use' (outcome is a predictor; right for inference), 'hide' (outcomes of
    # the rows being imputed are masked from the engine; for prediction
    # validation/deployment), 'exclude' (outcomes never reach the engine).
    outcome_policy: str = "use"
    # Outcome columns (globs). Empty -> the schema's survival time/event columns.
    outcome_columns: list[str] = field(default_factory=list)
    # Columns the imputer never sees and does not impute (globs), e.g.
    # post-baseline or treatment-response variables.
    exclude_columns: list[str] = field(default_factory=list)
    # kNN / MICE / missForest: 'stacked' (models also learn from the rows being
    # imputed) or 'reference' (models learn from training/reference rows only).
    fit_scope: str = "stacked"
    # 'kfold': fit on training folds, impute masked held-out folds (no leakage).
    # 'transductive': fit on the whole masked table and impute it.
    mode: str = "kfold"
    folds: int = 5
    repeats: int = 1
    # In kfold mode, also apply the scenario mask to training rows (simulates
    # an incomplete training cohort). Off -> models are fit once per fold and
    # reused across scenarios.
    mask_training_rows: bool = False
    # Never mask these columns (globs); survival time/event are protected too
    # unless protect_survival is False.
    protect_columns: list[str] = field(default_factory=list)
    protect_survival: bool = True
    # Named column groups for block masking, e.g. {"expr": ["expr_latent_*"]}.
    block_groups: dict[str, list[str]] = field(default_factory=dict)
    # Treat each block group as all-or-nothing, as for latents derived from one
    # omics modality: benchmark masks and tvae-masked training never hide part
    # of a block, and `apply` warns if real data has partly missing blocks.
    atomic_blocks: bool = True
    # Masking scenarios; see pode.imputation.masking.MaskSpec for keys.
    scenarios: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "mcar_10", "kind": "mcar", "rate": 0.1},
            {"name": "mcar_30", "kind": "mcar", "rate": 0.3},
        ]
    )
    engines: list[EngineSpec] = field(
        default_factory=lambda: [
            EngineSpec("mean_mode"),
            EngineSpec("mice"),
            EngineSpec("tvae-decoder"),
        ]
    )
    # Epoch override for TVAE fits inside the benchmark (None -> PODE config).
    epochs: int | None = None
    latent: LatentConfig = field(default_factory=LatentConfig)
    masked_tvae: MaskedTVAEConfig = field(default_factory=MaskedTVAEConfig)
    gain: GainConfig = field(default_factory=GainConfig)
    cox: CoxConfig = field(default_factory=CoxConfig)
    delta: DeltaConfig = field(default_factory=DeltaConfig)
    plots: bool = True

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ImputationConfig":
        data = dict(data or {})
        engines = [
            EngineSpec(name=e) if isinstance(e, str) else EngineSpec(**e)
            for e in data.pop("engines", [])
        ] or None
        nested = {
            "latent": LatentConfig,
            "masked_tvae": MaskedTVAEConfig,
            "gain": GainConfig,
            "cox": CoxConfig,
            "delta": DeltaConfig,
        }
        kwargs = {k: klass(**data.pop(k, {}) or {}) for k, klass in nested.items()}
        cfg = cls(**data, **kwargs)
        if engines:
            cfg.engines = engines
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ImputationConfig":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def quick(self) -> "ImputationConfig":
        """A cheaper copy for smoke runs (fewer draws, folds, epochs, samples)."""
        cfg = ImputationConfig.from_dict(self.to_dict())
        cfg.m = min(cfg.m, 5)
        cfg.folds = min(cfg.folds, 3)
        cfg.repeats = 1
        cfg.epochs = min(cfg.epochs or 50, 50)
        cfg.latent.k = min(cfg.latent.k, 500)
        cfg.latent.pcn_steps = min(cfg.latent.pcn_steps, 50)
        cfg.latent.smc_max_stages = min(cfg.latent.smc_max_stages, 30)
        cfg.masked_tvae.epochs = min(cfg.masked_tvae.epochs or 50, 50)
        cfg.gain.epochs = min(cfg.gain.epochs, 200)
        cfg.cox.max_covariates = min(cfg.cox.max_covariates or 5, 5)
        cfg.delta.deltas = [-1.0, 0.0, 1.0]
        return cfg


def load_imputation_config(path: str | Path | None) -> ImputationConfig:
    """Load an imputation config from YAML, or return defaults when ``path`` is None."""
    if path is None:
        return ImputationConfig()
    return ImputationConfig.from_yaml(path)
