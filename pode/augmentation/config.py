"""Configuration for generative data augmentation and its benchmark.

Kept in its own YAML (``--augment-config``) rather than as a section of
:class:`~pode.config.PodeConfig`, so existing PODE configs and the
train/generate/tune workflow are untouched by this feature. The TVAE's own
architecture still comes from the PODE config passed with ``--config``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STRATEGY_KINDS = ("tvae_natural", "tvae_targeted", "tvae_subgroup", "duplicate", "jitter", "weight")
TVAE_KINDS = ("tvae_natural", "tvae_targeted", "tvae_subgroup")
MODEL_KINDS = ("metacpg_fixed", "metacpg_search", "coxnet", "cox_ph", "logistic_landmark")
GENERATOR_BACKENDS = ("tvae", "hybrid")


@dataclass
class EndpointConfig:
    """Survival endpoint used for modelling and scoring."""

    # Time/event columns. None -> the schema's primary survival pair.
    time: str | None = None
    event: str | None = None
    # Prediction horizon, in the units of ``time`` (e.g. months).
    horizon: float = 24.0


@dataclass
class SplitConfig:
    """Repeated stratified K-fold over the real development rows."""

    folds: int = 5
    repeats: int = 5
    # Columns stratified on, in addition to the event indicator (e.g. cohort).
    stratify: list[str] = field(default_factory=list)
    # Rows sharing a value (e.g. patient or family id) never straddle a split.
    group_column: str | None = None


@dataclass
class ExternalConfig:
    """Rows held out entirely as an external test cohort (never fitted on)."""

    column: str | None = None
    values: list[Any] = field(default_factory=list)


@dataclass
class HybridConfig:
    """``generator.backend: hybrid`` - omics matrix -> latents -> TVAE -> decode.

    Per split, a compressor (the synthvae survival VAE, or PCA) is fitted on the
    split's real matrix rows, the TVAE models clinical columns + latents, and
    synthetic latents are decoded back to the full matrix. This is what lets
    augmentation reach feature selection over thousands of CpGs.
    """

    # The omics matrix columns (globs), e.g. ["cg*"]. They are never given to the TVAE.
    matrix_columns: list[str] = field(default_factory=list)
    # methylation (beta values; logit-transformed), expression, copy_number.
    matrix_kind: str = "methylation"
    # selected_config.json from `pode synthesis hybrid tune` (its best_config is used).
    compressor_config: str | None = None
    # Overrides / settings when no file is given: compressor (survival_vae | pca),
    # n_latent, epochs, w_surv, beta_max, free_bits, n_top_genes, seed, ...
    compressor: dict[str, Any] = field(default_factory=dict)
    latent_prefix: str = "methylation_latent_"
    # Decoded matrices are decoder means, much smoother than real data. Add
    # per-feature Gaussian noise with the training reconstruction residual SD
    # (on the transformed scale) so synthetic samples have realistic spread.
    residual_noise: bool = True
    # Matrix-level fidelity uses the top-variance features (training rows).
    fidelity_top_features: int = 500


@dataclass
class GeneratorConfig:
    """How the per-split TVAE is fitted and sampled."""

    # Columns the TVAE never sees or synthesises (ids, cohort labels, ...).
    exclude_columns: list[str] = field(default_factory=list)
    # Stratum label of synthetic rows for model strata columns:
    # 'own' -> every synthetic row gets ``synthetic_label`` (synthetic patients
    # never share a baseline hazard with a real cohort); 'generated' -> use the
    # synthesised value when the strata column was generated, else 'own'.
    synthetic_strata: str = "own"
    synthetic_label: str = "__synthetic__"
    # Epoch override for TVAE fits inside the benchmark (None -> PODE config).
    epochs: int | None = None
    # Each split samples one pool per strategy of size
    # pool_factor * max(ratios) * n_train, and every draw subsamples it.
    pool_factor: float = 2.0
    # Plausibility rules for synthetic rows; violating rows are dropped and
    # counted. Keys: ranges {col: {min, max}}, positive [cols],
    # ordered [[a, b]] (a <= b when both present), forbid [{col: value, ...}].
    constraints: dict[str, Any] = field(default_factory=dict)
    # Reuse per-split models saved in out_dir/generators on re-runs.
    cache: bool = True
    # tvae: the TVAE models the table directly. hybrid: see HybridConfig.
    backend: str = "tvae"
    hybrid: HybridConfig = field(default_factory=HybridConfig)


@dataclass
class StrategySpec:
    """One augmentation arm.

    ``kind`` is one of :data:`STRATEGY_KINDS`. ``target`` selects the rows a
    strategy enriches, as ``{equals: {col: value}, ranges: {col: {min, max}}}``
    or a full condition config ``{groups: [{proportion, equals, ranges}, ...]}``
    (the ``pode synthesis generate --conditions-config`` format). ``params``
    holds kind-specific options: ``columns``/``target_count`` (tvae_subgroup),
    ``sd_frac`` (jitter).
    """

    kind: str
    label: str | None = None
    target: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.label or self.kind

    @property
    def uses_tvae(self) -> bool:
        return self.kind in TVAE_KINDS


@dataclass
class ModelSpec:
    """One downstream prognostic model family (see :mod:`.models`)."""

    kind: str
    label: str | None = None
    # Feature columns/globs. Empty -> AugmentationConfig.features.
    features: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.label or self.kind


@dataclass
class LearningCurveConfig:
    """Real-only fits on subsamples, for the real-data-equivalent efficiency."""

    enabled: bool = True
    # Fractions of each split's real training rows (1.0 is the real-only arm).
    fractions: list[float] = field(default_factory=lambda: [0.5, 0.75])
    draws: int = 3


@dataclass
class PrivacyConfig:
    enabled: bool = True
    # Columns compared (globs). Empty -> every generated non-outcome column.
    columns: list[str] = field(default_factory=list)
    # Distances use at most this many synthetic rows per split.
    max_synthetic: int = 2000
    # Categorical combinations seen in at most rare_k training rows are "rare".
    rare_k: int = 3


@dataclass
class GateConfig:
    """Scorecard thresholds (see :mod:`.scorecard`)."""

    # Discrimination metric the utility gate is judged on.
    primary_metric: str = "c_harrell"
    # Utility: improvement when the lower CI of ΔC > 0; non-inferior when > -margin.
    noninferiority_margin: float = 0.01
    # Calibration: how much further from 1 the calibration slope may move
    # (Δ|slope - 1|; moving towards 1 never fails), and the Δ integrated Brier limit.
    max_abs_delta_slope: float = 0.10
    max_delta_ibs: float = 0.005
    # Subgroups: worst subgroup ΔC allowed (subgroups meeting the minimum size).
    min_subgroup_delta_c: float = -0.05
    # Fidelity of synthetic vs training rows.
    max_discriminator_auc: float = 0.80
    max_target_tvd: float = 0.15
    # Privacy.
    max_exact_duplicates: int = 0
    max_membership_auc: float = 0.60
    min_dcr_p5_ratio: float = 0.80


@dataclass
class AugmentationConfig:
    """Top-level augmentation / benchmark configuration."""

    seed: int = 42
    # CPU cores the benchmark may use (worker processes, BLAS and torch CPU
    # threads). -1 = all cores; keep it modest on shared servers.
    n_jobs: int = 4
    # Torch device for TVAE fits: auto (GPU if available), cpu, or cuda.
    device: str = "auto"
    # Patient identifier: kept through data preparation, never synthesised or
    # modelled, and used to record split membership in splits.csv.
    id_column: str | None = None
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    external: ExternalConfig = field(default_factory=ExternalConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    # Model features (globs). Empty -> generated feature columns minus strata.
    features: list[str] = field(default_factory=list)
    # Columns forming Cox baseline strata (metacpg_fixed, cox_ph).
    strata: list[str] = field(default_factory=list)
    # Columns for the subgroup audit, and its minimum sizes.
    subgroups: list[str] = field(default_factory=list)
    min_subgroup_n: int = 15
    min_subgroup_events: int = 5
    # Risk thresholds (event by the horizon) for decision-curve net benefit.
    net_benefit_thresholds: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    # Amount of synthetic data, as a multiple of the real training rows.
    ratios: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0])
    # Random subsamples of the pool per (split, strategy, ratio).
    draws: int = 5
    strategies: list[StrategySpec] = field(
        default_factory=lambda: [
            StrategySpec("tvae_natural"),
            StrategySpec("duplicate"),
            StrategySpec("jitter", params={"sd_frac": 0.1}),
        ]
    )
    models: list[ModelSpec] = field(
        default_factory=lambda: [ModelSpec("coxnet"), ModelSpec("cox_ph")]
    )
    learning_curve: LearningCurveConfig = field(default_factory=LearningCurveConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    fidelity: bool = True
    gates: GateConfig = field(default_factory=GateConfig)
    # Reference feature list for the selection-stability summary of
    # metacpg_search: a list of feature names, or a MetaCPG model.json path.
    reference_signature: Any = None
    # Paired test-set bootstrap replicates for the external cohort.
    bootstrap: int = 1000
    plots: bool = True

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AugmentationConfig":
        data = dict(data or {})
        strategies = [
            StrategySpec(kind=s) if isinstance(s, str) else StrategySpec(**s)
            for s in data.pop("strategies", [])
        ] or None
        models = [
            ModelSpec(kind=m) if isinstance(m, str) else ModelSpec(**m)
            for m in data.pop("models", [])
        ] or None
        nested = {
            "endpoint": EndpointConfig,
            "splits": SplitConfig,
            "external": ExternalConfig,
            "generator": GeneratorConfig,
            "learning_curve": LearningCurveConfig,
            "privacy": PrivacyConfig,
            "gates": GateConfig,
        }
        gen = dict(data.get("generator") or {})
        if isinstance(gen.get("hybrid"), dict):
            gen["hybrid"] = HybridConfig(**gen["hybrid"])
        data["generator"] = gen
        kwargs = {k: klass(**data.pop(k, {}) or {}) for k, klass in nested.items()}
        cfg = cls(**data, **kwargs)
        if strategies:
            cfg.strategies = strategies
        if models:
            cfg.models = models
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AugmentationConfig":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def validate(self) -> None:
        for s in self.strategies:
            if s.kind not in STRATEGY_KINDS:
                raise ValueError(f"Unknown strategy kind '{s.kind}'; choose from {STRATEGY_KINDS}.")
            if s.kind == "tvae_targeted" and not s.target:
                raise ValueError(f"Strategy '{s.key}' (tvae_targeted) needs a 'target'.")
            if s.kind == "tvae_subgroup" and not s.params.get("columns"):
                raise ValueError(f"Strategy '{s.key}' (tvae_subgroup) needs params.columns.")
        for m in self.models:
            if m.kind not in MODEL_KINDS:
                raise ValueError(f"Unknown model kind '{m.kind}'; choose from {MODEL_KINDS}.")
        for label, items in (("strategy", self.strategies), ("model", self.models)):
            keys = [i.key for i in items]
            if len(keys) != len(set(keys)):
                raise ValueError(f"Duplicate {label} labels {keys}; give each a unique 'label'.")
        if self.generator.backend not in GENERATOR_BACKENDS:
            raise ValueError(f"generator.backend must be one of {GENERATOR_BACKENDS}.")
        if self.generator.backend == "hybrid" and not self.generator.hybrid.matrix_columns:
            raise ValueError("generator.backend 'hybrid' needs generator.hybrid.matrix_columns.")
        if self.generator.synthetic_strata not in ("own", "generated"):
            raise ValueError("generator.synthetic_strata must be 'own' or 'generated'.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda.")
        if not self.ratios or min(self.ratios) <= 0:
            raise ValueError("ratios must be a non-empty list of positive numbers.")

    def quick(self) -> "AugmentationConfig":
        """A cheaper copy for smoke runs (fewer splits, draws, epochs, ratios)."""
        cfg = AugmentationConfig.from_dict(self.to_dict())
        cfg.splits.folds = min(cfg.splits.folds, 3)
        cfg.splits.repeats = 1
        cfg.draws = min(cfg.draws, 2)
        cfg.ratios = sorted(cfg.ratios)[:: max(1, len(cfg.ratios) // 2)][:2]
        cfg.generator.epochs = min(cfg.generator.epochs or 200, 200)
        cfg.learning_curve.draws = 1
        cfg.learning_curve.fractions = cfg.learning_curve.fractions[:1]
        cfg.privacy.max_synthetic = min(cfg.privacy.max_synthetic, 500)
        cfg.bootstrap = min(cfg.bootstrap, 200)
        return cfg


def load_augmentation_config(path: str | Path | None) -> AugmentationConfig:
    """Load an augmentation config from YAML, or return defaults when ``path`` is None."""
    if path is None:
        return AugmentationConfig()
    return AugmentationConfig.from_yaml(path)
