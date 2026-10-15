"""Configuration for federated learning across simulated (or, later, real) sites.

Kept in its own YAML (``--federated-config``) rather than as a section of
:class:`~pode.config.PodeConfig`, so existing PODE configs are untouched. The
TVAE *architecture* (layer sizes, embedding, batch size, loss factor) still
comes from the PODE config's ``synthesizer`` section, so tuned values carry
over; this file only describes the federation around it.

Sections:

* ``sites``       - which column defines a site, which sites take part, and
                    which are held out entirely for leave-one-site-out checks;
* ``generator``   - the federated TVAE (rounds, local epochs, FedAvg/FedProx,
                    site weighting, secure aggregation, personalisation);
* ``transformer`` - the shared column encoder built from per-site summaries
                    (small-cell suppression, histogram bins, public bounds);
* ``dp``          - DP-SGD (per-sample clipping + Gaussian noise, Opacus
                    accounting) for local and federated training;
* ``exchange``    - local TVAEs whose synthetic data is released only when it
                    passes the privacy gate;
* ``evaluation``  - the benchmark (endpoint, models, splits, conditions).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALGORITHMS = ("fedavg", "fedprox")
WEIGHTINGS = ("n", "sqrt_n", "uniform", "capped")
EXCHANGE_GENERATORS = ("fed_tvae", "sdv_tvae")
CONDITIONS = (
    "local",             # each site's own TVAE, evaluated at that site
    "exchange",          # pooled privacy-gated releases of local TVAEs
    "fedavg",            # federated TVAE, FedAvg
    "fedprox",           # federated TVAE, FedProx
    "fed_personalised",  # federated TVAE + local decoder fine-tuning
    "central_fed_tvae",  # same architecture on pooled rows (upper bound)
    "central_sdv_tvae",  # the existing PODE TVAEGenerator on pooled rows
    "central_copula",    # SDV Gaussian copula on pooled rows (non-neural)
)
# Plus ``fed_dp_eps<E>`` (federated + DP-SGD at target epsilon E), e.g. fed_dp_eps3.
DP_PREFIX = "fed_dp_eps"


@dataclass
class SitesConfig:
    # Column whose levels are the sites (simulation mode).
    column: str = "site"
    # Sites taking part (empty -> every level of ``column``).
    include: list[str] = field(default_factory=list)
    # Sites smaller than this are refused (too small to train or evaluate).
    min_site_n: int = 10
    # Sites held out of the federation entirely in the leave-one-site-out arm.
    loso: list[str] = field(default_factory=list)


@dataclass
class TransformerConfig:
    # A category seen in fewer than ``min_cell`` patients at a site is not
    # disclosed by that site; it is pooled into ``rare_label`` instead.
    min_cell: int = 3
    rare_label: str = "__rare__"
    # Histogram bins per numeric column.
    bins: int = 50
    # Per-site quantiles that set the shared numeric range when no public
    # ``bounds`` are given (robust to single extreme patients).
    range_quantiles: list[float] = field(default_factory=lambda: [0.01, 0.99])
    # Public numeric bounds, column -> [low, high]. Needed for a fully
    # differentially private transformer.
    bounds: dict[str, list[float]] = field(default_factory=dict)
    # Public category vocabularies, column -> levels (optional).
    categories: dict[str, list[Any]] = field(default_factory=dict)
    # Rows in the pseudo-sample the column encoder is fitted on.
    pseudo_rows: int = 5000
    # Max Gaussian modes per continuous column (ctgan default 10).
    max_clusters: int = 10


@dataclass
class DPConfig:
    enabled: bool = False
    # Either a fixed noise multiplier, or a target epsilon to solve for.
    noise_multiplier: float | None = None
    target_epsilon: float | None = None
    # None -> min(1e-5, 1 / (10 * n_site)) per site.
    delta: float | None = None
    # Per-sample gradient clipping norm C.
    max_grad_norm: float = 1.0
    # Expected Poisson batch size B (sampling rate q = B / n_site). Small
    # batches amplify privacy and keep per-sample gradients in memory.
    batch_size: int = 64
    # Laplace budget for the transformer's per-site summaries (added to the
    # per-site total by basic composition).
    transformer_epsilon: float = 0.5
    # Opacus accountant: rdp or prv.
    accountant: str = "rdp"


@dataclass
class GeneratorConfig:
    # Federation rounds and local epochs per round. A one-site run with
    # rounds x local_epochs epochs is a plain (local or central) TVAE.
    rounds: int = 60
    local_epochs: int = 5
    algorithm: str = "fedavg"
    # FedProx proximal weight (ignored by fedavg).
    prox_mu: float = 0.01
    # How site updates are weighted: n, sqrt_n, uniform, or capped (n, with no
    # site above ``cap_fraction`` of the total).
    weighting: str = "n"
    cap_fraction: float = 0.5
    # Pairwise-masked aggregation: the coordinator only sees the sum.
    secure_aggregation: bool = True
    # Model the site as a column (lets one generator sample per site).
    site_in_model: bool = False
    # Local decoder fine-tuning epochs after federation (personalised generators).
    personalise_epochs: int = 0
    learning_rate: float = 1e-3
    # Columns (globs) never given to the generator, e.g. ids, post-outcome fields.
    exclude_columns: list[str] = field(default_factory=list)
    # Plausibility rules, as in augmentation (ranges / positive / ordered / forbid).
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeConfig:
    # Privacy YAML (pode.privacy) used for each site's release gate.
    privacy_config: str | None = None
    # Verdicts that may be released.
    release_verdicts: list[str] = field(default_factory=lambda: ["PASS"])
    # Local generator: fed_tvae (this package, one site; DP-capable) or
    # sdv_tvae (the existing PODE TVAEGenerator).
    generator: str = "fed_tvae"
    # Share of each site held out to test its release for membership leakage.
    holdout_frac: float = 0.3
    # Synthetic rows per site (None -> as many as the site's training rows).
    num_samples: int | None = None


@dataclass
class EvaluationConfig:
    # Survival endpoint (None -> the schema's primary survival pair).
    time: str | None = None
    event: str | None = None
    # Horizon for AUC/Brier/calibration (None -> median observed time).
    horizon: float | None = None
    # Predictor columns (globs) for the downstream models (empty -> every
    # generator column except the endpoint).
    features: list[str] = field(default_factory=list)
    # Downstream prognostic models, as augmentation ModelSpecs.
    models: list[dict[str, Any]] = field(default_factory=lambda: [
        {"kind": "cox_ph", "params": {"penalizer": 0.1}}])
    # Share of each site held out as its real test set.
    holdout_frac: float = 0.3
    repeats: int = 3
    # Synthetic rows drawn for TSTR per evaluation (None -> the condition's
    # real training size).
    n_synthetic: int | None = None
    conditions: list[str] = field(default_factory=lambda: [
        "local", "exchange", "fedavg", "fedprox", "fed_personalised", "fed_dp_eps3",
        "central_fed_tvae", "central_sdv_tvae", "central_copula"])
    # Paired-bootstrap resamples for confidence intervals.
    n_bootstrap: int = 200
    # A condition is non-inferior when the lower CI of its C-index difference
    # versus the reference is above -margin.
    non_inferiority_margin: float = 0.02
    # Privacy YAML for the per-site privacy assessment (None -> defaults).
    privacy_config: str | None = None
    # Fidelity discriminator above this -> synthetic rows are easily spotted.
    max_discriminator_auc: float = 0.8


@dataclass
class FederatedConfig:
    sites: SitesConfig = field(default_factory=SitesConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # Torch device: auto, cpu or cuda.
    device: str = "auto"
    # CPU cores (threads, BLAS, torch). Keep it modest on shared servers.
    n_jobs: int = 4
    seed: int = 0

    def validate(self) -> "FederatedConfig":
        g, d, t, e = self.generator, self.dp, self.transformer, self.evaluation
        if g.algorithm not in ALGORITHMS:
            raise ValueError(f"generator.algorithm must be one of {ALGORITHMS}; got '{g.algorithm}'.")
        if g.weighting not in WEIGHTINGS:
            raise ValueError(f"generator.weighting must be one of {WEIGHTINGS}; got '{g.weighting}'.")
        if g.rounds < 1 or g.local_epochs < 1:
            raise ValueError("generator.rounds and generator.local_epochs must be >= 1.")
        if not 0 < g.cap_fraction <= 1:
            raise ValueError("generator.cap_fraction must be in (0, 1].")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"device must be auto, cpu or cuda; got '{self.device}'.")
        if d.enabled and d.noise_multiplier is None and d.target_epsilon is None:
            raise ValueError("dp.enabled needs dp.noise_multiplier or dp.target_epsilon.")
        if d.accountant not in ("rdp", "prv"):
            raise ValueError("dp.accountant must be rdp or prv.")
        if d.max_grad_norm <= 0:
            raise ValueError("dp.max_grad_norm must be > 0.")
        if t.min_cell < 1 or t.bins < 2 or t.pseudo_rows < 100:
            raise ValueError("transformer.min_cell >= 1, bins >= 2 and pseudo_rows >= 100 are required.")
        if len(t.range_quantiles) != 2 or not 0 <= t.range_quantiles[0] < t.range_quantiles[1] <= 1:
            raise ValueError("transformer.range_quantiles must be [low, high] with 0 <= low < high <= 1.")
        if self.exchange.generator not in EXCHANGE_GENERATORS:
            raise ValueError(f"exchange.generator must be one of {EXCHANGE_GENERATORS}.")
        for frac in (self.exchange.holdout_frac, e.holdout_frac):
            if not 0 < frac < 1:
                raise ValueError("holdout fractions must be between 0 and 1.")
        for c in e.conditions:
            if c not in CONDITIONS and not _is_dp_condition(c):
                raise ValueError(f"Unknown condition '{c}'. Known: {CONDITIONS} and {DP_PREFIX}<epsilon>.")
        if e.repeats < 1:
            raise ValueError("evaluation.repeats must be >= 1.")
        return self

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FederatedConfig":
        data = dict(data or {})
        sections = {
            "sites": SitesConfig, "generator": GeneratorConfig, "transformer": TransformerConfig,
            "dp": DPConfig, "exchange": ExchangeConfig, "evaluation": EvaluationConfig,
        }
        kwargs = {name: kind(**(data.pop(name, {}) or {})) for name, kind in sections.items()}
        return cls(**data, **kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FederatedConfig":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def copy(self) -> "FederatedConfig":
        return FederatedConfig.from_dict(self.to_dict())

    def quick(self) -> "FederatedConfig":
        """A cheaper copy for smoke runs."""
        cfg = self.copy()
        cfg.generator.rounds = min(cfg.generator.rounds, 4)
        cfg.generator.local_epochs = min(cfg.generator.local_epochs, 2)
        cfg.generator.personalise_epochs = min(cfg.generator.personalise_epochs, 2)
        cfg.transformer.pseudo_rows = min(cfg.transformer.pseudo_rows, 1000)
        cfg.evaluation.repeats = 1
        cfg.evaluation.n_bootstrap = min(cfg.evaluation.n_bootstrap, 50)
        return cfg


def _is_dp_condition(name: str) -> bool:
    return dp_epsilon(name) is not None


def dp_epsilon(name: str) -> float | None:
    """The target epsilon of a ``fed_dp_eps<E>`` condition name, else None."""
    if not name.startswith(DP_PREFIX):
        return None
    try:
        eps = float(name[len(DP_PREFIX):].replace("p", "."))
    except ValueError:
        return None
    return eps if eps > 0 else None


def load_federated_config(path: str | Path | None) -> FederatedConfig:
    """Load a federated config from YAML, or return defaults when ``path`` is None."""
    if path is None:
        return FederatedConfig()
    return FederatedConfig.from_yaml(path)
