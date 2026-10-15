"""Configuration for the privacy assessment of synthetic data.

Kept in its own YAML (``--privacy-config``) rather than as a section of
:class:`~pode.config.PodeConfig`, so existing PODE configs and the
train/generate/tune workflow are untouched by this feature.

The most important settings describe the *threat model*: which quasi-identifiers
a plausible intruder could know (``quasi_identifiers``, one entry per attacker
tier), which attributes would be harmful to disclose (``sensitive``), and which
rare strata deserve separate scrutiny (``strata``). Privacy numbers are only
meaningful relative to that model, so the report always restates it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GateConfig:
    """Release-gate thresholds. See ``docs/synthesis/PRIVACY.md`` for the reasoning."""

    # Whole-record copies of training patients tolerated before FAIL.
    max_exact_copies: int = 0
    # 5th percentile of synthetic DCR / 5th percentile of the real reference
    # DCR. Below this, synthetic rows sit systematically closer to training
    # patients than genuinely new patients do.
    min_dcr_p5_ratio: float = 0.8
    # Observed / expected fraction of synthetic rows inside the reference's
    # near-copy band (e.g. 9% where new patients give 5% -> 1.8).
    max_near_copy_excess: float = 2.0
    # Distance-attack membership AUC. FAIL above; REVIEW when the 95% CI
    # upper bound crosses it (small holdouts cannot rule leakage out).
    max_membership_auc: float = 0.60
    # Attribute-inference accuracy on training patients minus that on unseen
    # patients (the disclosure attributable to membership).
    max_attribute_member_gap: float = 0.05
    # Linkage rate of training patients minus that of unseen patients.
    max_linkage_excess: float = 0.02
    # Synthetic rows that reproduce a rare quasi-identifier combination AND
    # sit within near-copy distance of a training patient.
    max_rare_combo_leaks: int = 0
    # Excess-type checks (near copies, linkage, rare combinations, outliers,
    # missingness) only FAIL when the excess is also statistically clear at
    # this one-sided level; an excess within chance is REVIEW. Kept strict
    # because ~10 tests run on every release candidate.
    alpha: float = 0.01


@dataclass
class AuditConfig:
    """``pode synthesis privacy audit``: split -> fit TVAE -> generate -> assess."""

    # Fraction of real patients held out (never seen by the TVAE).
    holdout_frac: float = 0.3
    # Columns (globs) to stratify the split on. Empty -> the primary survival
    # event column, when there is one.
    stratify: list[str] = field(default_factory=list)
    # Synthetic rows to generate per repeat. None -> as many as training rows.
    num_samples: int | None = None
    # Independent split/fit/assess repeats; the verdict is the worst one.
    repeats: int = 1
    # TVAE epoch override (None -> the PODE config's synthesizer.epochs).
    epochs: int | None = None
    # Torch device for the TVAE fit: auto, cpu or cuda.
    device: str = "auto"


@dataclass
class PrivacyConfig:
    """Top-level privacy-assessment configuration."""

    # Columns to assess (globs). Empty -> every non-identifier column shared by
    # the real and synthetic tables.
    columns: list[str] = field(default_factory=list)
    exclude_columns: list[str] = field(default_factory=list)
    # Optional patient ID column, used only to label review rows; never a feature.
    id_column: str | None = None
    # Attacker tiers: name -> quasi-identifier columns (globs) that intruder
    # plausibly knows. Empty -> one 'categorical' tier of all non-sensitive
    # categorical columns.
    quasi_identifiers: dict[str, list[str]] = field(default_factory=dict)
    # Attributes whose disclosure would harm a child (globs). Empty -> the
    # schema's survival event columns.
    sensitive: list[str] = field(default_factory=list)
    # Rarity strata (globs) for the stratified checks, e.g. subtype, age band.
    strata: list[str] = field(default_factory=list)
    # Gower weight per column (globs allowed); unlisted columns weigh 1.
    weights: dict[str, float] = field(default_factory=dict)
    # Quantile bins for numeric quasi-identifiers/strata when they form keys.
    numeric_bins: int = 5
    # A combination/stratum seen in <= rare_k training patients is "rare".
    rare_k: int = 5
    # Synthetic rows assessed (random subsample above this).
    max_synthetic: int = 2000
    # Reference-distance quantiles that define the near-copy bands.
    near_copy_quantiles: list[float] = field(default_factory=lambda: [0.01, 0.05])
    # Decimal places numeric values are rounded to for exact-copy matching.
    exact_decimals: int = 6
    # Bootstrap resamples for confidence intervals.
    n_bootstrap: int = 200
    # Smallest group (per class) that gets an AUC or accuracy.
    min_group: int = 10
    # Numeric sensitive targets: a guess within this fraction of the training
    # range counts as correct.
    attribute_tolerance: float = 0.05
    # Rows listed in each case-review table.
    review_top_n: int = 25
    # CPU cores (threads, BLAS, torch). -1 = all cores; keep it modest on
    # shared servers.
    n_jobs: int = 4
    seed: int = 0
    gates: GateConfig = field(default_factory=GateConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

    def validate(self) -> "PrivacyConfig":
        if not 0 < self.audit.holdout_frac < 1:
            raise ValueError("audit.holdout_frac must be between 0 and 1.")
        if self.audit.repeats < 1:
            raise ValueError("audit.repeats must be >= 1.")
        if self.audit.device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"audit.device must be auto, cpu or cuda; got '{self.audit.device}'.")
        if not self.near_copy_quantiles or not all(0 < q < 1 for q in self.near_copy_quantiles):
            raise ValueError("near_copy_quantiles must be non-empty values in (0, 1).")
        if self.numeric_bins < 2:
            raise ValueError("numeric_bins must be >= 2.")
        if self.rare_k < 1 or self.max_synthetic < 1 or self.min_group < 2:
            raise ValueError("rare_k and max_synthetic must be >= 1, min_group >= 2.")
        if not isinstance(self.quasi_identifiers, dict):
            raise ValueError("quasi_identifiers must map a tier name to a list of column globs.")
        return self

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PrivacyConfig":
        data = dict(data or {})
        gates = GateConfig(**(data.pop("gates", {}) or {}))
        audit = AuditConfig(**(data.pop("audit", {}) or {}))
        return cls(**data, gates=gates, audit=audit)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PrivacyConfig":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def quick(self) -> "PrivacyConfig":
        """A cheaper copy for smoke runs."""
        cfg = PrivacyConfig.from_dict(self.to_dict())
        cfg.max_synthetic = min(cfg.max_synthetic, 500)
        cfg.n_bootstrap = min(cfg.n_bootstrap, 50)
        cfg.audit.repeats = 1
        cfg.audit.epochs = min(cfg.audit.epochs or 50, 50)
        return cfg


def load_privacy_config(path: str | Path | None) -> PrivacyConfig:
    """Load a privacy config from YAML, or return defaults when ``path`` is None."""
    if path is None:
        return PrivacyConfig()
    return PrivacyConfig.from_yaml(path)
