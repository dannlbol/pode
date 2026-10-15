"""Engine registry: build an :class:`Imputer` from an :class:`EngineSpec`."""

from __future__ import annotations

import dataclasses

from ..config import PodeConfig
from .base import Imputer
from .config import EngineSpec, ImputationConfig, LatentConfig, MaskedTVAEConfig

ENGINES = ("mean_mode", "knn", "mice", "missforest", "gain", "tvae-decoder", "tvae-masked", "hybrid")
# Engines whose models can learn from the rows being imputed (see fit_scope).
STACKED_ENGINES = ("knn", "mice", "missforest")
# Per-engine params handled by Imputer.configure rather than the constructor.
LEAKAGE_PARAMS = ("outcome_policy", "outcome_columns", "exclude_columns")


def _merge(base, overrides: dict):
    """dataclasses.replace with only the keys that belong to ``base``."""
    names = {f.name for f in dataclasses.fields(base)}
    return dataclasses.replace(base, **{k: v for k, v in overrides.items() if k in names})


def find_spec(name: str, icfg: ImputationConfig) -> EngineSpec:
    """The configured spec whose label or name matches, else a default spec."""
    for spec in icfg.engines:
        if spec.key == name:
            return spec
    for spec in icfg.engines:
        if spec.name == name:
            return spec
    return EngineSpec(name)


def build_imputer(
    spec: EngineSpec,
    pode_config: PodeConfig,
    icfg: ImputationConfig,
    seed: int | None = None,
    apply_leakage: bool = True,
) -> Imputer:
    """Instantiate the engine in ``spec`` with config + per-engine params.

    Leakage controls (outcome policy, excluded columns) come from the spec's
    params, falling back to the config's global settings. ``fit_scope`` for
    stacked engines falls back to ``icfg.fit_scope`` the same way.
    """
    from .baselines import KNNEngine, MeanModeImputer, MICEEngine, MissForestEngine

    p = dict(spec.params)
    leak = {k: p.pop(k) for k in LEAKAGE_PARAMS if k in p}
    if spec.name in STACKED_ENGINES:
        p.setdefault("fit_scope", icfg.fit_scope)
    name = spec.name
    if name == "mean_mode":
        imp = MeanModeImputer()
    elif name == "knn":
        imp = KNNEngine(**p)
    elif name == "mice":
        imp = MICEEngine(**{"n_jobs": icfg.n_jobs, **p})
    elif name == "missforest":
        imp = MissForestEngine(**p)
    elif name == "gain":
        from .gain import GAINEngine

        imp = GAINEngine(_merge(icfg.gain, p) if p else icfg.gain, fit_seed=seed, device=icfg.device)
    elif name == "tvae-decoder":
        from .decoder_imputer import TVAEDecoderImputer

        latent: LatentConfig = _merge(icfg.latent, p)
        synth = _merge(pode_config.synthesizer, p)
        imp = TVAEDecoderImputer(synth, latent, epochs=p.get("epochs", icfg.epochs), fit_seed=seed,
                                 device=icfg.device)
    elif name == "tvae-masked":
        from .masked_tvae import MaskedTVAEImputer

        latent = _merge(icfg.latent, p)
        masked: MaskedTVAEConfig = _merge(icfg.masked_tvae, p)
        synth = _merge(pode_config.synthesizer, p)
        epochs = p.get("epochs", masked.epochs or icfg.epochs)
        imp = MaskedTVAEImputer(synth, masked, latent, icfg.block_groups, epochs=epochs, fit_seed=seed,
                                device=icfg.device, atomic_blocks=icfg.atomic_blocks)
    elif name == "hybrid":
        from .hybrid import HybridImputer

        # Sub-engines see whatever the hybrid lets through, so they don't
        # apply leakage controls themselves.
        gap = build_imputer(
            find_spec(p.get("gap_engine", "mice"), icfg), pode_config, icfg, seed, apply_leakage=False
        )
        block = build_imputer(
            find_spec(p.get("block_engine", "tvae-masked"), icfg), pode_config, icfg, seed, apply_leakage=False
        )
        imp = HybridImputer(gap, block, icfg.block_groups, p.get("block_min_frac", 1.0))
    else:
        raise ValueError(f"Unknown imputation engine '{name}'. Choose from: {', '.join(ENGINES)}.")
    imp.n_jobs = icfg.n_jobs
    if apply_leakage:
        imp.configure(
            outcome_policy=leak.get("outcome_policy", icfg.outcome_policy),
            outcome_columns=leak.get("outcome_columns", icfg.outcome_columns),
            exclude_columns=leak.get("exclude_columns", icfg.exclude_columns),
        )
    return imp
