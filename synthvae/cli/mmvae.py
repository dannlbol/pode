#!/usr/bin/env python3
"""Train, sample, and evaluate the joint multimodal VAE (deliverable 2).

Toy cohort::

    python scripts/run_mmvae.py run --toy --epochs 30

Bring-your-own CSVs (same contract as run_hybrid.py)::

    python scripts/run_mmvae.py run \
        --clinical clinical.csv --expression expression.csv --survival survival.csv \
        --config config/mmvae.yaml

Overrides (fusion strategy, likelihood, weighting) layer on top of the config::

    python scripts/run_mmvae.py run --toy --fusion mopoe --likelihood nb

Controlled workflow — sweep, train the winner, then generate from the saved model::

    python scripts/run_mmvae.py tune --sweep config/mmvae_sweep.yaml --out-dir runs/tune \
        --clinical clinical.csv --expression expression.csv --survival survival.csv
    python scripts/run_mmvae.py train --config runs/tune/selected_config.json \
        --clinical clinical.csv --expression expression.csv --survival survival.csv \
        --out-dir runs/final
    python scripts/run_mmvae.py generate --model runs/final/model.pt \
        --n 1000 --out-dir runs/synth
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import hybrid as run_hybrid  # reused: load_sweep, expand_grid, aggregate_seed_results, jsonable
from ._io import read_table

from synthvae.data.toy import make_cohort, ToyConfig
from synthvae.evaluation.report import evaluate, format_report
from synthvae.mmvae import MMVAEConfig, JointVAESynthesizer
from synthvae.mmvae.metrics import (
    cross_omic_fidelity,
    modality_geometry,
    quality_guard_failures,
)


def load_config(path: str | None) -> dict:
    """Flatten the nested config/mmvae.yaml into MMVAEConfig kwargs."""
    kwargs: dict = {}
    if not path:
        return kwargs
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    def take(section, key, dest=None, tuple_=False):
        sec = raw.get(section, {}) or {}
        if key in sec and sec[key] is not None:
            v = sec[key]
            kwargs[dest or key] = tuple(v) if tuple_ else v

    take("dims", "n_time_bins")
    take("dims", "latent_dim")
    take("dims", "latent_structure")
    take("dims", "private_latent_dims")
    for k in ("encoder_hidden", "tabular_head_hidden",
              "expression_head_hidden", "methylation_head_hidden",
              "survival_head_hidden"):
        take("architecture", k, tuple_=True)
    take("architecture", "dropout")
    take("fusion", "strategy", "fusion")
    take("fusion", "gating")
    take("fusion", "include_prior_expert")
    take("likelihoods", "expression", "expression_likelihood")
    take("likelihoods", "methylation", "methylation_likelihood")
    for k in ("weighting_scheme", "fixed_weights", "beta_max", "free_bits",
              "kl_warmup_epochs", "recon_reduction", "gradnorm_alpha",
              "private_kl_weight"):
        take("objective", k)
    for k in ("lr", "weight_decay", "batch_size", "max_epochs", "patience",
              "val_fraction", "seed", "device", "verbose"):
        take("optimisation", k)
    take("generation", "latent_source")
    take("generation", "gmm_components")
    take("generation", "posterior_samples_per_patient")
    take("generation", "observation_sampler")
    take("generation", "residual_rank")
    take("generation", "observation_noise_scale")
    if raw.get("modalities") is not None:
        kwargs["modalities"] = raw["modalities"]

    # keep only valid MMVAEConfig fields
    valid = {f.name for f in fields(MMVAEConfig)}
    return {k: v for k, v in kwargs.items() if k in valid}


def _parse_omics(values):
    out = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--omic must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or not path or name in {"expression", "methylation"} or name in out:
            raise ValueError("--omic names must be unique and cannot be expression/methylation")
        out[name] = path
    return out


def load_data(args):
    """Return aligned clinical/expression/survival, legacy methylation, and omics."""
    methylation_path = getattr(args, "methylation", None)
    omic_paths = _parse_omics(getattr(args, "omic", None))
    if args.clinical or args.expression or args.survival or methylation_path or omic_paths:
        if not (args.expression and args.survival):
            raise ValueError("BYO runs require --expression and --survival.")
        expression = read_table(args.expression)
        survival = read_table(args.survival)
        clinical = read_table(args.clinical) if args.clinical else \
            pd.DataFrame(index=np.arange(len(expression)))
        if not (len(clinical) == len(expression) == len(survival)):
            raise ValueError("clinical/expression/survival must share row count.")
        methylation = None
        if methylation_path:
            methylation = read_table(methylation_path)
            if len(methylation) != len(expression):
                raise ValueError("methylation must share row count with expression.")
            methylation = methylation.reset_index(drop=True)
        omics = {}
        for name, path in omic_paths.items():
            frame = read_table(path)
            if len(frame) != len(expression):
                raise ValueError(f"{name} must share row count with expression.")
            omics[name] = frame.reset_index(drop=True)
        return (clinical.reset_index(drop=True), expression.reset_index(drop=True),
                survival.reset_index(drop=True), methylation, omics)
    clinical, expression, survival, _ = make_cohort(
        ToyConfig(n=args.n, n_genes=args.n_genes, seed=args.seed)
    )
    return clinical, expression, survival, None, {}


# Dotted config/mmvae.yaml sweep keys -> MMVAEConfig field names. Reused by
# scripts/benchmark.py to expand the `sweeps:` block.
SWEEP_KEY_MAP = {
    "dims.n_time_bins": "n_time_bins",
    "dims.latent_dim": "latent_dim",
    "dims.latent_structure": "latent_structure",
    "dims.private_latent_dims": "private_latent_dims",
    "architecture.dropout": "dropout",
    "fusion.strategy": "fusion",
    "fusion.gating": "gating",
    "fusion.include_prior_expert": "include_prior_expert",
    "likelihoods.expression": "expression_likelihood",
    "likelihoods.methylation": "methylation_likelihood",
    "architecture.methylation_head_hidden": "methylation_head_hidden",
    "objective.weighting_scheme": "weighting_scheme",
    "objective.beta_max": "beta_max",
    "objective.free_bits": "free_bits",
    "objective.kl_warmup_epochs": "kl_warmup_epochs",
    "objective.recon_reduction": "recon_reduction",
    "objective.gradnorm_alpha": "gradnorm_alpha",
    "objective.private_kl_weight": "private_kl_weight",
    "optimisation.lr": "lr",
    "optimisation.batch_size": "batch_size",
    "optimisation.max_epochs": "max_epochs",
    "optimisation.seed": "seed",
    "generation.latent_source": "latent_source",
    "generation.gmm_components": "gmm_components",
    "generation.posterior_samples_per_patient": "posterior_samples_per_patient",
    "generation.observation_sampler": "observation_sampler",
    "generation.residual_rank": "residual_rank",
    "generation.observation_noise_scale": "observation_noise_scale",
}


def _eval_dict(clinical, latents, survival):
    return {
        "clinical": clinical.reset_index(drop=True),
        "latents": np.asarray(latents, dtype=float),
        "time": survival["time"].to_numpy(dtype=float),
        "event": survival["event"].to_numpy(dtype=float),
    }


def split_frames(clinical, expression, survival, seed, test_fraction=0.25,
                 methylation=None, omics=None):
    """Seeded train/test split of the aligned frames.

    Identical logic to ``run_hybrid.split_data`` so the benchmark can hand both
    architectures the very same split. ``methylation`` is optional; the returned
    tuple always carries its train/test slots (``None`` when absent) so downstream
    unpacking is uniform.
    """
    rng = np.random.default_rng(seed)
    n = len(clinical)
    perm = rng.permutation(n)
    n_test = max(1, int(test_fraction * n))
    test_ix, train_ix = perm[:n_test], perm[n_test:]

    def sub(df, ix):
        return df.iloc[ix].reset_index(drop=True) if df is not None else None

    omics = omics or {}
    sub_omics = lambda ix: {name: sub(frame, ix) for name, frame in omics.items()}
    return (sub(clinical, train_ix), sub(expression, train_ix), sub(survival, train_ix),
            sub(methylation, train_ix),
            sub_omics(train_ix),
            sub(clinical, test_ix), sub(expression, test_ix), sub(survival, test_ix),
            sub(methylation, test_ix), sub_omics(test_ix))


def stratified_folds(clinical, expression, survival, seed, n_splits=3,
                     methylation=None, omics=None):
    """Fixed folds stratified by subgroup and event where sample counts allow."""
    from sklearn.model_selection import KFold, StratifiedKFold

    n_splits = max(2, min(int(n_splits), len(expression)))
    event = survival["event"].astype(str).to_numpy()
    subgroup_col = next((c for c in clinical.columns
                         if c.lower() in {"mnp_subgroup", "subgroup"}), None)
    labels = (clinical[subgroup_col].astype(str).to_numpy() + "|" + event
              if subgroup_col else event)
    counts = pd.Series(labels).value_counts()
    if counts.min() < n_splits and subgroup_col:
        labels = clinical[subgroup_col].astype(str).to_numpy()
        counts = pd.Series(labels).value_counts()
    splitter = (StratifiedKFold(n_splits, shuffle=True, random_state=seed)
                if counts.min() >= n_splits
                else KFold(n_splits, shuffle=True, random_state=seed))
    omics = omics or {}

    def sub(frame, ix):
        return frame.iloc[ix].reset_index(drop=True) if frame is not None else None

    folds = []
    for train_ix, test_ix in splitter.split(np.arange(len(expression)), labels):
        folds.append((
            sub(clinical, train_ix), sub(expression, train_ix), sub(survival, train_ix),
            sub(methylation, train_ix), {n: sub(f, train_ix) for n, f in omics.items()},
            sub(clinical, test_ix), sub(expression, test_ix), sub(survival, test_ix),
            sub(methylation, test_ix), {n: sub(f, test_ix) for n, f in omics.items()},
        ))
    return folds


def _subgroup_values(frame):
    column = next((c for c in frame.columns
                   if c.lower() in {"mnp_subgroup", "subgroup"}), None)
    return frame[column].astype(str).to_numpy() if column else None


def run_joint(cfg, splits):
    """Fit the joint model on a split and evaluate. Single source of truth for the
    joint arm, used by both ``cmd_run`` and ``scripts/benchmark.py``.

    ``splits`` is ``(clin_tr, expr_tr, surv_tr, meth_tr,
    clin_te, expr_te, surv_te, meth_te)`` (methylation slots may be ``None``).
    Returns ``(metrics, synth, result)``.
    """
    (clin_tr, expr_tr, surv_tr, meth_tr, omics_tr,
     clin_te, expr_te, surv_te, meth_te, omics_te) = splits
    synth = JointVAESynthesizer(cfg=cfg).fit(clin_tr, expr_tr, surv_tr,
                                             methylation=meth_tr, omics=omics_tr)
    metrics, result = evaluate_joint(synth, splits)
    return metrics, synth, result


def evaluate_joint(synth, splits):
    """Evaluate an already fitted model on one held-out aligned fold."""
    (clin_tr, expr_tr, surv_tr, meth_tr, omics_tr,
     clin_te, expr_te, surv_te, meth_te, omics_te) = splits
    train_real = _eval_dict(
        clin_tr, synth.encode_real(clin_tr, expr_tr, meth_tr, omics_tr), surv_tr)
    real = _eval_dict(
        clin_te, synth.encode_real(clin_te, expr_te, meth_te, omics_te), surv_te)
    result = synth.sample(len(clin_te))
    synth_d = _eval_dict(result.clinical, result.latents, result.survival)
    metrics = evaluate(real, synth_d, train_real=train_real)
    real_modalities = {"expression": expr_te}
    if meth_te is not None:
        real_modalities["methylation"] = meth_te
    real_modalities.update(omics_te)
    synth_modalities = {"expression": result.expression}
    if result.methylation is not None:
        synth_modalities["methylation"] = result.methylation
    synth_modalities.update(result.modalities)
    specs = synth.cfg.resolved_modalities(real_modalities)
    real_groups = _subgroup_values(clin_te)
    synth_groups = _subgroup_values(result.clinical)
    for name, real_frame in real_modalities.items():
        values = modality_geometry(
            real_frame, synth_modalities[name], specs[name].get("kind", "continuous"),
            real_groups=real_groups, synthetic_groups=synth_groups,
        )
        metrics.update({f"{name}_{key}": value for key, value in values.items()})
    names = list(real_modalities)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            values = cross_omic_fidelity(
                real_modalities[left], synth_modalities[left],
                real_modalities[right], synth_modalities[right])
            prefix = f"cross_omic_{left}__{right}"
            metrics.update({f"{prefix}_{key}": value for key, value in values.items()})
            metrics[f"{prefix}_gap"] = values["covariance_gap"]
    return metrics, result


_GENERATION_SWEEP_KEYS = {
    "generation.latent_source", "generation.gmm_components",
    "generation.posterior_samples_per_patient", "generation.observation_sampler",
    "generation.residual_rank", "generation.observation_noise_scale",
    "generation.preset",
}


def _recalibrate_generation(synth, cfg):
    """Apply generation-only settings to a fitted model without neural retraining."""
    for field in ("latent_source", "gmm_components", "posterior_samples_per_patient",
                  "observation_sampler", "residual_rank", "observation_noise_scale"):
        value = getattr(cfg, field)
        if field == "observation_sampler" and value == "auto":
            value = ("residual_factor" if len(synth.model.modality_names) > 1
                     else "independent")
        setattr(synth.cfg, field, value)
        setattr(synth.model.cfg, field, value)
    dataset = synth._fit_dataset
    z_train = synth._encode_dataset(
        dataset, posterior_samples=cfg.posterior_samples_per_patient)
    synth.model.fit_latent_density(z_train, n_components=cfg.gmm_components)
    synth._fit_residual_sampler(dataset)


def cfg_from_args(args):
    cfg_kwargs = load_config(args.config)
    if args.fusion:      cfg_kwargs["fusion"] = args.fusion
    if args.likelihood:  cfg_kwargs["expression_likelihood"] = args.likelihood
    if args.weighting:   cfg_kwargs["weighting_scheme"] = args.weighting
    if args.latent_dim:  cfg_kwargs["latent_dim"] = args.latent_dim
    if args.epochs:      cfg_kwargs["max_epochs"] = args.epochs
    if args.device:      cfg_kwargs["device"] = args.device
    cfg_kwargs["seed"] = args.seed
    cfg_kwargs.setdefault("verbose", args.verbose)
    return MMVAEConfig(**cfg_kwargs)


# --------------------------------------------------------------------------- #
# tuning: grid expansion, scoring, config bridge
# --------------------------------------------------------------------------- #
def composite_score(metrics: dict, weights: dict | None = None) -> float:
    """Rank joint configs (lower is better). Single source of truth shared with
    scripts/benchmark.py so tune and the benchmark agree.

    Default = ``abs(concordance_gap) + km_iad_late + (1 - hr_logcorr)`` with NaN
    terms mapped to a penalty. ``weights`` may override the per-term weights over
    ``concordance_gap`` / ``km_iad_late`` / ``hr_logcorr_penalty`` /
    ``latent_cov_relgap`` (``hr_logcorr_penalty = 1 - hr_logcorr``).
    """
    def g(key, default):
        v = metrics.get(key, np.nan)
        return float(v) if np.isfinite(v) else default

    hr = metrics.get("hr_logcorr", np.nan)
    terms = {
        "concordance_gap": abs(g("concordance_gap", 1.0)),
        "km_iad_late": g("km_iad_late", 1.0),
        "hr_logcorr_penalty": (1.0 - hr) if np.isfinite(hr) else 2.0,
        "latent_cov_relgap": g("latent_cov_relgap", 1.0),
    }
    weights = weights or {
        "concordance_gap": 1.0,
        "km_iad_late": 1.0,
        "hr_logcorr_penalty": 1.0,
    }
    return float(sum(w * terms.get(k, 0.0) for k, w in weights.items()))


def sweep_to_cfg_kwargs(candidate: dict) -> dict:
    """Translate one dotted-key sweep candidate into MMVAEConfig kwargs."""
    valid = {f.name for f in fields(MMVAEConfig)}
    kwargs = {}
    for dotted, value in candidate.items():
        if dotted == "generation.preset":
            if not isinstance(value, dict):
                raise ValueError("generation.preset sweep values must be mappings")
            for key, preset_value in value.items():
                field = SWEEP_KEY_MAP.get(f"generation.{key}", key)
                if field in valid:
                    kwargs[field] = preset_value
            continue
        field = SWEEP_KEY_MAP.get(dotted, dotted.split(".")[-1])
        if field in valid:
            kwargs[field] = value
    return kwargs


def load_selected_config(path: str | None) -> dict:
    """Config bridge for train/generate: accept either a tune ``selected_config.json``
    (``best_config`` = MMVAEConfig field names) or the nested ``config/mmvae.yaml``.
    """
    if path and str(path).endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        best = payload.get("best_config", payload)
        valid = {f.name for f in fields(MMVAEConfig)}
        return {k: v for k, v in best.items() if k in valid}
    return load_config(path)


_GENERATION_CFG_FIELDS = {
    "latent_source", "gmm_components", "posterior_samples_per_patient",
    "observation_sampler", "residual_rank", "observation_noise_scale",
}


def _stable_candidate_id(config):
    payload = json.dumps(run_hybrid.jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _rank_staged_rows(rows, modality_names, group_key="candidate_id",
                      enforce_guards=True):
    """Balanced survival/molecular ranks with four equal modality sub-blocks."""
    numeric = [c for c in rows if pd.api.types.is_numeric_dtype(rows[c])
               and c not in {"fold", "replicate", "seed"}]
    grouped = rows.groupby(group_key, sort=False)
    result = grouped[numeric].mean().reset_index()
    result["config_json"] = grouped["config_json"].first().values
    for metadata in ("candidate_id", "selection_id", "parent_id"):
        if metadata in rows and metadata not in result:
            result[metadata] = grouped[metadata].first().values

    rank_parts = []
    for column, higher_better in (
            ("concordance_gap", False), ("km_iad_late", False),
            ("hr_logcorr", True)):
        if column in result:
            rank_parts.append(result[column].rank(
                pct=True, method="average", ascending=not higher_better))
    result["survival_block_rank"] = pd.concat(rank_parts, axis=1).mean(axis=1)

    modality_blocks = []
    for name in modality_names:
        subblocks = []
        definitions = [
            [(f"{name}_covariance_gap", False),
             (f"{name}_pca_eigenspectrum_gap", False)],
            [(f"{name}_source_overlap_mixing_worst", True),
             (f"{name}_source_overlap_mixing_min_worst", True),
             (f"{name}_source_overlap_auc_worst", False),
             (f"{name}_source_overlap_auc_max_worst", False),
             (f"{name}_source_overlap_boundary_penalty_worst", False),
             (f"{name}_source_overlap_precision_worst", True),
             (f"{name}_source_overlap_recall_worst", True),
             (f"{name}_source_overlap_group_precision_worst", True),
             (f"{name}_source_overlap_group_recall_worst", True)],
            [(f"{name}_variance_ratio_median", True),
             (f"{name}_coverage_ratio", True)],
            [(f"{name}_subgroup_effect_corr", True),
             (f"{name}_subgroup_effect_sign_agreement", True),
             (f"{name}_subgroup_top_feature_overlap", True),
             (f"{name}_subgroup_centroid_distance_corr", True)],
        ]
        for definition in definitions:
            ranks = []
            for column, higher_better in definition:
                if column not in result:
                    continue
                values = result[column]
                if column.endswith(("variance_ratio_median", "coverage_ratio")):
                    values = np.abs(np.log(values.clip(lower=1e-8)))
                    higher_better = False
                ranks.append(values.rank(
                    pct=True, method="average", ascending=not higher_better))
            if ranks:
                subblocks.append(pd.concat(ranks, axis=1).mean(axis=1))
        block = pd.concat(subblocks, axis=1).mean(axis=1)
        result[f"{name}_block_rank"] = block
        modality_blocks.append(block)

    cross_columns = [c for c in result if c.startswith("cross_omic_")
                     and c.endswith("_geometry_penalty")]
    if cross_columns:
        cross = result[cross_columns].rank(pct=True, method="average").mean(axis=1)
        result["cross_omic_block_rank"] = cross
        modality_blocks.append(cross)
    result["molecular_block_rank"] = pd.concat(modality_blocks, axis=1).mean(axis=1)
    result["rank_score"] = (
        0.5 * result["survival_block_rank"] + 0.5 * result["molecular_block_rank"])
    prefixes = [name for name in modality_names
                if f"{name}_variance_ratio_median" in result]
    result["quality_guard_failures"] = result.apply(
        lambda row: ";".join(quality_guard_failures(row.to_dict(), prefixes)), axis=1)
    result["quality_rejected"] = result["quality_guard_failures"].str.len() > 0
    if enforce_guards:
        result.loc[result["quality_rejected"], "rank_score"] = np.inf
    return result.sort_values("rank_score").reset_index(drop=True)


def _evaluate_staged_candidates(candidates, stage_path, folds_for_seed,
                                modality_names, synthetic_replicates=1,
                                resume=False, enforce_guards=True):
    """Train each neural architecture once per fold and calibrate its samplers."""
    existing = pd.read_csv(stage_path) if resume and stage_path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    complete = set()
    if not existing.empty:
        complete = set(zip(existing["candidate_id"].astype(str),
                           existing["fold"].astype(int),
                           existing["replicate"].astype(int)))

    training_groups = {}
    for item in candidates:
        config = dict(item["config"])
        training = {k: v for k, v in config.items() if k not in _GENERATION_CFG_FIELDS}
        key = json.dumps(run_hybrid.jsonable(training), sort_keys=True)
        training_groups.setdefault(key, []).append(item)

    for group_number, group in enumerate(training_groups.values(), 1):
        first = group[0]
        seed = int(first["config"].get("seed", 0))
        folds = folds_for_seed(seed)
        print(f"  neural fit {group_number}/{len(training_groups)}; "
              f"{len(group)} sampler setting(s), seed={seed}")
        for fold, splits in enumerate(folds):
            needed = [item for item in group if any(
                (item["candidate_id"], fold, replicate) not in complete
                for replicate in range(synthetic_replicates))]
            if not needed:
                continue
            cfg = MMVAEConfig(**first["config"])
            clin_tr, expr_tr, surv_tr, meth_tr, omics_tr, *_ = splits
            synth = JointVAESynthesizer(cfg=cfg).fit(
                clin_tr, expr_tr, surv_tr, methylation=meth_tr, omics=omics_tr)
            for item in needed:
                candidate_cfg = MMVAEConfig(**item["config"])
                _recalibrate_generation(synth, candidate_cfg)
                for replicate in range(synthetic_replicates):
                    marker = (item["candidate_id"], fold, replicate)
                    if marker in complete:
                        continue
                    draw_seed = seed + replicate * 100003
                    synth.cfg.seed = draw_seed
                    synth.model.cfg.seed = draw_seed
                    metrics, _ = evaluate_joint(synth, splits)
                    rows.append({
                        "candidate_id": item["candidate_id"],
                        "selection_id": item.get("selection_id", item["candidate_id"]),
                        "parent_id": item.get("parent_id", ""),
                        "fold": fold, "replicate": replicate, "seed": seed,
                        "config_json": json.dumps(
                            run_hybrid.jsonable(item["config"]), sort_keys=True),
                        **metrics,
                    })
                    complete.add(marker)
                    pd.DataFrame(rows).to_csv(stage_path, index=False)
                synth.cfg.seed = seed
                synth.model.cfg.seed = seed
    frame = pd.DataFrame(rows)
    ranked = _rank_staged_rows(frame, modality_names, enforce_guards=enforce_guards)
    ranked.to_csv(stage_path.with_name(stage_path.stem + "_ranked.csv"), index=False)
    return ranked


def _config_from_ranked(row):
    return json.loads(row["config_json"])


def _residual_noise_presets(stages, modality_names):
    presets = [{"default": float(value)}
               for value in stages.get("common_noise_scales", [0.25, 0.5, 0.75, 1.0])]
    asymmetric = stages.get("asymmetric_noise_scales")
    if asymmetric is None and "methylation" in modality_names:
        asymmetric = [
            {"expression": 0.5, "methylation": 0.75},
            {"expression": 0.75, "methylation": 1.0},
        ]
    for values in asymmetric or []:
        resolved = {str(name): float(value) for name, value in dict(values).items()}
        unknown = set(resolved) - set(modality_names) - {"default"}
        if unknown:
            raise ValueError(
                "asymmetric_noise_scales contains inactive modalities: "
                + ", ".join(sorted(unknown)))
        presets.append(resolved)
    # Preserve order while avoiding duplicate mappings.
    unique = []
    seen = set()
    for preset in presets:
        key = json.dumps(preset, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(preset)
    return unique


def _run_calibration_only(args, sweep, out, selected_path, base,
                          folds_for_seed, modality_names):
    """Recalibrate generation around one fixed neural architecture.

    The neural fit is grouped across all residual settings, so each fold is
    trained once during screening.  This provides a cheap, reversible follow-up
    when a broad architecture sweep has already selected shared/private latents.
    """
    stages = sweep["stages"]
    screening_epochs = int(stages.get("screening_max_epochs", 150))
    noise_presets = _residual_noise_presets(stages, modality_names)
    candidates = []
    for rank in stages.get("residual_ranks", [0, 8, 16]):
        for scales in noise_presets:
            config = dict(
                base, residual_rank=int(rank), observation_noise_scale=scales,
                observation_sampler="residual_factor", max_epochs=screening_epochs,
                seed=0,
            )
            candidates.append({
                "config": config,
                "candidate_id": _stable_candidate_id(config),
                "parent_id": "calibration_base",
            })
    print(f"Calibration 1/2: {len(candidates)} residual settings")
    ranked = _evaluate_staged_candidates(
        candidates, out / "stage1_calibration.csv", folds_for_seed,
        modality_names, synthetic_replicates=int(
            stages.get("screening_replicates", 3)),
        resume=args.resume,
        # Screening models are intentionally short and held-out subgroup
        # supports can be small. Boundary metrics rank them here; hard guards
        # apply only after the retained settings receive robust finalist fits.
        enforce_guards=False)
    retained = ranked.head(int(stages.get("retain_finalists", 3)))
    if retained.empty or not np.isfinite(retained.iloc[0]["rank_score"]):
        raise RuntimeError("Every low-noise calibration candidate failed the quality guards")

    finalist_epochs = int(stages.get("finalist_max_epochs", 1500))
    finalists = []
    for _, row in retained.iterrows():
        config0 = _config_from_ranked(row)
        selection_id = row["candidate_id"]
        for seed in stages.get("finalist_seeds", [0, 1, 2]):
            config = dict(config0, seed=int(seed), max_epochs=finalist_epochs)
            finalists.append({
                "config": config,
                "candidate_id": _stable_candidate_id(config),
                "selection_id": selection_id,
                "parent_id": "calibration_base",
            })
    print(f"Calibration 2/2: {len(finalists)} seeded finalist configurations")
    _evaluate_staged_candidates(
        finalists, out / "stage2_finalists.csv", folds_for_seed,
        modality_names, synthetic_replicates=int(
            stages.get("finalist_replicates", 3)),
        resume=args.resume,
        enforce_guards=bool(sweep.get("enforce_quality_guards", True)))
    raw = pd.read_csv(out / "stage2_finalists.csv")
    ranked_final = _rank_staged_rows(
        raw, modality_names, group_key="selection_id",
        enforce_guards=bool(sweep.get("enforce_quality_guards", True)))
    ranked_final.to_csv(out / "stage2_finalists_ranked.csv", index=False)
    if ranked_final.empty or not np.isfinite(ranked_final.iloc[0]["rank_score"]):
        raise RuntimeError("Every low-noise calibration finalist failed the quality guards")
    winner_id = ranked_final.iloc[0]["selection_id"]
    winner_rows = raw[raw["selection_id"] == winner_id]
    seed0 = winner_rows[winner_rows["seed"] == 0]
    winner_config = json.loads(
        (seed0 if len(seed0) else winner_rows).iloc[0]["config_json"])
    winner_config["seed"] = 0
    winner_config["max_epochs"] = finalist_epochs
    with open(selected_path, "w", encoding="utf-8") as fh:
        json.dump(run_hybrid.jsonable({
            "best_config": winner_config,
            "best_metrics": ranked_final.iloc[0].to_dict(),
            "tuning_objective": "low_noise_worst_subgroup_calibration",
            "calibration_base": getattr(args, "config", None),
            "aggregation": {
                "over": ["fold", "seed", "replicate"],
                "seed_values": stages.get("finalist_seeds", [0, 1, 2]),
            },
        }), fh, indent=2, sort_keys=True)
    print(f"Calibrated winner written to {selected_path}")
    return 0


def _cmd_tune_staged(args, sweep):
    """Four-stage shared/private search. Existing flat sweeps remain untouched."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected_path = out / "selected_config.json"
    if selected_path.exists() and not args.resume:
        selected_path.unlink()
    base = asdict(MMVAEConfig())
    # Calibration-only runs intentionally accept the selected_config.json from
    # the preceding broad sweep.  YAML configs continue to resolve identically.
    base.update(load_selected_config(args.config))
    base.update(sweep_to_cfg_kwargs(sweep.get("fixed", {})))
    if args.device:
        base["device"] = args.device
    base["verbose"] = bool(args.verbose)
    clinical, expression, survival, methylation, omics = load_data(args)
    modality_names = ["expression"]
    if methylation is not None:
        modality_names.append("methylation")
    modality_names.extend(omics)
    manifest = json.loads(json.dumps(run_hybrid.jsonable({
        "workflow": ("shared_private_low_noise_calibration_v1"
                     if sweep.get("calibration_only", False)
                     else "staged_shared_private_v1"),
        "base_config": base,
        "sweep": sweep,
        "inputs": {
            "clinical": getattr(args, "clinical", None),
            "expression": getattr(args, "expression", None),
            "methylation": getattr(args, "methylation", None),
            "survival": getattr(args, "survival", None),
            "omics": list(getattr(args, "omic", []) or []),
            "n_patients": len(expression),
            "modality_shapes": {
                "expression": list(expression.shape),
                **({"methylation": list(methylation.shape)}
                   if methylation is not None else {}),
                **{name: list(frame.shape) for name, frame in omics.items()},
            },
        },
    })))
    manifest_path = out / "run_manifest.json"
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("--resume inputs/config/sweep do not match run_manifest.json")
    else:
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
    # Model initialization seeds must not change held-out patients. Every stage
    # and finalist therefore uses one fixed fold assignment.
    fold_seed = int(sweep.get("fold_seed", 0))
    fixed_folds = stratified_folds(
        clinical, expression, survival, fold_seed,
        n_splits=int(sweep.get("cv_folds", 3)),
        methylation=methylation, omics=omics)

    def folds_for_seed(_seed):
        return fixed_folds

    if sweep.get("calibration_only", False):
        return _run_calibration_only(
            args, sweep, out, selected_path, base, folds_for_seed, modality_names)

    stages = sweep["stages"]
    screening_epochs = int(stages.get("screening_max_epochs", 150))
    latent_dims = stages.get("shared_latent_dims", [16, 32, 64])
    private_presets = stages.get("private_latent_presets", [
        {"default": 4}, {"default": 8},
        {"expression": 8, "methylation": 16}, {"default": 16},
    ])
    private_weights = stages.get("private_kl_weights", [0.25, 0.5, 1.0])

    stage1 = []
    for latent_dim in latent_dims:
        shared = dict(base, latent_dim=int(latent_dim), latent_structure="shared",
                      private_latent_dims={}, max_epochs=screening_epochs, seed=0)
        stage1.append({"config": shared, "candidate_id": _stable_candidate_id(shared)})
        for preset in private_presets:
            resolved = {name: int(value) for name, value in dict(preset).items()
                        if name == "default" or name in modality_names}
            for weight in private_weights:
                config = dict(base, latent_dim=int(latent_dim),
                              latent_structure="shared_private",
                              private_latent_dims=resolved,
                              private_kl_weight=float(weight),
                              max_epochs=screening_epochs, seed=0)
                stage1.append({"config": config, "candidate_id": _stable_candidate_id(config)})
    print(f"Stage 1/4: {len(stage1)} latent-structure candidates")
    ranked1 = _evaluate_staged_candidates(
        stage1, out / "stage1_structure.csv", folds_for_seed, modality_names,
        synthetic_replicates=1, resume=args.resume, enforce_guards=False)
    private1 = ranked1[ranked1["config_json"].map(
        lambda value: json.loads(value).get("latent_structure") == "shared_private")]
    shared1 = ranked1[ranked1["config_json"].map(
        lambda value: json.loads(value).get("latent_structure") == "shared")]
    retained1 = pd.concat([private1.head(3), shared1.head(1)], ignore_index=True)

    stage2 = []
    for _, row in retained1.iterrows():
        config0 = _config_from_ranked(row)
        if config0["latent_structure"] == "shared":
            variants = [(base.get("fusion", "moe"), base.get("weighting_scheme", "uncertainty"),
                         base.get("beta_max", 0.25))]
        else:
            variants = [(fusion, weighting, beta)
                        for fusion in stages.get("fusion", ["poe", "moe", "mopoe"])
                        for weighting in stages.get("weighting", ["fixed", "uncertainty"])
                        for beta in stages.get("beta_max", [0.1, 0.25, 0.5])]
        for fusion, weighting, beta in variants:
            config = dict(config0, fusion=fusion, weighting_scheme=weighting,
                          beta_max=float(beta), max_epochs=screening_epochs, seed=0)
            stage2.append({"config": config, "candidate_id": _stable_candidate_id(config),
                           "parent_id": row["candidate_id"]})
    print(f"Stage 2/4: {len(stage2)} fusion/objective candidates")
    ranked2 = _evaluate_staged_candidates(
        stage2, out / "stage2_architecture.csv", folds_for_seed, modality_names,
        synthetic_replicates=1, resume=args.resume, enforce_guards=False)
    retained2 = ranked2.head(int(stages.get("retain_architectures", 6)))

    stage3_latent = []
    for _, row in retained2.iterrows():
        config0 = _config_from_ranked(row)
        parent = row["candidate_id"]
        for draws in stages.get("posterior_draws", [1, 2, 4]):
            for components in stages.get("gmm_components", [2, 5, 10]):
                config = dict(config0, posterior_samples_per_patient=int(draws),
                              gmm_components=int(components), max_epochs=screening_epochs)
                stage3_latent.append({
                    "config": config, "candidate_id": _stable_candidate_id(config),
                    "parent_id": parent})
    print(f"Stage 3a/4: {len(stage3_latent)} latent-density calibrations")
    ranked3a = _evaluate_staged_candidates(
        stage3_latent, out / "stage3_latent.csv", folds_for_seed, modality_names,
        synthetic_replicates=1, resume=args.resume, enforce_guards=False)
    retained3a = ranked3a.groupby("parent_id", group_keys=False).head(2)

    noise_presets = _residual_noise_presets(stages, modality_names)
    stage3_residual = []
    for _, row in retained3a.iterrows():
        config0 = _config_from_ranked(row)
        for rank in stages.get("residual_ranks", [0, 8, 16, 32]):
            for scales in noise_presets:
                config = dict(config0, residual_rank=int(rank),
                              observation_noise_scale=scales,
                              observation_sampler="residual_factor",
                              max_epochs=screening_epochs)
                stage3_residual.append({
                    "config": config, "candidate_id": _stable_candidate_id(config),
                    "parent_id": row["parent_id"]})
    print(f"Stage 3b/4: {len(stage3_residual)} residual calibrations")
    ranked3b = _evaluate_staged_candidates(
        stage3_residual, out / "stage3_residual.csv", folds_for_seed, modality_names,
        synthetic_replicates=3, resume=args.resume,
        enforce_guards=bool(sweep.get("enforce_quality_guards", True)))
    retained3b = ranked3b.head(int(stages.get("retain_finalists", 3)))

    finalist_epochs = int(stages.get("finalist_max_epochs", 1500))
    stage4 = []
    for _, row in retained3b.iterrows():
        config0 = _config_from_ranked(row)
        selection_id = row["candidate_id"]
        for seed in stages.get("finalist_seeds", [0, 1, 2]):
            config = dict(config0, seed=int(seed), max_epochs=finalist_epochs)
            stage4.append({
                "config": config, "candidate_id": _stable_candidate_id(config),
                "selection_id": selection_id, "parent_id": row.get("parent_id", "")})
    print(f"Stage 4/4: {len(stage4)} seeded finalist configurations")
    _evaluate_staged_candidates(
        stage4, out / "stage4_finalists.csv", folds_for_seed, modality_names,
        synthetic_replicates=3, resume=args.resume,
        enforce_guards=bool(sweep.get("enforce_quality_guards", True)))
    raw4 = pd.read_csv(out / "stage4_finalists.csv")
    ranked4 = _rank_staged_rows(
        raw4, modality_names, group_key="selection_id",
        enforce_guards=bool(sweep.get("enforce_quality_guards", True)))
    ranked4.to_csv(out / "stage4_finalists_ranked.csv", index=False)
    if ranked4.empty or (bool(sweep.get("enforce_quality_guards", True))
                         and not np.isfinite(ranked4.iloc[0]["rank_score"])):
        raise RuntimeError("Every shared/private finalist failed the quality guards")
    winner_id = ranked4.iloc[0]["selection_id"]
    winner_rows = raw4[raw4["selection_id"] == winner_id]
    seed0 = winner_rows[winner_rows["seed"] == 0]
    winner_config = json.loads((seed0 if len(seed0) else winner_rows).iloc[0]["config_json"])
    winner_config["seed"] = 0
    winner_config["max_epochs"] = finalist_epochs
    with open(selected_path, "w", encoding="utf-8") as fh:
        json.dump(run_hybrid.jsonable({
            "best_config": winner_config,
            "best_metrics": ranked4.iloc[0].to_dict(),
            "tuning_objective": "staged_shared_private_source_overlap",
            "aggregation": {"over": ["fold", "seed", "replicate"],
                            "seed_values": stages.get("finalist_seeds", [0, 1, 2])},
        }), fh, indent=2, sort_keys=True)
    print(f"Staged winner written to {selected_path}")
    return 0


def cmd_run(args):
    cfg = cfg_from_args(args)
    clinical, expression, survival, methylation, omics = load_data(args)
    splits = split_frames(clinical, expression, survival, args.seed,
                          methylation=methylation, omics=omics)

    print(f"Training joint VAE on {len(splits[0])} patients "
          f"(fusion={cfg.fusion}, likelihood={cfg.expression_likelihood}, "
          f"weighting={cfg.weighting_scheme}) ...")
    metrics, synth, result = run_joint(cfg, splits)

    print()
    print(format_report(metrics))

    last = synth.model.history_[-1] if synth.model.history_ else {}
    print(f"\nTrained {len(synth.model.history_)} epochs; "
          f"final val objective = {last.get('val_obj', float('nan')):.4f}")

    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        result.clinical.to_csv(Path(args.out) / "synthetic_clinical.csv", index=False)
        result.expression.to_csv(Path(args.out) / "synthetic_expression.csv", index=False)
        result.survival.to_csv(Path(args.out) / "synthetic_survival.csv", index=False)
        if result.methylation is not None:
            result.methylation.to_csv(
                Path(args.out) / "synthetic_methylation.csv", index=False)
        for name, frame in result.modalities.items():
            frame.to_csv(Path(args.out) / f"synthetic_{name}.csv", index=False)
        print(f"Wrote synthetic cohort to {args.out}")


def cmd_tune(args):
    sweep = run_hybrid.load_sweep(args.sweep)
    if sweep.get("staged", False):
        return _cmd_tune_staged(args, sweep)
    weights = sweep.get("metric_weights")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Never leave a previous winner looking valid after a new all-rejected sweep.
    selected_path = out / "selected_config.json"
    if selected_path.exists():
        selected_path.unlink()

    base = asdict(MMVAEConfig())
    base.update(load_config(args.config))
    if args.epochs:  base["max_epochs"] = args.epochs
    if args.device:  base["device"] = args.device
    base.setdefault("verbose", args.verbose)

    clinical, expression, survival, methylation, omics = load_data(args)
    has_clinical = len(clinical.columns) > 0
    folds_by_seed: dict[int, list] = {}
    candidates = run_hybrid.expand_grid(sweep)
    rows = []
    cv_folds = int(sweep.get("cv_folds", 3))
    synthetic_replicates = int(sweep.get("synthetic_replicates", 3))
    # Generation-only settings share a trained neural model. This turns one
    # otherwise enormous Cartesian grid into an architecture stage followed by
    # cheap sampler calibration on the fitted finalists.
    training_groups = {}
    for i, candidate in enumerate(candidates):
        cfg_kwargs = dict(base, **sweep_to_cfg_kwargs(candidate))
        seed = int(cfg_kwargs.get("seed", args.seed))
        cfg_kwargs["seed"] = seed
        cfg = MMVAEConfig(**cfg_kwargs)
        training_candidate = {k: v for k, v in candidate.items()
                              if k not in _GENERATION_SWEEP_KEYS}
        key = json.dumps(run_hybrid.jsonable(training_candidate), sort_keys=True)
        training_groups.setdefault(key, []).append((i, candidate, cfg))

    for group_number, group in enumerate(training_groups.values(), 1):
        seed = group[0][2].seed
        if seed not in folds_by_seed:
            folds_by_seed[seed] = stratified_folds(
                clinical, expression, survival, seed, n_splits=cv_folds,
                methylation=methylation, omics=omics)
        first_cfg = group[0][2]
        print(f"\nArchitecture {group_number}/{len(training_groups)}: "
              f"fusion={first_cfg.fusion} weighting={first_cfg.weighting_scheme} "
              f"beta_max={first_cfg.beta_max} free_bits={first_cfg.free_bits} "
              f"latent_dim={first_cfg.latent_dim} seed={seed}; "
              f"{len(group)} sampler configuration(s)")
        for fold, splits in enumerate(folds_by_seed[seed]):
            (clin_tr, expr_tr, surv_tr, meth_tr, omics_tr, *_rest) = splits
            synth = JointVAESynthesizer(cfg=first_cfg).fit(
                clin_tr, expr_tr, surv_tr, methylation=meth_tr, omics=omics_tr)
            for i, candidate, cfg in group:
                _recalibrate_generation(synth, cfg)
                for replicate in range(max(1, synthetic_replicates)):
                    draw_seed = seed + replicate * 100003
                    synth.cfg.seed = draw_seed
                    synth.model.cfg.seed = draw_seed
                    metrics, _ = evaluate_joint(synth, splits)
                    rows.append({
                        "survival_score": composite_score(metrics, weights),
                        "tuning_objective": "balanced_survival_molecular",
                        "candidate": i,
                        "fold": fold,
                        "replicate": replicate,
                        "has_clinical": has_clinical,
                        **sweep_to_cfg_kwargs(candidate),
                        "seed": seed,
                        **metrics,
                    })
                synth.cfg.seed = seed
                synth.model.cfg.seed = seed

    fold_results = pd.DataFrame(rows)
    fold_results.to_csv(out / "sweep_results_folds.csv", index=False)
    # Average held-out folds before ranking. Config fields are constant inside a
    # candidate; numeric evaluation fields receive the fold mean.
    config_field_names = {field.name for field in fields(MMVAEConfig)}
    # Hyperparameters are constant within a candidate. Averaging them turns
    # integer architecture settings (e.g. latent_dim) into floats and later
    # breaks torch module construction. Keep their original typed value.
    first_cols = [c for c in fold_results
                  if (not pd.api.types.is_numeric_dtype(fold_results[c])
                      or c in config_field_names)]
    numeric_cols = [c for c in fold_results if pd.api.types.is_numeric_dtype(fold_results[c])
                    and c not in {"candidate", "fold", "replicate", "seed"}
                    and c not in config_field_names]
    grouped_rows = []
    for candidate, frame in fold_results.groupby("candidate", sort=False):
        row = {c: frame[c].iloc[0] for c in first_cols}
        row.update({"candidate": candidate, "seed": int(frame["seed"].iloc[0]),
                    "n_folds": int(frame["fold"].nunique()),
                    "n_synthetic_replicates": int(frame["replicate"].nunique())})
        row.update({c: float(frame[c].mean()) for c in numeric_cols})
        grouped_rows.append(row)
    results = pd.DataFrame(grouped_rows)

    head_cols = [c for c in results
                 if c.endswith("_geometry_penalty") and not c.startswith("cross_omic_")]
    cross_cols = [c for c in results
                  if c.startswith("cross_omic_") and c.endswith("_geometry_penalty")]
    molecular_cols = head_cols + cross_cols
    by = ["seed"] if "seed" in results else None
    survival_rank = results.groupby(by, group_keys=False)["survival_score"].rank(
        pct=True, method="average") if by else results["survival_score"].rank(pct=True)
    molecular_ranks = results.groupby(by, group_keys=False)[molecular_cols].rank(
        pct=True, method="average") if by else results[molecular_cols].rank(pct=True)
    results["survival_block_rank"] = survival_rank
    results["molecular_block_rank"] = molecular_ranks.mean(axis=1)
    results["rank_score"] = 0.5 * results["survival_block_rank"] + 0.5 * results["molecular_block_rank"]
    prefixes = [c[:-len("_variance_ratio_median")] for c in results
                if c.endswith("_variance_ratio_median")]
    results["quality_guard_failures"] = results.apply(
        lambda row: ";".join(quality_guard_failures(row.to_dict(), prefixes)), axis=1)
    results["quality_rejected"] = results["quality_guard_failures"].str.len() > 0
    if bool(sweep.get("enforce_quality_guards", False)):
        results.loc[results["quality_rejected"], "rank_score"] = np.inf
    results = results.sort_values("rank_score").reset_index(drop=True)
    results.to_csv(out / "sweep_results.csv", index=False)
    if not np.isfinite(results["rank_score"]).any():
        raise RuntimeError(
            "Every sweep candidate failed the molecular-fidelity guards; no "
            "selected_config.json was written. Increase training capacity or "
            "repair the generative variance before promoting a final model."
        )

    dotted_config_keys = sorted(set(sweep.get("grid", {})) | set(sweep.get("fixed", {})))
    config_keys = [SWEEP_KEY_MAP.get(k, k.split(".")[-1])
                   for k in dotted_config_keys if k != "generation.preset"]
    for preset in sweep.get("grid", {}).get("generation.preset", []) or []:
        if isinstance(preset, dict):
            config_keys.extend(SWEEP_KEY_MAP.get(f"generation.{key}", key)
                               for key in preset)
    config_keys = sorted(set(config_keys)) + ["seed"]
    for key in config_keys:
        if key in results and results[key].map(lambda value: isinstance(value, (dict, list))).any():
            results[key] = results[key].map(
                lambda value: json.dumps(run_hybrid.jsonable(value), sort_keys=True)
                if isinstance(value, (dict, list)) else value)
    aggregated = run_hybrid.aggregate_seed_results(results, config_keys)
    if not aggregated.empty:
        aggregated.to_csv(out / "sweep_results_aggregated.csv", index=False)
        best = aggregated.iloc[0].to_dict()
        selected = {k: best[k] for k in config_keys
                    if k in best and pd.notna(best[k]) and k != "seed"}
        selected["seed"] = int(args.seed)
        best_metrics = {k: best[k] for k in best if k not in selected}
        aggregation = {
            "over": "seed",
            "seed_values": json.loads(best["seed_values"]),
            "refit_seed": int(args.seed),
        }
    else:
        best = results.iloc[0].to_dict()
        selected = {k: best[k] for k in config_keys if k in best and pd.notna(best[k])}
        best_metrics = {k: best[k] for k in best if k not in selected}
        aggregation = None

    if isinstance(selected.get("observation_noise_scale"), str):
        selected["observation_noise_scale"] = json.loads(selected["observation_noise_scale"])

    valid = {f.name for f in fields(MMVAEConfig)}
    # Persist every resolved setting, not just swept keys.  Dimensions are reset
    # from input data during train, while the remaining settings reproduce tune.
    selected = {k: v for k, v in dict(base, **selected).items() if k in valid}
    with open(selected_path, "w", encoding="utf-8") as fh:
        json.dump(run_hybrid.jsonable({
            "best_config": selected,
            "best_metrics": best_metrics,
            "metric_weights": weights,
            "tuning_objective": "balanced_survival_molecular",
            "aggregation": aggregation,
        }), fh, indent=2, sort_keys=True)
    print(f"\nBest config written to {selected_path}")
    print(f"Ranked sweep results written to {out / 'sweep_results.csv'}")


def cmd_train(args):
    cfg_kwargs = load_selected_config(args.config)
    if args.epochs:  cfg_kwargs["max_epochs"] = args.epochs
    if args.device:  cfg_kwargs["device"] = args.device
    cfg_kwargs["seed"] = args.seed
    cfg_kwargs.setdefault("verbose", args.verbose)
    cfg = MMVAEConfig(**cfg_kwargs)

    clinical, expression, survival, methylation, omics = load_data(args)
    print(f"Training final joint VAE on all {len(clinical)} patients "
          f"(fusion={cfg.fusion}, likelihood={cfg.expression_likelihood}, "
          f"weighting={cfg.weighting_scheme}"
          f"{', +methylation' if methylation is not None else ''}) ...")
    synth = JointVAESynthesizer(cfg=cfg).fit(clinical, expression, survival,
                                             methylation=methylation, omics=omics)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.pt"
    synth.save(model_path)

    epochs_trained = len(synth.model.history_)
    last = synth.model.history_[-1] if synth.model.history_ else {}
    with open(out / "train_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(run_hybrid.jsonable({
            "config": {f.name: getattr(cfg, f.name) for f in fields(MMVAEConfig)},
            "n_patients": int(len(clinical)),
            "epochs_trained": epochs_trained,
            "final_val_obj": last.get("val_obj"),
        }), fh, indent=2, sort_keys=True)
    print(f"Trained {epochs_trained} epochs; final val objective = "
          f"{last.get('val_obj', float('nan')):.4f}")
    print(f"Saved model to {model_path}")
    print(f"Next: python scripts/run_mmvae.py generate --model {model_path} "
          f"--n 1000 --out-dir runs/synth")


def cmd_generate(args):
    synth = JointVAESynthesizer.load(args.model)
    print(f"Loaded model from {args.model}; sampling {args.n} synthetic patients ...")
    result = synth.sample(args.n)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.clinical.to_csv(out / "synthetic_clinical.csv", index=False)
    result.expression.to_csv(out / "synthetic_expression.csv", index=False)
    result.survival.to_csv(out / "synthetic_survival.csv", index=False)
    if result.methylation is not None:
        result.methylation.to_csv(out / "synthetic_methylation.csv", index=False)
    for name, frame in result.modalities.items():
        frame.to_csv(out / f"synthetic_{name}.csv", index=False)
    print(f"Wrote synthetic cohort ({args.n} rows) to {out}")


def _parse_named_dirs(values):
    cohorts = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--synthetic must use NAME=DIR")
        name, directory = value.split("=", 1)
        if not name or not directory or name in cohorts:
            raise ValueError("--synthetic names must be non-empty and unique")
        cohorts[name] = Path(directory)
    return cohorts


def _real_fitted_expression_latents(real, synthetic, n_components=10):
    from sklearn.decomposition import PCA
    common = real.columns.intersection(synthetic.columns)
    if not len(common):
        raise ValueError("real and synthetic expression have no shared columns")
    r, s = real[common].to_numpy(float), synthetic[common].to_numpy(float)
    variance = np.var(r, axis=0)
    idx = np.argsort(variance)[-min(512, len(common)):]
    mean, std = r[:, idx].mean(0), r[:, idx].std(0) + 1e-8
    rz, sz = (r[:, idx] - mean) / std, (s[:, idx] - mean) / std
    k = max(1, min(n_components, rz.shape[1], len(rz) - 1))
    pca = PCA(k, random_state=0).fit(rz)
    return pca.transform(rz), pca.transform(sz)


def cmd_evaluate(args):
    """Compare one or more generated cohort directories against real inputs."""
    clinical = read_table(args.clinical)
    expression = read_table(args.expression)
    survival = read_table(args.survival)
    real_modalities = {"expression": expression}
    if args.methylation:
        real_modalities["methylation"] = read_table(args.methylation)
    for name, path in _parse_omics(args.omic).items():
        real_modalities[name] = read_table(path)
    if any(len(frame) != len(expression) for frame in [clinical, survival, *real_modalities.values()]):
        raise ValueError("all real evaluation inputs must share row count")
    cohorts = _parse_named_dirs(args.synthetic)
    if not cohorts:
        raise ValueError("at least one --synthetic NAME=DIR is required")
    if args.group_column and args.group_column not in clinical.columns:
        raise ValueError(f"group column {args.group_column!r} is absent from real clinical data")
    real_groups = clinical[args.group_column].astype(str).to_numpy() if args.group_column else None
    rows, summary = [], {}
    for cohort_name, directory in cohorts.items():
        required = {
            "clinical": directory / "synthetic_clinical.csv",
            "expression": directory / "synthetic_expression.csv",
            "survival": directory / "synthetic_survival.csv",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise ValueError(f"{cohort_name} is missing: {', '.join(missing)}")
        synth_clinical = pd.read_csv(required["clinical"])
        synth_survival = pd.read_csv(required["survival"])
        synth_modalities = {"expression": pd.read_csv(required["expression"])}
        for modality in real_modalities:
            if modality == "expression":
                continue
            path = directory / f"synthetic_{modality}.csv"
            if path.exists():
                synth_modalities[modality] = pd.read_csv(path)
        synth_groups = None
        if args.group_column and args.group_column in synth_clinical.columns:
            synth_groups = synth_clinical[args.group_column].astype(str).to_numpy()
        cohort_metrics = {"modalities": {}, "cross_omic": {}, "survival": {}}
        for modality, real_frame in real_modalities.items():
            if modality not in synth_modalities:
                cohort_metrics["modalities"][modality] = {"status": "missing"}
                continue
            synth_frame = synth_modalities[modality]
            common = real_frame.columns.intersection(synth_frame.columns)
            real_values = real_frame[common].to_numpy(float)
            kind = ("methylation" if modality == "methylation"
                    else "binary" if np.isin(real_values, (0.0, 1.0)).all()
                    else "continuous")
            values = modality_geometry(
                real_frame[common], synth_frame[common], kind=kind,
                real_groups=real_groups, synthetic_groups=synth_groups,
            )
            cohort_metrics["modalities"][modality] = values
            rows.extend({"cohort": cohort_name, "scope": "modality", "name": modality,
                         "metric": key, "value": value}
                        for key, value in values.items())
        available = [name for name in real_modalities if name in synth_modalities]
        for i, left in enumerate(available):
            for right in available[i + 1:]:
                la = real_modalities[left].columns.intersection(synth_modalities[left].columns)
                rb = real_modalities[right].columns.intersection(synth_modalities[right].columns)
                values = cross_omic_fidelity(
                    real_modalities[left][la], synth_modalities[left][la],
                    real_modalities[right][rb], synth_modalities[right][rb],
                )
                name = f"{left}__{right}"
                cohort_metrics["cross_omic"][name] = values
                rows.extend({"cohort": cohort_name, "scope": "cross_omic", "name": name,
                             "metric": key, "value": value}
                            for key, value in values.items())
        real_latent, synth_latent = _real_fitted_expression_latents(
            expression, synth_modalities["expression"])
        survival_metrics = evaluate(
            _eval_dict(clinical, real_latent, survival),
            _eval_dict(synth_clinical, synth_latent, synth_survival),
        )
        cohort_metrics["survival"] = survival_metrics
        rows.extend({"cohort": cohort_name, "scope": "survival", "name": "survival",
                     "metric": key, "value": value}
                    for key, value in survival_metrics.items())
        summary[cohort_name] = cohort_metrics

    acceptance = None
    if all(name in summary for name in
           (args.joint_name, args.expression_reference, args.hybrid_reference)):
        joint = summary[args.joint_name]
        expression_ref = summary[args.expression_reference]
        hybrid_ref = summary[args.hybrid_reference]
        checks = {}

        def check(name, value, limit, relation="max"):
            passed = bool(np.isfinite(value) and np.isfinite(limit) and
                          (value <= limit if relation == "max" else value >= limit))
            checks[name] = {"value": value, "limit": limit,
                            "relation": relation, "passed": passed}

        je = joint["modalities"].get("expression", {})
        ee = expression_ref["modalities"].get("expression", {})
        check("expression_covariance_noninferiority", je.get("covariance_gap", np.nan),
              1.10 * ee.get("covariance_gap", np.nan))
        check("expression_mixing_noninferiority", je.get("knn_mixing_normalized", np.nan),
              ee.get("knn_mixing_normalized", np.nan) - 0.10, "min")
        check("expression_effect_noninferiority", je.get("subgroup_effect_corr", np.nan),
              ee.get("subgroup_effect_corr", np.nan) - 0.05, "min")
        check("expression_conditional_mixing_noninferiority",
              je.get("source_overlap_mixing_worst", np.nan),
              ee.get("source_overlap_mixing_worst", np.nan) - 0.05, "min")
        check("expression_radial_noninferiority",
              je.get("source_overlap_radial_worst", np.nan),
              1.10 * ee.get("source_overlap_radial_worst", np.nan))
        check("expression_worst_subgroup_boundary_noninferiority",
              je.get("source_overlap_boundary_penalty_worst", np.nan),
              1.10 * ee.get("source_overlap_boundary_penalty_worst", np.nan))
        check("expression_worst_subgroup_source_auc_noninferiority",
              je.get("source_overlap_auc_max_worst", np.nan),
              ee.get("source_overlap_auc_max_worst", np.nan) + 0.05)
        check("expression_manifold_precision_noninferiority",
              je.get("source_overlap_precision_worst", np.nan),
              ee.get("source_overlap_precision_worst", np.nan) - 0.05, "min")
        check("expression_manifold_recall_noninferiority",
              je.get("source_overlap_recall_worst", np.nan),
              ee.get("source_overlap_recall_worst", np.nan) - 0.05, "min")
        check("expression_worst_subgroup_precision_noninferiority",
              je.get("source_overlap_group_precision_worst", np.nan),
              ee.get("source_overlap_group_precision_worst", np.nan) - 0.05, "min")
        check("expression_worst_subgroup_recall_noninferiority",
              je.get("source_overlap_group_recall_worst", np.nan),
              ee.get("source_overlap_group_recall_worst", np.nan) - 0.05, "min")
        jm = joint["modalities"].get("methylation", {})
        hm = hybrid_ref["modalities"].get("methylation", {})
        have_methylation = bool(jm and hm and jm.get("status") != "missing"
                                and hm.get("status") != "missing")
        if have_methylation:
            check("methylation_covariance_noninferiority", jm.get("covariance_gap", np.nan),
                  1.15 * hm.get("covariance_gap", np.nan))
            check("methylation_mixing_noninferiority", jm.get("knn_mixing_normalized", np.nan),
                  hm.get("knn_mixing_normalized", np.nan) - 0.10, "min")
            check("methylation_effect_noninferiority", jm.get("subgroup_effect_corr", np.nan),
                  hm.get("subgroup_effect_corr", np.nan) - 0.05, "min")
            check("methylation_conditional_mixing_noninferiority",
                  jm.get("source_overlap_mixing_worst", np.nan),
                  hm.get("source_overlap_mixing_worst", np.nan) - 0.05, "min")
            check("methylation_radial_noninferiority",
                  jm.get("source_overlap_radial_worst", np.nan),
                  1.15 * hm.get("source_overlap_radial_worst", np.nan))
            check("methylation_worst_subgroup_boundary_noninferiority",
                  jm.get("source_overlap_boundary_penalty_worst", np.nan),
                  1.15 * hm.get("source_overlap_boundary_penalty_worst", np.nan))
            check("methylation_worst_subgroup_source_auc_noninferiority",
                  jm.get("source_overlap_auc_max_worst", np.nan),
                  hm.get("source_overlap_auc_max_worst", np.nan) + 0.05)
            check("methylation_manifold_precision_noninferiority",
                  jm.get("source_overlap_precision_worst", np.nan),
                  hm.get("source_overlap_precision_worst", np.nan) - 0.05, "min")
            check("methylation_manifold_recall_noninferiority",
                  jm.get("source_overlap_recall_worst", np.nan),
                  hm.get("source_overlap_recall_worst", np.nan) - 0.05, "min")
            check("methylation_worst_subgroup_precision_noninferiority",
                  jm.get("source_overlap_group_precision_worst", np.nan),
                  hm.get("source_overlap_group_precision_worst", np.nan) - 0.05, "min")
            check("methylation_worst_subgroup_recall_noninferiority",
                  jm.get("source_overlap_group_recall_worst", np.nan),
                  hm.get("source_overlap_group_recall_worst", np.nan) - 0.05, "min")
        cross_name = "expression__methylation"
        jc = joint["cross_omic"].get(cross_name, {})
        hc = hybrid_ref["cross_omic"].get(cross_name, {})
        if jc and hc:
            check("cross_omic_noninferiority", jc.get("covariance_gap", np.nan),
                  1.15 * hc.get("covariance_gap", np.nan))
        guarded_modalities = [("expression", je)]
        if have_methylation:
            guarded_modalities.append(("methylation", jm))
        for modality, values in guarded_modalities:
            ratio = values.get("variance_ratio_median", np.nan)
            checks[f"{modality}_variance_guard"] = {
                "value": ratio, "range": [0.5, 1.5],
                "passed": bool(np.isfinite(ratio) and 0.5 <= ratio <= 1.5),
            }
        if args.joint_baseline:
            if args.joint_baseline not in summary:
                raise ValueError(
                    f"joint baseline {args.joint_baseline!r} was not supplied via --synthetic")
            baseline_survival = summary[args.joint_baseline]["survival"]
            candidate_survival = joint["survival"]
            check("survival_tstr_noninferiority",
                  candidate_survival.get("tstr_concordance", np.nan),
                  baseline_survival.get("tstr_concordance", np.nan) - 0.03, "min")
            check("survival_concordance_gap_noninferiority",
                  abs(candidate_survival.get("concordance_gap", np.nan)),
                  abs(baseline_survival.get("concordance_gap", np.nan)) + 0.03)
            check("survival_late_km_noninferiority",
                  candidate_survival.get("km_iad_late", np.nan),
                  baseline_survival.get("km_iad_late", np.nan) + 0.03)
        acceptance = {
            "checks": checks,
            "passed": bool(all(item["passed"] for item in checks.values())),
            "visual_review_required": True,
            "promotion_status": "quantitative_pass_visual_review_required"
            if all(item["passed"] for item in checks.values())
            else "quantitative_fail",
        }
        # Architecture escalation used to be recommended for every failure,
        # even when the evaluated candidate already used shared/private latents.
        # Failed promotion now keeps the references immutable and explicitly
        # recommends the reversible sampler-calibration workflow.
        acceptance["shared_private_latents_recommended"] = False
        acceptance["recommended_action"] = (
            "visual_review" if acceptance["passed"]
            else "retain_reference_and_run_low_noise_calibration")
        summary["acceptance"] = acceptance

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "quality_metrics.csv", index=False)
    with open(out / "quality_summary.json", "w", encoding="utf-8") as fh:
        json.dump(run_hybrid.jsonable(summary), fh, indent=2, sort_keys=True)
    print(f"Wrote comparative quality metrics to {out}")
    if acceptance is not None:
        print("Acceptance gates: " + (
            "PASS -- visual review required" if acceptance["passed"] else "FAIL"))


def add_shared_args(p):
    """Data + config flags shared by run / tune / train."""
    p.add_argument("--config", default=None, help="config/mmvae.yaml or a tune "
                   "selected_config.json (train/generate)")
    p.add_argument("--toy", action="store_true", help="use the synthetic cohort")
    p.add_argument("--clinical", default=None)
    p.add_argument("--expression", default=None)
    p.add_argument("--methylation", default=None,
                   help="optional methylation beta-value CSV (features as columns); "
                        "adds a joint methylation head sharing the latent")
    p.add_argument("--omic", action="append", default=[], metavar="NAME=PATH",
                   help="repeatable declared molecular head, e.g. --omic cnv=cnv.csv")
    p.add_argument("--survival", default=None)
    p.add_argument("--n", type=int, default=800, help="toy cohort size")
    p.add_argument("--n-genes", type=int, default=300, help="toy gene count")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")


def _add_subcommands(sub) -> None:
    run = sub.add_parser("run", help="train + sample + evaluate")
    add_shared_args(run)
    run.add_argument("--fusion", choices=["poe", "moe", "mopoe"], default=None)
    run.add_argument("--likelihood", choices=["gaussian", "nb", "zinb"], default=None)
    run.add_argument("--weighting", choices=["fixed", "uncertainty", "gradnorm"],
                     default=None)
    run.add_argument("--latent-dim", type=int, default=None)
    run.add_argument("--out", default=None, help="dir to write the synthetic cohort")
    run.set_defaults(func=cmd_run)

    tune = sub.add_parser("tune", help="grid sweep, rank by composite, pick a winner")
    add_shared_args(tune)
    tune.add_argument("--sweep", required=True, help="YAML sweep spec (grid/fixed)")
    tune.add_argument("--out-dir", required=True, help="directory for sweep results")
    tune.add_argument("--resume", action="store_true",
                      help="continue a staged sweep from completed candidate rows")
    tune.set_defaults(func=cmd_tune)

    train = sub.add_parser("train", help="fit the winning config on all data and save it")
    add_shared_args(train)
    train.add_argument("--out-dir", required=True, help="directory for model.pt")
    train.set_defaults(func=cmd_train)

    gen = sub.add_parser("generate", help="load a saved model and write synthetic CSVs")
    gen.add_argument("--model", required=True, help="model.pt from `train`")
    gen.add_argument("--n", type=int, default=1000, help="patients to generate")
    gen.add_argument("--out-dir", required=True, help="directory for synthetic CSVs")
    gen.set_defaults(func=cmd_generate)

    quality = sub.add_parser(
        "evaluate", help="compare named synthetic cohort directories with real data")
    quality.add_argument("--clinical", required=True)
    quality.add_argument("--expression", required=True)
    quality.add_argument("--methylation", default=None)
    quality.add_argument("--omic", action="append", default=[], metavar="NAME=PATH")
    quality.add_argument("--survival", required=True)
    quality.add_argument("--synthetic", action="append", required=True,
                         metavar="NAME=DIR", help="repeatable generated cohort directory")
    quality.add_argument("--group-column", default="MNP_Subgroup")
    quality.add_argument("--joint-name", default="joint")
    quality.add_argument("--expression-reference", default="expression_only")
    quality.add_argument("--hybrid-reference", default="hybrid")
    quality.add_argument("--joint-baseline", default=None,
                         help="named prior joint cohort used for survival non-inferiority")
    quality.add_argument("--out-dir", required=True)
    quality.set_defaults(func=cmd_evaluate)


def register(subparsers) -> None:
    """Mount the ``mmvae`` namespace under a parent ``add_subparsers`` (pode synthesis)."""
    mmvae = subparsers.add_parser(
        "mmvae", help="joint multimodal VAE (end-to-end, no external synthesizer).")
    _add_subcommands(mmvae.add_subparsers(dest="mmvae_command", required=True))


def main(argv=None) -> int:
    """Standalone entry: ``python -m synthvae.cli.mmvae <run|tune|train|generate|evaluate>``."""
    parser = argparse.ArgumentParser(
        prog="synthvae-mmvae", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_subcommands(parser.add_subparsers(dest="mmvae_command", required=True))
    args = parser.parse_args(argv)
    rc = args.func(args)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
