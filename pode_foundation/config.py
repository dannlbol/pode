"""Configuration loading, validation and override-merging.

Mirrors ``prognostk/config.py``'s style: plain-dict defaults, a recursive
override merge, and a small validator -- no JSON-Schema machinery, so the
whole schema is readable in one file.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml


def _defaults() -> Dict[str, Any]:
    return {
        "matrix": {
            "path": None,
            "orientation": "samples_by_features",  # samples_by_features | features_by_samples
            "scale": "beta",  # beta | m_value
            "feature_id_column": None,
        },
        "preprocess": {
            "annotation_path": None,
            "n_keep": 50_000,
            "n_candidates": 200_000,
            "r_thresh": 0.5,
            "window_bp": 1_000,
            "epic_overlap_path": None,
            "cross_reactive_path": None,
            "output_dir": "outputs/foundation/preprocess",
        },
        "model": {
            "encoder_dims": [1024, 512],
            "latent_dim": 256,
            "dropout": 0.2,
            "output_activation": "sigmoid",  # sigmoid for beta scale; linear for m_value
        },
        "training": {
            "mask_ratio": 0.15,
            "batch_size": 64,
            "epochs": 100,
            "patience": 10,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "val_frac": 0.1,
            "val_mask_draws": 3,
            "num_workers": 0,
            "amp": False,
            "seed": 42,
            "device": "auto",  # auto | cpu | cuda | cuda:N
            "output_dir": "outputs/foundation/pretrain",
            "prognostic_weight": {"path": None, "weight": 1.0},
        },
        "pretrain_stage1": {
            "enabled": False,
            "combine": {
                "tcga_glob": None,
                "tcga_dir": None,
                "geo_dir": None,
                "feature_ids_path": None,
                "chunk_size": 25_000,
            },
            "output_dir": "outputs/foundation/stage1",
        },
        "finetune_stage2": {
            "enabled": False,
            "init_ckpt": None,
            "lr": 1e-5,
            "epochs": 50,
            "patience": 20,
            "output_dir": "outputs/foundation/stage2",
        },
        "survival": {
            "enabled": False,
            "clinical_path": None,
            "id_column": "sample_id",
            "time_column": None,
            "event_column": None,
            "fold_column": None,
            "n_folds": 5,
            "seed": 42,
            "mode": "frozen",  # frozen | finetune
            "penalizer": 0.1,
            "l1_ratio": 0.5,
            "finetune_mode": "partial",  # frozen | partial | full (which encoder layers train)
            "finetune_epochs": 100,
            "head_lr": 1e-3,
            "encoder_lr": 1e-5,
            "output_dir": "outputs/foundation/survival",
        },
        "embed": {
            "feature_ids_path": None,
            "output_path": "outputs/foundation/embeddings.csv",
        },
    }


def _merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base``; None values are ignored."""
    if not override:
        return base
    for key, ov in override.items():
        if ov is None:
            continue
        if isinstance(ov, dict) and isinstance(base.get(key), dict):
            base[key] = _merge(base[key], ov)
        else:
            base[key] = ov
    return base


class Config(dict):
    """A validated pode_foundation configuration (a nested dict with helpers)."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        m = self["matrix"]
        return f"<Config matrix={m.get('path')} orientation={m.get('orientation')} scale={m.get('scale')}>"


def validate_config(cfg: Dict[str, Any]) -> Config:
    """Validate internal consistency; returns a :class:`Config`."""
    matrix = cfg.get("matrix", {})
    if matrix.get("orientation") not in ("samples_by_features", "features_by_samples"):
        raise ValueError("config: matrix.orientation must be 'samples_by_features' or 'features_by_samples'.")
    if matrix.get("scale") not in ("beta", "m_value"):
        raise ValueError("config: matrix.scale must be 'beta' or 'm_value'.")

    pp = cfg.get("preprocess", {})
    if pp.get("n_keep", 0) <= 0:
        raise ValueError("config: preprocess.n_keep must be > 0.")
    if pp.get("n_candidates", 0) < pp.get("n_keep", 0):
        raise ValueError("config: preprocess.n_candidates must be >= preprocess.n_keep.")

    model = cfg.get("model", {})
    if model.get("output_activation") not in ("sigmoid", "linear"):
        raise ValueError("config: model.output_activation must be 'sigmoid' or 'linear'.")

    tr = cfg.get("training", {})
    if not (0 < tr.get("mask_ratio", 0) < 1):
        raise ValueError("config: training.mask_ratio must be in (0, 1).")
    if not (0 < tr.get("val_frac", 0) < 1):
        raise ValueError("config: training.val_frac must be in (0, 1).")

    surv = cfg.get("survival", {})
    if surv.get("mode") not in ("frozen", "finetune"):
        raise ValueError("config: survival.mode must be 'frozen' or 'finetune'.")

    return Config(cfg)


def load_config(path: Optional[str] = None, **overrides: Any) -> Config:
    """Load a YAML config, merge over defaults + keyword overrides, and validate.

    Parameters
    ----------
    path : str or None
        Path to a YAML file, or ``None`` to start from the built-in defaults.
    **overrides
        Nested-dict overrides applied last, e.g.
        ``load_config("cfg.yaml", training={"epochs": 5})``.
    """
    cfg = _defaults()
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as fh:
            user = yaml.safe_load(fh) or {}
        cfg = _merge(cfg, user)
    if overrides:
        cfg = _merge(cfg, copy.deepcopy(overrides))
    return validate_config(cfg)
