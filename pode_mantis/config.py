"""Configuration loading, validation and override-merging.

Mirrors ``prognostk/config.py``'s style: plain-dict defaults, a recursive
override merge, and a small validator. This is **pode_mantis's own config**
(what drives every ``pode mantis <command>``) -- distinct from the MANTIS
reference repo's own ``config.yaml``, which ``generate_config`` produces as
an *output* artefact, not something this file describes.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml


def _defaults() -> Dict[str, Any]:
    return {
        "paths": {
            # Where sstefan01/methylation-survival (checked out to v1.0.0) lives.
            "reference_repo": None,
            # Interpreter inside the ISOLATED external MANTIS environment (see
            # docs/mantis/README.md) -- e.g. ".../envs/mantis/bin/python". This is
            # never PODE's own interpreter: MANTIS's pinned deps (sparselinear,
            # torch-scatter, torch-sparse, scikit-survival==0.23.1) cannot coexist
            # with PODE's own environment.
            "mantis_python": None,
            # Directory holding the Zenodo record 20818365 assets: conn_mat_beta.csv,
            # the released CpG/gene order lists, training_probe_means.csv.
            "zenodo_assets_dir": None,
        },
        "beta": {
            "cohort_path": None,
            "orientation": "samples_by_features",  # samples_by_features | features_by_samples
            "cpg_order_path": None,      # <zenodo_assets_dir>/beta_cpg_order.txt
            "training_means_path": None,  # <zenodo_assets_dir>/training_probe_means.csv
            "output_dir": "outputs/mantis/beta",
        },
        "cnv": {
            "msets": {},          # name -> RDS path, one or more query Msets
            "control_mset": None,  # REQUIRED, no default -- see docs/mantis/README.md
            "gene_order_path": None,  # <zenodo_assets_dir>/cnv_gene_order.txt
            "noise_threshold": None,  # REQUIRED, no default -- see docs/mantis/README.md
            "array_type": "450k",
            "phenotype_path": None,  # optional: sample sheet for MYC/MYCN QC ground truth
            "output_dir": "outputs/mantis/cnv",
        },
        "survival": {
            "clinical_path": None,
            "id_column": None,
            "time_column": None,
            "event_column": None,
            "pfs_time_column": None,
            "pfs_event_column": None,
            "metastasis_column": None,
            "output_dir": "outputs/mantis/survival",
        },
        "reference_config": {
            "endpoint": "os",  # os | pfs -- generate-config sets BOTH time_column/event_column
                                # AND os_time_column/os_event_column consistently either way,
                                # closing evaluate.py's documented footgun (see
                                # NOTES_implementation.md) by construction.
            "fixed_sections_from": None,  # reference repo's own config.yaml
            "output_path": "outputs/mantis/mantis_config.yaml",
        },
        "inference": {
            # Released checkpoint from the Zenodo record. Defaults to
            # "<zenodo_assets_dir>/mantismb_model_weights.pt" if unset.
            "model_weights_path": None,
            "output_predictions_path": "outputs/mantis/predictions.csv",
            "batch_size": 64,
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
    """A validated pode_mantis configuration (a nested dict with helpers)."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        b = self["beta"]
        return f"<Config beta={b.get('cohort_path')} endpoint={self['reference_config']['endpoint']}>"


def validate_config(cfg: Dict[str, Any]) -> Config:
    """Validate internal consistency; returns a :class:`Config`.

    Deliberately does **not** require ``cnv.control_mset``/``cnv.noise_threshold``
    or ``paths.*`` at load time -- those are only needed by the specific
    subcommands that use them (``prepare-cnv``, ``infer``/``evaluate``), and each
    raises its own clear error if a value it specifically needs is still unset
    when it runs. Requiring everything up front would make it impossible to use
    ``prepare-beta`` alone without also having a control Mset picked out yet.
    """
    beta = cfg.get("beta", {})
    if beta.get("orientation") not in ("samples_by_features", "features_by_samples"):
        raise ValueError("config: beta.orientation must be 'samples_by_features' or 'features_by_samples'.")

    cnv = cfg.get("cnv", {})
    noise = cnv.get("noise_threshold")
    if noise is not None and not (0 < float(noise) < 5):
        raise ValueError(
            "config: cnv.noise_threshold looks implausible (expected roughly (0, 5) for conumee2 "
            "noise values) -- double check you derived it from your own cohort's noise distribution, "
            "not copied a value tuned for a different cohort."
        )

    rc = cfg.get("reference_config", {})
    if rc.get("endpoint") not in ("os", "pfs"):
        raise ValueError("config: reference_config.endpoint must be 'os' or 'pfs'.")

    return Config(cfg)


def load_config(path: Optional[str] = None, **overrides: Any) -> Config:
    """Load a YAML config, merge over defaults + keyword overrides, and validate.

    Parameters
    ----------
    path : str or None
        Path to a YAML file, or ``None`` to start from the built-in defaults.
    **overrides
        Nested-dict overrides applied last, e.g.
        ``load_config("cfg.yaml", cnv={"noise_threshold": 0.77})``.
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
