"""Configuration loading, validation and override-merging.

Mirrors the R package's config.R: the same YAML file works in both languages.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml


def _defaults() -> Dict[str, Any]:
    return {
        "columns": {
            "id": None, "time": "time", "status": "status", "score": None,
            "group": None, "covariates": None, "categorical": None,
        },
        "survival": {
            "horizon": 5, "time_unit": "years", "landmark_offset": None,
            "bootstrap": {"n": 1000, "conf": 0.95, "seed": 42},
        },
        "risk_groups": {
            "thresholds": [0.90, 0.75, 0.50],
            "labels": ["Favourable", "Standard", "High", "VeryHigh"],
        },
        "palettes": {
            "risk": {"Favourable": "#2A9D8F", "Standard": "#457B9D",
                     "High": "#F4A261", "VeryHigh": "#E63946"},
            "cohort": {"discovery": "#08519C", "validation": "#D6604D"},
            "heatmap": ["#00008B", "#EEE9E9", "#8B0000"],
            "categorical": {},
        },
        "oncoplot": {"tracks": []},
        "figure": {
            "page": {"width_in": 7.09, "height_in": 10.51, "base_size": 8},
            "device": "pdf", "dpi": 300, "outdir": "./figures",
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
    """A validated prognostk configuration (a nested dict with helpers)."""

    def col(self, role: str, override: Optional[str] = None) -> str:
        """Resolve a column role (e.g. ``"time"``) to a column name."""
        if override is not None:
            return override
        val = self["columns"].get(role)
        if val is None:
            raise ValueError(
                f"config: no column configured for role '{role}' and no override given.")
        return val

    @property
    def horizon(self) -> float:
        return self["survival"]["horizon"]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        c = self["columns"]
        return (f"<Config time/status={c['time']}/{c['status']} "
                f"score={c.get('score')} horizon={self.horizon} "
                f"tiers={self['risk_groups']['labels']}>")


def validate_config(cfg: Dict[str, Any]) -> Config:
    """Validate required keys and internal consistency; returns a :class:`Config`."""
    cols = cfg.get("columns", {})
    if not cols.get("time") or not cols.get("status"):
        raise ValueError("config: columns.time and columns.status are required.")

    rg = cfg.get("risk_groups", {})
    th, lab = rg.get("thresholds"), rg.get("labels")
    if th is not None:
        th = [float(x) for x in th]
        if any(x <= 0 or x >= 1 for x in th):
            raise ValueError("config: risk_groups.thresholds must all be in (0, 1).")
        if len(th) > 1 and any(th[i + 1] >= th[i] for i in range(len(th) - 1)):
            raise ValueError("config: risk_groups.thresholds must be strictly descending.")
        if len(lab) != len(th) + 1:
            raise ValueError(
                f"config: risk_groups.labels must have length(thresholds)+1 "
                f"({len(th) + 1}), got {len(lab)}.")
        cfg["risk_groups"]["thresholds"] = th

    surv = cfg.get("survival", {})
    if surv.get("horizon") is not None and surv["horizon"] <= 0:
        raise ValueError("config: survival.horizon must be > 0.")
    conf = surv.get("bootstrap", {}).get("conf")
    if conf is not None and not (0 < conf < 1):
        raise ValueError("config: survival.bootstrap.conf must be in (0, 1).")

    valid_types = {"points", "categorical", "line"}
    for tr in cfg.get("oncoplot", {}).get("tracks", []):
        if tr.get("type") not in valid_types:
            raise ValueError(
                f"config: oncoplot track '{tr.get('name', '?')}' has invalid type "
                f"'{tr.get('type')}' (allowed: {sorted(valid_types)}).")
        if not tr.get("column"):
            raise ValueError(
                f"config: oncoplot track '{tr.get('name', '?')}' is missing 'column'.")
    return Config(cfg)


def load_config(path: Optional[str] = None, **overrides: Any) -> Config:
    """Load a YAML config, merge over defaults + keyword overrides, and validate.

    Parameters
    ----------
    path : str or None
        Path to a YAML file, or ``None`` to start from the built-in defaults.
    **overrides
        Nested-dict overrides applied last, e.g.
        ``load_config("cfg.yaml", survival={"horizon": 10})``.
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
