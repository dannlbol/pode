"""``assess_privacy``: run the full privacy suite on one synthetic release candidate.

Inputs are the real patients the generator was *trained* on, the synthetic
table, and (strongly recommended) real patients from the same population that
the generator never saw. Tables should be in the form the generator saw them
(after ``prepare_data``); identifier checks can also be given the raw tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..imputation.base import limit_cpu, resolve_columns
from . import checks as C
from .config import PrivacyConfig
from .gate import evaluate_gate, verdict

TABLES = ("gate", "identifiers", "strata", "membership", "attribute_inference", "linkage", "rare_combinations",
          "outliers", "closest_pairs")


@dataclass
class PrivacyReport:
    summary: dict
    tables: dict[str, pd.DataFrame]
    verdict: str
    config: PrivacyConfig
    threat_model: dict = field(default_factory=dict)

    @property
    def gate(self) -> pd.DataFrame:
        return self.tables["gate"]

    def to_markdown(self, title: str = "Synthetic-data privacy assessment") -> str:
        from .report import render

        return render(self, title)

    def save(self, out_dir: str | Path, title: str = "Synthetic-data privacy assessment") -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, t in self.tables.items():
            if isinstance(t, pd.DataFrame) and not t.empty:
                t.to_csv(out / f"{name}.csv", index=False)
        with open(out / "privacy_summary.json", "w") as f:
            json.dump(jsonable({"verdict": self.verdict, "threat_model": self.threat_model,
                                "summary": self.summary}), f, indent=2)
        self.config.to_yaml(out / "privacy_config_used.yaml")
        (out / "report.md").write_text(self.to_markdown(title))
        return out


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def _numeric_columns(df: pd.DataFrame, columns: list[str], schema) -> list[str]:
    from pandas.api import types as ptypes

    out = []
    for c in columns:
        if schema is not None and c in schema.types:
            if schema.types[c].is_numeric:
                out.append(c)
        elif ptypes.is_numeric_dtype(df[c]) and df[c].nunique() > 20:
            out.append(c)
    return out


def _default_sensitive(columns: list[str], schema) -> list[str]:
    if schema is None:
        return []
    from ..schema.column_types import ColumnType

    return [c for c in schema.columns_of_type(ColumnType.SURVIVAL_EVENT) if c in columns]


def assess_privacy(
    train: pd.DataFrame,
    synthetic: pd.DataFrame,
    holdout: pd.DataFrame | None = None,
    schema=None,
    config: PrivacyConfig | None = None,
    raw_train: pd.DataFrame | None = None,
    raw_synthetic: pd.DataFrame | None = None,
    train_ids=None,
    log: Callable[[str], None] = print,
) -> PrivacyReport:
    """Assess one synthetic table against its training patients (and unseen holdout patients)."""
    from ..schema.column_types import ColumnRole

    cfg = (config or PrivacyConfig()).validate()
    if synthetic.empty or train.empty:
        raise ValueError("assess_privacy needs non-empty training and synthetic tables.")
    if schema is None:
        from ..schema.schema import DatasetSchema

        schema = DatasetSchema.from_dataframe(train)

    # ---- columns and threat model --------------------------------------------
    shared = [c for c in train.columns if c in synthetic.columns]
    cols = resolve_columns(cfg.columns, shared) if cfg.columns else shared
    drop = set(resolve_columns(cfg.exclude_columns, cols)) | {cfg.id_column}
    if schema is not None:
        drop |= set(schema.columns_of_role(ColumnRole.IDENTIFIER))
    cols = [c for c in cols if c not in drop and train[c].notna().any()]
    if not cols:
        raise ValueError("No columns to assess: the real and synthetic tables share no usable columns.")
    if holdout is not None:
        missing = [c for c in cols if c not in holdout.columns]
        if missing:
            raise ValueError(f"Holdout table lacks assessed columns: {missing[:5]}")
    numeric = _numeric_columns(train, cols, schema)
    categorical = [c for c in cols if c not in numeric]
    sensitive = resolve_columns(cfg.sensitive, cols) if cfg.sensitive else _default_sensitive(cols, schema)
    notes = []
    if cfg.sensitive and not sensitive:
        notes.append("none of the configured sensitive columns are present")
    if cfg.quasi_identifiers:
        tiers = {name: resolve_columns(globs, cols) for name, globs in cfg.quasi_identifiers.items()}
        for name, tcols in tiers.items():
            if not tcols:
                notes.append(f"attacker tier '{name}' matches no assessed column")
        tiers = {k: v for k, v in tiers.items() if v}
    else:
        default = [c for c in categorical if c not in sensitive]
        tiers = {"categorical": default} if default else {}
        notes.append("no quasi_identifiers configured: one default tier of all non-sensitive categorical "
                     "columns was used; configure attacker tiers that match your threat model")
    strata = resolve_columns(cfg.strata, cols)
    for n in notes:
        log(f"warning: {n}")

    if holdout is not None and len(holdout) < cfg.min_group:
        notes.append(f"holdout has {len(holdout)} rows (< min_group={cfg.min_group}); it was not used")
        log(f"warning: {notes[-1]}")
        holdout = None
    if holdout is None:
        log("warning: no usable holdout: distances are referenced to real leave-one-out neighbours and "
            "membership inference, the attribute member gap and linkage excess are NOT assessed. Use "
            "`pode synthesis privacy audit`, or pass real patients the generator never saw.")

    rng = np.random.default_rng(cfg.seed)
    synth_full = synthetic.reset_index(drop=True)
    rows = np.arange(len(synth_full))
    if len(rows) > cfg.max_synthetic:
        rows = np.sort(rng.choice(len(rows), cfg.max_synthetic, replace=False))
    if train_ids is None and cfg.id_column:
        src = raw_train if raw_train is not None and len(raw_train) == len(train) else train
        if cfg.id_column in src:
            train_ids = src[cfg.id_column].to_numpy()

    ctx = C.Context(train=train.reset_index(drop=True), holdout=None if holdout is None else holdout.reset_index(drop=True),
                    synth=synth_full.iloc[rows].reset_index(drop=True), synth_full=synth_full, synth_rows=rows,
                    numeric=numeric, categorical=categorical, sensitive=sensitive, tiers=tiers, strata=strata,
                    cfg=cfg, train_ids=None if train_ids is None else np.asarray(train_ids), log=log)

    summary: dict = {"n_train": len(ctx.train), "n_holdout": 0 if ctx.holdout is None else len(ctx.holdout),
                     "n_synthetic_assessed": len(ctx.synth)}
    tables: dict[str, pd.DataFrame] = {}
    s, t = C.check_identifiers(raw_train if raw_train is not None else train,
                               raw_synthetic if raw_synthetic is not None else synthetic, schema, cfg.id_column)
    summary.update(s)
    tables["identifiers"] = t
    with limit_cpu(cfg.n_jobs):
        log(f"privacy: {len(ctx.train)} training, {0 if holdout is None else len(holdout)} holdout, "
            f"{len(ctx.synth)} synthetic rows; {len(numeric)} numeric + {len(categorical)} categorical columns")
        ctx.prepare()
        for name, fn in (("exact_copies", C.check_exact_copies), ("dcr", C.check_dcr), ("strata", C.check_strata),
                         ("membership", C.check_membership), ("attribute_inference", C.check_attribute_inference),
                         ("linkage", C.check_linkage), ("rare_combinations", C.check_rare_combinations),
                         ("outliers", C.check_outliers), ("missingness", C.check_missingness)):
            s, t = fn(ctx)
            summary.update(s)
            if name == "outliers":
                tables["outliers"], tables["closest_pairs"] = t
            elif t is not None:
                tables[name] = t
    gate = evaluate_gate(summary, cfg.gates, ctx.has_holdout)
    tables = {"gate": gate, **tables}
    threat = {"n_train": len(ctx.train), "n_holdout": 0 if holdout is None else len(holdout),
              "n_synthetic": len(synth_full), "n_synthetic_assessed": len(ctx.synth), "reference": ctx.ref_kind,
              "columns": cols, "numeric": numeric, "categorical": categorical, "sensitive": sensitive,
              "quasi_identifier_tiers": tiers, "strata": strata,
              "near_copy_thresholds": {C.qname(q): v for q, v in ctx.thresholds.items()}, "notes": notes}
    return PrivacyReport(summary=summary, tables=tables, verdict=verdict(gate), config=cfg, threat_model=threat)
