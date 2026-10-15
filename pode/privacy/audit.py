"""End-to-end privacy audit of the TVAE: split -> fit on training patients -> generate -> assess.

A synthetic table produced from *all* real patients cannot be tested for
membership: there is no one left who was not a member. The audit therefore
holds out a random (stratified) share of real patients, fits a fresh TVAE with
the same PODE configuration on the rest, and assesses the generated release
candidate against both groups. Repeats re-run this on new splits; the verdict
is the worst repeat, because small paediatric cohorts make single splits noisy.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..imputation.base import limit_cpu, resolve_columns
from .assess import assess_privacy, jsonable
from .config import PrivacyConfig
from .gate import ORDER
from .report import _table


def split_holdout(data: pd.DataFrame, schema, cfg: PrivacyConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of (training, holdout) patients, stratified when every stratum has >= 2 patients."""
    from sklearn.model_selection import train_test_split

    from .distance import _cat_strings

    cols = resolve_columns(cfg.audit.stratify, list(data.columns))
    if not cols and schema is not None:
        pair = schema.primary_survival_pair()
        if pair is not None and pair.event_col in data.columns:
            cols = [pair.event_col]
    labels = None
    if cols:
        parts = [_cat_strings(data[c].to_numpy()) for c in cols]
        labels = np.array(["|".join(r) for r in zip(*parts)], dtype=object)
        if pd.Series(labels).value_counts().min() < 2:
            labels = None
    idx = np.arange(len(data))
    tr, ho = train_test_split(idx, test_size=cfg.audit.holdout_frac, random_state=seed, stratify=labels)
    return np.sort(tr), np.sort(ho)


def fit_and_sample(train: pd.DataFrame, schema, pode_config, n: int, seed: int, cfg: PrivacyConfig) -> pd.DataFrame:
    """Fit a TVAE on ``train`` with the PODE synthesizer settings and sample ``n`` rows."""
    from ..generation.synthesizer import TVAEGenerator
    from ..imputation.decoder_imputer import _fit_sdv_on_cpu

    syn = copy.deepcopy(pode_config.synthesizer)
    if cfg.audit.epochs:
        syn.epochs = cfg.audit.epochs
    gen = TVAEGenerator(schema, syn)
    if cfg.audit.device == "cpu":
        gen.synthesizer = _fit_sdv_on_cpu(schema, syn, train, seed)
    else:
        gen.fit(train, seed=seed)
    return gen.sample(n, seed=seed, censor_threshold=pode_config.data.censor_time_threshold)


def audit_tvae(
    data: pd.DataFrame,
    schema,
    pode_config,
    cfg: PrivacyConfig,
    out_dir: str | Path,
    ids=None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run ``cfg.audit.repeats`` split/fit/assess rounds, writing ``repeat_<i>/`` and a summary report."""
    from ..data.io import save_dataset

    cfg.validate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = data.reset_index(drop=True)
    ids = None if ids is None else np.asarray(ids)
    rows, gates, reports = [], [], []
    for r in range(cfg.audit.repeats):
        seed = cfg.seed + r
        tr, ho = split_holdout(data, schema, cfg, seed)
        train, holdout = data.iloc[tr].reset_index(drop=True), data.iloc[ho].reset_index(drop=True)
        n = cfg.audit.num_samples or len(train)
        log(f"audit repeat {r}: fitting TVAE on {len(train)} patients ({len(holdout)} held out), sampling {n} rows")
        with limit_cpu(cfg.n_jobs):
            synth = fit_and_sample(train, schema, pode_config, n, seed, cfg)
        rep_dir = out / f"repeat_{r}"
        rep_dir.mkdir(exist_ok=True)
        split = pd.DataFrame({"row": np.r_[tr, ho], "role": ["train"] * len(tr) + ["holdout"] * len(ho)})
        if ids is not None:
            split["id"] = ids[split["row"]]
        split.sort_values("row").to_csv(rep_dir / "split.csv", index=False)
        save_dataset(synth, rep_dir / "synthetic.csv")
        rcfg = copy.deepcopy(cfg)
        rcfg.seed = seed
        report = assess_privacy(train, synth, holdout, schema, rcfg, train_ids=None if ids is None else ids[tr],
                                log=log)
        report.save(rep_dir, title=f"Privacy audit - repeat {r}")
        log(f"audit repeat {r}: {report.verdict}")
        reports.append(report)
        rows.append({"repeat": r, "seed": seed, "verdict": report.verdict,
                     **{k: v for k, v in report.summary.items() if isinstance(v, (int, float, np.number))
                        and not isinstance(v, bool)}})
        gates.append(report.gate.assign(repeat=r))

    table = pd.DataFrame(rows)
    table.to_csv(out / "repeats.csv", index=False)
    overall = max((rep.verdict for rep in reports), key=lambda v: ORDER[v])
    numeric = table.drop(columns=["repeat", "seed", "verdict"]).apply(pd.to_numeric, errors="coerce")
    agg = numeric.agg(["mean", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
    agg.to_csv(out / "repeats_summary.csv", index=False)
    gate_all = pd.concat(gates, ignore_index=True)
    gate_all = gate_all[gate_all["status"] != "MANUAL"]
    pivot = gate_all.groupby(["check", "repeat"])["status"].agg(
        lambda s: max(s, key=lambda v: ORDER[v])).unstack("repeat")
    pivot.columns = [f"repeat_{c}" for c in pivot.columns]
    pivot = pivot.reset_index()
    pivot.to_csv(out / "gate_by_repeat.csv", index=False)
    _write_audit_report(out, overall, table, agg, pivot, reports[0], cfg)
    with open(out / "audit_summary.json", "w") as f:
        import json

        json.dump(jsonable({"verdict": overall, "repeats": rows}), f, indent=2)
    cfg.to_yaml(out / "privacy_config_used.yaml")
    return {"verdict": overall, "repeats": table, "summary": agg, "reports": reports}


KEY_METRICS = ["exact_copies", "dcr_p5_ratio", "near_copy_excess_p5", "membership_auc", "membership_auc_ci_high",
               "membership_auc_max", "attribute_member_gap_max", "linkage_excess_max", "rare_combo_leaks",
               "outliers_replicated"]


def _write_audit_report(out: Path, overall: str, table, agg, pivot, first, cfg) -> None:
    from .report import VERDICT_TEXT

    tm = first.threat_model
    lines = ["# TVAE privacy audit", "", f"**Verdict: {overall}** (worst of {len(table)} repeat(s)) - "
             f"{VERDICT_TEXT[overall]}", "",
             f"Each repeat held out {cfg.audit.holdout_frac:.0%} of the {tm['n_train'] + tm['n_holdout']} real "
             f"patients, fitted a fresh TVAE on the rest with the PODE configuration, generated "
             f"{tm['n_synthetic']} rows and assessed them against both groups. Full per-repeat reports: "
             + ", ".join(f"[repeat_{r}](repeat_{r}/report.md)" for r in table["repeat"]) + ".", "",
             "The audit measures the *model and configuration*. A release generated from all patients by the "
             "same configuration carries at least this risk (more training data per patient usually lowers "
             "it; more epochs raise it).", "",
             "## Gate status by repeat", "", _table(pivot), "## Key metrics", ""]
    keep = [m for m in KEY_METRICS if m in table.columns]
    lines += [_table(table[["repeat", "verdict", *keep]])]
    if len(table) > 1:
        lines += ["## Across repeats", "", _table(agg[agg["metric"].isin(keep)])]
    (out / "report.md").write_text("\n".join(lines) + "\n")
