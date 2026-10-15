"""Markdown summary of an imputation benchmark run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df.empty:
        return "_(none)_\n"

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


KEY_COLUMNS = [
    ("cont_nrmse", "NRMSE ↓"),
    ("cont_crps_norm", "CRPS/SD ↓"),
    ("cont_coverage90", "Cov90 (→0.90)"),
    ("cont_w1_norm", "W1/SD ↓"),
    ("disc_accuracy", "Acc ↑"),
    ("disc_log_score", "LogS ↓"),
    ("disc_brier", "Brier ↓"),
    ("spearman_mad", "|ΔSpearman| ↓"),
    ("cox_median_abs_bias", "Cox median |bias| ↓"),
    ("cox_abs_bias", "Cox mean |bias| ↓"),
    ("cox_truth_coverage", "Cox cov (→0.95)"),
    ("cox_fmi", "FMI"),
]


def write_markdown_report(tables: dict[str, pd.DataFrame], report: dict, path: str | Path) -> None:
    summary = tables["summary"]
    lines = [
        "# Imputation benchmark",
        "",
        f"Data: {report['data']['rows']} rows × {report['data']['columns']} columns. "
        f"Mode: `{report['config']['mode']}`, M = {report['config']['m']} draws, "
        f"folds = {report['config']['folds']}, repeats = {report['config']['repeats']}.",
        "",
        "Metrics are computed on benchmark-hidden cells only. Continuous metrics are "
        "normalised by the column SD and averaged over columns; discrete metrics are "
        "averaged over columns. Coverage should sit near its nominal level: well below it "
        "means over-confident imputations (typical of single imputation).",
        "",
    ]
    if not summary.empty:
        for scen, g in summary.groupby("scenario", sort=False):
            avail = [(k, label) for k, label in KEY_COLUMNS if k in g.columns]
            ranked = [k for k in ("cont_crps_norm", "disc_log_score") if k in g.columns and g[k].notna().any()]
            sort_key = ranked[0] if ranked else "engine"
            t = g.sort_values(sort_key)[["engine", *[k for k, _ in avail]]]
            t.columns = ["engine", *[label for _, label in avail]]
            lines += [f"## Scenario `{scen}`", "", _md_table(t), ""]

    tip = tables.get("delta_tipping", pd.DataFrame())
    if not tip.empty:
        fragile = tip[~tip["robust_over_grid"]]
        lines += [
            "## MNAR delta sensitivity",
            "",
            f"{len(fragile)} of {len(tip)} pooled Cox terms change significance or direction "
            "within the delta grid. Smallest tipping deltas:",
            "",
            _md_table(fragile.sort_values("tipping_delta", key=np.abs).head(20)),
            "",
        ]

    leak = report.get("leakage_settings") or {}
    if leak:
        rows = [{"engine": e, "outcome_policy": v["outcome_policy"],
                 "outcome_columns": ", ".join(v["outcome_columns"]) or "-",
                 "excluded_columns": ", ".join(v["excluded_columns"]) or "-",
                 "fit_scope": v.get("fit_scope") or "-"} for e, v in leak.items()]
        lines += [
            "## Leakage settings",
            "",
            "What each engine was allowed to see. `use` lets outcomes inform imputations "
            "(appropriate for inference); `hide`/`exclude` keep a row's own outcome away from "
            "its imputed covariates (needed for honest prediction validation).",
            "",
            _md_table(pd.DataFrame(rows)),
            "",
        ]

    diag = tables.get("diagnostics", pd.DataFrame())
    if not diag.empty:
        cols = [c for c in ["fit_seconds", "impute_seconds", "ess_median", "frac_ess_below_m", "frac_refined", "mcmc_acceptance_median", "smc_stages_median"] if c in diag]
        lines += ["## Engine diagnostics", "", _md_table(diag.groupby("engine")[cols].mean().reset_index()), ""]

    if report.get("failures"):
        lines += ["## Failures", "", _md_table(pd.DataFrame(report["failures"])), ""]

    lines += [
        "## Caveats",
        "",
        "- Masks are synthetic. MCAR/MAR results do not validate behaviour under real MNAR "
        "missingness; use the delta analysis to gauge sensitivity.",
        "- `tvae-decoder` is trained on complete rows only (SDV mean-fills NaNs); "
        "`tvae-masked` trains on incomplete rows directly.",
        "- Single-imputation engines (mean_mode, knn, missforest) give M identical draws, "
        "so their Rubin between-imputation variance is zero by construction.",
        "",
    ]
    Path(path).write_text("\n".join(lines))
