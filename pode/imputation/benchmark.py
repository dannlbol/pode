"""Masking benchmark: hide known values, impute them, score the imputations.

For each repeat, every scenario's mask is drawn once on the reference table.
In ``kfold`` mode each engine is fitted on the training folds and imputes the
masked held-out fold (no model ever sees a row it imputes); the imputed folds
are reassembled into M completed copies of the full table. In
``transductive`` mode each engine is fitted on the whole masked table and
imputes it (the hidden values are never visible).

Completed tables are then scored with :mod:`pode.imputation.metrics`.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import delayed

from ..config import PodeConfig
from ..schema.schema import DatasetSchema
from .base import finalize_dtypes, limit_cpu, n_workers, parallel_map, resolve_columns, subset_schema
from .config import ImputationConfig
from .engines import build_imputer
from .masking import MaskSpec, apply_mask, make_mask
from .metrics import (
    apply_delta,
    cell_metrics,
    compare_to_truth,
    dependence_metrics,
    fit_cox,
    pooled_cox,
)

CONT_SUMMARY = ["nmae", "nrmse", "crps_norm", "coverage50", "coverage90", "width90_norm", "w1_norm"]
DISC_SUMMARY = ["accuracy", "balanced_accuracy", "brier", "log_score", "tvd"]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _cox_covariates(df: pd.DataFrame, schema: DatasetSchema, icfg: ImputationConfig) -> list[str]:
    features = [c for c in schema.feature_columns if c in df.columns]
    covs = resolve_columns(icfg.cox.covariates, features) if icfg.cox.covariates else features
    covs = [c for c in covs if df[c].nunique(dropna=True) > 1]
    if icfg.cox.max_covariates:
        covs = covs[: icfg.cox.max_covariates]
    return covs


def run_benchmark(
    df: pd.DataFrame,
    schema: DatasetSchema,
    pode_config: PodeConfig,
    icfg: ImputationConfig,
    out_dir: str | Path,
    log: Callable[[str], None] = _log,
) -> dict:
    """Run the benchmark and write all tables/plots/reports into ``out_dir``.

    CPU use is capped at ``icfg.n_jobs`` cores: worker processes, BLAS/OpenMP
    thread pools and torch's CPU threads all respect the limit.
    """
    with limit_cpu(icfg.n_jobs):
        return _run_benchmark(df, schema, pode_config, icfg, out_dir, log)


def _run_benchmark(
    df: pd.DataFrame,
    schema: DatasetSchema,
    pode_config: PodeConfig,
    icfg: ImputationConfig,
    out_dir: str | Path,
    log: Callable[[str], None] = _log,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in schema.columns if c in df.columns]
    df = df[cols].reset_index(drop=True)
    schema = subset_schema(schema, cols)
    scenarios = [MaskSpec.from_dict(s) for s in icfg.scenarios]
    m = icfg.m

    pair = schema.primary_survival_pair()
    covs = _cox_covariates(df, schema, icfg) if (icfg.cox.enabled and pair) else []
    truth_fit = (
        fit_cox(df, covs, schema, pair.time_col, pair.event_col, df, icfg.cox.mode, icfg.cox.penalizer)
        if covs
        else pd.DataFrame()
    )

    cell_rows, dep_rows, cox_rows, delta_rows, diag_rows, failures = [], [], [], [], [], []
    # Per-engine record of what each imputer was allowed to see.
    leakage: dict[str, dict] = {}
    # For plots: pooled draws of the most-masked continuous column per scenario.
    distributions: dict[str, dict] = {}

    for rep in range(icfg.repeats):
        seed_r = icfg.seed + 1000 * rep
        masks = {
            s.name: make_mask(
                df, s, schema, np.random.default_rng(seed_r + i), icfg.protect_columns,
                icfg.protect_survival, icfg.block_groups, icfg.atomic_blocks,
            )
            for i, s in enumerate(scenarios)
        }
        for s in scenarios:
            log(f"[rep {rep}] scenario {s.name}: {int(masks[s.name].sum())} cells hidden")

        # parts[(scenario, engine)] -> list of per-fold draw lists
        parts: dict[tuple[str, str], list[list[pd.DataFrame]]] = {}
        failed: set[tuple[str, str]] = set()

        def record(scen: str, eng: str, fold: int, imp, fit_s: float, imp_s: float) -> None:
            if eng not in leakage:
                leakage[eng] = imp.leakage_settings()
            diag_rows.append({
                "repeat": rep, "scenario": scen, "engine": eng, "fold": fold,
                "fit_seconds": fit_s, "impute_seconds": imp_s,
                **{k: v for k, v in imp.diagnostics().items() if k not in ("fit_seconds", "impute_seconds")},
            })

        def fail(scen: str, eng: str, exc: Exception) -> None:
            if (scen, eng) not in failed:
                failures.append({"repeat": rep, "scenario": scen, "engine": eng, "error": f"{type(exc).__name__}: {exc}"})
                log(f"  ! {eng} failed on {scen}: {exc}")
                log(traceback.format_exc(limit=3))
            failed.add((scen, eng))

        if icfg.mode == "kfold":
            from sklearn.model_selection import KFold

            splits = list(KFold(icfg.folds, shuffle=True, random_state=seed_r).split(df))
            for fold, (tr, te) in enumerate(splits):
                for spec in icfg.engines:
                    eng = spec.key
                    fit_seed = seed_r + 31 * fold
                    shared = None
                    if not icfg.mask_training_rows:
                        try:
                            t0 = time.perf_counter()
                            shared = build_imputer(spec, pode_config, icfg, fit_seed).fit(df.iloc[tr], schema)
                            shared_fit = time.perf_counter() - t0
                        except Exception as exc:  # noqa: BLE001 - recorded, benchmark continues
                            for s in scenarios:
                                fail(s.name, eng, exc)
                            continue
                    for s in scenarios:
                        if (s.name, eng) in failed:
                            continue
                        try:
                            if shared is None:
                                t0 = time.perf_counter()
                                imp = build_imputer(spec, pode_config, icfg, fit_seed)
                                imp.fit(apply_mask(df.iloc[tr], masks[s.name][tr]), schema)
                                fit_s = time.perf_counter() - t0
                            else:
                                imp, fit_s = shared, shared_fit
                            t0 = time.perf_counter()
                            draws = imp.impute(apply_mask(df.iloc[te], masks[s.name][te]), m, seed_r + fold)
                            record(s.name, eng, fold, imp, fit_s, time.perf_counter() - t0)
                            parts.setdefault((s.name, eng), []).append(draws)
                        except Exception as exc:  # noqa: BLE001
                            fail(s.name, eng, exc)
                log(f"[rep {rep}] fold {fold + 1}/{len(splits)} done")
        elif icfg.mode == "transductive":
            for s in scenarios:
                masked = apply_mask(df, masks[s.name])
                for spec in icfg.engines:
                    eng = spec.key
                    try:
                        t0 = time.perf_counter()
                        imp = build_imputer(spec, pode_config, icfg, seed_r)
                        if imp.outcome_policy == "hide":
                            # Fitting on the rows being imputed would expose
                            # their outcomes through training: exclude instead.
                            imp.outcome_policy = "exclude"
                        imp.fit(masked, schema)
                        fit_s = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        draws = imp.impute(masked, m, seed_r)
                        record(s.name, eng, -1, imp, fit_s, time.perf_counter() - t0)
                        parts[(s.name, eng)] = [draws]
                    except Exception as exc:  # noqa: BLE001
                        fail(s.name, eng, exc)
                log(f"[rep {rep}] scenario {s.name} done")
        else:
            raise ValueError(f"Unknown benchmark mode '{icfg.mode}' (use 'kfold' or 'transductive').")

        # ---- evaluation ------------------------------------------------------
        # Each (scenario, engine) is scored independently (Cox pooling and the
        # delta refits dominate), so score them in parallel worker processes.
        jobs = []
        for (scen, eng), fold_draws in parts.items():
            if (scen, eng) in failed:
                continue
            completed = [
                finalize_dtypes(pd.concat([fd[i] for fd in fold_draws]).sort_index(), schema)
                for i in range(m)
            ]
            if rep == 0:
                _keep_distribution(distributions, scen, eng, df, completed, masks[scen], schema)
            ids = {"repeat": rep, "scenario": scen, "engine": eng}
            jobs.append((ids, completed, masks[scen]))
        log(f"[rep {rep}] scoring {len(jobs)} scenario x engine results on {n_workers(icfg.n_jobs)} cores")
        scored = parallel_map(
            [delayed(_score_pair)(ids, completed, mask, df, schema, icfg, covs, pair, truth_fit, seed_r)
             for ids, completed, mask in jobs],
            icfg.n_jobs,
        )
        for res in scored:
            cell_rows.extend(res["cell"])
            dep_rows.extend(res["dependence"])
            cox_rows.extend(res["cox"])
            delta_rows.extend(res["delta"])
        log(f"[rep {rep}] scoring done")

    tables = {
        "cell_metrics": pd.DataFrame(cell_rows),
        "dependence": pd.DataFrame(dep_rows),
        "rubin_cox": pd.DataFrame(cox_rows),
        "delta_sensitivity": pd.DataFrame(delta_rows),
        "diagnostics": pd.DataFrame(diag_rows),
        "failures": pd.DataFrame(failures),
    }
    tables["summary"] = summarise(tables)
    tables["delta_tipping"] = delta_tipping(tables["delta_sensitivity"], icfg.cox.alpha)
    for name, table in tables.items():
        if not table.empty:
            table.to_csv(out_dir / f"{name}.csv", index=False)

    report = {
        "data": {"rows": len(df), "columns": len(cols)},
        "config": icfg.to_dict(),
        "cox_covariates": covs,
        "leakage_settings": leakage,
        "summary": tables["summary"].to_dict("records"),
        "failures": failures,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    from .report import write_markdown_report

    write_markdown_report(tables, report, out_dir / "report.md")
    if icfg.plots:
        from .plots import write_plots

        try:
            write_plots(tables, out_dir / "plots", distributions, [e.key for e in icfg.engines])
        except Exception as exc:  # noqa: BLE001 - plots are best-effort
            log(f"warning: plotting failed: {exc}")
    return {"tables": tables, "report": report}


def _score_pair(ids, completed, mask, df, schema, icfg, covs, pair, truth_fit, seed) -> dict[str, list]:
    """All metrics for one (scenario, engine) set of completed tables."""
    out = {"cell": [], "dependence": [], "cox": [], "delta": []}
    out["cell"] = [{**ids, **row} for row in cell_metrics(df, completed, mask, schema)]
    out["dependence"] = [{**ids, **dependence_metrics(df, completed, schema)}]
    if covs and not truth_fit.empty:
        masked_per_col = dict(zip(df.columns, mask.sum(axis=0)))
        pooled = pooled_cox(completed, covs, schema, pair.time_col, pair.event_col, df,
                            icfg.cox.mode, icfg.cox.penalizer, icfg.cox.alpha)
        cmp = compare_to_truth(pooled, truth_fit, icfg.cox.alpha)
        if not cmp.empty:
            cmp["imputed_cells"] = cmp["covariate"].map(masked_per_col).fillna(0).astype(int)
            out["cox"] = [{**ids, **r} for r in cmp.to_dict("records")]
        if icfg.delta.enabled and (not icfg.delta.engines or ids["engine"] in icfg.delta.engines):
            out["delta"] = _delta_analysis(df, completed, mask, schema, icfg, covs, pair, ids, seed)
    return out


def _keep_distribution(store, scen, eng, df, completed, mask, schema) -> None:
    counts = {
        c: int(mask[:, j].sum())
        for j, c in enumerate(df.columns)
        if schema.types[c].is_numeric and mask[:, j].any()
    }
    if not counts:
        return
    if scen not in store:
        col = max(counts, key=counts.get)
        hidden = mask[:, list(df.columns).index(col)]
        store[scen] = {"column": col, "truth": pd.to_numeric(df[col]).to_numpy(float)[hidden], "draws": {}, "_hidden": hidden}
    entry = store[scen]
    entry["draws"][eng] = np.concatenate(
        [pd.to_numeric(d[entry["column"]], errors="coerce").to_numpy(float)[entry["_hidden"]] for d in completed]
    )


def _delta_analysis(df, completed, mask, schema, icfg, covs, pair, ids, seed) -> list[dict]:
    imputed = [c for j, c in enumerate(df.columns) if mask[:, j].any()]
    cols = resolve_columns(icfg.delta.columns, imputed) if icfg.delta.columns else imputed
    cols = [c for c in cols if schema.types[c].is_numeric or df[c].nunique(dropna=True) == 2]
    target_covs = [c for c in covs if c in cols]
    if not cols or not target_covs:
        return []
    rows = []
    for delta in icfg.delta.deltas:
        shifted = apply_delta(completed, mask, cols, schema, df, float(delta), np.random.default_rng(seed))
        pooled = pooled_cox(shifted, target_covs, schema, pair.time_col, pair.event_col, df,
                            icfg.cox.mode, icfg.cox.penalizer, icfg.cox.alpha)
        for r in pooled.to_dict("records"):
            rows.append({**ids, "delta": float(delta), "covariate": r["covariate"], "term": r["term"],
                         "coef": r["coef"], "hr": float(np.exp(r["coef"])), "ci_lo": r["ci_lo"],
                         "ci_hi": r["ci_hi"], "p": r["p"]})
    return rows


def summarise(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (scenario, engine), averaged over columns, covariates and repeats."""
    cell = tables["cell_metrics"]
    if cell.empty:
        return pd.DataFrame()
    keys = ["scenario", "engine"]
    is_cont = cell["type"].isin(["continuous", "survival_time"])
    parts = []
    cont = cell[is_cont]
    if not cont.empty:
        c = cont.copy()
        c["abs_bias_norm"] = c["bias_norm"].abs()
        parts.append(c.groupby(keys)[CONT_SUMMARY + ["abs_bias_norm"]].mean().add_prefix("cont_"))
    disc = cell[~is_cont]
    if not disc.empty:
        parts.append(disc.groupby(keys)[DISC_SUMMARY].mean().add_prefix("disc_"))
    parts.append(cell.groupby(keys)["n_masked"].sum().rename("cells_masked"))
    dep = tables["dependence"]
    if not dep.empty:
        parts.append(dep.groupby(keys)[["spearman_mad", "cramers_v_mad"]].mean())
    cox = tables["rubin_cox"]
    if not cox.empty:
        aff = cox[cox["imputed_cells"] > 0].copy()
        if not aff.empty:
            aff["abs_bias"] = aff["bias"].abs()
            g = aff.groupby(keys)
            parts.append(pd.DataFrame({
                "cox_abs_bias": g["abs_bias"].mean(),
                # Robust to a single collapsed term (e.g. a rare level losing its events).
                "cox_median_abs_bias": g["abs_bias"].median(),
                "cox_abs_hr_error": g["abs_hr_error"].mean(),
                "cox_truth_coverage": g["covers_truth"].mean(),
                "cox_ci_width_ratio": g["ci_width_ratio"].mean(),
                "cox_significance_agreement": g["significance_agrees"].mean(),
                "cox_fmi": g["fmi"].mean(),
            }))
    diag = tables["diagnostics"]
    if not diag.empty:
        cols = [c for c in ["fit_seconds", "impute_seconds", "ess_median", "frac_refined"] if c in diag]
        parts.append(diag.groupby(keys)[cols].mean())
    return pd.concat(parts, axis=1).reset_index()


def delta_tipping(delta: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Smallest |delta| at which a term's significance or direction changes."""
    if delta.empty:
        return pd.DataFrame()
    rows = []
    for key, g in delta.groupby(["scenario", "engine", "covariate", "term"]):
        g = g.groupby("delta", as_index=False).mean(numeric_only=True)
        base = g[g["delta"] == 0]
        if base.empty:
            continue
        b = base.iloc[0]
        sig0, sign0 = b["p"] < alpha, np.sign(b["coef"])
        flips = g[((g["p"] < alpha) != sig0) | (np.sign(g["coef"]) != sign0)]
        tip = flips.loc[flips["delta"].abs().idxmin()] if not flips.empty else None
        rows.append({
            "scenario": key[0], "engine": key[1], "covariate": key[2], "term": key[3],
            "hr_at_0": float(np.exp(b["coef"])), "p_at_0": float(b["p"]),
            "tipping_delta": float(tip["delta"]) if tip is not None else np.nan,
            "hr_at_tipping": float(np.exp(tip["coef"])) if tip is not None else np.nan,
            "robust_over_grid": tip is None,
        })
    return pd.DataFrame(rows)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
