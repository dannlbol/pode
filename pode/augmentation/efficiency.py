"""How much is augmentation worth? Paired gains, real-data equivalents,
dose-response and compute cost.

**Real-data equivalent.** The real-only learning curve (each split refitted on
50%, 75% and 100% of its real training rows) is summarised as
``metric ≈ a + b·log(n_real)``, with ``b`` estimated *within* splits (each
split's points centred on their own mean, so between-split differences in
difficulty do not masquerade as a size effect) and given a standard error. An
augmented arm whose metric beats real-only by Δ is then "worth"
``n_equiv = n_real · exp(Δ / b)`` real patients, and

    efficiency = (n_equiv − n_real) / n_synthetic

is the number of real patients each synthetic row is worth: 0.1 means ten
synthetic rows buy as much as one more real patient; negative means the
synthetic rows cost performance (like discarding real patients).

The conversion divides by ``b``, so it is only reported when the learning curve
rises distinguishably from flat (one-sided 95%: ``b − 1.645·SE > 0``), and its
CI combines the Δ CI with the slope's uncertainty. When more
real data does not measurably help, "how many real patients is it worth" has no
stable answer and the row is flagged instead. Because a gain over real-only
always lies beyond the observed curve, ``n_equiv`` is an extrapolation;
``extrapolation_x`` (= n_equiv / n_real) above 2 is flagged too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import corrected_cv_ci

# Metrics where higher is better; the rest (Brier-type, calibration error) are lower-is-better.
HIGHER_BETTER = {"c_harrell", "c_uno", "auc_h", "ipa_h", "nb_mean", "auprc_minority"}
# cal_slope / oe_ratio are ideal at 1, so "better" is judged on their errors
# (cal_slope_err = |slope - 1|, oe_log_err = |log O/E|) and left blank for the raw values.
PAIRED_METRICS = ["c_harrell", "c_uno", "auc_h", "ipa_h", "brier_h", "ibs", "cal_slope", "cal_slope_err",
                  "oe_ratio", "oe_log_err", "cal_abs_err", "nb_mean", "auprc_minority"]
NO_DIRECTION = {"cal_slope", "oe_ratio"}
KEYS = ["model", "strategy", "ratio"]


def split_means(results: pd.DataFrame) -> pd.DataFrame:
    """Average draws within each (split, model, arm, strategy, ratio, fraction)."""
    keys = ["split", "model", "arm", "strategy", "ratio", "fraction"]
    num = [c for c in results.columns if c not in keys and pd.api.types.is_numeric_dtype(results[c])]
    return results.groupby(keys, dropna=False)[num].mean().reset_index()


def paired_differences(results: pd.DataFrame) -> pd.DataFrame:
    """Augmented minus real-only, paired within split, with corrected CV CIs."""
    internal = results[results["evaluation"] == "internal"]
    if internal.empty:
        return pd.DataFrame()
    sm = split_means(internal)
    base = sm[sm["arm"] == "real_only"].set_index(["split", "model"])
    aug = sm[sm["arm"] == "augmented"]
    rows = []
    for (model, strategy, ratio), g in aug.groupby(KEYS):
        g = g[[(s, model) in base.index for s in g["split"]]]
        if g.empty:
            continue
        b = base.loc[[(s, model) for s in g["split"]]]
        n_train, n_test = b["n_real"].mean(), b["n_test"].mean()
        for metric in PAIRED_METRICS:
            if metric not in g or g[metric].isna().all():
                continue
            d = g[metric].to_numpy(float) - b[metric].to_numpy(float)
            ci = corrected_cv_ci(d, n_train, n_test)
            if metric in NO_DIRECTION:
                ci["p_better"] = np.nan
            elif metric not in HIGHER_BETTER and np.isfinite(ci["p_better"]):
                ci["p_better"] = 1 - ci["p_better"]
            rows.append({"model": model, "strategy": strategy, "ratio": ratio, "metric": metric,
                         "real_only": float(np.nanmean(b[metric])), "augmented": float(np.nanmean(g[metric])),
                         "delta": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                         "p_better": ci["p_better"], "n_splits": ci["n_splits"],
                         "n_real": n_train, "n_synthetic": float(g["n_added"].mean())})
    return pd.DataFrame(rows)


def learning_curve(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per model: mean metric against mean real training size (incl. real-only)."""
    internal = results[(results["evaluation"] == "internal") & results["arm"].isin(["real_only", "learning_curve"])]
    if internal.empty or metric not in internal:
        return pd.DataFrame()
    sm = split_means(internal)
    return (sm.groupby(["model", "fraction"])
              .agg(n_real=("n_real", "mean"), value=(metric, "mean"), sd=(metric, "std"))
              .reset_index())


def curve_slope(results: pd.DataFrame, metric: str, model: str) -> dict:
    """Within-split least-squares slope of ``metric`` on log(n_real), with its SE."""
    internal = results[(results["evaluation"] == "internal") & (results["model"] == model)
                       & results["arm"].isin(["real_only", "learning_curve"])]
    if internal.empty or metric not in internal:
        return {"slope": np.nan, "se": np.nan}
    sm = split_means(internal)
    sm = sm[np.isfinite(sm[metric]) & (sm["n_real"] > 0)]
    x = np.log(sm["n_real"].to_numpy(float))
    y = sm[metric].to_numpy(float)
    g = sm["split"].to_numpy()
    xc = x - pd.Series(x).groupby(g).transform("mean").to_numpy()
    yc = y - pd.Series(y).groupby(g).transform("mean").to_numpy()
    sxx = float(np.sum(xc ** 2))
    dof = len(y) - sm["split"].nunique() - 1
    if sxx <= 0 or dof < 1:
        return {"slope": np.nan, "se": np.nan}
    b = float(np.sum(xc * yc) / sxx)
    resid = yc - b * xc
    return {"slope": b, "se": float(np.sqrt(np.sum(resid ** 2) / dof / sxx))}


def real_data_equivalent(paired: pd.DataFrame, results: pd.DataFrame,
                         metrics=("c_harrell", "ibs")) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        sign = 1.0 if metric in HIGHER_BETTER else -1.0
        for model in paired["model"].unique():
            fit = curve_slope(results, metric, model)
            b, se = sign * fit["slope"], fit["se"]
            p = paired[(paired["model"] == model) & (paired["metric"] == metric)]
            for _, r in p.iterrows():
                row = {"model": model, "strategy": r["strategy"], "ratio": r["ratio"], "metric": metric,
                       "curve_slope": b, "curve_slope_se": se, "doubling_gain": b * np.log(2),
                       "n_real": r["n_real"], "n_synthetic": r["n_synthetic"]}
                if not np.isfinite(b) or not np.isfinite(se) or b - 1.645 * se <= 0:
                    row.update(n_equiv=np.nan, efficiency=np.nan,
                               flag=f"flat learning curve (doubling the real data: {b * np.log(2):+.4f} "
                                    f"± {se * np.log(2):.4f}); real-data equivalent undefined")
                else:
                    def equiv(delta, slope=b):
                        return r["n_real"] * np.exp(sign * delta / slope)
                    n_eq = equiv(r["delta"])
                    eff = lambda ne: (ne - r["n_real"]) / r["n_synthetic"] if r["n_synthetic"] else np.nan  # noqa: E731
                    # The CI carries both the Δ CI and the slope's (90%) range, which is
                    # strictly positive here: a shallow curve inflates every equivalent.
                    corners = [eff(equiv(d, sl)) for d in (r["ci_low"], r["ci_high"])
                               for sl in (b - 1.645 * se, b + 1.645 * se)]
                    lo, hi = float(np.nanmin(corners)), float(np.nanmax(corners))
                    row.update(n_equiv=n_eq, real_equiv_gain=n_eq - r["n_real"], efficiency=eff(n_eq),
                               efficiency_ci_low=lo, efficiency_ci_high=hi,
                               extrapolation_x=n_eq / r["n_real"],
                               flag="extrapolated >2x" if n_eq / r["n_real"] > 2 else "")
                rows.append(row)
    return pd.DataFrame(rows)


def learning_curve_slopes(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per model: how much doubling the real training data changes ``metric``."""
    sign = 1.0 if metric in HIGHER_BETTER else -1.0
    rows = []
    for model in sorted(results["model"].unique()):
        fit = curve_slope(results, metric, model)
        rows.append({"model": model, "metric": metric, "doubling_gain": fit["slope"] * np.log(2),
                     "doubling_gain_se": fit["se"] * np.log(2),
                     # Direction-aware: for IBS "improves" means the error falls.
                     "improves": bool(np.isfinite(fit["se"]) and sign * fit["slope"] - 1.645 * fit["se"] > 0)})
    return pd.DataFrame(rows)


def dose_response(paired: pd.DataFrame, metric: str = "c_harrell") -> pd.DataFrame:
    """Best ratio per (model, strategy) and whether gains plateau or reverse."""
    rows = []
    p = paired[paired["metric"] == metric]
    for (model, strategy), g in p.groupby(["model", "strategy"]):
        g = g.sort_values("ratio")
        best = g.loc[g["delta"].idxmax()]
        half_width = (best["ci_high"] - best["ci_low"]) / 2 if np.isfinite(best["ci_high"]) else 0.0
        after = g[g["ratio"] > best["ratio"]]
        if after.empty:
            shape = "still rising at the largest ratio" if len(g) > 1 else "single ratio"
        elif (after["delta"] < best["delta"] - half_width).any():
            shape = "reverses beyond the optimum"
        else:
            shape = "plateaus"
        rows.append({"model": model, "strategy": strategy, "metric": metric,
                     "best_ratio": best["ratio"], "best_delta": best["delta"],
                     "best_ci_low": best["ci_low"], "best_ci_high": best["ci_high"], "shape": shape,
                     "deltas_by_ratio": "; ".join(f"{r:g}x: {d:+.3f}" for r, d in zip(g["ratio"], g["delta"]))})
    return pd.DataFrame(rows)


def compute_cost(timings: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Wall time per (model, strategy, ratio) relative to the real-only fit.

    ``generation_s_per_split`` is what producing that augmented training set
    costs in practice: one generator fit (TVAE strategies) plus the strategy's
    pool sampling.
    """
    if results.empty:
        return pd.DataFrame()
    internal = results[results["evaluation"] == "internal"]
    fit = internal.groupby(["model", "arm", "strategy", "ratio"])["fit_seconds"].mean().reset_index()
    base = fit[fit["arm"] == "real_only"].set_index("model")["fit_seconds"]
    gen = timings[timings["stage"] == "generator_fit"]
    pool = timings[timings["stage"] == "pool"]
    gen_per_split = gen["seconds"].mean() if not gen.empty else 0.0
    pool_s = pool.groupby("strategy")["seconds"].mean() if not pool.empty else pd.Series(dtype=float)
    tvae = set(pool[pool["tvae"]]["strategy"]) if not pool.empty else set()
    rows = []
    for _, r in fit[fit["arm"] == "augmented"].iterrows():
        g = (gen_per_split if r["strategy"] in tvae else 0.0) + float(pool_s.get(r["strategy"], 0.0))
        b = float(base.get(r["model"], np.nan))
        rows.append({"model": r["model"], "strategy": r["strategy"], "ratio": r["ratio"],
                     "fit_s": r["fit_seconds"], "real_only_fit_s": b, "generation_s_per_split": g,
                     "relative_cost": (r["fit_seconds"] + g) / b if b else np.nan})
    return pd.DataFrame(rows)
