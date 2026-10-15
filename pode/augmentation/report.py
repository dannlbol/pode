"""Markdown summary and reproducibility manifest of an augmentation benchmark."""

from __future__ import annotations

import datetime as _dt
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..imputation.report import _md_table
from .config import AugmentationConfig


def run_manifest(cfg: AugmentationConfig) -> dict:
    """Git commit, package versions and config, for reproducing a run."""
    import importlib

    versions = {}
    for mod in ("pode", "metacpg", "sdv", "torch", "sksurv", "lifelines", "sklearn", "numpy", "pandas"):
        try:
            versions[mod] = getattr(importlib.import_module(mod), "__version__", "?")
        except Exception:
            versions[mod] = None
    try:
        root = Path(__file__).resolve().parents[2]
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=10).stdout.strip())
    except Exception:  # pragma: no cover
        sha, dirty = None, None
    return {"created": _dt.datetime.now().isoformat(timespec="seconds"), "git_commit": sha,
            "git_dirty": dirty, "python": platform.python_version(), "versions": versions,
            "config": cfg.to_dict()}


def _ci(d, lo, hi) -> str:
    if not np.isfinite(d):
        return ""
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else f"{d:+.3f}"


def _bound(x: float, cap: float = 10.0) -> str:
    """CI bound for display; huge values only arise from near-flat curves."""
    return f"> +{cap:g}" if x > cap else f"< -{cap:g}" if x < -cap else f"{x:+.3f}"


def _efficiency_lines(tables: dict[str, pd.DataFrame], metric: str) -> list[str]:
    slopes = tables.get("learning_curve_slopes", pd.DataFrame())
    eff = tables.get("efficiency", pd.DataFrame())
    card = tables.get("scorecard", pd.DataFrame())
    out: list[str] = []
    if not slopes.empty:
        sl = slopes[slopes["metric"] == metric]
        out += [f"How much **doubling the real training data** changes {metric} (real-only fits on 50/75/100% "
                "of each split's real rows; slope estimated within splits, ± SE). A real-data equivalent is "
                "only defined for models whose curve rises distinguishably from flat; the shallower the curve, "
                "the more a small Δ is magnified, so read the CIs (they include the slope's uncertainty), not "
                "the point estimates:", "",
                _md_table(sl[["model", "doubling_gain", "doubling_gain_se", "improves"]], "{:+.4f}"), ""]
    if eff.empty:
        return out
    e = eff[(eff["metric"] == metric) & np.isfinite(eff.get("n_equiv", pd.Series(np.nan, index=eff.index)))]
    if e.empty:
        return out + ["No model's learning curve rises distinguishably from flat, so \"how many real patients is "
                      "a synthetic row worth\" has no stable answer here: more real data of the same kind does "
                      "not measurably help these models either.", ""]
    verdict = (card.set_index(["model", "strategy", "ratio"])["verdict"] if not card.empty else pd.Series(dtype=str))
    for model, g in e.groupby("model"):
        out.append(f"`{model}`, best three configurations:")
        for _, r in g.sort_values("efficiency", ascending=False).head(3).iterrows():
            flag = f"; {r['flag']}" if r.get("flag") else ""
            v = verdict.get((r["model"], r["strategy"], r["ratio"]), "-")
            lo, hi = r["efficiency_ci_low"], r["efficiency_ci_high"]
            reading = ("a real gain" if lo > 0 else "a real loss" if hi < 0 else "consistent with no effect")
            out.append(
                f"- `{r['strategy']}` at {r['ratio']:g}×: efficiency {r['efficiency']:+.3f} real patients per "
                f"synthetic row, CI {_bound(lo)} to {_bound(hi)}: **{reading}** (point estimate: "
                f"+{r['n_synthetic']:.0f} synthetic ≈ {r['real_equiv_gain']:+.0f} real{flag}); verdict {v}.")
        out.append("")
    return out


def write_markdown_report(tables: dict[str, pd.DataFrame], report: dict, path: str | Path) -> None:
    cfg = report["config"]
    pm = cfg["gates"]["primary_metric"]
    d = report["data"]
    card = tables.get("scorecard", pd.DataFrame())
    lines = [
        "# Augmentation benchmark",
        "",
        f"Development cohort: {d['development_rows']} patients ({d['development_events']} events)"
        + (f"; external cohort: {d['external_rows']} patients ({d['external_events']} events)."
           if d["external_rows"] else "; no external cohort."),
        f"Internal validation: {cfg['splits']['folds']}-fold × {cfg['splits']['repeats']} repeats, stratified on "
        f"the event{(' + ' + ', '.join(cfg['splits']['stratify'])) if cfg['splits']['stratify'] else ''}. "
        f"A TVAE is fitted inside every split on its real training rows only; synthetic rows never enter a "
        f"test set. Ratios {cfg['ratios']} × the real training size, {cfg['draws']} draws each.",
        f"Endpoint `{report['columns']['time']}` / `{report['columns']['event']}`, horizon "
        f"{cfg['endpoint']['horizon']:g}. {len(report['columns']['features'])} model features; strata "
        f"{report['columns']['strata'] or 'none'} (synthetic rows: `{cfg['generator']['synthetic_strata']}` stratum).",
        "",
        "## How to read this",
        "",
        "- **ΔC** is the paired change in Harrell's C-index (augmented − real-only, same splits). 0.5 is a coin "
        "toss and 0.7 a useful prognostic score; +0.01–0.02 is a small but real gain. CIs use the "
        "Nadeau–Bengio correction for overlapping CV training sets.",
        "- **IBS** (integrated Brier score) and **Brier@h**: prediction error, lower is better. **IPA** rescales "
        "Brier like R² (0 = no better than the Kaplan–Meier average).",
        "- **Calibration slope** ideal 1 (< 1: predictions too extreme). **O/E** ideal 1.",
        "- **Efficiency**: real patients each synthetic row is worth, read off the real-only learning curve.",
        "- **Verdicts**: `RECOMMENDED` = clear gain with every gate passing; `NON-INFERIOR` = no reliable "
        "harm; `NOT RECOMMENDED` = a utility, calibration, subgroup or fidelity gate fails; `UNSAFE` = a "
        "privacy gate fails (TVAE strategies only).",
        "",
    ]

    if not card.empty:
        lines += ["## Scorecard", ""]
        for model, g in card.groupby("model", sort=False):
            t = pd.DataFrame({
                "strategy": g["strategy"], "ratio": g["ratio"].map("{:g}×".format),
                f"Δ{pm} [95% CI]": [_ci(a, b, c) for a, b, c in
                                     zip(g[f"{pm}:delta"], g[f"{pm}:ci_low"], g[f"{pm}:ci_high"])],
                "external ΔC [95% CI]": [_ci(a, b, c) for a, b, c in zip(
                    g.get("external:delta", pd.Series(np.nan, index=g.index)),
                    g.get("external:ci_low", pd.Series(np.nan, index=g.index)),
                    g.get("external:ci_high", pd.Series(np.nan, index=g.index)))],
                "ΔIBS": g.get("ibs:delta", pd.Series(np.nan, index=g.index)),
                "Δslope": g.get("cal_slope:delta", pd.Series(np.nan, index=g.index)),
                "worst subgroup ΔC": g.get("worst_subgroup_delta_c", pd.Series(np.nan, index=g.index)),
                "verdict": g["verdict"], "why": g["reasons"],
            })
            lines += [f"### Model `{model}`", "", _md_table(t), ""]
        if d["external_rows"]:
            lines += ["Internal and external results can disagree; the external cohort is the stronger test of "
                      "transportability and takes priority, but it is a single small cohort, so read its CIs.", ""]

    lines += ["## Efficiency (real-data equivalent)", "", *_efficiency_lines(tables, pm)]
    lc = tables.get("learning_curve", pd.DataFrame())
    if not lc.empty:
        lines += ["Real-only learning curve (mean over splits):", "",
                  _md_table(lc.rename(columns={"value": pm})), ""]

    dr = tables.get("dose_response", pd.DataFrame())
    if not dr.empty:
        lines += ["## Dose-response", "",
                  "Best augmentation ratio per strategy and whether gains plateau or reverse beyond it.", "",
                  _md_table(dr[["model", "strategy", "best_ratio", "best_delta", "best_ci_low", "best_ci_high",
                                "shape", "deltas_by_ratio"]]), ""]

    sel = tables.get("selection_stability", pd.DataFrame())
    if not sel.empty:
        lines += ["## Feature selection (models that choose their own features)", "",
                  "How augmentation changes *which* features are selected. `jaccard_vs_real_only`: overlap with "
                  "the real-only selection in the same split (1 = unchanged). `across_split_jaccard`: how "
                  "stable the selection is across splits (higher = more stable; compare with the real-only "
                  "row). `reference_overlap`: share of the reference signature selected.", "",
                  _md_table(sel), ""]
        freq = tables.get("selection_frequency", pd.DataFrame())
        if not freq.empty:
            ro = freq[freq["arm"] == "real_only"].sort_values("frequency", ascending=False)
            for model, g in ro.groupby("model"):
                lines += [f"Most often selected by real-only `{model}` across splits "
                          f"({int(g['in_reference'].head(35).sum())} of its top 35 are in the reference):", "",
                          _md_table(g.head(15)[["feature", "frequency", "in_reference"]]), ""]
    fid = tables.get("fidelity", pd.DataFrame())
    if not fid.empty:
        cols = [c for c in ["discriminator_auc", "tvd_mean", "ks_mean", "corr_gap", "km_diff_h", "event_time_ks",
                            "target_tvd_mean", "target_km_diff_h", "matrix_discriminator_auc", "matrix_mean_gap",
                            "matrix_sd_ratio", "matrix_corr_gap"] if c in fid]
        lines += ["## Fidelity of TVAE pools (mean over splits)", "",
                  _md_table(fid.groupby(["evaluation", "strategy"])[cols].mean().reset_index()), ""]
    pools = tables.get("pools", pd.DataFrame())
    if not pools.empty and "shortfall" in pools:
        p = pools.groupby("strategy")[[c for c in ("requested", "produced", "shortfall", "constraint_dropped")
                                       if c in pools]].mean().reset_index()
        lines += ["Pool generation (mean per split): a shortfall means the conditional target was too rare "
                  "for the generator to produce on request.", "", _md_table(p, "{:.1f}"), ""]
        if "produced" in pools:
            empty = pools[(pools["produced"] == 0) & (pools["kind"] != "weight")]
            if not empty.empty:
                names = ", ".join(f"`{n}` ({k} splits)" for n, k in empty["strategy"].value_counts().items())
                lines += [f"**Warning:** no usable rows for {names}. Those splits are skipped for that "
                          "strategy (see Failures). A TVAE that never emits a rare level usually needs more "
                          "epochs, or the target is too rare to model.", ""]
    priv = tables.get("privacy", pd.DataFrame())
    if not priv.empty:
        cols = [c for c in ["exact_duplicates", "near_copy_frac", "dcr_p5_ratio", "nndr_mean", "membership_auc",
                            "rare_combo_leaks"] if c in priv]
        lines += ["## Privacy of TVAE pools (mean over splits)", "",
                  "Internal splits only: each split's held-out patients are the never-seen reference, drawn "
                  "from the same cohorts as the generator's training rows. A DCR p5 ratio near 1 and a "
                  "membership AUC near 0.5 mean synthetic rows are no closer to training patients than new "
                  "patients are. (The external cohort is not a valid reference: it differs systematically, "
                  "which would make synthetic rows look spuriously close.)", "",
                  _md_table(priv.groupby(["evaluation", "strategy"])[cols].mean().reset_index()), ""]

    cost = tables.get("compute_cost", pd.DataFrame())
    if not cost.empty:
        c = cost.groupby(["model", "strategy"])[["fit_s", "real_only_fit_s", "generation_s_per_split",
                                                 "relative_cost"]].mean().reset_index()
        lines += ["## Compute cost", "",
                  f"Wall time per fit and per split's data generation; relative cost = (augmented fit + "
                  f"generation) / real-only fit. Peak GPU memory {report.get('gpu_peak_mib', 0):.0f} MiB; "
                  f"total run {report.get('elapsed_seconds', 0) / 60:.1f} min.", "", _md_table(c, "{:.2f}"), ""]

    ext = tables.get("external_results", pd.DataFrame())
    if not ext.empty:
        cols = [c for c in ["c_harrell", "c_uno", "auc_h", "ibs", "ipa_h", "cal_slope", "oe_ratio", "nb_mean",
                            "auprc_minority"] if c in ext]
        lines += ["## External cohort (mean over draws)", "",
                  _md_table(ext[["model", "arm", "strategy", "ratio", *cols]]), ""]

    if report.get("failures"):
        lines += ["## Failures", "", _md_table(pd.DataFrame(report["failures"])), ""]

    lines += [
        "## Caveats",
        "",
        "- Synthetic rows are training auxiliaries only. They never enter an evaluation set and give no "
        "evidence about incidence, prevalence or treatment effects.",
        "- If model features were selected on the full development cohort (e.g. a fixed 35-CpG "
        "signature), internal estimates are optimistic for **every** arm alike. Paired comparisons stay "
        "fair, but absolute values do not; the external cohort is unaffected.",
        "- Survival time and event are generated as ordinary TVAE columns (no censoring model). The KM and "
        "event-time checks in the fidelity table show whether that coupling survived.",
        "- Case weights (`weight`) only reach models that accept them (cox_ph, logistic_landmark).",
        "- Duplicate/jitter baselines are copies of real patients and are never subject to the privacy "
        "gates; they must never leave the training pipeline either.",
        "",
    ]
    Path(path).write_text("\n".join(lines))
