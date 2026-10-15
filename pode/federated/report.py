"""Summaries, scorecard and ``report.md`` for the federated generator benchmark.

Reference points used throughout (the report repeats them next to the numbers):

* **C-index**: 0.5 = no discrimination, 0.6 modest, 0.7 good. A paired ΔC of
  +0.02 is a small but real gain; the CI tells whether it is distinguishable
  from zero.
* **Discriminator AUC** (real vs synthetic): 0.5 = indistinguishable, > 0.8 =
  synthetic rows easily spotted.
* **Membership AUC**: 0.5 = an attacker cannot tell who was in the training
  data; > 0.6 = worrying (the privacy gate's FAIL threshold).
* **Epsilon** (DP): ~1 strong, 3-8 common for useful models, > 10 weak.

Verdicts per generator condition (first downstream model):

* ``UNSAFE`` - the privacy gate FAILed at some site in some repeat;
* ``RECOMMENDED`` - non-inferior to the pooled (central) TVAE *and* not worse
  than each site's own local TVAE, with acceptable fidelity;
* ``NON-INFERIOR`` - non-inferior to the pooled TVAE only;
* ``NOT RECOMMENDED`` - otherwise;
* ``BASELINE`` - the local and central comparators themselves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FederatedConfig

ORDER = {"PASS": 0, "NOT_ASSESSED": 0, "REVIEW": 1, "FAIL": 2}
BASELINES = ("local", "central_fed_tvae", "central_sdv_tvae", "central_copula")


def _ci(diffs, cfg: FederatedConfig) -> dict:
    from ..augmentation.metrics import corrected_cv_ci

    f = cfg.evaluation.holdout_frac
    return corrected_cv_ci(np.asarray(diffs, float), 1 - f, f)


def summarise(tables: dict, cfg: FederatedConfig) -> dict:
    res, pairs = tables["results"], tables["pairs"]
    if res.empty:
        return {}
    first = res["model"].iloc[0]
    within = res[res["arm"] == "within_site"]
    num = within.select_dtypes("number").columns.drop(["repeat"], errors="ignore")
    by_cond = within.groupby(["condition", "model"])[list(num)].mean(numeric_only=True).reset_index()
    by_site = within.groupby(["condition", "model", "site"])[["c_harrell", "c_uno", "auc_h", "ipa_h"]] \
        .mean().reset_index() if "ipa_h" in within else \
        within.groupby(["condition", "model", "site"])[["c_harrell"]].mean().reset_index()
    deltas = []
    if not pairs.empty:
        for (cond, ref, model), g in pairs.groupby(["condition", "reference", "model"]):
            ci = _ci(g["delta_c"], cfg)
            deltas.append({"condition": cond, "reference": ref, "model": model, "delta_c": ci["mean"],
                           "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n": len(g),
                           "worst_site_delta": g.groupby("site")["delta_c"].mean().min()})
    deltas = pd.DataFrame(deltas)
    privacy = []
    if "priv_verdict" in within:
        for cond, g in within[within["model"] == first].groupby("condition"):
            v = g["priv_verdict"].dropna()
            if v.empty:
                continue
            privacy.append({"condition": cond, "worst_verdict": max(v, key=lambda x: ORDER.get(x, 0)),
                            "fail_rate": float((v == "FAIL").mean()), "review_rate": float((v == "REVIEW").mean()),
                            "membership_auc_mean": pd.to_numeric(g.get("priv_membership_auc"), errors="coerce").mean(),
                            "membership_auc_max": pd.to_numeric(g.get("priv_membership_auc"), errors="coerce").max(),
                            "dcr_p5_ratio_min": pd.to_numeric(g.get("priv_dcr_p5_ratio"), errors="coerce").min(),
                            "exact_copies_max": pd.to_numeric(g.get("priv_exact_copies"), errors="coerce").max(),
                            "failed_checks": ";".join(sorted({c for x in g.get("priv_failed_checks", pd.Series())
                                                              .dropna() for c in str(x).split(";") if c}))})
    privacy = pd.DataFrame(privacy)
    score = scorecard(by_cond[by_cond["model"] == first], deltas[deltas["model"] == first] if len(deltas) else deltas,
                      privacy, tables.get("generators", pd.DataFrame()), cfg)
    return {"summary": by_cond, "by_site": by_site, "deltas": deltas, "privacy": privacy, "scorecard": score}


def _get(deltas, cond, ref):
    if deltas is None or deltas.empty:
        return None
    d = deltas[(deltas["condition"] == cond) & (deltas["reference"] == ref)]
    return None if d.empty else d.iloc[0]


def scorecard(summary, deltas, privacy, gens, cfg: FederatedConfig) -> pd.DataFrame:
    m = cfg.evaluation.non_inferiority_margin
    rows = []
    for _, r in summary.iterrows():
        cond = r["condition"]
        if cond in ("real_local", "real_pooled", "real_federated_cox"):
            continue
        p = privacy[privacy["condition"] == cond].iloc[0] if len(privacy) and (privacy["condition"] == cond).any() \
            else None
        vc, vl, vr = _get(deltas, cond, "central_fed_tvae"), _get(deltas, cond, "local"), _get(deltas, cond, "real_local")
        disc = r.get("fid_discriminator_auc", np.nan)
        ni_central = vc is not None and vc["ci_low"] > -m
        ok_local = vl is not None and vl["ci_low"] > -m
        fid_ok = not np.isfinite(disc) or disc <= cfg.evaluation.max_discriminator_auc
        if p is not None and p["worst_verdict"] == "FAIL":
            verdict = "UNSAFE"
        elif cond in BASELINES:
            verdict = "BASELINE"
        elif ni_central and ok_local and fid_ok:
            verdict = "RECOMMENDED"
        elif ni_central:
            verdict = "NON-INFERIOR"
        else:
            verdict = "NOT RECOMMENDED"
        g = gens[gens["condition"] == cond] if len(gens) else pd.DataFrame()
        rows.append({
            "condition": cond, "verdict": verdict, "c_harrell": r.get("c_harrell"),
            "delta_vs_real_local": None if vr is None else vr["delta_c"],
            "delta_vs_local_tvae": None if vl is None else vl["delta_c"],
            "delta_vs_local_ci": None if vl is None else f"[{vl['ci_low']:.3f}, {vl['ci_high']:.3f}]",
            "delta_vs_central_tvae": None if vc is None else vc["delta_c"],
            "delta_vs_central_ci": None if vc is None else f"[{vc['ci_low']:.3f}, {vc['ci_high']:.3f}]",
            "discriminator_auc": disc,
            "membership_auc_max": None if p is None else p["membership_auc_max"],
            "privacy_worst": None if p is None else p["worst_verdict"],
            "privacy_fail_rate": None if p is None else p["fail_rate"],
            "epsilon_max": g["epsilon_max"].max() if "epsilon_max" in g else None,
            "seconds": g["seconds"].mean() if "seconds" in g else None,
            "mb_communicated": (g["bytes"].mean() / 1e6) if "bytes" in g and g["bytes"].notna().any() else None,
        })
    return pd.DataFrame(rows)


def _fmt(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}"
    return str(v)


def _plots(tables: dict, out: Path) -> list[str]:
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pode-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    pdir = out / "plots"
    pdir.mkdir(exist_ok=True)
    blue, ink, muted, grid = "#2a78d6", "#1f1f1e", "#6b6a64", "#e4e3dc"
    d = tables.get("deltas")
    if d is not None and len(d):
        first = d["model"].iloc[0]
        d = d[(d["reference"] == "real_local") & (d["model"] == first)].sort_values("delta_c")
        if len(d):
            fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(d) + 1.4))
            y = np.arange(len(d))
            ax.hlines(y, d["ci_low"], d["ci_high"], color=blue, lw=2)
            ax.plot(d["delta_c"], y, "o", color=blue, ms=8, mec="white", mew=2)
            ax.axvline(0, color=muted, lw=1)
            ax.set_yticks(y, d["condition"], color=ink)
            ax.set_xlabel("Paired ΔC vs a model trained on the site's own real rows (95% CI)", color=ink)
            ax.grid(axis="x", color=grid, lw=0.8)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.set_title("Utility on real held-out patients (TSTR)", loc="left", color=ink)
            fig.tight_layout()
            fig.savefig(pdir / "utility_delta.png", dpi=150)
            plt.close(fig)
            made.append("plots/utility_delta.png")
    sc = tables.get("scorecard")
    if sc is not None and len(sc) and sc["membership_auc_max"].notna().any():
        s = sc.dropna(subset=["membership_auc_max", "c_harrell"])
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.axvspan(0.6, max(0.62, float(s["membership_auc_max"].max()) + 0.02), color="#f6e3dc", lw=0)
        ax.plot(s["membership_auc_max"], s["c_harrell"], "o", color=blue, ms=8, mec="white", mew=2)
        for _, r in s.iterrows():
            ax.annotate(r["condition"], (r["membership_auc_max"], r["c_harrell"]), xytext=(6, 4),
                        textcoords="offset points", fontsize=8, color=ink)
        ax.set_xlabel("Worst-site membership AUC (0.5 = no leakage; shaded = gate FAIL)", color=ink)
        ax.set_ylabel("Mean TSTR C-index", color=ink)
        ax.grid(color=grid, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title("Privacy vs utility", loc="left", color=ink)
        fig.tight_layout()
        fig.savefig(pdir / "privacy_utility.png", dpi=150)
        plt.close(fig)
        made.append("plots/privacy_utility.png")
    return made


def write_report(tables: dict, cfg: FederatedConfig, meta: dict, out: Path) -> Path:
    plots = _plots(tables, out)
    sc = tables.get("scorecard", pd.DataFrame())
    ep = meta["endpoint"]
    L = ["# Federated TVAE benchmark", "",
         f"Sites: {', '.join(f'{k} (n={v})' for k, v in meta['sites'].items())}. "
         f"Endpoint `{ep['time']}` / `{ep['event']}`, horizon {ep['horizon']:.3g}. "
         f"{cfg.evaluation.repeats} repeat(s), {cfg.evaluation.holdout_frac:.0%} of each site held out as real test "
         f"patients. Federation: {cfg.generator.rounds} rounds x {cfg.generator.local_epochs} local epochs, "
         f"weighting `{cfg.generator.weighting}`, secure aggregation {'on' if cfg.generator.secure_aggregation else 'off'}.",
         "", "Every utility number is measured on **real held-out patients of each site**; synthetic rows are only "
         "ever training data. C-index: 0.5 = chance, 0.6 modest, 0.7 good. Discriminator AUC: 0.5 = synthetic "
         "rows indistinguishable from real, > 0.8 easily spotted. Membership AUC: 0.5 = no leakage, > 0.6 = gate "
         "FAIL. Epsilon: ~1 strong, 3-8 typical, > 10 weak.", "", "## Scorecard", "",
         "| condition | verdict | TSTR C | ΔC vs own real | ΔC vs local TVAE [CI] | ΔC vs pooled TVAE [CI] | "
         "discriminator | membership AUC (worst) | privacy | ε (max) | MB sent |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in sc.iterrows():
        L.append(f"| {r['condition']} | **{r['verdict']}** | {_fmt(r['c_harrell'])} | {_fmt(r['delta_vs_real_local'])} | "
                 f"{_fmt(r['delta_vs_local_tvae'])} {r['delta_vs_local_ci'] or ''} | "
                 f"{_fmt(r['delta_vs_central_tvae'])} {r['delta_vs_central_ci'] or ''} | {_fmt(r['discriminator_auc'])} | "
                 f"{_fmt(r['membership_auc_max'])} | {r['privacy_worst'] or '-'} "
                 f"({_fmt(r['privacy_fail_rate'], 2)} FAIL) | {_fmt(r['epsilon_max'], 2)} | {_fmt(r['mb_communicated'], 1)} |")
    L += ["", "Verdicts: UNSAFE = privacy gate FAIL at some site; RECOMMENDED = non-inferior to the pooled TVAE "
          f"(margin {cfg.evaluation.non_inferiority_margin}) and not worse than each site's local TVAE; NON-INFERIOR = "
          "non-inferior to pooled only; BASELINE = local / central comparators.", ""]
    for p in plots:
        L += [f"![{Path(p).stem}]({p})", ""]
    bs = tables.get("by_site")
    if bs is not None and len(bs):
        first = bs["model"].iloc[0]
        piv = bs[bs["model"] == first].pivot_table(index="condition", columns="site", values="c_harrell")
        L += ["## C-index by evaluation site", "",
              "Small sites gain most if federation helps; compare `local` (own TVAE) with the federated rows.", "",
              "| condition | " + " | ".join(piv.columns) + " |", "|---|" + "---|" * len(piv.columns)]
        for cond, row in piv.iterrows():
            L.append(f"| {cond} | " + " | ".join(_fmt(v) for v in row) + " |")
        L.append("")
    pv = tables.get("privacy")
    if pv is not None and len(pv):
        L += ["## Privacy by condition", "",
              "Members = each site's training patients, non-members = its held-out patients.", "",
              "| condition | worst | FAIL rate | membership AUC mean / max | DCR p5 ratio (min) | exact copies | failed checks |",
              "|---|---|---|---|---|---|---|"]
        for _, r in pv.iterrows():
            L.append(f"| {r['condition']} | {r['worst_verdict']} | {_fmt(r['fail_rate'], 2)} | "
                     f"{_fmt(r['membership_auc_mean'])} / {_fmt(r['membership_auc_max'])} | {_fmt(r['dcr_p5_ratio_min'])} | "
                     f"{_fmt(r['exact_copies_max'], 0)} | {r['failed_checks'] or '-'} |")
        L.append("")
    lo = tables.get("loso")
    if lo is not None and len(lo):
        first = lo["model"].iloc[0]
        t = lo[lo["model"].isin([first, "federated_cox"])].groupby(["heldout_site", "condition"])["c_harrell"] \
            .mean().unstack("heldout_site")
        L += ["## Leave-one-site-out", "",
              "The held-out site contributed nothing to any generator or model; all of its patients are test "
              "patients. `real_pooled` = trained on the other sites' real rows.", "",
              "| condition | " + " | ".join(t.columns) + " |", "|---|" + "---|" * len(t.columns)]
        for cond, row in t.iterrows():
            L.append(f"| {cond} | " + " | ".join(_fmt(v) for v in row) + " |")
        L.append("")
    L += ["## Synthetic-data card", "",
          "- **Intended use:** development environment (code, phenotype definitions, feasibility, teaching, "
          "model pretraining / selection); final effect estimates and validation on real data, locally or federated.",
          "- **Prohibited use:** clinical inference or biomarker claims from synthetic rows alone; re-identification "
          "or linkage attempts.",
          "- **Known failure modes:** small sites and rare strata are where memorisation concentrates; a global "
          "generator can blur genuine site effects; DP noise costs utility most at small sites.",
          "- **Not covered:** leakage through model updates themselves (secure aggregation is simulated; DP-SGD is "
          "the protection that covers updates), and survival-aware or longitudinal generation.", ""]
    path = out / "report.md"
    path.write_text("\n".join(L) + "\n")
    return path
