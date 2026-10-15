"""Gated verdict per (model, strategy, ratio).

Gates, all configurable in ``gates:`` (defaults in :class:`.config.GateConfig`):

* **utility**     - paired ΔC (augmented − real-only) with a corrected repeated-CV
  CI: *improves* when the lower bound > 0, *non-inferior* when it is above
  −margin, *worse* when the whole CI is below 0, otherwise *inconclusive*;
* **calibration** - the calibration slope moves no further from 1 than the limit,
  and the integrated Brier score does not worsen beyond its limit (improvements
  never fail a gate);
* **subgroups**   - no audited subgroup loses more than the allowed C-index;
* **fidelity**    - synthetic rows not trivially distinguishable from real ones
  and, for targeted pools, the target stratum reproduced (TVAE strategies only);
* **privacy**     - no exact copies, no membership signal, no near-copies
  (TVAE strategies only: the duplicate/jitter baselines are copies of real
  patients by design and are never meant to leave the training pipeline).

Verdicts: ``UNSAFE`` (any privacy gate fails) > ``RECOMMENDED`` (improves and
every other gate passes) > ``NON-INFERIOR`` (non-inferior, calibration and
subgroups pass) > ``NOT RECOMMENDED``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AugmentationConfig


def _utility(lo: float, hi: float, margin: float, missing: str = "undetermined") -> str:
    """improves (CI above 0) / non-inferior (CI above -margin) / worse (CI below 0) / inconclusive."""
    if not np.isfinite(lo):
        return missing
    if lo > 0:
        return "improves"
    if lo > -margin:
        return "non-inferior"
    if np.isfinite(hi) and hi < 0:
        return "worse"
    return "inconclusive"


def _delta(paired: pd.DataFrame, metric: str) -> pd.DataFrame:
    p = paired[paired["metric"] == metric].set_index(["model", "strategy", "ratio"])
    return p[["delta", "ci_low", "ci_high"]].add_prefix(f"{metric}:")


def subgroup_deltas(subgroups: pd.DataFrame) -> pd.DataFrame:
    """Worst mean subgroup ΔC per (model, strategy, ratio), paired within split."""
    if subgroups.empty:
        return pd.DataFrame(columns=["model", "strategy", "ratio", "worst_subgroup", "worst_subgroup_delta_c"])
    sg = subgroups[subgroups["evaluation"] == "internal"]
    keys = ["split", "model", "subgroup", "level"]
    base = (sg[sg["arm"] == "real_only"].groupby(keys)["c_harrell"].mean().rename("base"))
    aug = sg[sg["arm"] == "augmented"].groupby([*keys, "strategy", "ratio"])["c_harrell"].mean().reset_index()
    aug = aug.join(base, on=keys)
    aug["d"] = aug["c_harrell"] - aug["base"]
    per = aug.groupby(["model", "strategy", "ratio", "subgroup", "level"])["d"].mean().reset_index()
    rows = []
    for k, g in per.groupby(["model", "strategy", "ratio"]):
        w = g.loc[g["d"].idxmin()] if g["d"].notna().any() else None
        rows.append({"model": k[0], "strategy": k[1], "ratio": k[2],
                     "worst_subgroup": f"{w['subgroup']}={w['level']}" if w is not None else "-",
                     "worst_subgroup_delta_c": float(w["d"]) if w is not None else np.nan})
    return pd.DataFrame(rows)


def build_scorecard(
    paired: pd.DataFrame,
    subgroups: pd.DataFrame,
    fidelity: pd.DataFrame,
    privacy: pd.DataFrame,
    external: pd.DataFrame,
    cfg: AugmentationConfig,
    tvae_strategies: set[str],
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    g = cfg.gates
    pm = g.primary_metric
    card = _delta(paired, pm)
    for m in ("cal_slope", "cal_slope_err", "ibs", "brier_h"):
        if (paired["metric"] == m).any():
            card = card.join(_delta(paired, m)[[f"{m}:delta"]], how="left")
    card = card.reset_index()
    card = card.merge(subgroup_deltas(subgroups), on=["model", "strategy", "ratio"], how="left")

    fid = (fidelity[fidelity["evaluation"] == "internal"].groupby("strategy")
           .agg({c: "mean" for c in ("discriminator_auc", "target_tvd_mean") if c in fidelity})
           if not fidelity.empty else pd.DataFrame())
    agg = {"exact_duplicates": "max", "membership_auc": "mean", "dcr_p5_ratio": "mean"}
    priv = (privacy[privacy["evaluation"] == "internal"].groupby("strategy")
            .agg({c: f for c, f in agg.items() if c in privacy})
            if not privacy.empty else pd.DataFrame())
    card = card.join(fid, on="strategy").join(priv, on="strategy")

    if not external.empty:
        ext = external[external["metric"] == pm].set_index(["model", "strategy", "ratio"])
        card = card.join(ext[["delta", "ci_low", "ci_high"]].add_prefix("external:"),
                         on=["model", "strategy", "ratio"])

    verdicts, reasons, utility, ext_utility = [], [], [], []
    for _, r in card.iterrows():
        why: list[str] = []
        util = _utility(r[f"{pm}:ci_low"], r[f"{pm}:ci_high"], g.noninferiority_margin)
        utility.append(util)
        if util not in ("improves", "non-inferior"):
            why.append(f"utility {util} (Δ{pm} CI {r[f'{pm}:ci_low']:+.3f} to {r[f'{pm}:ci_high']:+.3f})")
        cal_ok = True
        # Judged on the distance of the slope from 1, so moving towards 1 (e.g. an
        # overfit model's slope of 0.2 rising to 0.4) never counts against augmentation.
        de = r.get("cal_slope_err:delta", np.nan)
        if np.isfinite(de) and de > g.max_abs_delta_slope:
            cal_ok = False
            why.append(f"calibration slope moves {de:.2f} further from 1 (Δslope {r.get('cal_slope:delta', np.nan):+.2f})")
        dbr = r.get("ibs:delta", np.nan)
        if not np.isfinite(dbr):
            dbr = r.get("brier_h:delta", np.nan)
        if np.isfinite(dbr) and dbr > g.max_delta_ibs:
            cal_ok = False
            why.append(f"Brier worsens {dbr:+.4f}")
        sub_ok = not (np.isfinite(r.get("worst_subgroup_delta_c", np.nan))
                      and r["worst_subgroup_delta_c"] < g.min_subgroup_delta_c)
        if not sub_ok:
            why.append(f"subgroup {r['worst_subgroup']} loses {r['worst_subgroup_delta_c']:+.3f} C")
        fid_ok = priv_ok = True
        if r["strategy"] in tvae_strategies:
            da = r.get("discriminator_auc", np.nan)
            if np.isfinite(da) and da > g.max_discriminator_auc:
                fid_ok = False
                why.append(f"discriminator AUC {da:.2f}")
            tt = r.get("target_tvd_mean", np.nan)
            if np.isfinite(tt) and tt > g.max_target_tvd:
                fid_ok = False
                why.append(f"target-stratum TVD {tt:.2f}")
            if r.get("exact_duplicates", 0) > g.max_exact_duplicates:
                priv_ok = False
                why.append(f"{int(r['exact_duplicates'])} exact copies of training patients")
            ma = r.get("membership_auc", np.nan)
            if np.isfinite(ma) and ma > g.max_membership_auc:
                priv_ok = False
                why.append(f"membership AUC {ma:.2f}")
            dr = r.get("dcr_p5_ratio", np.nan)
            if np.isfinite(dr) and dr < g.min_dcr_p5_ratio:
                priv_ok = False
                why.append(f"near-copies (DCR p5 ratio {dr:.2f})")
        if not priv_ok:
            v = "UNSAFE"
        elif util == "improves" and cal_ok and sub_ok and fid_ok:
            v = "RECOMMENDED"
        elif util in ("improves", "non-inferior") and cal_ok and sub_ok:
            v = "NON-INFERIOR"
        else:
            v = "NOT RECOMMENDED"
        verdicts.append(v)
        reasons.append("; ".join(why) or "all gates pass")
        ext_utility.append(_utility(r.get("external:ci_low", np.nan), r.get("external:ci_high", np.nan),
                                    g.noninferiority_margin, missing="-"))
    card["utility"] = utility
    card["external_utility"] = ext_utility
    card["verdict"] = verdicts
    card["reasons"] = reasons
    order = {"RECOMMENDED": 0, "NON-INFERIOR": 1, "NOT RECOMMENDED": 2, "UNSAFE": 3}
    return card.sort_values(["model", "verdict", f"{pm}:delta"],
                            key=lambda s: s.map(order) if s.name == "verdict" else
                            (-s if s.name == f"{pm}:delta" else s)).reset_index(drop=True)
