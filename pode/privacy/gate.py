"""Turn the check summaries into a release gate: PASS / REVIEW / FAIL per check.

The overall verdict is the worst automated status. A required check that could
not be run (membership inference without a holdout) counts as REVIEW: missing
evidence is not evidence of safety. The governance items from the release
checklist cannot be automated and are listed as MANUAL so the report never
reads as a complete sign-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GateConfig

ORDER = {"PASS": 0, "NOT_ASSESSED": 0, "REVIEW": 1, "FAIL": 2}

MANUAL_ITEMS = [
    ("data_minimisation", "Direct identifiers, free text, accession IDs and exact dates removed or transformed "
                          "before training; geography, centre and rare categories coarsened."),
    ("release_tier", "Release pathway matches residual risk: public only for low-risk data, otherwise "
                     "controlled access / trusted research environment with a data-access agreement."),
    ("utility_checked", "Utility checked separately (e.g. train on synthetic, test on held-out real) so the "
                        "data are not misleading clinical artefacts."),
    ("independent_review", "Someone who did not tune the TVAE has reviewed this audit and the case-review tables."),
    ("governance", "Written privacy assessment, consent/ethics review and information-governance decision "
                   "support the stated sharing purpose."),
    ("formal_guarantee", "Decide whether the use requires differentially private training (DP-SGD with a "
                         "reported epsilon/delta); this assessment is evidence, not a guarantee."),
]


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def excess_status(value: float, limit: float, p: float, alpha: float) -> str:
    """PASS within the limit; above it FAIL if statistically clear, else REVIEW."""
    if np.isnan(value) or value <= limit:  # inf (no chance events at all) is an excess
        return "PASS"
    return "FAIL" if np.isfinite(p) and p < alpha else "REVIEW"


def _chance(status: str, p: float) -> str:
    if status == "REVIEW":
        return f"; above the limit but within chance for this sample size (p={p:.2g})"
    if status == "FAIL":
        return f" (p={p:.2g})"
    return ""


def evaluate_gate(s: dict, g: GateConfig, has_holdout: bool) -> pd.DataFrame:
    rows = []

    def add(check, metric, value, threshold, status, reason, required=False):
        rows.append({"check": check, "metric": metric, "value": _fmt(value), "threshold": threshold,
                     "status": status, "reason": reason, "required": required})

    def num(key):
        v = s.get(key, np.nan)
        return float(v) if v is not None else np.nan

    # 1. Identifiers.
    f, r = s.get("identifier_issues_fail", 0), s.get("identifier_issues_review", 0)
    add("identifiers", "issues (fail/review)", f"{f}/{r}", "0", "FAIL" if f else "REVIEW" if r else "PASS",
        "identifier columns or real ID values in the synthetic table" if f else
        "date or free-text columns need transforming" if r else "no identifier, date or free-text columns")

    # 2. Exact copies.
    n, ft, fh = s.get("exact_copies", 0), num("exact_copy_frac_train"), num("exact_copy_frac_holdout")
    if n <= g.max_exact_copies:
        add("exact_copies", "training patients copied", n, f"<= {g.max_exact_copies}", "PASS",
            "no whole-record copies of training patients")
    elif np.isfinite(fh) and ft <= fh:
        add("exact_copies", "training patients copied", n, f"<= {g.max_exact_copies}", "REVIEW",
            f"copies occur at the chance rate for unseen patients ({ft:.1%} vs {fh:.1%}): the table is coarse "
            "enough for coincidental matches; inspect, especially rare ones")
    else:
        extra = f" ({s.get('exact_copies_small_strata', 0)} in small strata)" if s.get("exact_copies_small_strata") else ""
        add("exact_copies", "training patients copied", n, f"<= {g.max_exact_copies}", "FAIL",
            f"{n} training patients reproduced as whole records{extra}"
            + (f", above the unseen-patient rate ({ft:.1%} vs {fh:.1%})" if np.isfinite(fh) else ""))

    # 3. Distance to closest record.
    ratio = num("dcr_p5_ratio")
    ref = s.get("reference", "holdout")
    if not np.isfinite(ratio):
        add("near_copies", "dcr_p5_ratio", ratio, f">= {g.min_dcr_p5_ratio}", "NOT_ASSESSED",
            "reference 5th percentile is 0 (many identical real records); rely on near-copy excess")
    else:
        add("near_copies", "dcr_p5_ratio", ratio, f">= {g.min_dcr_p5_ratio}",
            "PASS" if ratio >= g.min_dcr_p5_ratio else "FAIL",
            f"closest synthetic rows vs {ref} reference: "
            + ("comparable" if ratio >= g.min_dcr_p5_ratio else "systematically closer to training patients"))
    bands = [k[len("near_copy_excess_"):] for k in s if k.startswith("near_copy_excess_")]
    bands = [b for b in bands if s.get(f"near_copy_band_rows_{b}", 0) >= 5 and np.isfinite(s[f"near_copy_excess_{b}"])]
    if bands:
        b = max(bands, key=lambda b: s[f"near_copy_excess_{b}"])
        v, p = s[f"near_copy_excess_{b}"], s[f"near_copy_pvalue_{b}"]
        st = excess_status(v, g.max_near_copy_excess, p, g.alpha)
        add("near_copies", f"near_copy_excess_{b}", v, f"<= {g.max_near_copy_excess}", st,
            f"{v:.2g}x the near-copy rate of {'new patients' if ref == 'holdout' else 'real neighbours'}"
            + _chance(st, p))
    else:
        add("near_copies", "near_copy_excess", None, f"<= {g.max_near_copy_excess}", "NOT_ASSESSED",
            "fewer than 5 reference patients inside every near-copy band; use a larger holdout")
    iso, p = num("isolated_near_copy_excess"), num("isolated_near_copy_pvalue")
    if np.isfinite(iso):
        st = excess_status(iso, g.max_near_copy_excess, p, g.alpha)
        add("near_copies", "isolated_near_copy_excess", iso, f"<= {g.max_near_copy_excess}",
            "REVIEW" if st == "FAIL" else st,
            "near-copies that sit on one training patient (small nearest / second-nearest ratio)" + _chance(st, p))

    # 4. Strata.
    fl = s.get("strata_flagged", np.nan)
    if isinstance(fl, float) and np.isnan(fl):
        add("strata", "flagged levels", None, "0", "NOT_ASSESSED", "no strata configured")
    else:
        add("strata", "flagged levels", fl, "0", "REVIEW" if fl else "PASS",
            "see strata.csv: near-copying or memorisation concentrated in a stratum" if fl else
            "no stratum shows concentrated near-copying")

    # 5. Membership.
    auc, hi = num("membership_auc_max"), num("membership_auc_max_ci_high")
    if not has_holdout or not np.isfinite(auc):
        add("membership", "membership_auc", None, f"<= {g.max_membership_auc}", "NOT_ASSESSED",
            "needs held-out real patients the generator never saw (use `privacy audit` or --holdout)",
            required=True)
    else:
        worst = s.get("membership_worst_attack", "all_columns")
        status = "FAIL" if auc > g.max_membership_auc else "REVIEW" if hi > g.max_membership_auc else "PASS"
        add("membership", f"membership_auc ({worst})", auc, f"<= {g.max_membership_auc}", status,
            f"95% CI upper {hi:.2f}" + ("" if status == "PASS" else
                                         "; the holdout is too small to rule leakage out" if status == "REVIEW" else
                                         "; training patients are distinguishable from unseen ones"))

    # 6. Attribute inference.
    gap, lo = num("attribute_member_gap_max"), num("attribute_gap_ci_low")
    if not np.isfinite(gap):
        add("attribute_inference", "member_gap", None, f"<= {g.max_attribute_member_gap}", "NOT_ASSESSED",
            "needs a holdout, quasi-identifiers and sensitive targets")
    else:
        status = ("PASS" if gap <= g.max_attribute_member_gap else "FAIL" if lo > 0 else "REVIEW")
        add("attribute_inference", f"member_gap ({s.get('attribute_worst', '')})", gap,
            f"<= {g.max_attribute_member_gap}", status,
            "synthetic data reveals sensitive values of training patients more than of unseen ones"
            if status != "PASS" else "no extra disclosure for training patients")

    # 7. Linkage: singling-out (rate excess) and disclosive links (unique match that is also a near-copy).
    le, p = num("linkage_excess_max"), num("linkage_pvalue")
    if not np.isfinite(le):
        add("linkage", "linkage_excess", None, f"<= {g.max_linkage_excess}", "NOT_ASSESSED",
            "needs a holdout and quasi-identifier tiers")
    else:
        st = excess_status(le, g.max_linkage_excess, p, g.alpha)
        add("linkage", f"singling_out_excess ({s.get('linkage_worst_tier', '')})", le, f"<= {g.max_linkage_excess}",
            st, f"{s.get('singled_out_members_max', 0)} training patients singled out by a unique synthetic match"
            + _chance(st, p))
        n, rm, rh, p = (s.get("disclosive_links", 0), num("disclosive_rate_members"),
                        num("disclosive_rate_holdout"), num("disclosive_pvalue"))
        ratio = rm / rh if rh > 0 else (np.inf if n else 0.0)
        st = excess_status(ratio, 2.0, p, g.alpha) if n else "PASS"
        add("linkage", f"disclosive_links ({s.get('disclosive_worst_tier', '')})", n, "<= 2x chance", st,
            f"{n} training patients uniquely matched by a near-copy of their record ({rm:.2%} vs {rh:.2%} "
            "for unseen patients)" + _chance(st, p))

    # 8. Rare combinations.
    lk, rs, rh = s.get("rare_combo_leaks", np.nan), num("rare_combo_leak_rate"), num("rare_combo_leak_rate_holdout")
    if isinstance(lk, float) and np.isnan(lk):
        add("rare_combinations", "rare_combo_leaks", None, f"<= {g.max_rare_combo_leaks}", "NOT_ASSESSED",
            "no quasi-identifier tiers")
    elif lk <= g.max_rare_combo_leaks:
        add("rare_combinations", "rare_combo_leaks", lk, f"<= {g.max_rare_combo_leaks}", "PASS",
            "no rare combination reproduced at near-copy distance")
    else:
        p = num("rare_combo_pvalue")
        if np.isfinite(rh):
            floor = 0.5 / max(s.get("n_holdout", 1), 1)
            st = excess_status(rs / max(rh, floor), 2.0, p, g.alpha)
        else:
            st = "REVIEW"
        add("rare_combinations", "rare_combo_leaks", lk, f"<= {g.max_rare_combo_leaks}", st,
            f"rare quasi-identifier combinations reproduced close to a training patient (rate {rs:.2%}"
            + (f" vs {rh:.2%} for unseen patients)" if np.isfinite(rh) else ", no holdout to compare)")
            + (_chance(st, p) if np.isfinite(p) else ""))

    # 9. Outliers (always a REVIEW item at worst: it calls for case-level inspection).
    rep, p = s.get("outliers_replicated", 0), num("outlier_pvalue")
    rate, rate_h = num("outlier_replication_rate"), num("outlier_replication_rate_holdout")
    st = "PASS" if not rep else "REVIEW" if (not np.isfinite(p) or p < g.alpha) else "PASS"
    add("outliers", "outliers_replicated", rep, "chance rate", st,
        f"{rep} unusual training patients have a synthetic row much closer than any real patient "
        f"({rate:.0%} of outliers" + (f" vs {rate_h:.0%} for unseen outliers)" if np.isfinite(rate_h) else ")")
        + ("; see outliers.csv" if st == "REVIEW" else ""))

    # 10. Missingness.
    if s.get("missingness_present"):
        mr, mh, p = num("rare_missingness_rate"), num("rare_missingness_rate_holdout"), num("rare_missingness_pvalue")
        ratio = mr / mh if np.isfinite(mh) and mh > 0 else (np.inf if mr > 0 else 0.0)
        st = excess_status(ratio, 2.0, p if np.isfinite(p) else 0.0, g.alpha)
        add("missingness", "rare_missingness_rate", mr, "<= 2x holdout", "REVIEW" if st == "FAIL" else st,
            f"synthetic rows with missingness patterns rare in training (holdout rate {_fmt(mh)})" + _chance(st, p))

    for item, text in MANUAL_ITEMS:
        rows.append({"check": item, "metric": "", "value": "", "threshold": "", "status": "MANUAL",
                     "reason": text, "required": False})
    return pd.DataFrame(rows)


def verdict(gate: pd.DataFrame) -> str:
    auto = gate[gate["status"] != "MANUAL"]
    worst = max((ORDER[s] for s in auto["status"]), default=0)
    if worst < 1 and ((auto["status"] == "NOT_ASSESSED") & auto["required"]).any():
        worst = 1
    return {0: "PASS", 1: "REVIEW", 2: "FAIL"}[worst]
