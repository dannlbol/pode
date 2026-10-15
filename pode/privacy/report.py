"""Markdown rendering of a :class:`~pode.privacy.assess.PrivacyReport`."""

from __future__ import annotations

import numpy as np
import pandas as pd

VERDICT_TEXT = {
    "PASS": "No automated check found a privacy problem. This is evidence for release, not a guarantee: "
            "complete the MANUAL items before any sharing.",
    "REVIEW": "No check failed outright, but at least one needs human review (or could not be run). "
              "Do not release until the REVIEW items are resolved.",
    "FAIL": "At least one check found memorisation or disclosure. Do not release this synthetic data; "
            "see the remediation section.",
}

READING = """\
| Check | What it asks | Reference points |
|---|---|---|
| identifiers | Are IDs, dates or free text present in the synthetic table? | any identifier = FAIL; dates/free text = REVIEW |
| exact_copies | Is any training patient reproduced as a whole record? | 0 expected; compared with the coincidental match rate for unseen patients |
| near_copies | Are synthetic rows closer to training patients than new patients are? | `dcr_p5_ratio` ~1 good, < 0.8 near-copies; near-copy excess ~1 good, > 2 bad |
| strata | Is near-copying hidden inside a rare diagnosis, age band or outcome? | flagged levels listed in strata.csv |
| membership | Can an attacker tell who was in the training data? | AUC 0.5 = no signal, > 0.6 worrying, 1.0 = fully revealed |
| attribute_inference | Does the synthetic data reveal a training child's sensitive value from their quasi-identifiers? | member gap ~0 good; > 0.05 = training patients leak more than unseen ones |
| linkage | Does a synthetic record uniquely point to a real child (singling-out), and is it a near-copy of their record (disclosive link)? | singling-out excess ~0 good, > 0.02 bad; disclosive links at most ~chance |
| rare_combinations | Are rare quasi-identifier combinations reproduced close to the real patient? | 0 leaks, or no more than 2x the unseen-patient rate |
| outliers | Are clinically unusual children closely replicated? | no more often than unseen unusual patients; case review of outliers.csv |
| missingness | Do rare missing-data patterns (e.g. unusual testing) reappear? | rate <= 2x unseen patients |
"""

REMEDIES = """\
* Coarsen quasi-identifiers (age and diagnosis-year bands, region instead of centre, broader subtypes) and
  drop free text, exact dates and IDs before training.
* Suppress or pool strata with very few patients; do not generate "synthetic children" from 1-5 real cases.
* Reduce memorisation (fewer epochs, a smaller network, more training patients per parameter) and re-audit.
* For a defensible formal claim, train with differential privacy (DP-SGD) and report epsilon/delta.
* Release higher-fidelity data only through controlled access (trusted research environment, data-access
  agreement, audit logging).
"""


def _table(df: pd.DataFrame, cols: list[str] | None = None, n: int | None = None) -> str:
    if df is None or df.empty:
        return "_none_\n"
    d = df[cols] if cols else df
    if n is not None:
        d = d.head(n)
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if not np.isfinite(v) else f"{v:.3g}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join(str(v).replace("|", "/") for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def render(report, title: str) -> str:
    s, t, tm = report.summary, report.tables, report.threat_model
    gate = t["gate"]
    lines = [f"# {title}", "", f"**Verdict: {report.verdict}** - {VERDICT_TEXT[report.verdict]}", ""]
    lines += ["Synthetic data from a TVAE is *synthetic*, not automatically anonymous. These checks test "
              "realistic re-identification threats for the threat model below; they do not prove safety.", ""]

    lines += ["## Release gate", "", _table(gate[gate["status"] != "MANUAL"],
                                           ["check", "metric", "value", "threshold", "status", "reason"])]
    lines += ["### Manual checklist (cannot be automated)", ""]
    lines += [f"- [ ] **{r.check}**: {r.reason}" for r in gate[gate["status"] == "MANUAL"].itertuples()]
    lines += [""]

    lines += ["## Threat model and data", ""]
    lines += [f"- Training patients: {tm['n_train']}; held-out patients: {tm['n_holdout']}; synthetic rows: "
              f"{tm['n_synthetic']} ({tm['n_synthetic_assessed']} assessed).",
              f"- Distance reference: **{tm['reference']}**"
              + (" (unseen patients from the same population)." if tm["reference"] == "holdout" else
                 " - no holdout, so membership, attribute member gap and linkage were not assessed."),
              f"- Columns assessed: {len(tm['columns'])} ({len(tm['numeric'])} numeric, "
              f"{len(tm['categorical'])} categorical).",
              f"- Sensitive attributes: {', '.join(tm['sensitive']) or '_none_'}.",
              f"- Strata: {', '.join(tm['strata']) or '_none_'}."]
    lines += ["- Attacker tiers (quasi-identifiers the intruder is assumed to know):"]
    lines += [f"  - `{k}`: {', '.join(v)}" for k, v in tm["quasi_identifier_tiers"].items()] or ["  - _none_"]
    lines += [f"- Note: {n}" for n in tm.get("notes", [])]
    lines += ["", "## How to read the checks", "", READING]

    lines += ["## Details", "", "### Distance to closest record", ""]
    keys = [k for k in s if k.startswith(("dcr_", "near_copy_", "nndr_", "isolated_"))]
    lines += [_table(pd.DataFrame({"metric": keys, "value": [s[k] for k in keys]}))]
    lines += ["### Exact copies", "",
              f"{s.get('exact_copies', 0)} training patients reproduced as whole records "
              f"({s.get('exact_copy_frac_train', 0):.2%}); "
              + (f"unseen-patient coincidence rate {s['exact_copy_frac_holdout']:.2%}; "
                 if "exact_copy_frac_holdout" in s else "")
              + f"{s.get('exact_copies_small_strata', 0)} of them in small strata.", ""]
    if "membership" in t:
        lines += ["### Membership inference", "",
                  "Distance attacks using all columns and each attacker tier's columns only.", "",
                  _table(t["membership"])]
    if "attribute_inference" in t:
        lines += ["### Attribute inference", "",
                  "`acc_baseline` guesses the most common synthetic value with no quasi-identifiers; "
                  "`gain_over_baseline` is what the synthetic data teaches about patients in general; "
                  "`member_gap` is the extra disclosure for training patients (the privacy loss).", "",
                  _table(t["attribute_inference"])]
    if "linkage" in t:
        lines += ["### Singling-out and linkage", "", _table(t["linkage"])]
    if "rare_combinations" in t:
        lines += ["### Rare combinations", "", _table(t["rare_combinations"])]
    if "strata" in t:
        flagged = t["strata"][t["strata"]["flag"] != ""]
        lines += ["### Strata", "", f"{len(flagged)} of {len(t['strata'])} stratum levels flagged "
                  "(full table in strata.csv).", "", _table(flagged)]
    if s.get("missingness_present"):
        lines += ["### Missingness patterns", "",
                  f"{s['rare_missingness_rows']} synthetic rows ({s['rare_missingness_rate']:.2%}) carry a "
                  "missingness pattern seen in few training patients"
                  + (f" (unseen-patient rate {s['rare_missingness_rate_holdout']:.2%})."
                     if np.isfinite(s.get("rare_missingness_rate_holdout", np.nan)) else "."), ""]
    if not t["identifiers"].empty:
        lines += ["### Identifier, date and free-text columns", "", _table(t["identifiers"])]

    lines += ["## Case review", "",
              "Row numbers refer to the assessed training table and the synthetic table (0-based); no real "
              "values are written here. Inspect these pairs side by side with the data custodian.", "",
              "### Unusual training patients and their nearest synthetic row", "",
              f"{s.get('outliers_replicated', 0)} of {s.get('outliers', 0)} outliers are closely replicated "
              "(nearest synthetic row inside the near-copy band, or at less than half the distance of the "
              "nearest real patient).", "", _table(t.get("outliers"), n=report.config.review_top_n),
              "### Closest synthetic-training pairs", "", _table(t.get("closest_pairs"))]
    if report.verdict != "PASS":
        lines += ["## Remediation options", "", REMEDIES]
    return "\n".join(lines) + "\n"
