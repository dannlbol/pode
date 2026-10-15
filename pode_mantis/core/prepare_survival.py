"""Build MANTIS-MB's expected survival/clinical CSV from a generic clinical table.

Generalizes a reusable *pattern* from an earlier study-specific script (read a
sample-order sidecar -> join a clinical sheet on a configurable key column ->
emit the reference implementation's expected columns -> assert row order
matches) while dropping the dataset-specific ID normalisation and
de-duplication steps, which are not generic requirements.

Required output columns, matching the reference repo's own convention
(``Patient_ID``, ``time``/``event`` for OS, ``pfs_time``/``pfs_event`` for
PFS, ``m`` for metastasis coded 0/1 -- see ``mantis_comparison_spec.md``
Phase 4C in the source project): row order is asserted to exactly match the
beta sample-order sidecar before writing, since nothing in the reference
``inference.py``/``evaluate.py`` checks this itself (see
``core.assertions`` and ``NOTES_implementation.md`` #4-#5 in the source
project for why this is the single most important thing to get right).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PrepareSurvivalResult:
    """Outcome of a ``run_prepare_survival`` call. ``out_path`` is what
    ``generate_config``/``infer``/``evaluate`` point at as the cohort's
    ``*_surv_path``."""

    out_path: Path
    n_samples: int
    n_events_os: int | None
    n_events_pfs: int | None


def _load_sample_order(path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def run_prepare_survival(cfg, clinical=None, sample_order_path=None, out_dir=None) -> PrepareSurvivalResult:
    surv = cfg["survival"]
    clinical_path = clinical or surv.get("clinical_path")
    if not clinical_path:
        raise ValueError("prepare-survival: no clinical CSV given (config survival.clinical_path or --clinical).")
    id_col = surv.get("id_column")
    time_col = surv.get("time_column")
    event_col = surv.get("event_column")
    if not id_col or not time_col or not event_col:
        raise ValueError(
            "prepare-survival: survival.id_column, survival.time_column and survival.event_column are required."
        )
    metastasis_col = surv.get("metastasis_column")
    if not metastasis_col:
        raise ValueError(
            "prepare-survival: survival.metastasis_column is required -- MANTIS-MB's clinical "
            "feature is metastasis status, a required model input, not optional."
        )

    clin = pd.read_csv(clinical_path)
    for col in (id_col, time_col, event_col, metastasis_col):
        if col not in clin.columns:
            raise ValueError(f"prepare-survival: clinical CSV is missing column '{col}'.")

    out_rows = {
        "Patient_ID": clin[id_col].astype(str),
        "time": clin[time_col],
        "event": clin[event_col].astype(int),
        # MANTIS's config recodes 0 -> -1 itself (run_setup.metastasis_recoding) --
        # do NOT pre-recode here, matching mantis_comparison_spec.md's explicit warning.
        "m": clin[metastasis_col],
    }
    n_events_pfs = None
    pfs_time_col, pfs_event_col = surv.get("pfs_time_column"), surv.get("pfs_event_column")
    if pfs_time_col and pfs_event_col:
        if pfs_time_col not in clin.columns or pfs_event_col not in clin.columns:
            raise ValueError("prepare-survival: pfs_time_column/pfs_event_column set but not found in clinical CSV.")
        out_rows["pfs_time"] = clin[pfs_time_col]
        out_rows["pfs_event"] = clin[pfs_event_col].astype(int)
        n_events_pfs = int(out_rows["pfs_event"].sum())

    n_missing_m = int(clin[metastasis_col].isna().sum())
    if n_missing_m:
        print(
            f"[warn] {n_missing_m} samples missing metastasis status -- metastasis is a required "
            f"model input; exclude these from primary analysis (M0 imputation is a sensitivity-only "
            f"option, per the source project's spec)."
        )

    surv_df = pd.DataFrame(out_rows)

    if sample_order_path:
        expected_order = _load_sample_order(sample_order_path)
        actual_order = list(surv_df["Patient_ID"])
        if actual_order != expected_order:
            common = set(actual_order) & set(expected_order)
            raise ValueError(
                f"prepare-survival: clinical CSV row order does not match the beta sample-order "
                f"sidecar ({sample_order_path}). {len(common)}/{len(expected_order)} IDs overlap at "
                f"all. This is the #1 MANTIS failure mode (silent row misalignment) -- reorder your "
                f"clinical CSV to exactly match the beta matrix's sample-column order before "
                f"proceeding; do not skip this check."
            )

    out = Path(out_dir or surv["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "survival.csv"
    surv_df.to_csv(out_path, index=False)

    n_events_os = int(surv_df["event"].sum())
    print(f"[saved] {out_path}  ({len(surv_df)} samples, {n_events_os} OS events" + (
        f", {n_events_pfs} PFS events)" if n_events_pfs is not None else ")"
    ))

    return PrepareSurvivalResult(
        out_path=out_path, n_samples=len(surv_df), n_events_os=n_events_os, n_events_pfs=n_events_pfs
    )
