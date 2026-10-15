"""Alignment and sanity checks guarding against MANTIS's #1 documented failure mode.

Nothing in the reference implementation checks that the beta matrix's sample
columns, the CNV matrix's sample columns, and the survival CSV's row order
actually agree with each other -- ``inference.py``/``evaluate.py`` trust
positional convention alone (see ``NOTES_implementation.md`` #1, #4, #5 in the
source project). A silently misaligned input produces plausible-looking,
confidently wrong predictions with no internal check. This module provides:

* ``check_pre_flight`` -- run BEFORE the (potentially slow) reference
  ``inference.py`` subprocess: do the beta/CNV sample-order sidecars and the
  survival CSV's ``Patient_ID`` column all agree, in order?
* ``check_post_flight`` -- generic port of the source project's
  ``scripts/check_inference_assertions.py``: row count, ID order, column
  count, NaN-row explanation, monotonicity of the predictions CSV against the
  survival CSV used to build it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class AlignmentReport:
    ok: bool
    problems: list[str] = field(default_factory=list)

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise ValueError(
                "Alignment check failed:\n  - " + "\n  - ".join(self.problems)
            )


def _load_ids(path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def check_pre_flight(beta_sample_order_path, survival_path, cnv_sample_order_path=None) -> AlignmentReport:
    """Verify beta/CNV/survival sample order agree before running inference."""
    problems: list[str] = []

    beta_ids = _load_ids(beta_sample_order_path)
    surv_df = pd.read_csv(survival_path)
    if "Patient_ID" not in surv_df.columns:
        problems.append(f"{survival_path} has no 'Patient_ID' column.")
        return AlignmentReport(ok=False, problems=problems)
    surv_ids = [str(x) for x in surv_df["Patient_ID"]]

    if len(beta_ids) != len(surv_ids):
        problems.append(
            f"beta sample count ({len(beta_ids)}) != survival row count ({len(surv_ids)})."
        )
    elif beta_ids != surv_ids:
        n_common = len(set(beta_ids) & set(surv_ids))
        first_mismatch = next((i for i, (a, b) in enumerate(zip(beta_ids, surv_ids)) if a != b), None)
        problems.append(
            f"beta sample order != survival Patient_ID order ({n_common}/{len(beta_ids)} IDs overlap "
            f"at all; first mismatch at row {first_mismatch}: beta='{beta_ids[first_mismatch]}' "
            f"vs survival='{surv_ids[first_mismatch]}')."
        )

    if cnv_sample_order_path:
        cnv_ids = _load_ids(cnv_sample_order_path)
        if cnv_ids != beta_ids:
            n_common = len(set(cnv_ids) & set(beta_ids))
            problems.append(
                f"CNV sample order != beta sample order ({n_common}/{len(beta_ids)} IDs overlap at all)."
            )

    return AlignmentReport(ok=not problems, problems=problems)


def check_post_flight(pred_path, survival_path, expected_n_time_bins: int = 16) -> AlignmentReport:
    """Generic port of check_inference_assertions.py's four post-inference checks."""
    problems: list[str] = []

    pred_df = pd.read_csv(pred_path, index_col=0)
    surv_df = pd.read_csv(survival_path)

    if len(pred_df) != len(surv_df):
        problems.append(f"prediction row count ({len(pred_df)}) != survival row count ({len(surv_df)}).")
    else:
        pred_ids = [str(x) for x in pred_df.index]
        surv_ids = [str(x) for x in surv_df["Patient_ID"]]
        if pred_ids != surv_ids:
            first_mismatch = next((i for i, (a, b) in enumerate(zip(pred_ids, surv_ids)) if a != b), None)
            problems.append(
                f"prediction index != survival Patient_ID, elementwise (first mismatch at row "
                f"{first_mismatch}: pred='{pred_ids[first_mismatch]}' vs survival='{surv_ids[first_mismatch]}')."
            )

    if pred_df.shape[1] != expected_n_time_bins:
        problems.append(f"expected {expected_n_time_bins} prediction columns, found {pred_df.shape[1]}.")

    nan_rows = pred_df.isna().any(axis=1)
    if nan_rows.any():
        fully_nan = pred_df.isna().all(axis=1)
        partially_nan = nan_rows & ~fully_nan
        if partially_nan.any():
            problems.append(
                f"{int(partially_nan.sum())} rows are PARTIALLY NaN (some but not all of the "
                f"{expected_n_time_bins} columns) -- this should never happen and indicates real "
                f"corruption, not just a missing clinical feature."
            )
        if "m" in surv_df.columns and len(pred_df) == len(surv_df):
            m_missing = surv_df["m"].isna().to_numpy()
            fully_nan_arr = fully_nan.to_numpy()
            if not np.array_equal(fully_nan_arr, m_missing):
                problems.append(
                    f"{int(fully_nan.sum())} fully-NaN prediction rows do not exactly match the "
                    f"{int(m_missing.sum())} rows with missing metastasis status ('m') in the survival "
                    f"CSV -- NaN predictions should be fully explained by missing clinical input."
                )
        else:
            print(
                f"[warn] {int(fully_nan.sum())} fully-NaN prediction rows found, but couldn't cross-check "
                f"against survival 'm' (row counts differ or column absent)."
            )

    numeric = pred_df.select_dtypes(include=[float, int]).to_numpy(dtype=float)
    if numeric.shape == pred_df.shape:
        diffs = np.diff(numeric, axis=1)
        violations = np.nansum(diffs > 1e-9)
        if violations:
            worst = np.nanmax(diffs)
            print(
                f"[warn] {int(violations)} monotonicity violations across all rows/columns "
                f"(survival curves should be non-increasing); worst violation magnitude {worst:.4f}. "
                f"Reported, not treated as fatal -- the reference implementation's own smoothing pass "
                f"already handles most of this upstream."
            )

    return AlignmentReport(ok=not problems, problems=problems)
