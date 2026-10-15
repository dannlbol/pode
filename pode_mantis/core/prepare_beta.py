"""Reindex a cohort's beta matrix to MANTIS-MB's exact 357,692-CpG order.

Generic port of an earlier study-specific input-preparation script into a
reusable function. The algorithm is
unchanged: reindex to the canonical order (missing probes become all-NaN
rows), impute preferring the MANTIS-training-cohort per-probe mean, falling
back to this cohort's own per-probe median only where a training mean isn't
available, and never drop a probe -- ``normalize_instance_wise`` in the
reference implementation is per-sample and per-block, so dropping a probe
would silently shift every other value for that sample (see
``NOTES_implementation.md`` #2 in the source project).

Writes the bare, no-header, no-index numeric CSV the reference `inference.py`
actually requires on disk (probes as rows, samples as columns -- see
``NOTES_implementation.md`` #1), plus a `_sample_order.txt` sidecar (the raw
sample IDs in that exact column order) that ``prepare_survival``/
``assertions`` use to verify beta/CNV/survival files all agree on sample
order before anything expensive runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PrepareBetaResult:
    """Outcome of a ``run_prepare_beta`` call.

    ``numeric_path`` is what MANTIS's ``inference.py`` actually reads;
    ``sample_order_path`` is the sidecar every other prepare-* step and
    ``assertions.py`` cross-checks against, so beta/CNV/survival never
    silently drift out of alignment.
    """

    numeric_path: Path
    sample_order_path: Path
    coverage_path: Path
    n_samples: int
    n_cpgs: int


def _load_cpg_order(path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _load_beta_matrix(path, orientation: str) -> pd.DataFrame:
    """Load a beta matrix, returned as probes (index) x samples (columns)."""
    path = str(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        if "probe_id" in df.columns:
            df = df.set_index("probe_id")
    else:
        sep = "\t" if path.endswith((".tsv", ".tsv.gz")) else ","
        df = pd.read_csv(path, sep=sep, index_col=0, low_memory=False)
    if orientation == "samples_by_features":
        df = df.T
    return df


def run_prepare_beta(cfg, betas=None, out_dir=None) -> PrepareBetaResult:
    b = cfg["beta"]
    betas_path = betas or b.get("cohort_path")
    if not betas_path:
        raise ValueError("prepare-beta: no beta matrix given (config beta.cohort_path or --betas).")
    cpg_order_path = b.get("cpg_order_path")
    if not cpg_order_path:
        raise ValueError(
            "prepare-beta: no canonical CpG order given (config beta.cpg_order_path -- "
            "from the Zenodo release, see docs/mantis/README.md)."
        )
    out = Path(out_dir or b["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    cpg_order = _load_cpg_order(cpg_order_path)
    n_cpg = len(cpg_order)
    print(f"[cpg-order] {n_cpg:,} probes")

    df = _load_beta_matrix(betas_path, b["orientation"])
    sample_ids = list(df.columns)
    print(f"[load] {betas_path}: {df.shape[0]:,} native probes x {len(sample_ids)} samples")

    df_reindexed = df.reindex(cpg_order)
    present_mask = df_reindexed.notna()
    n_present_per_sample = present_mask.sum(axis=0)
    n_imputed_per_sample = n_cpg - n_present_per_sample

    train_means = None
    if b.get("training_means_path"):
        train_means = pd.read_csv(b["training_means_path"]).set_index("probe_id")["mean_beta"]
        train_means = train_means.reindex(cpg_order)
        print(f"[training-means] {int(train_means.notna().sum())}/{n_cpg} probes have a value")

    own_median = df_reindexed.median(axis=1, skipna=True)
    if train_means is not None:
        fill_value = train_means.where(train_means.notna(), own_median)
        n_from_training = int(train_means.notna().sum())
    else:
        fill_value = own_median
        n_from_training = 0
        print("[warn] no training_means_path given -- imputing from this cohort's own median only")

    df_filled = df_reindexed.apply(lambda col: col.fillna(fill_value), axis=0)
    still_na = int(df_filled.isna().sum().sum())
    if still_na:
        print(
            f"[warn] {still_na} cells remain NaN after imputation (no training mean or own-cohort "
            f"median available for those probes) -- these probes have zero signal anywhere in this data."
        )
        df_filled = df_filled.fillna(0.5)  # neutral fallback; see warning above

    numeric_path = out / "beta_numeric.csv"
    df_filled.loc[cpg_order].to_csv(numeric_path, header=False, index=False, float_format="%.6f")

    sample_order_path = out / "beta_sample_order.txt"
    with open(sample_order_path, "w") as f:
        f.write("\n".join(str(s) for s in sample_ids) + "\n")

    coverage_rows = []
    for sid in sample_ids:
        n_present = int(n_present_per_sample[sid])
        n_imputed = int(n_imputed_per_sample[sid])
        pct_imputed = 100.0 * n_imputed / n_cpg
        coverage_rows.append(
            {
                "sample_id": sid,
                "n_present": n_present,
                "n_imputed": n_imputed,
                "pct_imputed": pct_imputed,
                "primary_analysis_flag": pct_imputed <= 5.0,
            }
        )
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = out / "probe_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)

    n_excluded = int((~coverage_df["primary_analysis_flag"]).sum())
    print(
        f"[coverage] mean pct_imputed={coverage_df['pct_imputed'].mean():.2f}  "
        f"max={coverage_df['pct_imputed'].max():.2f}  "
        f"n_samples_over_5pct_imputed={n_excluded} (flagged, not dropped -- see probe_coverage.csv)"
    )
    print(f"[saved] {numeric_path}  ({df_filled.shape[0]} x {df_filled.shape[1]})")
    print(f"[saved] {sample_order_path}")
    print(f"[saved] {coverage_path}")

    return PrepareBetaResult(
        numeric_path=numeric_path,
        sample_order_path=sample_order_path,
        coverage_path=coverage_path,
        n_samples=len(sample_ids),
        n_cpgs=n_cpg,
    )
