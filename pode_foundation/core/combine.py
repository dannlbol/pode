"""EXPERIMENTAL: combine multiple public reference sources into one Stage-1
pretraining corpus, aligned to a fixed feature set.

Upstream (the private research pipeline this theme was ported from) marks
this script "written, not run" -- the logic is sound and was exercised on
small inputs, but the full pan-cancer pretraining run it enables has not
itself been evaluated end to end. Use it if you're sourcing a broad public
reference corpus yourself for Stage-1 pretraining (see
:mod:`pode_foundation.core.train`); skip it entirely (``pretrain_stage1.enabled:
false``, the default) if you're only pretraining on your own cohort.

Combines two source layouts onto the same fixed ``feature_ids`` set (any
probe missing from a given source file is filled with 0.5, the MAE's neutral
prior for a masked/unknown value):

  1. Wide TSV.gz files (probes x samples), matched by a glob pattern
     (``pretrain_stage1.combine.tcga_glob``) against a directory
     (``pretrain_stage1.combine.tcga_dir``) -- named after its original TCGA
     use case but works for any wide-TSV.gz source.
  2. Parquet files under ``<geo_dir>/<series>/betas.parquet`` (probes x
     samples) -- named after its original GEO use case but works for any
     directory-of-parquet-series source.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CombineResult:
    """Outcome of a ``run_combine`` call. ``x_path`` is the combined-corpus
    matrix to pass as ``--x-path`` to ``pretrain --stage-tag 1``;
    ``probe_overlap_path`` reports what fraction of the fixed feature set
    each source actually covered, worth checking before trusting a source
    with low overlap."""

    x_path: Path
    sample_ids_path: Path
    source_labels_path: Path
    probe_overlap_path: Path
    n_samples: int
    n_features: int


def _process_wide_tsv(gz_path: Path, feature_ids: np.ndarray, chunk_size: int) -> tuple[np.ndarray, list[str], int, int]:
    feat_set = set(feature_ids)
    feat_idx = {p: i for i, p in enumerate(feature_ids)}
    n_feat = len(feature_ids)

    header = pd.read_csv(gz_path, sep="\t", nrows=0, index_col=0)
    sample_ids = list(header.columns)
    mat = np.full((len(sample_ids), n_feat), 0.5, dtype=np.float32)

    n_probes_seen = 0
    n_overlap = 0
    for chunk in pd.read_csv(gz_path, sep="\t", index_col=0, chunksize=chunk_size):
        in_feat = chunk.index[chunk.index.isin(feat_set)]
        n_probes_seen += len(chunk)
        n_overlap += len(in_feat)
        if len(in_feat) == 0:
            continue
        sub = chunk.loc[in_feat]
        row_idx = [feat_idx[p] for p in sub.index]
        mat[:, row_idx] = sub.values.T.astype(np.float32)

    return mat, sample_ids, n_probes_seen, n_overlap


def _process_parquet(parquet_path: Path, feature_ids: np.ndarray) -> tuple[np.ndarray, list[str], int, int]:
    feat_set = set(feature_ids)
    feat_idx = {p: i for i, p in enumerate(feature_ids)}
    n_feat = len(feature_ids)

    df = pd.read_parquet(parquet_path)
    if df.index.name != "probe" and "probe" in df.columns:
        df = df.set_index("probe")

    n_probes_seen = len(df)
    in_feat = df.index[df.index.isin(feat_set)]
    n_overlap = len(in_feat)
    sample_ids = list(df.columns)
    mat = np.full((len(sample_ids), n_feat), 0.5, dtype=np.float32)

    if n_overlap > 0:
        sub = df.loc[in_feat]
        row_idx = [feat_idx[p] for p in sub.index]
        vals = sub.values.T.astype(np.float64)
        vals = np.where(np.isnan(vals), 0.5, vals)
        mat[:, row_idx] = vals.astype(np.float32)

    return mat, sample_ids, n_probes_seen, n_overlap


def run_combine(cfg, out_dir=None) -> CombineResult:
    s1 = cfg["pretrain_stage1"]
    combine_cfg = s1["combine"]
    feature_ids_path = combine_cfg.get("feature_ids_path")
    if not feature_ids_path:
        raise ValueError("combine: pretrain_stage1.combine.feature_ids_path is required.")

    feature_ids = np.load(feature_ids_path, allow_pickle=True)
    print(f"[features] {len(feature_ids):,} probe IDs loaded from {feature_ids_path}")

    out = Path(out_dir or s1["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    chunk_size = combine_cfg.get("chunk_size", 25_000)

    all_mats: list[np.ndarray] = []
    all_ids: list[str] = []
    all_sources: list[str] = []
    overlap_rows: list[dict] = []

    tcga_dir = combine_cfg.get("tcga_dir")
    tcga_glob = combine_cfg.get("tcga_glob")
    if tcga_dir and tcga_glob:
        tcga_files = sorted(Path(tcga_dir).glob(tcga_glob))
        print(f"\n[wide-tsv] {len(tcga_files)} files matching '{tcga_glob}' in {tcga_dir}")
        for gz in tcga_files:
            label = gz.name.split(".")[0]
            try:
                mat, sids, n_probes, n_overlap = _process_wide_tsv(gz, feature_ids, chunk_size)
                print(f"  {label:20s}  n={len(sids):5d}  overlap={n_overlap:,}/{n_probes:,}")
                all_mats.append(mat)
                all_ids.extend(sids)
                all_sources.extend([label] * len(sids))
                overlap_rows.append(
                    {
                        "source": label,
                        "n_samples": len(sids),
                        "n_probes": n_probes,
                        "n_overlap": n_overlap,
                        "overlap_pct": round(n_overlap / max(n_probes, 1) * 100, 1),
                    }
                )
            except Exception as exc:
                print(f"  {label:20s}  ERROR: {exc}")

    geo_dir = combine_cfg.get("geo_dir")
    if geo_dir:
        geo_parquets = sorted(Path(geo_dir).glob("*/betas.parquet"))
        print(f"\n[parquet] {len(geo_parquets)} series found in {geo_dir}")
        for parquet in geo_parquets:
            series = parquet.parent.name
            try:
                mat, sids, n_probes, n_overlap = _process_parquet(parquet, feature_ids)
                print(f"  {series:12s}  n={len(sids):5d}  overlap={n_overlap:,}/{n_probes:,}")
                all_mats.append(mat)
                all_ids.extend([f"{series}_{s}" for s in sids])
                all_sources.extend([series] * len(sids))
                overlap_rows.append(
                    {
                        "source": series,
                        "n_samples": len(sids),
                        "n_probes": n_probes,
                        "n_overlap": n_overlap,
                        "overlap_pct": round(n_overlap / max(n_probes, 1) * 100, 1),
                    }
                )
            except Exception as exc:
                print(f"  {series:12s}  ERROR: {exc}")

    if not all_mats:
        sys.exit("[combine] No data found -- check pretrain_stage1.combine.{tcga_dir,tcga_glob,geo_dir}.")

    print("\n[combine] Stacking matrices ...")
    x = np.concatenate(all_mats, axis=0)
    print(f"  shape: {x.shape}  dtype={x.dtype}")

    x_path = out / "X_stage1.npy"
    sample_ids_path = out / "sample_ids_stage1.npy"
    source_labels_path = out / "source_labels.tsv"
    probe_overlap_path = out / "probe_overlap.tsv"

    np.save(x_path, x)
    np.save(sample_ids_path, np.array(all_ids, dtype=object))
    pd.DataFrame({"sample_id": all_ids, "source": all_sources}).to_csv(source_labels_path, sep="\t", index=False)
    pd.DataFrame(overlap_rows).to_csv(probe_overlap_path, sep="\t", index=False)

    print(f"\n[saved] {x_path}  {x.shape}")
    print(f"[saved] {sample_ids_path}")
    print(f"[saved] {source_labels_path}")
    print(f"[saved] {probe_overlap_path}")

    return CombineResult(
        x_path=x_path,
        sample_ids_path=sample_ids_path,
        source_labels_path=source_labels_path,
        probe_overlap_path=probe_overlap_path,
        n_samples=x.shape[0],
        n_features=x.shape[1],
    )
