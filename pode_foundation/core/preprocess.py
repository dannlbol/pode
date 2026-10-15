"""Quality filtering + genomic LD pruning of a methylation beta matrix.

Turns an arbitrary probes-by-samples beta-value matrix (any array platform,
any tumour type) plus a chr/pos probe annotation into a fixed-size,
approximately independent feature set for masked-autoencoder pretraining:

  1. Drop sex-chromosome probes (chrX/chrY) — otherwise sex becomes a
     dominant, confounding axis of variation.
  2. Drop any probe with a missing value in any sample.
  3. Optionally restrict to an EPIC/450K probe overlap list, and/or drop a
     cross-reactive probe list (both just plain one-column probe-ID files).
  4. Rank the survivors by standard deviation and take the top
     ``n_candidates``.
  5. Greedily prune correlated neighbours within a genomic window: walking
     candidates in descending-SD order, exclude any not-yet-excluded probe on
     the same chromosome within ``window_bp`` base pairs whose |Pearson r|
     with the kept probe exceeds ``r_thresh``.
  6. Keep the first ``n_keep`` survivors.

This keeps the final probe set both high-variance AND approximately
independent within the window, which avoids the effective dimensionality
being inflated by redundant, highly-correlated CpG clusters.

Memory note: loading n_candidates x n_samples as float32 — e.g.
200,000 x 300 x 4 bytes ~= 240 MB, well within reach on a laptop; scale
n_candidates down if you're memory constrained on a much larger cohort.
"""

from __future__ import annotations

import bisect
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PreprocessResult:
    """Paths and shape of a completed ``run_preprocess`` call.

    ``feature_ids_path`` is the one downstream commands need most -- it's the
    fixed feature set ``pretrain``/``finetune`` train against and ``embed``
    aligns any future cohort to. ``feature_stats_path`` (per-probe mean/SD)
    and ``pruning_report_path`` (per-candidate kept/excluded reason) are
    diagnostic, not required by any other command.
    """

    out_dir: Path
    x_path: Path
    sample_ids_path: Path
    feature_ids_path: Path
    feature_stats_path: Path
    pruning_report_path: Path
    n_samples: int
    n_features: int


def load_beta(path: Path, orientation: str = "features_by_samples") -> pd.DataFrame:
    """Load a beta matrix; always returns it as probes (index) x samples (columns)
    regardless of the file's own orientation."""
    path = Path(path)
    print(f"[load] {path}")
    t0 = time.time()
    if path.suffix == ".parquet":
        beta = pd.read_parquet(path)
        if "probe_id" in beta.columns:
            beta = beta.set_index("probe_id")
    else:
        sep = "," if path.suffixes and path.suffixes[0] == ".csv" else "\t"
        beta = pd.read_csv(path, sep=sep, index_col=0, low_memory=False)
    print(f"  raw shape: {beta.shape}  ({time.time() - t0:.0f}s)")
    pval_cols = [c for c in beta.columns if str(c).startswith("Detection Pval")]
    if pval_cols:
        beta = beta.drop(columns=pval_cols)
        print(f"  dropped {len(pval_cols)} detection p-value columns -> {beta.shape[1]} samples")
    if orientation == "samples_by_features":
        beta = beta.T
        print(f"  transposed to probes x samples: {beta.shape}")
    return beta


def load_anno(path: Path) -> pd.DataFrame:
    """Load a probe annotation file; auto-detects Name/chr/pos-style column names."""
    anno = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in anno.columns}
    name_col = cols_lower.get("name") or cols_lower.get("ilmnid") or anno.columns[0]
    chr_col = cols_lower.get("chr") or cols_lower.get("chromosome")
    pos_col = cols_lower.get("pos") or cols_lower.get("mapinfo") or cols_lower.get("position")
    if chr_col is None:
        raise ValueError(f"No chromosome column found in {path}")
    if pos_col is None:
        raise ValueError(f"No position column found in {path} - required for LD pruning")
    anno = anno.rename(columns={name_col: "Name", chr_col: "chr", pos_col: "pos"})
    return anno[["Name", "chr", "pos"]].dropna(subset=["chr", "pos"])


def _ld_prune(
    betas_t: np.ndarray,  # (n_candidates, n_samples) float32, row = probe
    probe_ids: np.ndarray,  # (n_candidates,) in descending-SD order
    chrs: np.ndarray,  # (n_candidates,) chromosome strings
    positions: np.ndarray,  # (n_candidates,) int64 genomic positions
    r_thresh: float,
    window_bp: int,
    n_keep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy LD pruning. Returns (kept probe IDs, kept candidate indices)."""
    n_candidates = len(probe_ids)
    excluded = np.zeros(n_candidates, dtype=bool)
    kept_indices: list[int] = []

    chr_pos_idx: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, (ch, pos) in enumerate(zip(chrs, positions)):
        chr_pos_idx[ch].append((int(pos), i))
    for ch in chr_pos_idx:
        chr_pos_idx[ch].sort()
    chr_sorted_pos = {ch: np.array([x[0] for x in lst]) for ch, lst in chr_pos_idx.items()}

    def _corr_vec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xz = x - x.mean()
        xz_norm = np.linalg.norm(xz)
        if xz_norm < 1e-10:
            return np.zeros(len(y))
        xz /= xz_norm
        yz = y - y.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(yz, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        yz /= norms
        return yz @ xz

    t0 = time.time()
    for rank, idx in enumerate(range(n_candidates)):
        if excluded[idx]:
            continue
        if len(kept_indices) >= n_keep:
            break
        kept_indices.append(idx)

        ch = chrs[idx]
        pos = int(positions[idx])
        pos_arr = chr_sorted_pos[ch]
        ch_probes = chr_pos_idx[ch]
        lo = bisect.bisect_left(pos_arr, pos - window_bp)
        hi = bisect.bisect_right(pos_arr, pos + window_bp)

        beta_i = betas_t[idx]
        nb_indices = [
            ch_probes[j][1]
            for j in range(lo, hi)
            if ch_probes[j][1] != idx and not excluded[ch_probes[j][1]]
        ]
        if nb_indices:
            nb_betas = betas_t[nb_indices]
            r_vals = _corr_vec(beta_i.copy(), nb_betas.copy())
            to_exclude = [nb_indices[j] for j in range(len(nb_indices)) if abs(r_vals[j]) > r_thresh]
            for ex_idx in to_exclude:
                excluded[ex_idx] = True

        if (rank + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(
                f"  processed {rank + 1}/{n_candidates}  kept={len(kept_indices)}  "
                f"excluded={excluded.sum()}  ({elapsed:.0f}s elapsed)"
            )

    kept_indices = np.array(kept_indices, dtype=np.int64)
    kept_probe_ids = probe_ids[kept_indices]
    print(
        f"[prune] kept {len(kept_probe_ids)} / {n_candidates} candidates "
        f"(excluded {excluded.sum()})  target={n_keep}"
    )
    if len(kept_probe_ids) < n_keep:
        print(f"[warn] only {len(kept_probe_ids)} probes survived pruning (< n_keep={n_keep}).")
    return kept_probe_ids, np.array(kept_indices)


def run_preprocess(cfg, beta_path=None, anno_path=None, out_dir=None) -> PreprocessResult:
    """Run QC filtering + LD pruning per ``cfg["preprocess"]`` / ``cfg["matrix"]``.

    Explicit ``beta_path``/``anno_path``/``out_dir`` arguments (e.g. from CLI
    flags) take priority over the config file's values.
    """
    pp = cfg["preprocess"]
    beta_path = beta_path or cfg["matrix"].get("path")
    anno_path = anno_path or pp.get("annotation_path")
    if not beta_path:
        raise ValueError("preprocess: no beta matrix given (config matrix.path or --beta).")
    if not anno_path:
        raise ValueError("preprocess: no annotation file given (config preprocess.annotation_path or --anno).")
    beta_path, anno_path = Path(beta_path), Path(anno_path)

    out = Path(out_dir or pp["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    beta = load_beta(beta_path, orientation=cfg["matrix"]["orientation"])
    anno = load_anno(anno_path)
    print(f"[anno] {anno.shape[0]} probes with chr+pos")

    # sex chromosomes (always dropped - a strong, confounding axis of variation)
    sex = set(anno.loc[anno["chr"].isin(["chrX", "chrY", "X", "Y"]), "Name"])
    n0 = beta.shape[0]
    beta = beta.loc[~beta.index.isin(sex)]
    print(f"[filter] sex chr:    {n0} -> {beta.shape[0]}")

    if pp.get("epic_overlap_path"):
        epic = set(pd.read_csv(pp["epic_overlap_path"], header=None)[0].astype(str))
        n0 = beta.shape[0]
        beta = beta.loc[beta.index.isin(epic)]
        print(f"[filter] EPIC overlap: {n0} -> {beta.shape[0]}")

    n0 = beta.shape[0]
    beta = beta.loc[beta.isna().sum(axis=1) == 0]
    print(f"[filter] NA probes:  {n0} -> {beta.shape[0]}")

    if pp.get("cross_reactive_path"):
        xr = set(pd.read_csv(pp["cross_reactive_path"], header=None)[0].astype(str))
        n0 = beta.shape[0]
        beta = beta.loc[~beta.index.isin(xr)]
        print(f"[filter] cross-react: {n0} -> {beta.shape[0]}")

    print(f"\n[SD] computing standard deviations for {beta.shape[0]} probes ...")
    sds = beta.std(axis=1)

    n_cand = min(pp["n_candidates"], beta.shape[0])
    top_cand = sds.nlargest(n_cand).index
    beta_cand = beta.loc[top_cand]
    sds_cand = sds[top_cand]
    print(f"[candidates] {n_cand} probes  SD range {sds_cand.min():.4f} - {sds_cand.max():.4f}")

    anno_idx = anno.set_index("Name")
    cand_anno = anno_idx.reindex(top_cand)
    missing_anno = cand_anno["chr"].isna().sum()
    if missing_anno > 0:
        print(f"[warn] {missing_anno} candidates lack chr/pos annotation - excluded from pruning")
        beta_cand = beta_cand.loc[cand_anno["chr"].notna()]
        cand_anno = cand_anno.loc[cand_anno["chr"].notna()]
        sds_cand = sds_cand[cand_anno.index]

    probe_ids = np.array(cand_anno.index)
    chrs = cand_anno["chr"].values
    positions = cand_anno["pos"].values.astype(np.int64)

    n_keep = pp["n_keep"]
    print(f"\n[prune] r_thresh={pp['r_thresh']}  window={pp['window_bp']} bp  target={n_keep}")
    betas_t = beta_cand.to_numpy(dtype=np.float32)
    kept_probe_ids, kept_indices = _ld_prune(
        betas_t=betas_t,
        probe_ids=probe_ids,
        chrs=chrs,
        positions=positions,
        r_thresh=pp["r_thresh"],
        window_bp=pp["window_bp"],
        n_keep=n_keep,
    )

    beta_out = beta_cand.loc[kept_probe_ids[:n_keep]]
    x = beta_out.T.to_numpy(dtype=np.float32)
    sample_ids = np.asarray(beta_out.columns)
    feature_ids = np.asarray(beta_out.index)

    x_path = out / "X.npy"
    sample_ids_path = out / "sample_ids.npy"
    feature_ids_path = out / "feature_ids.npy"
    feature_stats_path = out / "feature_stats.tsv"
    pruning_report_path = out / "pruning_report.tsv"

    np.save(x_path, x)
    np.save(sample_ids_path, sample_ids)
    np.save(feature_ids_path, feature_ids)

    kept_sds = sds_cand[feature_ids]
    kept_means = beta_cand.loc[feature_ids].mean(axis=1)
    stats = pd.DataFrame({"sd": kept_sds.values, "mean": kept_means.values}, index=feature_ids)
    stats.to_csv(feature_stats_path, sep="\t")

    all_flags = pd.Series("excluded_correlated", index=probe_ids)
    all_flags.iloc[kept_indices] = "kept"
    report = pd.DataFrame(
        {"probe": probe_ids, "chr": chrs, "pos": positions, "sd": sds_cand.values, "status": all_flags.values}
    )
    report.to_csv(pruning_report_path, sep="\t", index=False)

    print(f"\n[saved] {x_path}  {x.shape}  {x.dtype}")
    print(f"[saved] {sample_ids_path}  {sample_ids.shape}")
    print(f"[saved] {feature_ids_path}  {feature_ids.shape}")
    print(f"[saved] {feature_stats_path}")
    print(f"[saved] {pruning_report_path}")

    return PreprocessResult(
        out_dir=out,
        x_path=x_path,
        sample_ids_path=sample_ids_path,
        feature_ids_path=feature_ids_path,
        feature_stats_path=feature_stats_path,
        pruning_report_path=pruning_report_path,
        n_samples=x.shape[0],
        n_features=x.shape[1],
    )
