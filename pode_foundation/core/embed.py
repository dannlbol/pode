"""Apply a frozen (or fine-tuned) MAE encoder to a beta matrix; dump embeddings.

Accepts wide CSV/TSV/TSV.gz or parquet beta matrices (probes x samples, or
samples x probes via ``matrix.orientation``), aligns them to the encoder's
training feature set, mean-imputes any probes missing from this particular
cohort, and writes a ``sample_id x z000..zN`` embeddings CSV.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .mae_model import MethylationMAE


@dataclass
class EmbedResult:
    """Outcome of a ``run_embed`` call -- the written CSV plus its shape,
    so callers (the CLI, tests) can report/assert on it without re-reading
    the file."""

    out_path: Path
    n_samples: int
    n_latent: int


def _load_feature_ids(path) -> np.ndarray:
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=True).astype(str)
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).to_numpy()


def _load_betas(path, orientation: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        b = pd.read_parquet(path)
    else:
        sep = "\t" if path.endswith((".tsv", ".tsv.gz")) else ","
        b = pd.read_csv(path, sep=sep, index_col=0)
    if "probe_id" in b.columns:
        b = b.set_index("probe_id")
    if orientation == "samples_by_features":
        b = b.T
    return b.astype(np.float32)


def align_betas(betas_path, orientation: str, feature_ids: np.ndarray, sample_ids=None) -> tuple[np.ndarray, list[str]]:
    """Load a beta matrix and align it to ``feature_ids`` (row order, imputing
    any probes missing from this cohort). Returns ``(X, sample_ids)`` with
    ``X`` shaped ``(n_samples, len(feature_ids))``, clipped to ``[0, 1]``.

    Shared by :func:`run_embed` and the survival ``finetune`` mode so both
    align a raw beta matrix to a checkpoint's training feature set the same
    way.
    """
    b = _load_betas(betas_path, orientation)
    ids = list(dict.fromkeys(b.columns)) if sample_ids is None else [s for s in sample_ids if s in set(b.columns)]
    print(f"[samples] {len(ids)}")

    sub = b.reindex(index=feature_ids, columns=ids)
    if sub.isna().values.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fill = np.nanmean(sub.to_numpy(), axis=1, keepdims=True)
        # A probe entirely absent from this cohort has no samples to average
        # over (fill is itself NaN) -- fall back to 0.5, the same neutral
        # prior the MAE uses for masked/unknown positions during training.
        n_no_support = int(np.isnan(fill).sum())
        fill = np.where(np.isnan(fill), 0.5, fill)
        arr = sub.to_numpy()
        nan = np.isnan(arr)
        arr[nan] = np.broadcast_to(fill, arr.shape)[nan]
        sub = pd.DataFrame(arr, index=feature_ids, columns=ids)
        print(f"[impute] filled {int(nan.sum())} NaNs with per-feature mean (missing probes for this cohort)")
        if n_no_support:
            print(f"[impute] {n_no_support} probes had zero support in this cohort -- filled with 0.5 (neutral prior)")

    x = np.clip(sub.to_numpy().T.astype(np.float32), 0, 1)
    return x, ids


def _load_encoder(path, n_features: int, device) -> MethylationMAE:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    mae = MethylationMAE(
        n_features=n_features,
        encoder_dims=tuple(a["encoder_dims"]) if "encoder_dims" in a else (1024, 512),
        latent_dim=a.get("latent_dim", 256),
        dropout=a.get("dropout", 0.0),
        output_activation=a.get("output_activation", "sigmoid"),
    )
    mae.load_state_dict({k: v for k, v in ckpt["model_state"].items() if k in mae.state_dict()}, strict=False)
    return mae.to(device).eval()


def run_embed(cfg, ckpt, betas=None, out=None, sample_ids=None) -> EmbedResult:
    emb = cfg["embed"]
    ckpt = Path(ckpt)
    betas_path = betas or cfg["matrix"].get("path")
    if not betas_path:
        raise ValueError("embed: no beta matrix given (config matrix.path or --betas).")
    out_path = Path(out or emb["output_path"])

    feature_ids_path = emb.get("feature_ids_path") or cfg["preprocess"].get("feature_ids_path")
    if not feature_ids_path:
        raise ValueError("embed: no feature-id list given (config embed.feature_ids_path).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat = _load_feature_ids(feature_ids_path)
    x, ids = align_betas(betas_path, cfg["matrix"]["orientation"], feat, sample_ids=sample_ids)
    mae = _load_encoder(ckpt, len(feat), device)
    with torch.no_grad():
        lat = mae.encode(torch.tensor(x, device=device)).cpu().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f"z{i:03d}" for i in range(lat.shape[1])]
    pd.DataFrame(lat, index=pd.Index(ids, name="sample_id"), columns=cols).to_csv(out_path)
    print(f"[saved] {out_path}  ({lat.shape[0]} x {lat.shape[1]})")
    return EmbedResult(out_path=out_path, n_samples=lat.shape[0], n_latent=lat.shape[1])
