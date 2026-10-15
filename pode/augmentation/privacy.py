"""Memorisation and disclosure checks for a synthetic pool.

Each split supplies a natural reference: its held-out real patients are
*non-members* the generator never saw, so "how close are synthetic rows to
training patients" can be compared with "how close are genuinely new patients
to training patients".

* ``dcr_*``: Gower distance from each synthetic row to its closest training row
  (DCR). ``dcr_p5_ratio`` = 5th percentile of synthetic DCR / 5th percentile of
  held-out DCR: ~1 means synthetic rows are no closer to training patients than
  new patients are; well below 1 means near-copies.
* ``exact_duplicates`` / ``near_copy_frac``: synthetic rows identical to, or
  closer than the held-out 5th percentile to, a training row.
* ``nndr_mean``: nearest/second-nearest distance ratio (low = sits on one record).
* ``membership_auc``: a distance attack. Each real patient is scored by its
  distance to the nearest synthetic row; AUC for telling training patients from
  held-out ones. 0.5 = no membership signal; > 0.6 = worrying.
* ``rare_combo_leaks``: synthetic rows reproducing a categorical combination seen
  in at most ``rare_k`` training rows, and lying within near-copy distance of one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gower_matrix(A: pd.DataFrame, B: pd.DataFrame, numeric: list[str], categorical: list[str],
                 ranges: pd.Series) -> np.ndarray:
    """Gower distance between every row of A and every row of B (NaN-aware)."""
    num = np.zeros((len(A), len(B)))
    cnt = np.zeros((len(A), len(B)))
    for c in numeric:
        a = pd.to_numeric(A[c], errors="coerce").to_numpy(float)[:, None]
        b = pd.to_numeric(B[c], errors="coerce").to_numpy(float)[None, :]
        d = np.abs(a - b) / (ranges[c] or 1.0)
        ok = np.isfinite(d)
        num += np.where(ok, d, 0.0)
        cnt += ok
    for c in categorical:
        a = A[c].astype(str).to_numpy()[:, None]
        b = B[c].astype(str).to_numpy()[None, :]
        num += a != b
        cnt += 1
    return num / np.maximum(cnt, 1)


def privacy_metrics(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    synth: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    rare_k: int = 3,
    max_synthetic: int = 2000,
    seed: int = 0,
) -> dict:
    from sklearn.metrics import roc_auc_score

    numeric = [c for c in numeric if c in synth.columns]
    categorical = [c for c in categorical if c in synth.columns]
    if synth.empty or (not numeric and not categorical):
        return {}
    rng = np.random.default_rng(seed)
    if len(synth) > max_synthetic:
        synth = synth.iloc[rng.choice(len(synth), max_synthetic, replace=False)]
    num = train[numeric].apply(pd.to_numeric, errors="coerce")
    ranges = (num.max() - num.min()).replace(0, 1.0)

    d_st = gower_matrix(synth, train, numeric, categorical, ranges)
    d_ht = gower_matrix(holdout, train, numeric, categorical, ranges)
    two = np.sort(d_st, axis=1)[:, :2]
    dcr_s, dcr_h = two[:, 0], d_ht.min(axis=1)
    p5_h = float(np.percentile(dcr_h, 5)) if len(dcr_h) else np.nan
    out = {
        "dcr_synth_median": float(np.median(dcr_s)),
        "dcr_holdout_median": float(np.median(dcr_h)) if len(dcr_h) else np.nan,
        "dcr_p5_ratio": float(np.percentile(dcr_s, 5) / p5_h) if p5_h and p5_h > 0 else np.nan,
        "exact_duplicates": int((dcr_s < 1e-9).sum()),
        "near_copy_frac": float((dcr_s < p5_h).mean()) if np.isfinite(p5_h) else np.nan,
        "nndr_mean": float(np.mean(two[:, 0] / np.maximum(two[:, 1], 1e-12))) if two.shape[1] == 2 else np.nan,
    }

    # Distance-to-synthetic membership attack: members are closer if memorised.
    if len(holdout) >= 5:
        d_tr_s = gower_matrix(train, synth, numeric, categorical, ranges).min(axis=1)
        d_ho_s = gower_matrix(holdout, synth, numeric, categorical, ranges).min(axis=1)
        y = np.r_[np.ones(len(d_tr_s)), np.zeros(len(d_ho_s))]
        out["membership_auc"] = float(roc_auc_score(y, -np.r_[d_tr_s, d_ho_s]))

    if categorical:
        tkey = train[categorical].astype(str).agg("|".join, axis=1)
        counts = tkey.value_counts()
        rare = set(counts[counts <= rare_k].index)
        skey = synth[categorical].astype(str).agg("|".join, axis=1).to_numpy()
        hit = np.array([k in rare for k in skey])
        close = dcr_s < (p5_h if np.isfinite(p5_h) else 0)
        out["rare_combo_rows"] = int(hit.sum())
        out["rare_combo_leaks"] = int((hit & close).sum())
    return out
