"""Per-feature-weighted masked-MSE loss for :class:`MethylationMAE`.

Reuses the architecture UNCHANGED (same symmetric MLP, same encode/forward,
same masking). The only addition is a per-feature-weighted reconstruction
loss: at moderate mask ratios the flat MSE can still let the bottleneck
compress a small set of features of particular interest toward the mean, so
this lets you up-weight the reconstruction error on a chosen subset of
columns by a factor ``w`` (e.g. a locus known to be relevant for your tumour
type). Set ``w=1.0`` to recover the standard flat MSE.

Usage:
    from pode_foundation.core.mae_model import MethylationMAE, make_mask
    from pode_foundation.core.mae_model_weighted import (
        weighted_masked_mse_loss, make_feature_weight,
    )

    feat_w = make_feature_weight(feature_mask, w=5.0, device=device)  # (F,)
    x_hat, z = model(x, mask)
    loss = weighted_masked_mse_loss(x_hat, x, mask, feat_w)
"""

from __future__ import annotations

import numpy as np
import torch

# re-export the architecture so callers can import everything from one place
from .mae_model import MethylationMAE, make_mask, masked_mse_loss  # noqa: F401


def make_feature_weight(feature_mask, w: float, device) -> torch.Tensor:
    """Per-feature weight vector: masked-in columns get `w`, others get 1.0.
    Normalised so the mean weight is 1.0, keeping the overall loss scale stable
    (so `w` doesn't secretly act as a learning-rate multiplier on the loss)."""
    mask = np.asarray(feature_mask).astype(bool)
    fw = np.ones(len(mask), dtype=np.float32)
    fw[mask] = float(w)
    fw = fw / fw.mean()  # mean-1 normalisation
    return torch.tensor(fw, device=device)


def weighted_masked_mse_loss(x_hat, x, mask, feat_weight) -> torch.Tensor:
    """MSE on hidden positions (mask==0), weighted per feature.

    feat_weight: (n_features,) tensor from make_feature_weight.
    Reduces to masked_mse_loss when feat_weight is all-ones.
    """
    hidden = 1.0 - mask  # (B, F)
    w = feat_weight.unsqueeze(0)  # (1, F) broadcast over batch
    sq = (x_hat - x).pow(2) * hidden * w
    denom = (hidden * w).sum().clamp_min(1.0)
    return sq.sum() / denom
