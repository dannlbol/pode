"""Discrete-time (logistic-hazard) survival likelihood.

Parameterisation (Gensheimer & Narasimhan "nnet-survival"; pycox LogisticHazard):
the head emits ``K`` logits, one per interval. Let ``h_k = sigmoid(logit_k)`` be
the *hazard* -- the conditional probability of the event in interval ``k`` given
survival to the start of ``k``. For a subject observed to interval ``j`` with
event indicator ``d``:

    log L = sum_{k<j} log(1 - h_k) + d*log(h_j) + (1-d)*log(1 - h_j)

This is exactly a **masked binary cross-entropy**: build a target ``y`` that is 1
only at ``(k == j and d == 1)`` and a mask that is 1 for ``k <= j`` and 0 after.
Because the loss is per subject (no risk set), it minibatches cleanly and, unlike
a Cox head, lets you *sample* an actual time-to-event at generation time.

Both a Torch implementation (for training) and a NumPy reference (for testing and
for the fallback synthesizer's survival sampling) are provided and are kept
numerically identical.
"""
from __future__ import annotations

import numpy as np

try:  # torch is optional; only needed for training the SurvivalVAE.
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only when torch is absent
    _HAS_TORCH = False


def build_targets(idx: np.ndarray, event: np.ndarray, n_bins: int):
    """Construct (target, mask) arrays for the masked-BCE formulation.

    Parameters
    ----------
    idx : np.ndarray
        Interval index ``j`` per subject (from :meth:`TimeGrid.index`).
    event : np.ndarray
        1/0 event indicator per subject.
    n_bins : int
        Number of intervals ``K``.

    Returns
    -------
    y : np.ndarray, shape (n, K)
        1 at the event interval for uncensored subjects, else 0.
    mask : np.ndarray, shape (n, K)
        1 for intervals ``k <= j`` (the subject contributes there), else 0.
    """
    idx = np.asarray(idx, dtype=int)
    event = np.asarray(event).astype(float)
    n = idx.shape[0]
    ks = np.arange(n_bins)[None, :]  # (1, K)
    j = idx[:, None]  # (n, 1)
    mask = (ks <= j).astype(float)
    y = np.zeros((n, n_bins), dtype=float)
    hit = event.astype(bool)
    y[hit, idx[hit]] = 1.0
    return y, mask


def nll_numpy(logits: np.ndarray, idx: np.ndarray, event: np.ndarray) -> float:
    """Mean discrete-time NLL over a batch (NumPy reference)."""
    logits = np.asarray(logits, dtype=float)
    n, K = logits.shape
    y, mask = build_targets(idx, event, K)
    # Numerically stable elementwise BCE-with-logits.
    # bce = max(x,0) - x*y + log(1+exp(-|x|))
    x = logits
    bce = np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))
    per_subject = (bce * mask).sum(axis=1)
    return float(per_subject.mean())


def survival_from_logits_numpy(logits: np.ndarray) -> np.ndarray:
    """Discrete survival function ``S_k = prod_{i<=k}(1 - h_i)`` (NumPy).

    Returns array of shape ``(n, K)`` where column ``k`` is the probability of
    surviving through the end of interval ``k``.
    """
    logits = np.asarray(logits, dtype=float)
    h = 1.0 / (1.0 + np.exp(-logits))
    return np.cumprod(1.0 - h, axis=1)


def risk_from_logits_numpy(logits: np.ndarray) -> np.ndarray:
    """Scalar risk per subject for ranking metrics: ``1 - S_{K-1}``.

    Monotone in cumulative hazard, so suitable for concordance.
    """
    S = survival_from_logits_numpy(logits)
    return 1.0 - S[:, -1]


if _HAS_TORCH:

    def nll_torch(
        logits: "torch.Tensor",
        idx: "torch.Tensor",
        event: "torch.Tensor",
    ) -> "torch.Tensor":
        """Mean discrete-time NLL (Torch, differentiable).

        Parameters
        ----------
        logits : (n, K) float tensor
        idx : (n,) long tensor of interval indices
        event : (n,) float/bool tensor of event indicators
        """
        n, K = logits.shape
        device = logits.device
        ks = torch.arange(K, device=device).unsqueeze(0)  # (1, K)
        j = idx.long().unsqueeze(1)  # (n, 1)
        mask = (ks <= j).float()
        y = torch.zeros_like(logits)
        ev = event.bool()
        if ev.any():
            rows = torch.nonzero(ev, as_tuple=False).squeeze(1)
            y[rows, idx.long()[rows]] = 1.0
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        per_subject = (bce * mask).sum(dim=1)
        return per_subject.mean()

    def survival_from_logits_torch(logits: "torch.Tensor") -> "torch.Tensor":
        h = torch.sigmoid(logits)
        return torch.cumprod(1.0 - h, dim=1)
