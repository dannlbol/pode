"""Decoder-space utilities for ctgan-style TVAE representations.

ctgan's ``DataTransformer`` encodes each raw column as a *span*:
  * continuous -> one tanh scalar (mode-normalised value) + a one-hot mode
    indicator from a Bayesian GMM (``ClusterBasedNormalizer``);
  * discrete   -> a one-hot category vector.

The functions here compute per-column log-likelihoods of observed values under
the decoder (a masked version of ctgan's ``_loss_function``) and sample
missing spans properly: the scalar from N(tanh(rec), sigma) and mode/category
indicators from the softmax, rather than taking an argmax, so repeated draws
reflect the model's uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.functional import log_softmax


@dataclass(frozen=True)
class ColumnSpan:
    name: str
    kind: str  # 'continuous' | 'discrete'
    start: int
    dim: int

    @property
    def end(self) -> int:
        return self.start + self.dim


def spans_from_transformer(transformer) -> list[ColumnSpan]:
    """Read column spans from a fitted ctgan ``DataTransformer``."""
    spans, st = [], 0
    for info in transformer._column_transform_info_list:
        spans.append(ColumnSpan(info.column_name, info.column_type, st, int(info.output_dimensions)))
        st += int(info.output_dimensions)
    if st != transformer.output_dimensions:
        raise RuntimeError("Transformer span layout does not match its output dimensions.")
    return spans


def seed_transformer(transformer, seed: int) -> None:
    """Make ctgan encodings reproducible.

    RDT's ``ClusterBasedNormalizer`` samples each value's GMM mode from a
    per-transformer RandomState that advances across calls (the global NumPy
    seed is ignored), so reseed every column's transform state from ``seed``.
    """
    for j, info in enumerate(transformer._column_transform_info_list):
        rdt = info.transform
        if hasattr(rdt, "set_random_state"):
            rdt.set_random_state(np.random.RandomState((int(seed) + j) % 2**32), "transform")


def expand_mask(col_mask: np.ndarray, spans: list[ColumnSpan]) -> np.ndarray:
    """Broadcast a (rows, columns) mask onto the (rows, data_dim) span layout."""
    data_dim = spans[-1].end if spans else 0
    out = np.zeros((col_mask.shape[0], data_dim), dtype=col_mask.dtype)
    for j, sp in enumerate(spans):
        out[:, sp.start : sp.end] = col_mask[:, j : j + 1]
    return out


def column_loglik(
    rec: torch.Tensor,
    sigmas: torch.Tensor,
    x: torch.Tensor,
    spans: list[ColumnSpan],
) -> torch.Tensor:
    """Per-column log p(x_j | z) for every row: shape (..., n_columns).

    ``rec`` is the raw decoder output, ``x`` the transformed data (same
    leading shape), ``sigmas`` the decoder's per-dimension scale. Continuous
    columns contribute a Gaussian term on the scalar plus the mode's
    categorical term; discrete columns a categorical term.
    """
    cols = []
    for sp in spans:
        if sp.kind == "continuous":
            s = sp.start
            sigma = sigmas[s]
            resid = x[..., s] - torch.tanh(rec[..., s])
            gauss = -0.5 * (resid / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
            if sp.dim > 1:
                lp = log_softmax(rec[..., s + 1 : sp.end], dim=-1)
                mode = (lp * x[..., s + 1 : sp.end]).sum(-1)
            else:
                mode = torch.zeros_like(gauss)
            cols.append(gauss + mode)
        else:
            lp = log_softmax(rec[..., sp.start : sp.end], dim=-1)
            cols.append((lp * x[..., sp.start : sp.end]).sum(-1))
    return torch.stack(cols, dim=-1)


def sample_spans(
    rec: torch.Tensor,
    sigmas: torch.Tensor,
    spans: list[ColumnSpan],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw a transformed-space row from the decoder output (one per row of ``rec``)."""
    out = torch.zeros_like(rec)
    for sp in spans:
        if sp.kind == "continuous":
            s = sp.start
            noise = torch.randn(rec.shape[:-1], generator=generator, device=rec.device)
            out[..., s] = torch.tanh(rec[..., s]) + sigmas[s] * noise
            if sp.dim > 1:
                out[..., s + 1 : sp.end] = _sample_onehot(rec[..., s + 1 : sp.end], generator)
        else:
            out[..., sp.start : sp.end] = _sample_onehot(rec[..., sp.start : sp.end], generator)
    return out


def _sample_onehot(logits: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    flat = logits.reshape(-1, logits.shape[-1])
    idx = torch.multinomial(torch.softmax(flat, dim=-1), 1, generator=generator).squeeze(-1)
    onehot = torch.zeros_like(flat)
    onehot[torch.arange(flat.shape[0], device=flat.device), idx] = 1.0
    return onehot.reshape(logits.shape)
