"""Decoder-space multiple imputation shared by the two TVAE engines."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .config import LatentConfig
from .latent import condition_latents
from .spans import ColumnSpan, column_loglik, sample_spans

# decoder(z (N, E)) -> (rec (N, D), sigmas (D,))
DecoderFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
# encoder(x (R, D), obs (R, C)) -> (mu (R, E), std (R, E))
EncoderFn = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@torch.no_grad()
def draw_completions(
    decoder: DecoderFn,
    x: np.ndarray,
    obs: np.ndarray,
    spans: list[ColumnSpan],
    emb_dim: int,
    m: int,
    cfg: LatentConfig,
    power: float,
    device,
    seed: int,
    encoder: EncoderFn | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sample ``m`` transformed-space completions per row.

    ``x`` is the transformed data (missing spans hold placeholders), ``obs``
    the (rows, columns) observed mask. Returns ``(samples (R, M, D), diag)``;
    observed spans in the samples are model draws and must be overwritten by
    the caller at the raw level.
    """
    n, data_dim = x.shape
    generator = torch.Generator(device=device).manual_seed(int(seed))
    chunk = max(1, cfg.max_batch // max(cfg.k, cfg.smc_particles or 0, m))
    out = np.zeros((n, m, data_dim), dtype=np.float32)
    diag = {
        "ess": np.full(n, np.nan),
        "refined": np.zeros(n, bool),
        "acceptance": np.full(n, np.nan),
        "smc_stages": np.full(n, np.nan),
    }

    for st in range(0, n, chunk):
        xc = torch.as_tensor(x[st : st + chunk], dtype=torch.float32, device=device)
        oc = torch.as_tensor(obs[st : st + chunk], dtype=torch.float32, device=device)

        def loglik(z: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
            xr = xc if rows is None else xc[rows]
            orr = oc if rows is None else oc[rows]
            r, k, e = z.shape
            rec, sig = decoder(z.reshape(-1, e))
            rec = rec.reshape(r, k, -1)
            per_col = column_loglik(rec, sig, xr[:, None, :].expand_as(rec), spans)
            return power * (per_col * orr[:, None, :]).sum(-1)

        proposal = encoder(xc, oc) if encoder is not None else None
        z, d = condition_latents(loglik, xc.shape[0], emb_dim, m, cfg, device, generator, proposal)
        rec, sig = decoder(z.reshape(-1, emb_dim))
        samples = sample_spans(rec, sig, spans, generator).reshape(xc.shape[0], m, data_dim)
        out[st : st + chunk] = samples.cpu().numpy()
        for key in diag:
            diag[key][st : st + chunk] = d[key]
    return out, diag


def summarise_diagnostics(diag: dict[str, np.ndarray], m: int) -> dict[str, float]:
    """Scalar summary of per-row latent diagnostics."""
    ess = diag.get("ess", np.array([]))
    acc = diag.get("acceptance", np.array([]))
    refined = diag.get("refined", np.array([]))
    stages = diag.get("smc_stages", np.array([]))
    return {
        "rows_conditioned": int(ess.size),
        "ess_median": float(np.nanmedian(ess)) if ess.size else float("nan"),
        "ess_min": float(np.nanmin(ess)) if ess.size else float("nan"),
        "frac_ess_below_m": float(np.mean(ess < m)) if ess.size else float("nan"),
        "frac_refined": float(np.mean(refined)) if refined.size else 0.0,
        "mcmc_acceptance_median": float(np.nanmedian(acc)) if np.isfinite(acc).any() else float("nan"),
        "smc_stages_median": float(np.nanmedian(stages)) if np.isfinite(stages).any() else float("nan"),
    }
