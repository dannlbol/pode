"""Conditioning a TVAE's latent code on the observed part of a row.

The target is the posterior p(z | x_obs) ∝ N(z; 0, I) · p(x_obs | z)^power.
Two samplers are provided, both vectorised over rows:

* **SIR** (sampling-importance-resampling): draw K particles from a proposal
  (the prior, or an encoder's q(z | x_obs, m)), weight by
  likelihood · prior / proposal, and resample M draws. The effective sample
  size (ESS) of the weights is reported per row.
* **pCN** (preconditioned Crank-Nicolson) MCMC: z' = sqrt(1-b²)·z + b·ξ is
  reversible w.r.t. the N(0, I) prior, so acceptance depends only on the
  likelihood ratio. It mixes well in high dimension (TVAE latents are 128-d)
  and refines SIR draws for rows where SIR degenerates.

* **Tempered SMC**: anneals the likelihood in from the prior with pCN moves
  at each temperature; robust when the likelihood is sharply peaked.

``method='auto'`` runs SIR and, for rows whose ESS falls below a threshold,
uses SMC (prior proposal) or pCN refinement of the SIR draws (encoder proposal).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from .config import LatentConfig

# ``fn(z, rows=None)``: z (R, K, E) -> tempered log p(x_obs | z) of shape (R, K).
# ``rows`` optionally selects a subset of the rows the function was built for.
LogLikFn = Callable[..., torch.Tensor]


def _log_normal(z: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (-0.5 * ((z - mu) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)


@torch.no_grad()
def sir(
    loglik_fn: LogLikFn,
    n_rows: int,
    emb_dim: int,
    k: int,
    m: int,
    device,
    generator: torch.Generator,
    proposal: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(z draws (R, M, E), ESS (R,))``."""
    eps = torch.randn((n_rows, k, emb_dim), generator=generator, device=device)
    if proposal is None:
        z = eps
        logw = loglik_fn(z)
    else:
        mu, std = proposal[0][:, None, :], proposal[1][:, None, :]
        z = mu + std * eps
        zeros = torch.zeros_like(z)
        logw = loglik_fn(z) + _log_normal(z, zeros, zeros + 1) - _log_normal(z, mu, std)
    logw = torch.nan_to_num(logw, nan=-torch.inf)
    w = torch.softmax(logw, dim=1)
    ess = 1.0 / (w**2).sum(1)
    idx = torch.multinomial(w, m, replacement=True, generator=generator)
    draws = torch.gather(z, 1, idx[..., None].expand(-1, -1, emb_dim))
    return draws, ess


@torch.no_grad()
def pcn(
    loglik_fn: LogLikFn,
    z0: torch.Tensor,
    steps: int,
    beta: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one pCN chain per (row, draw) from ``z0`` (R, M, E).

    The step size is adapted per row during the first half of the run towards
    a ~25% acceptance rate; the returned acceptance rate (R,) is measured over
    the second half.
    """
    z = z0.clone()
    ll = loglik_fn(z)
    r = z.shape[0]
    b = torch.full((r, 1, 1), float(beta), device=z.device)
    burn = steps // 2
    acc_sum = torch.zeros(r, device=z.device)
    window = torch.zeros(r, device=z.device)
    for t in range(steps):
        xi = torch.randn(z.shape, generator=generator, device=z.device)
        prop = torch.sqrt(1 - b**2) * z + b * xi
        ll_prop = loglik_fn(prop)
        log_u = torch.log(torch.rand(ll.shape, generator=generator, device=z.device))
        accept = log_u < (ll_prop - ll)
        z = torch.where(accept[..., None], prop, z)
        ll = torch.where(accept, ll_prop, ll)
        rate = accept.float().mean(1)
        if t < burn:
            window += rate
            if (t + 1) % 10 == 0:
                w = window / 10
                b = torch.where((w < 0.15)[:, None, None], b * 0.7, b)
                b = torch.where((w > 0.40)[:, None, None], b * 1.3, b)
                b = b.clamp(0.005, 1.0)
                window.zero_()
        else:
            acc_sum += rate
    acc = acc_sum / max(steps - burn, 1)
    return z, acc


@torch.no_grad()
def smc(
    loglik_fn: LogLikFn,
    n_rows: int,
    emb_dim: int,
    k: int,
    m: int,
    device,
    generator: torch.Generator,
    mutation_steps: int = 5,
    beta: float = 0.3,
    target_ess: float = 0.5,
    max_stages: int = 100,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Tempered sequential Monte Carlo from the prior to the posterior.

    Targets pi_t(z) ∝ N(z; 0, I) · L(z)^tau_t with tau rising 0 -> 1. Each stage
    picks the largest increment keeping the incremental-weight ESS at
    ``target_ess`` x K (per row, by bisection), resamples, then mutates every
    particle with ``mutation_steps`` pCN moves, which leave pi_t invariant. This
    copes with sharply peaked likelihoods (many observed columns, long-trained
    decoders) where one-shot SIR collapses to a single particle.
    Returns ``(z (R, M, E), {"stages": (R,), "acceptance": (R,)})``.
    """
    z = torch.randn((n_rows, k, emb_dim), generator=generator, device=device)
    ll = torch.nan_to_num(loglik_fn(z), nan=-1e30)
    tau = torch.zeros(n_rows, device=device)
    b = torch.full((n_rows, 1, 1), float(beta), device=device)
    stages = torch.zeros(n_rows, device=device)
    acc_total = torch.zeros(n_rows, device=device)
    rows = torch.arange(n_rows, device=device)[:, None]

    def ess_of(delta: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(delta[:, None] * ll, dim=1)
        return 1.0 / (w**2).sum(1)

    for _ in range(max_stages):
        active = tau < 1.0
        if not active.any():
            break
        remaining = 1.0 - tau
        # Bisection for the tempering increment hitting the target ESS.
        lo, hi = torch.zeros_like(tau), remaining.clone()
        ok_full = ess_of(remaining) >= target_ess * k
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            good = ess_of(mid) >= target_ess * k
            lo = torch.where(good, mid, lo)
            hi = torch.where(good, hi, mid)
        delta = torch.where(ok_full, remaining, torch.clamp(lo, min=1e-6))
        delta = torch.where(active, delta, torch.zeros_like(delta))

        w = torch.softmax(delta[:, None] * ll, dim=1)
        idx = torch.multinomial(w, k, replacement=True, generator=generator)
        z = z[rows, idx]
        ll = ll[rows, idx]
        tau = torch.clamp(tau + delta, max=1.0)
        stages += active.float()

        acc_stage = torch.zeros(n_rows, device=device)
        for _ in range(mutation_steps):
            xi = torch.randn(z.shape, generator=generator, device=device)
            prop = torch.sqrt(1 - b**2) * z + b * xi
            ll_prop = torch.nan_to_num(loglik_fn(prop), nan=-1e30)
            log_u = torch.log(torch.rand(ll.shape, generator=generator, device=device))
            accept = log_u < tau[:, None] * (ll_prop - ll)
            z = torch.where(accept[..., None], prop, z)
            ll = torch.where(accept, ll_prop, ll)
            acc_stage += accept.float().mean(1)
        rate = acc_stage / max(mutation_steps, 1)
        acc_total += rate * active.float()
        b = torch.where((rate < 0.15)[:, None, None], b * 0.7, b)
        b = torch.where((rate > 0.40)[:, None, None], b * 1.3, b)
        b = b.clamp(0.005, 1.0)

    if m <= k:
        pick = torch.stack([torch.randperm(k, generator=generator, device=device)[:m] for _ in range(n_rows)])
    else:
        pick = torch.randint(k, (n_rows, m), generator=generator, device=device)
    draws = z[rows, pick]
    return draws, {"stages": stages, "acceptance": acc_total / torch.clamp(stages, min=1.0)}


def condition_latents(
    loglik_fn: LogLikFn,
    n_rows: int,
    emb_dim: int,
    m: int,
    cfg: LatentConfig,
    device,
    generator: torch.Generator,
    proposal: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """Sample ``m`` latent draws per row; returns ``(z (R, M, E), diagnostics)``.

    ``auto`` runs SIR and, for rows whose ESS falls below the threshold, falls
    back to tempered SMC (prior proposal) or pCN refinement of the SIR draws
    (encoder proposal, whose draws are usually already close).
    """
    diag: dict[str, np.ndarray] = {}
    method = cfg.method
    if method not in ("sir", "pcn", "smc", "auto"):
        raise ValueError(f"Unknown latent method '{method}'.")

    z, ess = sir(loglik_fn, n_rows, emb_dim, cfg.k, m, device, generator, proposal)
    diag["ess"] = ess.cpu().numpy()
    diag["acceptance"] = np.full(n_rows, np.nan)
    diag["smc_stages"] = np.full(n_rows, np.nan)
    refine = torch.zeros(n_rows, dtype=torch.bool, device=device)
    if method in ("pcn", "smc"):
        refine[:] = True
    elif method == "auto":
        threshold = cfg.ess_min if cfg.ess_min is not None else m
        refine = ess < threshold
    diag["refined"] = refine.cpu().numpy()
    if not refine.any():
        return z, diag

    idx = torch.nonzero(refine).squeeze(-1)
    idx_np = idx.cpu().numpy()

    def sub_ll(zz: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        return loglik_fn(zz, rows=idx if rows is None else idx[rows])

    use_smc = method == "smc" or (method == "auto" and proposal is None)
    if use_smc:
        z_ref, d = smc(
            sub_ll, idx.numel(), emb_dim, cfg.smc_particles or cfg.k, m, device, generator,
            cfg.smc_mutation_steps, cfg.pcn_beta, cfg.smc_target_ess, cfg.smc_max_stages,
        )
        diag["smc_stages"][idx_np] = d["stages"].cpu().numpy()
        diag["acceptance"][idx_np] = d["acceptance"].cpu().numpy()
    else:
        z_ref, acc = pcn(sub_ll, z[idx], cfg.pcn_steps, cfg.pcn_beta, generator)
        diag["acceptance"][idx_np] = acc.cpu().numpy()
    z[idx] = z_ref
    return z, diag
