"""Latent fusion strategies for the joint multimodal VAE.

Given per-modality Gaussian posteriors ``q(z | x_m) = N(mu_m, var_m)`` this module
combines them into a single joint posterior. Three strategies are provided plus a
data-dependent gating network; the choice is a top-level config switch so you can
benchmark them under identical encoders/heads (see docs/SPECIFICATION.md sec. 4).

* **PoE** (Product-of-Experts, Wu & Goodman 2018): multiply the Gaussians. Closed
  form; precision-weighted, so a confident modality dominates. Include a "prior
  expert" N(0, I) for numerical stability and to define the posterior when a
  modality is missing (drop its expert from the product).
* **MoE** (Mixture-of-Experts, Shi et al. 2019): average the experts. Better one-
  way cross-generation, weaker joint coherence.
* **MoPoE** (Sutter et al. 2021): mixture over the PoE of every subset of
  modalities; balances PoE and MoE at higher cost.
* **GatingNetwork**: learns per-sample, per-modality scalar weights that scale each
  expert's precision (PoE) or mixture weight (MoE), giving adaptive control over
  each modality's contribution -- the mechanism to down-weight the high-dimensional
  expression modality relative to clinical when appropriate.

All functions operate on ``logvar`` (log variance) for numerical stability.
Requires torch; import-guarded.
"""
from __future__ import annotations

import itertools

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:

    def _prior_expert(shape, device):
        mu = torch.zeros(shape, device=device)
        logvar = torch.zeros(shape, device=device)  # var = 1
        return mu, logvar

    def product_of_experts(mus, logvars, weights=None, include_prior=True, eps=1e-8):
        """Combine Gaussian experts by product.

        Parameters
        ----------
        mus, logvars : list of (n, D) tensors, one per present modality.
        weights : optional list of (n, 1) tensors in (0, inf) scaling each expert's
            precision (from the gating network). ``None`` => all ones.
        include_prior : prepend a unit-Gaussian prior expert (recommended).

        Returns
        -------
        mu, logvar : (n, D) joint posterior parameters.
        """
        device = mus[0].device
        mus = list(mus)
        logvars = list(logvars)
        if include_prior:
            pm, pv = _prior_expert(mus[0].shape, device)
            mus = [pm] + mus
            logvars = [pv] + logvars
            if weights is not None:
                ones = torch.ones_like(weights[0])
                weights = [ones] + list(weights)

        # precision (inverse variance) of each expert
        precisions = [torch.exp(-lv) for lv in logvars]
        if weights is not None:
            precisions = [w * p for w, p in zip(weights, precisions)]

        prec_sum = sum(precisions) + eps
        mu_num = sum(p * m for p, m in zip(precisions, mus))
        joint_mu = mu_num / prec_sum
        joint_var = 1.0 / prec_sum
        joint_logvar = torch.log(joint_var + eps)
        return joint_mu, joint_logvar

    def mixture_of_experts(mus, logvars, weights=None):
        """Combine experts by (weighted) mixture at the *distribution* level.

        Returns the parameters of a moment-matched single Gaussian (mean and
        variance of the mixture), which is the common practical approximation used
        for a single-latent decoder. For a strict MoE-VAE you would instead sample
        one expert per forward pass; that variant is left as a documented option.
        """
        n_exp = len(mus)
        if weights is None:
            w = [torch.full_like(mus[0][:, :1], 1.0 / n_exp) for _ in range(n_exp)]
        else:
            wsum = sum(weights) + 1e-8
            w = [wi / wsum for wi in weights]
        mix_mu = sum(wi * m for wi, m in zip(w, mus))
        # law of total variance for the mixture
        var = [torch.exp(lv) for lv in logvars]
        second = sum(wi * (v + m ** 2) for wi, v, m in zip(w, var, mus))
        mix_var = second - mix_mu ** 2
        mix_logvar = torch.log(torch.clamp(mix_var, min=1e-8))
        return mix_mu, mix_logvar

    def mixture_of_products_of_experts(mus, logvars, weights=None):
        """MoPoE: uniform mixture over the PoE of every non-empty modality subset.

        Moment-matched to a single Gaussian for a shared-latent decoder. Cost grows
        as 2^M in the number of modalities M; fine for M=2..3 (our case).
        """
        M = len(mus)
        subset_mus, subset_logvars = [], []
        for r in range(1, M + 1):
            for combo in itertools.combinations(range(M), r):
                cm = [mus[i] for i in combo]
                cl = [logvars[i] for i in combo]
                cw = None if weights is None else [weights[i] for i in combo]
                pm, pl = product_of_experts(cm, cl, cw, include_prior=True)
                subset_mus.append(pm)
                subset_logvars.append(pl)
        return mixture_of_experts(subset_mus, subset_logvars)

    class GatingNetwork(torch.nn.Module):
        """Per-sample, per-modality gating weights in (0, inf).

        Takes the concatenated per-modality posterior means and emits one positive
        scalar per modality (softplus). Use the weights to scale expert precisions
        in PoE or mixture weights in MoE. This is the adaptive per-omic contribution
        mechanism from the POEMS-style design.
        """

        def __init__(self, latent_dim: int, n_modalities: int, hidden: int = 64):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(latent_dim * n_modalities, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.LeakyReLU(0.01),
                torch.nn.Linear(hidden, n_modalities),
                torch.nn.Softplus(),
            )
            self.n_modalities = n_modalities

        def forward(self, mus):
            """mus: list of (n, D). Returns list of (n, 1) positive weights."""
            h = torch.cat(mus, dim=1)
            w = self.net(h) + 1e-4  # (n, M), strictly positive
            return [w[:, i : i + 1] for i in range(self.n_modalities)]


FUSION = {
    "poe": "product_of_experts",
    "moe": "mixture_of_experts",
    "mopoe": "mixture_of_products_of_experts",
}
