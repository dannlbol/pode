"""VAE loss building blocks (Torch).

Three pieces the hybrid needs and the MMVAE will reuse:

* ``gaussian_nll`` -- reconstruction NLL for continuous targets (expression scores
  or vst/log-CPM expression). Supports a learned or fixed observation scale.
* ``kl_standard_normal`` with **free bits** -- KL of a diagonal-Gaussian posterior
  against N(0, I), with a per-dimension floor ``tau`` so that low-variance but
  survival-relevant dimensions are not driven to collapse. This directly targets
  the failure mode reported for survival-supervised VAEs (regularisation magnitude
  matters more than its exact form).
* ``UncertaintyWeighter`` -- Kendall & Gal homoscedastic task weighting, turning
  the fragile hand-tuned (lambda_recon, lambda_surv) into learned parameters.

All functions are import-guarded so the rest of the package is usable without
torch installed.
"""
from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:

    def gaussian_nll(
        x: "torch.Tensor",
        mean: "torch.Tensor",
        log_scale: "torch.Tensor | float" = 0.0,
        reduction: str = "mean_over_batch",
    ) -> "torch.Tensor":
        """Negative log-likelihood of ``x`` under N(mean, exp(log_scale)^2).

        ``reduction``:
          * ``"mean_over_batch"`` -- sum over features, mean over batch (default;
            keeps the term on a per-sample scale so it is comparable to the KL).
          * ``"mean_per_feature"`` -- mean over both features and batch (use this
            when balancing a high-dim modality against a low-dim one so a big
            feature count does not dominate the objective).
          * ``"none"`` -- elementwise.
        """
        if not torch.is_tensor(log_scale):
            log_scale = torch.as_tensor(
                float(log_scale), dtype=x.dtype, device=x.device
            )
        var = torch.exp(2.0 * log_scale)
        elem = 0.5 * (
            math.log(2.0 * math.pi) + 2.0 * log_scale + (x - mean) ** 2 / var
        )
        if reduction == "none":
            return elem
        if reduction == "mean_over_batch":
            return elem.sum(dim=1).mean()
        if reduction == "mean_per_feature":
            return elem.mean()
        raise ValueError(f"unknown reduction {reduction!r}")

    def kl_standard_normal(
        mu: "torch.Tensor",
        logvar: "torch.Tensor",
        free_bits: float = 0.0,
        reduction: str = "mean_over_batch",
    ) -> "torch.Tensor":
        """KL(q(z|x) || N(0, I)) with optional per-dimension free bits.

        Per-dimension analytic KL for a diagonal Gaussian is
        ``0.5 * (mu^2 + var - 1 - logvar)``. With ``free_bits = tau > 0`` each
        dimension's KL is floored at ``tau`` nats via ``max(kl_d, tau)`` computed
        on the batch-mean per-dimension KL (the standard "free bits" trick).
        """
        kl_d = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # (n, D)
        if free_bits and free_bits > 0.0:
            kl_dim_mean = kl_d.mean(dim=0)  # (D,)
            floored = torch.clamp(kl_dim_mean, min=free_bits)
            # Broadcast the floored per-dim value back; keep batch-mean semantics.
            total = floored.sum()
            if reduction in ("mean_over_batch", "mean_per_feature"):
                return total
            raise ValueError("free_bits requires a mean reduction")
        if reduction == "mean_over_batch":
            return kl_d.sum(dim=1).mean()
        if reduction == "mean_per_feature":
            return kl_d.mean()
        if reduction == "none":
            return kl_d
        raise ValueError(f"unknown reduction {reduction!r}")

    class UncertaintyWeighter(nn.Module):
        """Homoscedastic multi-task weighting (Kendall & Gal, 2018).

        For each named task it learns ``log_sigma^2`` and returns the weighted
        sum ``sum_k [ exp(-s_k) * L_k + 0.5 * s_k ]`` where ``s_k = log sigma_k^2``.
        This is the regression-flavoured form; it works well for the Gaussian
        reconstruction term and is a reasonable, stable choice for the survival
        NLL too. Set ``enabled=False`` to fall back to fixed weights supplied at
        call time.
        """

        def __init__(self, task_names, enabled: bool = True):
            super().__init__()
            self.task_names = list(task_names)
            self.enabled = enabled
            self.log_sigma_sq = nn.Parameter(torch.zeros(len(self.task_names)))

        def forward(self, losses: dict, fixed_weights: dict | None = None):
            total = 0.0
            logs = {}
            for i, name in enumerate(self.task_names):
                if name not in losses:
                    continue
                L = losses[name]
                if self.enabled:
                    s = self.log_sigma_sq[i]
                    contrib = torch.exp(-s) * L + 0.5 * s
                    logs[f"w_{name}"] = float(torch.exp(-s).detach())
                else:
                    w = 1.0 if fixed_weights is None else float(
                        fixed_weights.get(name, 1.0)
                    )
                    contrib = w * L
                    logs[f"w_{name}"] = w
                total = total + contrib
            return total, logs
