"""Multimodal ELBO assembly and weighting schemes.

The joint objective is::

    L = sum_m w_m * recon_nll_m(z, x_m)     # per-modality reconstruction / survival
      + beta(t) * KL(q(z|X) || N(0, I))     # annealed, with free bits

The per-modality weights ``w_m`` are the crux (the literature is clear that the
*magnitude* of survival/regularisation weighting governs whether prognostic signal
survives). Three schemes are exposed:

* ``"fixed"``       -- constant weights from config.
* ``"uncertainty"`` -- Kendall & Gal homoscedastic weights (learned), reusing
  ``synthvae.losses.vae.UncertaintyWeighter``.
* ``"gradnorm"``    -- GradNorm dynamic balancing (STUB: signature + hook provided;
  agent to implement, see docs/SPECIFICATION.md sec. 5.2).

KL uses free bits from ``synthvae.losses.vae.kl_standard_normal``.
Requires torch; import-guarded.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from ..losses.vae import kl_standard_normal, UncertaintyWeighter

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:

    def beta_schedule(epoch: int, beta_max: float, warmup_epochs: int) -> float:
        return beta_max * min(1.0, (epoch + 1) / max(1, warmup_epochs))

    class GradNormWeighter(nn.Module):
        """GradNorm dynamic task balancing (Chen et al. 2018).

        Keeps learnable positive task weights ``w_m`` (softplus of raw params,
        renormalised to sum to ``M``). The main objective uses *detached* weights
        so the network is updated by ``sum_m w_m.detach() * L_m``; the weights
        themselves are updated by a separate optimizer against the GradNorm
        penalty (see :meth:`gradnorm_penalty`), which pulls each task's gradient
        norm on the shared latent toward a common, training-rate-adjusted target.
        """

        def __init__(self, task_names):
            super().__init__()
            self.task_names = list(task_names)
            self.M = len(self.task_names)
            self.raw = nn.Parameter(torch.zeros(self.M))
            self.register_buffer("initial", torch.zeros(self.M))
            self._have_initial = False

        def weights(self):
            w = torch.nn.functional.softplus(self.raw) + 1e-4
            return self.M * w / w.sum()

        def weighted(self, losses: dict):
            """Return ``(sum_m w_m.detach() * L_m, logs)`` for the network step."""
            w = self.weights().detach()
            total = 0.0
            logs = {}
            for i, name in enumerate(self.task_names):
                total = total + w[i] * losses[name]
                logs[f"w_{name}"] = float(w[i])
            return total, logs

        def gradnorm_penalty(self, losses: dict, shared, alpha: float):
            """L1 penalty aligning per-task grad norms; call ``.backward()`` on it.

            ``shared`` is the shared representation (the sampled latent ``z``) whose
            gradient norm defines each task's training signal.
            """
            L = [losses[name] for name in self.task_names]
            if not self._have_initial:
                self.initial = torch.stack([l.detach() for l in L]).clamp(min=1e-8)
                self._have_initial = True
            w = self.weights()
            g_norms = []
            for i in range(self.M):
                g = torch.autograd.grad(w[i] * L[i], shared,
                                        retain_graph=True, create_graph=True)[0]
                g_norms.append(g.norm(2))
            g_norms = torch.stack(g_norms)
            mean_norm = g_norms.mean().detach()
            # relative inverse training rate r_m
            loss_ratio = torch.stack([l.detach() for l in L]) / self.initial
            r = loss_ratio / loss_ratio.mean().clamp(min=1e-8)
            target = (mean_norm * r.pow(alpha)).detach()
            return torch.abs(g_norms - target).sum()

        def renormalize(self):
            # keep raw params well-scaled so softplus(raw) stays sensible
            with torch.no_grad():
                w = self.weights()
                self.raw.copy_(torch.log(torch.expm1(w.clamp(min=1e-4))))

    class ObjectiveAssembler(torch.nn.Module):
        def __init__(self, modality_names, scheme="uncertainty",
                     fixed_weights=None, free_bits=0.02,
                     private_kl_weight=1.0):
            super().__init__()
            self.modality_names = list(modality_names)
            self.scheme = scheme
            self.free_bits = free_bits
            self.private_kl_weight = float(private_kl_weight)
            self.fixed_weights = fixed_weights or {m: 1.0 for m in modality_names}
            self.is_gradnorm = scheme == "gradnorm"
            if scheme == "uncertainty":
                self.weighter = UncertaintyWeighter(modality_names, enabled=True)
            elif scheme == "fixed":
                self.weighter = UncertaintyWeighter(modality_names, enabled=False)
            elif scheme == "gradnorm":
                self.weighter = GradNormWeighter(modality_names)
            else:
                raise ValueError(f"unknown scheme {scheme!r}")

        def forward(self, recon_losses: dict, mu, logvar, beta: float,
                    private_posteriors=None):
            """Return (total_loss, logs).

            recon_losses: {modality_name: scalar tensor}
            mu, logvar: joint posterior parameters for the KL term.
            """
            if self.is_gradnorm:
                data_term, logs = self.weighter.weighted(recon_losses)
            else:
                data_term, logs = self.weighter(recon_losses, self.fixed_weights)
            kl, kl_logs = self.kl_loss(mu, logvar, private_posteriors)
            total = data_term + beta * kl
            logs = dict(logs)
            logs.update(kl_logs)
            logs["beta"] = beta
            for k, v in recon_losses.items():
                logs[f"nll_{k}"] = float(v.detach())
            return total, logs

        def kl_loss(self, mu, logvar, private_posteriors=None):
            """Shared KL plus the mean weighted KL of active private heads."""
            shared = kl_standard_normal(mu, logvar, free_bits=self.free_bits)
            private_values = [
                kl_standard_normal(pmu, plogvar, free_bits=self.free_bits)
                for pmu, plogvar in (private_posteriors or {}).values()
            ]
            if private_values:
                private = torch.stack(private_values).mean()
            else:
                private = shared.new_zeros(())
            weight = self.private_kl_weight
            total = shared + weight * private
            return total, {
                "kl": float(total.detach()),
                "kl_shared": float(shared.detach()),
                "kl_private": float(private.detach()),
                "private_kl_weight": weight,
            }
