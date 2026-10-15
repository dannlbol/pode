"""Differentially private SGD for the TVAE: per-sample clipping, Gaussian noise, accounting.

**Mechanism** (Abadi et al., 2016). Each step draws a Poisson batch (every
patient independently with probability ``q = B / n``), computes each patient's
gradient separately, clips it to norm ``C`` (``dp.max_grad_norm``), sums,
adds Gaussian noise ``N(0, (sigma C)^2)`` and divides by the expected batch
size. No single patient can then move the model by more than a bounded,
noise-masked amount.

**Per-sample gradients** come from ``torch.func`` (``vmap(grad(...))``) rather
than Opacus' layer hooks: the TVAE decoder's ``sigma`` is a free parameter that
is returned from ``forward`` and only used inside the loss, which hooks cannot
attribute per patient. The reparameterisation noise is drawn outside and passed
in, so the vmapped function is deterministic.

**Accounting** uses Opacus (``pip install -e '.[federated]'``): an RDP (or PRV)
accountant composes all of a site's steps, across every federation round, into
one ``(epsilon, delta)`` for that site. A target epsilon is converted into a
noise multiplier up front. ``delta`` defaults to ``min(1e-5, 1 / (10 n))``.

Reading epsilon: it bounds how much any one patient's presence can change what
the site's updates reveal. Smaller is stronger; ~1 is strong, ~3-8 is common
practice for useful deep models, > 10 is weak. The guarantee covers the
training updates; the encoder's summaries add ``dp.transformer_epsilon``.
"""

from __future__ import annotations

import math

import torch

INSTALL = "DP-SGD needs Opacus for privacy accounting: pip install -e '.[federated]' (or pip install 'opacus>=1.5,<2')."


def require_opacus():
    try:
        import opacus  # noqa: F401
    except ImportError as exc:
        raise ImportError(INSTALL) from exc
    return opacus


def default_delta(n: int) -> float:
    return min(1e-5, 1.0 / (10.0 * max(int(n), 1)))


def noise_for_target(target_epsilon: float, delta: float, sample_rate: float, steps: int,
                     accountant: str = "rdp") -> float:
    """Noise multiplier that spends ``target_epsilon`` over ``steps`` at ``sample_rate``."""
    require_opacus()
    from opacus.accountants.utils import get_noise_multiplier

    return float(get_noise_multiplier(target_epsilon=target_epsilon, target_delta=delta,
                                      sample_rate=sample_rate, steps=steps, accountant=accountant))


def epsilon_spent(noise_multiplier: float, sample_rate: float, steps: int, delta: float,
                  accountant: str = "rdp") -> float:
    """Epsilon after ``steps`` noisy steps at ``sample_rate`` (inf without noise)."""
    if noise_multiplier <= 0:
        return math.inf
    if steps <= 0:
        return 0.0
    require_opacus()
    from opacus.accountants import create_accountant

    acc = create_accountant(accountant)
    acc.history = [(float(noise_multiplier), float(sample_rate), int(steps))]
    return float(acc.get_epsilon(delta))


def per_sample_grads(net: torch.nn.Module, loss_one, x: torch.Tensor, eps: torch.Tensor,
                     chunk_size: int | None = 64) -> dict[str, torch.Tensor]:
    """``{param_name: [B, *param_shape]}`` gradients of ``loss_one(outputs, x_row)`` per row."""
    from torch.func import functional_call, grad, vmap

    params = {k: v.detach() for k, v in net.named_parameters()}

    def single(p, xi, ei):
        out = functional_call(net, p, (xi.unsqueeze(0), ei.unsqueeze(0)))
        return loss_one(out, xi.unsqueeze(0))[0]

    return vmap(grad(single), in_dims=(None, 0, 0), chunk_size=chunk_size)(params, x, eps)


def clip_and_noise(grads: dict[str, torch.Tensor], max_norm: float, noise_multiplier: float,
                   expected_batch: float, generator: torch.Generator | None = None) -> tuple[dict, torch.Tensor]:
    """Clip each row's gradient to ``max_norm``, sum, add noise, divide by the expected batch.

    Returns the privatised gradient and the per-row pre-clipping norms.
    """
    keys = list(grads)
    b = grads[keys[0]].shape[0]
    norms = torch.sqrt(sum(grads[k].reshape(b, -1).pow(2).sum(1) for k in keys)) if b else torch.zeros(0)
    factor = (max_norm / (norms + 1e-6)).clamp(max=1.0)
    out = {}
    for k in keys:
        g = grads[k]
        summed = (g * factor.view(-1, *([1] * (g.dim() - 1)))).sum(0) if b else torch.zeros_like(g[0:1]).sum(0)
        noise = torch.normal(0.0, noise_multiplier * max_norm, size=summed.shape, generator=generator,
                             device="cpu").to(summed.device, summed.dtype)
        out[k] = (summed + noise) / expected_batch
    return out, norms
