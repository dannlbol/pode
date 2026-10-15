"""Decoder heads for the joint multimodal VAE.

Each head maps the shared latent ``z`` back to one modality's likelihood
parameters and exposes a ``nll(z, target)`` method returning a scalar loss on a
per-sample scale. Keeping the loss inside the head lets the objective assembler
stay agnostic to per-modality likelihoods.

Heads
-----
* ``TabularHead``   -- mixed per-column likelihoods (Gaussian for continuous,
  categorical cross-entropy for discrete). This mirrors what TVAE does internally;
  for full fidelity you may later swap in TVAE-style mode-specific normalisation.
* ``ExpressionHead`` -- Gaussian by default; negative-binomial / zero-inflated NB
  are provided as selectable likelihoods for count data (NB/ZINB marked TODO).
* ``SurvivalHead``  -- discrete-time logistic hazard, reusing
  ``synthvae.survival.losses`` so the hybrid and the MMVAE share one survival loss.

Requires torch; import-guarded.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from ..losses.vae import gaussian_nll
    from ..survival.losses import nll_torch as survival_nll_torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


@dataclass
class ColumnSpec:
    """Describes one tabular column for the TabularHead."""
    name: str
    kind: str          # "continuous" | "categorical"
    n_categories: int = 0  # for categorical


if _HAS_TORCH:

    def _mlp(in_dim, hidden, out_dim, dropout):
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.LeakyReLU(0.01)]
            if dropout:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        return nn.Sequential(*layers)

    def _log_nb_positive(x, mu, theta, eps=1e-8):
        """Log-likelihood of counts ``x`` under NB(mean=mu, dispersion=theta).

        Uses the mean/dispersion parameterisation (scvi-tools ``log_nb_positive``):
        variance = mu + mu^2 / theta. Returns an elementwise (n, G) tensor.
        """
        log_theta_mu_eps = torch.log(theta + mu + eps)
        return (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1.0)
        )

    def _log_zinb_positive(x, mu, theta, pi_logits, eps=1e-8):
        """Zero-inflated NB log-likelihood (scvi-tools ``log_zinb_positive``).

        ``pi_logits`` are the logits of the zero-inflation gate. Returns (n, G).
        """
        softplus_pi = F.softplus(-pi_logits)
        log_theta_eps = torch.log(theta + eps)
        log_theta_mu_eps = torch.log(theta + mu + eps)
        pi_theta_log = -pi_logits + theta * (log_theta_eps - log_theta_mu_eps)

        case_zero = F.softplus(pi_theta_log) - softplus_pi
        mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

        case_non_zero = (
            -softplus_pi
            + pi_theta_log
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1.0)
        )
        mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)
        return mul_case_zero + mul_case_non_zero

    class TabularHead(nn.Module):
        """Mixed-likelihood decoder for the clinical table.

        Continuous columns are decoded as Gaussians (shared learned log-scale);
        categorical columns as softmax logits. ``target`` is a dict with keys
        ``continuous`` (n, C_cont) and ``categorical`` (list of (n,) long tensors).
        """

        def __init__(self, latent_dim, column_specs, hidden=(128,), dropout=0.1):
            super().__init__()
            self.specs = column_specs
            self.cont_idx = [i for i, s in enumerate(column_specs) if s.kind == "continuous"]
            self.cat_specs = [s for s in column_specs if s.kind == "categorical"]
            n_cont = len(self.cont_idx)
            self.backbone = _mlp(latent_dim, hidden, hidden[-1] if hidden else latent_dim, dropout)
            feat = hidden[-1] if hidden else latent_dim
            self.cont_out = nn.Linear(feat, n_cont) if n_cont else None
            self.cont_log_scale = nn.Parameter(torch.zeros(1)) if n_cont else None
            self.cat_out = nn.ModuleList(
                [nn.Linear(feat, s.n_categories) for s in self.cat_specs]
            )

        def forward(self, z):
            h = self.backbone(z)
            cont = self.cont_out(h) if self.cont_out is not None else None
            cats = [layer(h) for layer in self.cat_out]
            return {"continuous": cont, "categorical": cats}

        def nll(self, z, target):
            out = self.forward(z)
            loss = 0.0
            if out["continuous"] is not None and "continuous" in target:
                loss = loss + gaussian_nll(
                    target["continuous"], out["continuous"],
                    self.cont_log_scale, reduction="mean_over_batch",
                )
            for logits, tgt in zip(out["categorical"], target.get("categorical", [])):
                loss = loss + F.cross_entropy(logits, tgt, reduction="mean")
            return loss

    class ExpressionHead(nn.Module):
        """Decoder for expression. ``likelihood`` in {"gaussian","nb","zinb"}.

        Gaussian is implemented. NB/ZINB heads are stubbed with the correct output
        parameterisation so an agent can fill in the likelihoods for count data
        (see docs/SPECIFICATION.md sec. 3.3).
        """

        def __init__(self, latent_dim, n_features, hidden=(256, 512),
                     dropout=0.1, likelihood="gaussian",
                     reduction="mean_over_batch"):
            super().__init__()
            self.likelihood = likelihood
            self.n_features = n_features
            # ``mean_per_feature`` puts this high-dim modality on a per-gene scale so
            # its feature count does not swamp the survival / tabular terms in the
            # shared latent (SPEC sec. 5; losses.vae.gaussian_nll docstring).
            self.reduction = reduction
            self.decoder = _mlp(latent_dim, hidden, n_features, dropout)
            if likelihood == "gaussian":
                self.log_scale = nn.Parameter(torch.zeros(1))
            elif likelihood in ("nb", "zinb"):
                # dispersion (per-gene) and, for zinb, dropout-logits head
                self.log_theta = nn.Parameter(torch.zeros(n_features))
                if likelihood == "zinb":
                    self.dropout_logits = _mlp(latent_dim, hidden, n_features, dropout)
            else:
                raise ValueError(f"unknown likelihood {likelihood!r}")

        def forward(self, z):
            return self.decoder(z)

        def nll(self, z, target, library=None):
            out = self.forward(z)
            if self.likelihood == "gaussian":
                return gaussian_nll(target, out, self.log_scale,
                                    reduction=self.reduction)
            # Count likelihoods. Decoder emits a rate; softplus keeps mu > 0. An
            # optional per-cell library size (size factor) scales the mean.
            mu = F.softplus(out)
            if library is not None:
                mu = mu * library
            mu = mu.clamp(min=1e-8)
            theta = torch.exp(self.log_theta).clamp(min=1e-4)  # per-gene dispersion
            if self.likelihood == "nb":
                ll = _log_nb_positive(target, mu, theta)
            else:  # zinb
                pi_logits = self.dropout_logits(z)
                ll = _log_zinb_positive(target, mu, theta, pi_logits)
            # match the gaussian reduction so weighting stays comparable
            if self.reduction == "mean_per_feature":
                return -ll.mean()
            return -ll.sum(dim=1).mean()

    class MethylationHead(ExpressionHead):
        """Decoder for methylation on the M-value (logit-of-beta) scale.

        Methylation beta-values in [0, 1] are logit-transformed to M-values and
        standardized upstream (see ``data.MethylationTransformer``), which makes a
        Gaussian likelihood the natural, symmetric choice -- so this is just an
        :class:`ExpressionHead` pinned to ``likelihood="gaussian"``. The ``library``
        argument of ``nll`` is inherited but unused on the gaussian path.
        """

        def __init__(self, latent_dim, n_features, hidden=(256, 512),
                     dropout=0.1, reduction="mean_over_batch"):
            super().__init__(latent_dim, n_features, hidden=hidden,
                             dropout=dropout, likelihood="gaussian",
                             reduction=reduction)

    class BinaryHead(nn.Module):
        """Independent Bernoulli decoder for a binary molecular matrix.

        Mutation calls are deliberately modelled as a molecular head rather than
        being folded into the clinical categorical decoder: there can be many
        genes and each output is an independent Bernoulli likelihood conditional
        on the shared latent.
        """

        def __init__(self, latent_dim, n_features, hidden=(256, 512),
                     dropout=0.1, reduction="mean_per_feature"):
            super().__init__()
            self.decoder = _mlp(latent_dim, hidden, n_features, dropout)
            self.reduction = reduction

        def forward(self, z):
            return self.decoder(z)

        def nll(self, z, target, library=None):
            logits = self.forward(z)
            reduction = "mean" if self.reduction == "mean_per_feature" else "none"
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction=reduction)
            return loss if reduction == "mean" else loss.sum(dim=1).mean()

    class SurvivalHead(nn.Module):
        """Discrete-time (logistic-hazard) survival decoder.

        Emits ``n_time_bins`` hazard logits from z and scores them with the shared
        discrete-time NLL. ``target`` is a dict with ``idx`` (long interval index)
        and ``event`` (float indicator).
        """

        def __init__(self, latent_dim, n_time_bins, hidden=(64,), dropout=0.1):
            super().__init__()
            self.n_time_bins = n_time_bins
            self.head = _mlp(latent_dim, hidden, n_time_bins, dropout)

        def forward(self, z):
            return self.head(z)

        def nll(self, z, target):
            logits = self.forward(z)
            return survival_nll_torch(logits, target["idx"], target["event"])
