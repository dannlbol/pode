"""Per-modality encoders for the joint multimodal VAE.

Each modality gets its own encoder mapping its raw features to the parameters of a
diagonal-Gaussian posterior ``q(z | x_m) = N(mu_m, diag(exp(logvar_m)))``. The
fusion module (experts.py) then combines these into the joint posterior.

Tabular and expression share the same MLP encoder here; keep them separate classes
so you can later specialise (e.g. an embedding front end for high-cardinality
categoricals, or a wider encoder for expression). Requires torch; import-guarded.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:

    class ModalityEncoder(nn.Module):
        """Generic MLP encoder -> (mu, logvar)."""

        def __init__(self, in_dim: int, latent_dim: int,
                     hidden=(512, 256), dropout: float = 0.1,
                     logvar_clamp: float = 10.0):
            super().__init__()
            layers = []
            d = in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.LeakyReLU(0.01)]
                if dropout:
                    layers += [nn.Dropout(dropout)]
                d = h
            self.backbone = nn.Sequential(*layers)
            self.to_mu = nn.Linear(d, latent_dim)
            self.to_logvar = nn.Linear(d, latent_dim)
            self.logvar_clamp = logvar_clamp

        def forward(self, x):
            h = self.backbone(x)
            mu = self.to_mu(h)
            logvar = torch.clamp(self.to_logvar(h), -self.logvar_clamp, self.logvar_clamp)
            return mu, logvar

    class TabularEncoder(ModalityEncoder):
        """Encoder for the (already numerically-encoded) clinical table.

        TODO(agent): optionally replace the plain Linear front end with per-column
        embeddings for categoricals (see docs/SPECIFICATION.md sec. 3.1). The
        current implementation assumes the caller has one-hot / ordinal-encoded
        categoricals upstream.
        """

    class ExpressionEncoder(ModalityEncoder):
        """Encoder for expression (raw HVG matrix or a pre-compressed block)."""

    class MethylationEncoder(ModalityEncoder):
        """Encoder for methylation on the M-value (logit-of-beta) scale.

        Mirrors :class:`ExpressionEncoder` -- the standardized M-values are a
        dense continuous matrix, so the generic MLP front end applies unchanged.
        Kept as its own class so it can later specialise (e.g. a CpG-island prior).
        """
