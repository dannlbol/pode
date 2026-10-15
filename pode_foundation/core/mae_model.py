"""Masked autoencoder for methylation beta values.

Symmetric MLP with a tight bottleneck. Reconstruction loss is computed only
on the positions that were masked (hidden) during the forward pass — this is
the BERT/MAE-style self-supervised objective adapted to tabular methylation
data.

Notes on design choices:
  - Hidden positions are replaced with 0.5 (mid-beta), not 0. Using 0 would
    conflate "masked" with "fully unmethylated" since both are valid beta
    values. 0.5 is a neutral prior. An even cleaner alternative is to pass
    the mask as an additional input channel (doubles input dim), but for a
    50-100k input that's an extra 50-100k params per hidden unit and not
    worth it at this scale.
  - Sigmoid output keeps reconstructions in [0,1], matching beta-value scale.
    If you switch to M-values, drop the final Sigmoid and use a linear head
    (``output_activation="linear"``).
  - LayerNorm + GELU is standard for MLP autoencoders on tabular data.

This module has no cohort/disease assumptions: it operates on any
``(n_samples, n_features)`` float tensor in ``[0, 1]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MethylationMAE(nn.Module):
    def __init__(
        self,
        n_features: int,
        encoder_dims: tuple[int, ...] = (1024, 512),
        latent_dim: int = 256,
        dropout: float = 0.2,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim

        enc_layers = []
        prev = n_features
        for d in encoder_dims:
            enc_layers += [
                nn.Linear(prev, d),
                nn.GELU(),
                nn.LayerNorm(d),
                nn.Dropout(dropout),
            ]
            prev = d
        enc_layers += [nn.Linear(prev, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = latent_dim
        for d in reversed(encoder_dims):
            dec_layers += [
                nn.Linear(prev, d),
                nn.GELU(),
                nn.LayerNorm(d),
                nn.Dropout(dropout),
            ]
            prev = d
        dec_layers += [nn.Linear(prev, n_features)]
        if output_activation == "sigmoid":
            dec_layers += [nn.Sigmoid()]
        elif output_activation == "linear":
            pass
        else:
            raise ValueError(f"unknown output_activation: {output_activation}")
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without masking - use at inference for downstream tasks."""
        return self.encoder(x)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with masking.

        Args:
            x: (B, n_features) beta values in [0,1].
            mask: (B, n_features) float tensor, 1 = visible, 0 = hidden.

        Returns:
            x_hat: reconstructed values.
            z: latent embedding.
        """
        # Hidden positions get the neutral value 0.5.
        x_in = x * mask + 0.5 * (1.0 - mask)
        z = self.encoder(x_in)
        x_hat = self.decoder(z)
        return x_hat, z


def masked_mse_loss(
    x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE computed only on hidden positions (mask == 0)."""
    hidden = 1.0 - mask
    sq = (x_hat - x).pow(2) * hidden
    denom = hidden.sum().clamp_min(1.0)
    return sq.sum() / denom


def make_mask(
    shape: tuple[int, ...], mask_ratio: float, device: torch.device
) -> torch.Tensor:
    """Bernoulli mask. 1 = visible, 0 = hidden. mask_ratio is fraction hidden."""
    return (torch.rand(shape, device=device) > mask_ratio).float()
