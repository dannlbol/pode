import numpy as np
import torch

from pode_foundation.core.mae_model import MethylationMAE, make_mask, masked_mse_loss
from pode_foundation.core.mae_model_weighted import make_feature_weight, weighted_masked_mse_loss


def test_forward_shapes():
    model = MethylationMAE(n_features=40, encoder_dims=(16, 8), latent_dim=4, dropout=0.0)
    x = torch.rand(6, 40)
    mask = make_mask(x.shape, mask_ratio=0.3, device=x.device)
    x_hat, z = model(x, mask)
    assert x_hat.shape == x.shape
    assert z.shape == (6, 4)


def test_encode_matches_forward_latent():
    model = MethylationMAE(n_features=20, encoder_dims=(8,), latent_dim=3, dropout=0.0)
    model.eval()
    x = torch.rand(5, 20)
    mask = torch.ones_like(x)  # fully visible -> encode() and forward()'s z should match
    with torch.no_grad():
        _, z_forward = model(x, mask)
        z_encode = model.encode(x)
    assert torch.allclose(z_forward, z_encode, atol=1e-6)


def test_make_mask_ratio_is_approximately_respected():
    mask = make_mask((1000, 1000), mask_ratio=0.3, device=torch.device("cpu"))
    hidden_frac = 1.0 - mask.mean().item()
    assert 0.27 < hidden_frac < 0.33


def test_masked_mse_only_penalises_hidden_positions():
    x = torch.rand(4, 10)
    x_hat = x.clone()
    mask = torch.ones_like(x)
    mask[:, 0] = 0  # hide column 0
    x_hat[:, 0] += 1.0  # only wrong on the hidden column
    loss = masked_mse_loss(x_hat, x, mask)
    assert loss.item() > 0
    x_hat2 = x.clone()
    x_hat2[:, 1] += 1.0  # wrong on a *visible* column instead
    assert masked_mse_loss(x_hat2, x, mask).item() == 0.0


def test_weighted_loss_reduces_to_plain_mse_at_weight_one():
    x = torch.rand(4, 10)
    x_hat = torch.rand(4, 10)
    mask = make_mask(x.shape, 0.4, x.device)
    feat_w = make_feature_weight(np.zeros(10, dtype=bool), w=1.0, device=x.device)
    plain = masked_mse_loss(x_hat, x, mask)
    weighted = weighted_masked_mse_loss(x_hat, x, mask, feat_w)
    assert torch.allclose(plain, weighted, atol=1e-5)


def test_weighted_loss_upweights_selected_features():
    # Feature 0 is way off, the rest are nearly right. Up-weighting feature 0
    # should pull the weighted loss above the plain (unweighted) loss, since
    # the noisiest feature now counts for more of the average.
    x = torch.zeros(1, 4)
    x_hat = torch.tensor([[2.0, 0.1, 0.1, 0.1]])
    mask = torch.zeros(1, 4)  # everything hidden
    feature_mask = np.array([True, False, False, False])
    feat_w = make_feature_weight(feature_mask, w=10.0, device=x.device)
    weighted = weighted_masked_mse_loss(x_hat, x, mask, feat_w)
    plain = masked_mse_loss(x_hat, x, mask)
    assert weighted.item() > plain.item()


def test_unknown_output_activation_rejected():
    import pytest

    with pytest.raises(ValueError):
        MethylationMAE(n_features=10, output_activation="tanh")
