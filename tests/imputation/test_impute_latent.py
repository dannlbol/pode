"""SIR and pCN should recover the exact posterior of a linear-Gaussian model."""

import numpy as np
import torch

from pode.imputation.config import LatentConfig
from pode.imputation.latent import condition_latents


def _problem():
    rng = np.random.default_rng(0)
    e, d, sigma = 3, 4, 0.5
    A = torch.as_tensor(rng.normal(0, 1, (d, e)), dtype=torch.float32)
    x = torch.as_tensor(rng.normal(0, 1, (2, d)), dtype=torch.float32)

    def loglik(z, rows=None):
        xr = x if rows is None else x[rows]
        pred = z @ A.T
        return (-0.5 * ((xr[:, None, :] - pred) / sigma) ** 2).sum(-1)

    prec = A.T @ A / sigma**2 + torch.eye(e)
    cov = torch.linalg.inv(prec)
    mean = (cov @ (A.T @ x.T / sigma**2)).T
    return loglik, mean.numpy(), cov.numpy(), e


def _check(z, mean, cov):
    z = z.numpy()
    assert np.allclose(z.mean(1), mean, atol=0.12)
    for r in range(z.shape[0]):
        assert np.allclose(np.cov(z[r].T), cov, atol=0.08)


def test_sir_recovers_linear_gaussian_posterior():
    loglik, mean, cov, e = _problem()
    gen = torch.Generator().manual_seed(0)
    cfg = LatentConfig(k=200_000, method="sir")
    z, diag = condition_latents(loglik, 2, e, 2000, cfg, "cpu", gen)
    assert (diag["ess"] > 100).all()
    _check(z, mean, cov)


def test_pcn_recovers_linear_gaussian_posterior():
    loglik, mean, cov, e = _problem()
    gen = torch.Generator().manual_seed(1)
    cfg = LatentConfig(k=10, method="pcn", pcn_steps=400, pcn_beta=0.3)
    z, diag = condition_latents(loglik, 2, e, 2000, cfg, "cpu", gen)
    assert diag["refined"].all()
    assert np.all((diag["acceptance"] > 0.05) & (diag["acceptance"] < 0.9))
    _check(z, mean, cov)


def test_smc_recovers_linear_gaussian_posterior():
    loglik, mean, cov, e = _problem()
    gen = torch.Generator().manual_seed(2)
    cfg = LatentConfig(k=4000, method="smc", smc_mutation_steps=10)
    z, diag = condition_latents(loglik, 2, e, 2000, cfg, "cpu", gen)
    assert diag["refined"].all() and (diag["smc_stages"] >= 1).all()
    _check(z, mean, cov)


def test_smc_handles_sharp_likelihood_where_sir_collapses():
    loglik, mean, cov, e = _problem()

    def sharp(z, rows=None):
        return 400.0 * loglik(z, rows)

    gen = torch.Generator().manual_seed(3)
    z_sir, d_sir = condition_latents(sharp, 2, e, 200, LatentConfig(k=2000, method="sir"), "cpu", gen)
    z_smc, d_smc = condition_latents(sharp, 2, e, 200, LatentConfig(k=2000, method="smc", smc_mutation_steps=10), "cpu", gen)
    assert (d_sir["ess"] < 5).all()
    # SMC keeps distinct particles; SIR returns copies of one or two.
    assert all(len(np.unique(z_smc[r].numpy().round(6), axis=0)) > 100 for r in range(2))
    assert all(len(np.unique(z_sir[r].numpy().round(6), axis=0)) < 10 for r in range(2))
