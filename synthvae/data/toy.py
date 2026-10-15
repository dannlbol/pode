"""Synthetic multimodal cohort generator.

Produces a cohort whose expression, clinical variables and survival all share a
small set of *true* latent factors, so that a good compressor+synthesizer should
be able to recover the joint structure and the evaluation harness has real signal
to measure. This lets the whole pipeline run end to end with no private data.

Structure
---------
* ``L`` true factors ``Z ~ N(0, I)``.
* Expression ``E = Z @ W + noise`` with sparse loadings ``W`` (G genes), returned
  on a log-expression-like scale.
* Clinical: ``age`` (driven by factor 0), ``biomarker`` (factor 1), ``stage``
  (ordinal, from factor 2), ``sex`` (independent binary).
* Survival: Weibull time with log-hazard linear in a couple of factors + stage,
  plus independent right-censoring.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ToyConfig:
    n: int = 800
    n_genes: int = 300
    n_factors: int = 6
    loading_sparsity: float = 0.7   # fraction of zero loadings
    noise_sd: float = 0.4
    weibull_shape: float = 1.3
    censor_rate: float = 0.3
    seed: int = 0


def make_cohort(config: ToyConfig | None = None):
    """Return ``(clinical_df, expression_df, survival_df, truth)``.

    ``truth`` carries the true factor scores and survival coefficients for
    debugging/oracle comparisons.
    """
    cfg = config or ToyConfig()
    rng = np.random.default_rng(cfg.seed)

    Z = rng.standard_normal((cfg.n, cfg.n_factors))

    W = rng.standard_normal((cfg.n_factors, cfg.n_genes))
    mask = rng.random((cfg.n_factors, cfg.n_genes)) < cfg.loading_sparsity
    W[mask] = 0.0
    base = rng.uniform(3.0, 8.0, size=cfg.n_genes)  # per-gene baseline expression
    E = base[None, :] + Z @ W + cfg.noise_sd * rng.standard_normal((cfg.n, cfg.n_genes))
    expr_cols = [f"gene_{i}" for i in range(cfg.n_genes)]
    expression = pd.DataFrame(E, columns=expr_cols)

    # Clinical variables tied to factors.
    age = 60 + 8 * Z[:, 0] + rng.normal(0, 3, cfg.n)
    biomarker = 2.0 + 1.2 * Z[:, 1] + rng.normal(0, 0.5, cfg.n)
    stage_latent = Z[:, 2] + rng.normal(0, 0.5, cfg.n)
    stage = np.digitize(stage_latent, np.quantile(stage_latent, [0.25, 0.5, 0.75])) + 1
    sex = rng.integers(0, 2, cfg.n)
    clinical = pd.DataFrame(
        {
            "age": age.round(1),
            "biomarker": biomarker.round(3),
            "stage": stage.astype(int),      # ordinal 1..4
            "sex": sex.astype(int),          # binary
        }
    )

    # Survival: log-hazard linear in factors 1,3 and stage.
    beta = np.array([0.0, 0.6, 0.0, 0.5, 0.0, 0.0])[: cfg.n_factors]
    lp = Z @ beta + 0.25 * (stage - stage.mean())
    # Weibull inverse-CDF sampling with risk exp(lp).
    u = rng.uniform(1e-6, 1 - 1e-6, cfg.n)
    scale = np.exp(-lp / cfg.weibull_shape)
    event_time = scale * (-np.log(u)) ** (1.0 / cfg.weibull_shape)
    # Independent censoring calibrated to hit approx censor_rate.
    c_scale = np.quantile(event_time, 1 - cfg.censor_rate)
    censor_time = rng.exponential(c_scale, cfg.n)
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    survival = pd.DataFrame({"time": time.round(4), "event": event})

    truth = {
        "Z": Z,
        "W": W,
        "surv_beta": beta,
        "expr_cols": expr_cols,
    }
    return clinical, expression, survival, truth
