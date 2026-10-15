"""Generators fitted on one table, behind one ``sample(n, seed)`` interface.

Used for the local and central baselines and for local exchange:

* ``fed_tvae``  - this package's TVAE trainer on one site (same architecture
  and training budget as the federated runs; DP-capable);
* ``sdv_tvae``  - PODE's existing SDV ``TVAEGenerator``, with its epochs set to
  the federated budget (``rounds x local_epochs``) so budgets match;
* ``copula``    - SDV's Gaussian copula, a non-neural baseline.

All fits happen where the table lives (a site, or a central server in the
"if pooling were permitted" baselines).
"""

from __future__ import annotations

import copy
import warnings

import pandas as pd

from ..config import SynthesizerConfig
from ..imputation.base import subset_schema
from .config import FederatedConfig


class FittedGenerator:
    def __init__(self, kind: str, model, columns: list[str], privacy: dict | None = None, fit_seconds: float = 0.0):
        self.kind, self.model, self.columns = kind, model, columns
        self.privacy = privacy or {}
        self.fit_seconds = fit_seconds

    def sample(self, n: int, seed: int = 0) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.kind == "fed_tvae":
                return self.model.sample(n, seed=seed)
            if self.kind == "sdv_tvae":
                return self.model.sample(n, seed=seed)[self.columns]
            df = self.model.sample(num_rows=n)
            return df[self.columns]


def fit_generator(kind: str, train: pd.DataFrame, columns: list[str], schema, synth: SynthesizerConfig,
                  fcfg: FederatedConfig, seed: int, censor_threshold=None, name: str = "local",
                  dp: bool = False) -> FittedGenerator:
    """Fit a ``kind`` generator on ``train[columns]``."""
    import time

    t0 = time.perf_counter()
    data = train[columns].reset_index(drop=True)
    sub = subset_schema(schema, columns) if schema is not None else None
    if kind == "fed_tvae":
        from .tvae import fit_on_frame

        cfg = fcfg.copy()
        cfg.seed = seed
        cfg.dp.enabled = bool(dp and cfg.dp.enabled)
        model = fit_on_frame(data, synth, cfg, sub, columns, name, censor_threshold)
        return FittedGenerator(kind, model, columns, model.record.privacy.get(name, {}),
                               time.perf_counter() - t0)
    if kind == "sdv_tvae":
        from ..generation.synthesizer import TVAEGenerator
        from ..imputation.decoder_imputer import _fit_sdv_on_cpu
        from .tvae import resolve_device

        s = copy.deepcopy(synth)
        s.epochs = fcfg.generator.rounds * fcfg.generator.local_epochs
        s.verbose = False
        gen = TVAEGenerator(sub, s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if resolve_device(fcfg.device) == "cpu":
                gen.synthesizer = _fit_sdv_on_cpu(sub, s, data, seed)
            else:
                gen.fit(data, seed=seed)
        _Censor.wrap(gen, censor_threshold)
        return FittedGenerator(kind, gen, columns, fit_seconds=time.perf_counter() - t0)
    if kind == "copula":
        from sdv.single_table import GaussianCopulaSynthesizer

        from ..reproducibility.seeds import set_all_seeds

        set_all_seeds(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianCopulaSynthesizer(sub.to_sdv_metadata())
            model.fit(data)
        return FittedGenerator(kind, model, columns, fit_seconds=time.perf_counter() - t0)
    raise ValueError(f"Unknown generator kind '{kind}'.")


class _Censor:
    """Bind the censoring threshold so ``TVAEGenerator.sample(n, seed=)`` applies it."""

    @staticmethod
    def wrap(gen, threshold) -> None:
        orig = gen.sample

        def sample(n, seed=None, **kw):
            return orig(n, seed=seed, censor_threshold=threshold, **kw)

        gen.sample = sample
