"""Tabular synthesizer interface.

The synthesizer models the joint distribution of ``[clinical | expr_latents |
survival]`` as one table and samples new rows. Swapping SDV's TVAE for a fallback
(or CTGAN, GaussianCopula, ...) must not touch the pipeline, so both implement
this contract.
"""
from __future__ import annotations

import abc
import pandas as pd


class Synthesizer(abc.ABC):
    @abc.abstractmethod
    def fit(self, table: pd.DataFrame, metadata: dict) -> "Synthesizer":
        """Fit on the assembled table. ``metadata`` describes column dtypes
        (see :func:`synthvae.pipeline.latents.build_metadata`)."""

    @abc.abstractmethod
    def sample(self, n: int) -> pd.DataFrame:
        """Return ``n`` synthetic rows with the same columns as the fit table."""
