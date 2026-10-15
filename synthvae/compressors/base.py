"""Compressor interface.

A *compressor* reduces a high-dimensional expression matrix to ``k`` latent
scores and can invert those scores back to expression space. Every compressor
must implement the same four-method contract so the pipeline can swap PCA for the
SurvivalVAE (or, later, a MOFA / NMF front end) without touching downstream code.

The contract deliberately mirrors scikit-learn (``fit`` / ``transform`` /
``inverse_transform``) plus ``save`` / ``load`` for the train->synthesis handoff.
"""
from __future__ import annotations

import abc
import numpy as np


class Compressor(abc.ABC):
    """Abstract expression compressor."""

    #: number of latent dimensions produced by ``transform``
    n_latent: int

    @abc.abstractmethod
    def fit(
        self,
        X: np.ndarray,
        time: np.ndarray | None = None,
        event: np.ndarray | None = None,
    ) -> "Compressor":
        """Fit on expression ``X`` (n, G). ``time``/``event`` are optional and are
        used only by survival-supervised compressors."""

    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map expression ``X`` (n, G) to latents (n, n_latent)."""

    @abc.abstractmethod
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map latents ``Z`` (n, n_latent) back to expression (n, G)."""

    def fit_transform(
        self,
        X: np.ndarray,
        time: np.ndarray | None = None,
        event: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.fit(X, time, event).transform(X)

    @abc.abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str) -> "Compressor":
        ...

    # -- convenience ---------------------------------------------------------
    def latent_columns(self, prefix: str = "expr_latent_") -> list[str]:
        return [f"{prefix}{i}" for i in range(self.n_latent)]

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean squared reconstruction error on ``X`` (diagnostic)."""
        Xr = self.inverse_transform(self.transform(X))
        return float(np.mean((np.asarray(X) - Xr) ** 2))
