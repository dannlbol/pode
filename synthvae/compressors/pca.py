"""PCA compressor -- the baseline the fancier front ends must beat.

For a fixed ``k`` this is the L2-optimal linear reconstruction, it is deterministic,
its scores are decorrelated (which plays to TVAE's per-column modelling), and its
linear decoder extrapolates gracefully to off-manifold latents the synthesizer may
emit. Always run this first and record its reconstruction error and its downstream
survival concordance as the floor to improve on.
"""
from __future__ import annotations

import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .base import Compressor


class PCACompressor(Compressor):
    def __init__(self, n_latent: int = 64, whiten: bool = False, random_state: int = 0):
        self.n_latent = int(n_latent)
        self.whiten = whiten
        self.random_state = random_state
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None

    def fit(self, X, time=None, event=None):
        X = np.asarray(X, dtype=float)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        k = min(self.n_latent, Xs.shape[1], Xs.shape[0])
        self._pca = PCA(
            n_components=k, whiten=self.whiten, random_state=self.random_state
        ).fit(Xs)
        self.n_latent = k
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return self._pca.transform(self._scaler.transform(X))

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=float)
        Xs = self._pca.inverse_transform(Z)
        return self._scaler.inverse_transform(Xs)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self._pca.explained_variance_ratio_

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
