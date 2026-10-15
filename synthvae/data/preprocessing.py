"""High-dimensional molecular matrix preprocessing.

Order of operations matters and is fixed here so train and synthesis stay
consistent:

    1. (optional) batch correction  -- hook only; wire in ComBat/pyComBat if you
       have batch labels, BEFORE variance-based gene selection.
    2. transform                    -- expression can use log1p as a convenience;
       methylation beta-values are clipped and logit-transformed to M-value scale.
    3. highly-variable-feature selection -- top-k by variance, denoises the
       reconstruction target and shrinks the problem before compression.

Everything needed to reproduce the transform on new data (selected gene indices)
is captured in :class:`ExpressionPreprocessor` for the train->synthesis handoff.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ExpressionPreprocessor:
    n_top_genes: int | None = 2000
    log1p: bool = False
    matrix_kind: str = "expression"
    beta_epsilon: float = 1e-6
    selected_: np.ndarray | None = field(default=None, init=False)
    fill_values_: np.ndarray | None = field(default=None, init=False)

    def fit(self, X: np.ndarray) -> "ExpressionPreprocessor":
        X = self._transform(np.asarray(X, dtype=float))
        self.fill_values_ = X.mean(axis=0)
        if self.n_top_genes is None or self.n_top_genes >= X.shape[1]:
            self.selected_ = np.arange(X.shape[1])
        else:
            var = X.var(axis=0)
            self.selected_ = np.argsort(var)[::-1][: self.n_top_genes]
            self.selected_.sort()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._transform(np.asarray(X, dtype=float))
        return X[:, self.selected_]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_selected: np.ndarray) -> np.ndarray:
        """Scatter selected-gene values back to the original column order."""
        X_selected = np.asarray(X_selected, dtype=float)
        X_full = np.tile(self.fill_values_, (X_selected.shape[0], 1))
        X_full[:, self.selected_] = X_selected
        return self._inverse_transform(X_full)

    def _transform(self, X):
        if not np.isfinite(X).all():
            raise ValueError("matrix contains non-finite values")
        if self.matrix_kind == "expression":
            return np.log1p(X) if self.log1p else X
        # Copy number log2 ratios (continuous) and ordinal calls are already
        # centred; the compressor standardises internally, so the transform is
        # identity here. log1p is only meaningful for expression counts.
        if self.matrix_kind in ("copy_number", "copy_number_discrete"):
            if self.log1p:
                raise ValueError("log1p is only valid for expression matrices")
            return X
        if self.matrix_kind == "methylation":
            if self.log1p:
                raise ValueError("log1p is not valid for methylation beta-values")
            if np.any((X < 0.0) | (X > 1.0)):
                raise ValueError("methylation beta-values must be in [0, 1]")
            eps = float(self.beta_epsilon)
            if not (0.0 < eps < 0.5):
                raise ValueError("beta_epsilon must be between 0 and 0.5")
            X = np.clip(X, eps, 1.0 - eps)
            return np.log(X / (1.0 - X))
        raise ValueError(
            "matrix_kind must be 'expression', 'methylation', 'copy_number', "
            "or 'copy_number_discrete'"
        )

    def _inverse_transform(self, X):
        if self.matrix_kind in ("expression", "copy_number"):
            return X
        if self.matrix_kind == "copy_number_discrete":
            # snap decoded latents back to the ordinal {-1, 0, 1} call scale
            return np.clip(np.rint(X), -1.0, 1.0)
        if self.matrix_kind == "methylation":
            return np.clip(1.0 / (1.0 + np.exp(-X)), 0.0, 1.0)
        raise ValueError(
            "matrix_kind must be 'expression', 'methylation', 'copy_number', "
            "or 'copy_number_discrete'"
        )


def combat_hook(X: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Placeholder for batch correction.

    Left unimplemented on purpose: plug in ``pycombat`` / ``scanpy.pp.combat`` or
    an equivalent here if you have batch labels, and call it before
    :class:`ExpressionPreprocessor`. Returning X unchanged keeps the demo runnable.
    """
    return X
