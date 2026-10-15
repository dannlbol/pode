"""End-to-end hybrid pipeline.

Fit path::

    matrix --feature-select--> compressor.fit(X, time, event) --> latents
    [clinical | latents | survival] --> synthesizer.fit

Sample path::

    synthesizer.sample(n) --> [clinical | latents | survival]
    latents --compressor.inverse_transform--> synthetic matrix

The compressor is any :class:`~synthvae.compressors.base.Compressor`
(PCACompressor or SurvivalVAE); the synthesizer is any
:class:`~synthvae.synthesizers.base.Synthesizer` (SdvTvaeSynthesizer or the
GaussianMixtureFallback). Nothing here is torch- or SDV-specific.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from ..data.preprocessing import ExpressionPreprocessor
from .latents import build_table, build_metadata, split_table


@dataclass
class SynthesisResult:
    clinical: pd.DataFrame
    expression: pd.DataFrame
    survival: pd.DataFrame
    latents: np.ndarray
    # Optional 2nd molecular modality (the joint MMVAE fills this when a
    # methylation head is present); None for the single-matrix hybrid path.
    methylation: pd.DataFrame | None = None
    # Extra molecular heads produced by the extensible MMVAE.  ``expression``
    # and ``methylation`` remain explicit for compatibility with existing users.
    modalities: dict[str, pd.DataFrame] = field(default_factory=dict)


class HybridPipeline:
    def __init__(self, compressor, synthesizer,
                 preprocessor: ExpressionPreprocessor | None = None,
                 categorical: list[str] | None = None,
                 latent_prefix: str = "expr_latent_"):
        self.compressor = compressor
        self.synthesizer = synthesizer
        self.preprocessor = preprocessor or ExpressionPreprocessor(n_top_genes=None)
        self.categorical = categorical or []
        self.latent_prefix = latent_prefix
        self._clinical_cols = None
        self._expr_cols = None

    def fit(self, clinical: pd.DataFrame, expression: pd.DataFrame,
            survival: pd.DataFrame) -> "HybridPipeline":
        self._clinical_cols = list(clinical.columns)
        self._expr_cols = list(expression.columns)

        Xall = expression.to_numpy(dtype=float)
        X = self.preprocessor.fit_transform(Xall)
        time = survival["time"].to_numpy(dtype=float)
        event = survival["event"].to_numpy(dtype=float)

        latents = self.compressor.fit_transform(X, time, event)

        table = build_table(clinical, latents, survival, latent_prefix=self.latent_prefix)
        metadata = build_metadata(
            clinical, self.compressor.n_latent, survival, self.categorical,
            latent_prefix=self.latent_prefix
        )
        self.synthesizer.fit(table, metadata)
        return self

    def sample(self, n: int) -> SynthesisResult:
        table = self.synthesizer.sample(n)
        clinical, latents, survival = split_table(
            table, self._clinical_cols, self.compressor.n_latent,
            latent_prefix=self.latent_prefix
        )
        # Decode latents to the selected-feature matrix space...
        X_sel = self.compressor.inverse_transform(latents)
        # ...and scatter back into the full feature set. Unmodelled features use
        # their training-set mean in the preprocessed matrix space.
        X_full = self.preprocessor.inverse_transform(X_sel)
        expression = pd.DataFrame(X_full, columns=self._expr_cols)
        return SynthesisResult(
            clinical=clinical, expression=expression,
            survival=survival, latents=latents
        )

    # Convenience: expose the real (train) latents for evaluation.
    def encode_real(self, expression: pd.DataFrame) -> np.ndarray:
        X = self.preprocessor.transform(expression.to_numpy(dtype=float))
        return self.compressor.transform(X)
