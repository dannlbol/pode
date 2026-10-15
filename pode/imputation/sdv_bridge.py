"""Bridge between raw rows and the decoder space of a fitted SDV TVAE.

A PODE ``model.pkl`` is an SDV ``TVAESynthesizer``. Its pipeline is::

    raw df -> SDV DataProcessor (RDT) -> ctgan DataTransformer -> decoder space

ctgan discards the TVAE encoder after training, so only the decoder and the
two transformers survive in a saved model. This module exposes exactly what
latent conditioning needs from them. It relies on private SDV/ctgan attributes
(``_data_processor``, ``_model.decoder``, ``_model.transformer``), so it
validates them up front and fails with a clear message if a library upgrade
changes the layout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .base import category_key
from .spans import ColumnSpan, spans_from_transformer

TESTED_VERSIONS = {"sdv": "1.25", "ctgan": "0.12"}


class BridgeError(RuntimeError):
    """The SDV/ctgan model layout is not what the bridge expects."""


class SDVDecoderBridge:
    """Encode raw rows into, and decode samples out of, a TVAE's decoder space."""

    def __init__(self, synthesizer):
        _validate(synthesizer)
        self.synth = synthesizer
        self.model = synthesizer._model
        self.transformer = self.model.transformer
        self.decoder = self.model.decoder
        self.decoder.eval()
        self.device = self.model._device
        self.emb_dim = int(self.model.embedding_dim)
        self.loss_factor = float(getattr(self.model, "loss_factor", 1.0))
        self.spans: list[ColumnSpan] = spans_from_transformer(self.transformer)
        self.columns = [sp.name for sp in self.spans]
        self._info = {i.column_name: i for i in self.transformer._column_transform_info_list}

    # ---- encode ------------------------------------------------------------
    def encode(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x (n, D), observed (n, C))`` for ``df``'s model columns.

        Missing cells get a valid placeholder before transformation; their
        spans are ignored by the likelihood through the observed mask.
        Discrete values unseen during training are treated as missing.
        """
        raw = df.reindex(columns=self.columns)
        observed = raw.notna().to_numpy()
        processed = self.synth._data_processor.transform(raw.copy())
        processed = processed.reindex(columns=self.columns)

        for j, col in enumerate(self.columns):
            info = self._info[col]
            if info.column_type == "discrete":
                dummies = list(info.transform.dummies)
                lookup = {k: d for k, d in zip(category_key(dummies), dummies)}
                keys = category_key(raw[col].to_numpy())
                mapped = [lookup.get(k) for k in keys]
                unknown = np.array([k is not None and k not in lookup for k in keys])
                observed[:, j] &= ~unknown
                placeholder = dummies[0]
                processed[col] = pd.Series(
                    [v if v is not None else placeholder for v in mapped],
                    index=processed.index,
                    dtype=object,
                )
                processed[col] = processed[col].infer_objects()
            else:
                vals = pd.to_numeric(processed[col], errors="coerce")
                fill = vals[observed[:, j]].median() if observed[:, j].any() else 0.0
                processed[col] = vals.where(observed[:, j], fill).fillna(0.0)

        x = self.transformer.transform(processed).astype(np.float32)
        return x, observed

    # ---- decode ------------------------------------------------------------
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(z)

    def to_frame(self, samples: np.ndarray) -> pd.DataFrame:
        """Map transformed-space samples (N, D) back to raw values."""
        inner = self.transformer.inverse_transform(samples.astype(np.float64), sigmas=None)
        raw = self.synth._data_processor.reverse_transform(inner)
        raw = raw.reindex(columns=self.columns)
        # SDV may re-insert NaNs at its training-time null rate; fall back to
        # the pre-RDT value for any such cell so imputations are never empty.
        for col in self.columns:
            if raw[col].isna().any() and col in inner.columns:
                raw[col] = raw[col].where(raw[col].notna(), inner[col].to_numpy())
        return raw.reset_index(drop=True)


def _validate(synthesizer) -> None:
    missing = []
    if not hasattr(synthesizer, "_data_processor"):
        missing.append("_data_processor")
    model = getattr(synthesizer, "_model", None)
    if model is None:
        missing.append("_model (is the synthesizer fitted?)")
    else:
        for attr in ("decoder", "transformer", "embedding_dim", "_device"):
            if not hasattr(model, attr):
                missing.append(f"_model.{attr}")
    if missing:
        raise BridgeError(
            "Cannot condition this SDV TVAE: missing " + ", ".join(missing) + ". "
            f"The decoder bridge was built against sdv {TESTED_VERSIONS['sdv']}.x / "
            f"ctgan {TESTED_VERSIONS['ctgan']}.x; check the installed versions."
        )
