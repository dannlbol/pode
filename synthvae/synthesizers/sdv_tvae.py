"""SDV TVAE wrapper -- the production path.

Thin adapter around ``sdv.single_table.TVAESynthesizer`` that translates our
lightweight column-type metadata into an SDV metadata object. This is the
synthesizer you already trust for clinico-pathological data; here it additionally
receives the expression latent columns as continuous features and the survival
columns, so the joint distribution of clinical + latents + survival is modelled
in one place.

Uses the ``SingleTableMetadata`` API (stable across SDV 1.x): auto-detect from the
dataframe, then override the sdtypes we care about. If your SDV version exposes the
newer unified ``Metadata`` class only, adjust ``_build_metadata`` accordingly --
this is the one spot that is SDV-version sensitive.

Requires ``sdv``. Kept import-guarded so the package imports without it.
"""
from __future__ import annotations

import pandas as pd
from joblib import parallel_backend
from .base import Synthesizer


class SdvTvaeSynthesizer(Synthesizer):
    def __init__(self, epochs: int = 300, cuda: bool | None = None, **tvae_kwargs):
        self.epochs = epochs
        self.cuda = cuda
        self.tvae_kwargs = tvae_kwargs
        self._model = None
        self._columns = None

    def _build_metadata(self, table: pd.DataFrame, metadata: dict):
        from sdv.metadata import SingleTableMetadata

        md = SingleTableMetadata()
        md.detect_from_dataframe(table)
        # Override auto-detected sdtypes with our explicit spec.
        for col, spec in metadata["columns"].items():
            if col not in table.columns:
                continue
            sdtype = spec["sdtype"]
            try:
                md.update_column(column_name=col, sdtype=sdtype)
            except Exception:
                # Some SDV versions require removing then re-adding; be defensive.
                md.columns[col] = {"sdtype": sdtype}
        return md

    def fit(self, table: pd.DataFrame, metadata: dict):
        from sdv.single_table import TVAESynthesizer

        self._columns = list(table.columns)
        md = self._build_metadata(table, metadata)
        kwargs = dict(self.tvae_kwargs)
        if self.cuda is not None:
            kwargs["cuda"] = self.cuda
        self._model = TVAESynthesizer(md, epochs=self.epochs, **kwargs)
        # CTGAN/TVAE uses joblib for preprocessing larger tables; prefer
        # threads so training also works in restricted environments.
        with parallel_backend("threading"):
            self._model.fit(table)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("SdvTvaeSynthesizer is not fitted.")
        out = self._model.sample(num_rows=n)
        return out[self._columns]
