"""Extension points for the future biomarker / feature-selection phase.

Nothing here implements biomarker logic yet (the metagene extractor is
deliberately deferred). This module defines the *contracts* that future feature
extractors and selectors will satisfy, so they can be dropped in without
touching the generation/evaluation core. All of them consume a
:class:`DatasetSchema`, keeping them consistent with the rest of PODE.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from ..schema.schema import DatasetSchema


@runtime_checkable
class FeatureExtractor(Protocol):
    """A transformer that derives new feature columns from a dataset.

    The future metagene extractor (weighted CpG signatures) will implement this.
    """

    def fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "FeatureExtractor": ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...


@runtime_checkable
class FeatureSelector(Protocol):
    """Selects an informative subset of columns (e.g. for biomarker discovery)."""

    def fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "FeatureSelector": ...

    def selected_features(self) -> list[str]: ...
