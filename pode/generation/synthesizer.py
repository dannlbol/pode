"""TVAE synthesizer wrapper.

Wraps SDV's :class:`TVAESynthesizer` with PODE's schema and config. SDV/torch
are imported lazily so the rest of the package (schema, data wrangling) stays
importable without the heavy backend installed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import SynthesizerConfig
from ..reproducibility.seeds import set_all_seeds, set_synthesizer_seed
from ..schema.schema import DatasetSchema
from .conditional import ConditionGroup, generate_conditional_data
from .postprocess import correct_survival_boundaries


class TVAEGenerator:
    """Train a TVAE on real data and sample synthetic data from it."""

    def __init__(self, schema: DatasetSchema, config: SynthesizerConfig | None = None):
        self.schema = schema
        self.config = config or SynthesizerConfig()
        self.synthesizer = None

    # ---- training ----------------------------------------------------------
    def fit(self, real_data: pd.DataFrame, seed: int | None = None) -> "TVAEGenerator":
        """Train the TVAE on ``real_data``."""
        from sdv.single_table import TVAESynthesizer

        metadata = self.schema.to_sdv_metadata()
        # Seed the global RNGs (torch/numpy) before training; the synthesizer's
        # own ``_set_random_state`` only exists after the model is built, so it
        # is applied at sampling time instead.
        if seed is not None:
            set_all_seeds(seed)
        self.synthesizer = TVAESynthesizer(
            metadata,
            epochs=self.config.epochs,
            verbose=self.config.verbose,
            **self.config.tvae_kwargs(),
        )
        self.synthesizer.fit(real_data)
        return self

    # ---- sampling ----------------------------------------------------------
    def sample(
        self,
        num_rows: int,
        conditions: str | list[ConditionGroup] | None = None,
        seed: int | None = None,
        censor_threshold: float | None = None,
    ) -> pd.DataFrame:
        """Sample synthetic rows, optionally conditioned, then post-process."""
        if self.synthesizer is None:
            raise RuntimeError("Generator is not fitted. Call fit() or load() first.")

        if seed is not None:
            set_synthesizer_seed(self.synthesizer, seed)

        if conditions:
            synthetic = generate_conditional_data(
                self.synthesizer, num_rows, conditions, self.schema, seed_value=seed
            )
            if synthetic is None:
                raise RuntimeError("Conditional generation produced no rows.")
        else:
            synthetic = self.synthesizer.sample(num_rows=num_rows)

        return correct_survival_boundaries(synthetic, self.schema, censor_threshold)

    # ---- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self.synthesizer is None:
            raise RuntimeError("Nothing to save; generator is not fitted.")
        self.synthesizer.save(filepath=str(path))

    @classmethod
    def load(cls, path: str | Path, schema: DatasetSchema | None = None) -> "TVAEGenerator":
        """Load a saved TVAE; rebuild the schema from its metadata if needed."""
        from sdv.single_table import TVAESynthesizer

        synthesizer = TVAESynthesizer.load(filepath=str(path))
        if schema is None:
            schema = _schema_from_sdv_metadata(synthesizer.get_metadata())
        obj = cls(schema=schema)
        obj.synthesizer = synthesizer
        return obj


def _schema_from_sdv_metadata(metadata) -> DatasetSchema:
    """Best-effort reconstruction of a DatasetSchema from SDV metadata.

    Maps numerical->CONTINUOUS and categorical->CATEGORICAL. Survival semantics
    are not recoverable from SDV metadata alone, so callers that need survival
    handling should pass an explicit schema to :meth:`TVAEGenerator.load`.
    """
    from ..schema.column_types import ColumnType
    from ..schema.detect import detect_survival_pairs
    from ..config import SchemaConfig

    meta_dict = metadata.to_dict()
    # SDV may wrap single-table metadata under 'tables'.
    if "columns" in meta_dict:
        columns = meta_dict["columns"]
    else:
        table = next(iter(meta_dict.get("tables", {}).values()), {})
        columns = table.get("columns", {})

    types = {}
    for col, info in columns.items():
        sdtype = info.get("sdtype")
        types[col] = ColumnType.CONTINUOUS if sdtype == "numerical" else ColumnType.CATEGORICAL

    schema = DatasetSchema.from_dict({"types": {c: t.value for c, t in types.items()}})
    schema.survival_pairs = detect_survival_pairs(list(types.keys()), SchemaConfig())
    return schema
