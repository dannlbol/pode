"""Engine ``hybrid``: route each incomplete row to the engine that suits it.

Benchmarks on hybrid omics tables show chained equations (MICE) are strongest
for scattered gaps, while the mask-aware TVAE is strongest when a whole
modality block is absent. ``HybridImputer`` fits both and routes each row:

* a row goes to the **block engine** if any configured block group has at
  least ``block_min_frac`` of its columns missing in that row (the block
  engine then also fills that row's scattered gaps);
* every other incomplete row goes to the **gap engine**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema.schema import DatasetSchema
from .base import Imputer, resolve_columns


class HybridImputer(Imputer):
    name = "hybrid"

    def __init__(
        self,
        gap_imputer: Imputer,
        block_imputer: Imputer,
        block_groups: dict[str, list[str]],
        block_min_frac: float = 1.0,
    ):
        super().__init__()
        if not block_groups:
            raise ValueError("hybrid needs block_groups in the imputation config to recognise absent blocks.")
        if not 0 < block_min_frac <= 1:
            raise ValueError("block_min_frac must be in (0, 1].")
        self.gap = gap_imputer
        self.block = block_imputer
        self.block_groups = block_groups
        self.block_min_frac = block_min_frac
        self.multiple = gap_imputer.multiple or block_imputer.multiple
        self.fit_scope = getattr(gap_imputer, "fit_scope", None)

    def _fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "HybridImputer":
        self.schema = schema
        self.columns = [c for c in schema.columns if c in df.columns]
        self._groups = {
            name: cols
            for name, globs in self.block_groups.items()
            if (cols := resolve_columns(globs, self.columns))
        }
        if not self._groups:
            raise ValueError("hybrid: no configured block group matches the data's columns.")
        self.gap.fit(df, schema)
        self.block.fit(df, schema)
        return self

    def route(self, df: pd.DataFrame) -> np.ndarray:
        """Boolean per row: True -> block engine."""
        to_block = np.zeros(len(df), dtype=bool)
        for cols in self._groups.values():
            present = [c for c in cols if c in df.columns]
            missing_frac = (
                df[present].isna().mean(axis=1).to_numpy() if present else np.ones(len(df))
            )
            # Columns of the group absent from df count as missing.
            missing_frac = (missing_frac * len(present) + (len(cols) - len(present))) / len(cols)
            to_block |= missing_frac >= self.block_min_frac - 1e-12
        return to_block

    def _impute(self, df: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
        incomplete = df.isna().any(axis=1).to_numpy()
        to_block = self.route(df) & incomplete
        to_gap = incomplete & ~to_block
        draws = [df.copy() for _ in range(m)]
        sub_diag = {}
        for engine, rows, label in ((self.block, to_block, "block"), (self.gap, to_gap, "gap")):
            if not rows.any():
                continue
            # Each engine sees the whole table (so stacked engines can recognise
            # their own fitting data and never duplicate rows); only the rows
            # routed to it are kept.
            part = engine.impute(df, m=m, seed=seed)
            for i in range(m):
                for col in self.columns:
                    draws[i][col] = draws[i][col].astype(object)
                    draws[i].loc[rows, col] = part[i].loc[rows, col].to_numpy()
            sub_diag[label] = engine.diagnostics()
        self._diagnostics.update({
            "rows_to_block_engine": int(to_block.sum()),
            "rows_to_gap_engine": int(to_gap.sum()),
            "block_engine": self.block.name,
            "gap_engine": self.gap.name,
            **{f"{label}_{k}": v for label, d in sub_diag.items() for k, v in d.items()
               if isinstance(v, (int, float, str))},
        })
        return [d.infer_objects() for d in draws]
