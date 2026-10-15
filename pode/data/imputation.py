"""Post-generation NaN imputation for synthetic data.

Adapted from ``impute_synthetic_nans_custom`` in
``standalone_synthetic_generator_cond.py``, but driven by a
:class:`DatasetSchema` instead of an SDV metadata dict.

Strategies:
  * numerical: ``'simple'`` -> median fitted on the real data.
  * categorical: ``'proportional_fill'`` -> sample from the synthetic column's
    own value distribution; ``'simple'`` -> mode of the real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema.schema import DatasetSchema


def impute_synthetic_nans(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    schema: DatasetSchema,
    cat_strategy: str = "proportional_fill",
    num_strategy: str = "simple",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Impute NaNs in ``synthetic_df`` using ``real_df`` as the reference.

    Returns a new dataframe with NaNs filled for columns known to the schema.
    """
    rng = rng or np.random.default_rng()
    out = synthetic_df.copy()

    numerical_cols = [
        c for c in schema.numerical_columns if c in out.columns and c in real_df.columns
    ]
    categorical_cols = [
        c for c in schema.categorical_columns if c in out.columns and c in real_df.columns
    ]

    if num_strategy == "simple":
        for col in numerical_cols:
            if not out[col].isna().any():
                continue
            median_val = pd.to_numeric(real_df[col], errors="coerce").median()
            if pd.isna(median_val):
                median_val = 0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(median_val)

    for col in categorical_cols:
        if not out[col].isna().any():
            continue
        nan_idx = out.index[out[col].isna()]
        if cat_strategy == "proportional_fill":
            non_nan = out[col].dropna()
            proportions = non_nan.value_counts(normalize=True)
            if not proportions.empty:
                out.loc[nan_idx, col] = rng.choice(
                    proportions.index.to_numpy(),
                    size=len(nan_idx),
                    p=proportions.to_numpy(),
                )
                continue
        # 'simple' strategy, or fallback when synthetic column is all-NaN.
        mode = real_df[col].mode()
        fill = mode.iloc[0] if not mode.empty else f"Unknown_{col}"
        out.loc[nan_idx, col] = fill

    return out
