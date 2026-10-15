"""Post-processing of freshly sampled synthetic data.

Generalises the survival boundary correction from notebook cell 31: a TVAE can
emit rows sitting exactly on the censoring horizon yet still flagged as events;
those are re-coded as censored. Event columns are also coerced back to integers.
"""

from __future__ import annotations

import pandas as pd

from ..schema.schema import DatasetSchema


def correct_survival_boundaries(
    df: pd.DataFrame,
    schema: DatasetSchema,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Coerce event columns to int and fix events on the censoring boundary.

    For each survival pair, any row whose time equals ``threshold`` but is still
    marked as an event is set to censored. Returns a new dataframe.
    """
    out = df.copy()
    for pair in schema.survival_pairs:
        event_col, time_col = pair.event_col, pair.time_col
        if event_col in out.columns:
            out[event_col] = (
                pd.to_numeric(out[event_col], errors="coerce").fillna(0).astype(int)
            )
        if threshold is not None and time_col in out.columns and event_col in out.columns:
            boundary = (out[time_col] == threshold) & (out[event_col] == 1)
            out.loc[boundary, event_col] = 0
    return out
