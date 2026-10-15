"""Administrative survival censoring.

Generalises the notebook censoring block (cell 23 / ``grid_search_v2.py``): any
event occurring after a horizon is re-coded as censored at the horizon, and any
time beyond the horizon is clipped. Works on arbitrary (time, event) column
pairs supplied directly or taken from a :class:`DatasetSchema`.
"""

from __future__ import annotations

import pandas as pd

from ..schema.schema import DatasetSchema


def apply_administrative_censoring(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    threshold: float,
) -> pd.DataFrame:
    """Censor a single (time, event) pair at ``threshold``.

    Events after the threshold become censored (event=0) at the threshold;
    follow-up times beyond the threshold are clipped. The event column is
    coerced to numeric first. Returns a new dataframe.
    """
    out = df.copy()
    if time_col not in out.columns or event_col not in out.columns:
        return out

    out[event_col] = pd.to_numeric(out[event_col], errors="coerce")
    event_mask = (out[time_col] > threshold) & (out[event_col] == 1)
    out.loc[event_mask, event_col] = 0.0
    out.loc[event_mask, time_col] = threshold
    time_mask = out[time_col] > threshold
    out.loc[time_mask, time_col] = threshold
    return out


def censor_by_schema(
    df: pd.DataFrame, schema: DatasetSchema, threshold: float
) -> pd.DataFrame:
    """Apply administrative censoring to every survival pair in the schema."""
    out = df
    for pair in schema.survival_pairs:
        out = apply_administrative_censoring(out, pair.time_col, pair.event_col, threshold)
    return out
