"""Shared CSV loading for the synthesis CLIs.

The hybrid/MMVAE pipelines consume ``expression``/``methylation``/``copy_number``
matrices positionally -- *every* column is a feature. Input CSVs may carry a
leading ``sample_id`` column so a row can be traced back to a sample and
joined across modalities; :func:`read_table` strips it on load so the id never
leaks in as a phantom feature. It is a no-op for id-less CSVs, so it stays
backward-compatible with matrices written before the id column existed.
"""
from __future__ import annotations

import pandas as pd

SAMPLE_ID_COL = "sample_id"


def read_table(path) -> pd.DataFrame:
    """Read a CSV and drop the unifying ``sample_id`` column if present."""
    df = pd.read_csv(path)
    if SAMPLE_ID_COL in df.columns:
        df = df.drop(columns=[SAMPLE_ID_COL])
    return df
