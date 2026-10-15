"""Dataset I/O helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """Load a tabular dataset from CSV.

    Thin wrapper around :func:`pandas.read_csv` kept as a seam so other formats
    (parquet, feather) can be added without touching callers.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, **read_csv_kwargs)


def save_dataset(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Save a dataframe to CSV, creating parent directories as needed."""
    path = Path(path)
    if path.parent:
        os.makedirs(path.parent, exist_ok=True)
    df.to_csv(path, index=index)


def ensure_dir(directory_path: str | Path) -> Path:
    """Create a directory if it does not exist; return it as a Path."""
    directory_path = Path(directory_path)
    os.makedirs(directory_path, exist_ok=True)
    return directory_path
