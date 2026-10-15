"""Data wrangling: quality analysis, variable filtering, category collapsing.

Generalised from the notebook helpers ``analyze_data_quality`` and
``filter_variables_by_quality`` (cells 17/19), plus the category-collapsing and
column-dropping steps requested in the project notes. All routines are
data-agnostic - column names come from arguments or the schema, never literals.
"""

from __future__ import annotations

import fnmatch
from typing import Iterable

import pandas as pd


def drop_columns_by_pattern(
    df: pd.DataFrame,
    patterns: Iterable[str] = (),
    columns: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns matching glob-style ``patterns`` or named in ``columns``.

    Returns the filtered frame and the list of dropped column names.
    """
    to_drop: set[str] = set(c for c in columns if c in df.columns)
    for pat in patterns:
        to_drop.update(fnmatch.filter(list(df.columns), pat))
    dropped = sorted(to_drop)
    return df.drop(columns=dropped, errors="ignore").copy(), dropped


def analyze_data_quality(
    df: pd.DataFrame, variables: Iterable[str] | None = None
) -> pd.DataFrame:
    """Per-column quality report: missingness, cardinality, sparsity.

    Returns a dataframe indexed by column with columns:
    ``n_missing``, ``na_fraction``, ``n_unique``, ``min_category_count``.
    """
    cols = list(variables) if variables is not None else list(df.columns)
    rows = []
    n = len(df)
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            min_cat = 0
        else:
            min_cat = int(non_null.value_counts().min())
        rows.append(
            {
                "column": col,
                "n_missing": int(series.isna().sum()),
                "na_fraction": float(series.isna().mean()) if n else 0.0,
                "n_unique": int(non_null.nunique()),
                "min_category_count": min_cat,
            }
        )
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.set_index("column")
    return report


def filter_variables_by_quality(
    df: pd.DataFrame,
    max_na_fraction: float = 1.0,
    min_category_count: int = 0,
    protected_columns: Iterable[str] = (),
    manual_exclude: Iterable[str] = (),
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Drop low-quality columns; never drop ``protected_columns``.

    A column is removed if it is in ``manual_exclude``, exceeds
    ``max_na_fraction`` missingness, or (for discrete columns) its rarest
    category has fewer than ``min_category_count`` observations.

    Returns the filtered frame and a ``{column: reason}`` dict of what was
    removed.
    """
    protected = set(protected_columns)
    removed: dict[str, str] = {}

    for col in manual_exclude:
        if col in df.columns and col not in protected:
            removed[col] = "manually excluded"

    quality = analyze_data_quality(df)
    for col, row in quality.iterrows():
        if col in protected or col in removed:
            continue
        if row["na_fraction"] > max_na_fraction:
            removed[col] = f"na_fraction {row['na_fraction']:.2f} > {max_na_fraction}"
        elif (
            min_category_count > 0
            # Only applies to discrete-like columns; never drop a high-cardinality
            # (continuous) column just because each value is rare.
            and row["n_unique"] <= 20
            and row["min_category_count"] < min_category_count
        ):
            removed[col] = (
                f"min_category_count {row['min_category_count']} < {min_category_count}"
            )

    filtered = df.drop(columns=list(removed.keys()), errors="ignore").copy()
    return filtered, removed


def collapse_categories(
    df: pd.DataFrame, column: str, mapping: dict, other_label: str | None = None
) -> pd.DataFrame:
    """Collapse categories in ``column`` according to ``mapping``.

    ``mapping`` maps original values -> collapsed values. Values not present in
    the mapping are left unchanged, or replaced with ``other_label`` if given.
    Returns a new dataframe.
    """
    out = df.copy()
    if column not in out.columns:
        raise KeyError(f"Column '{column}' not in dataframe")

    def _map(value):
        if value in mapping:
            return mapping[value]
        if other_label is not None and pd.notna(value):
            return other_label
        return value

    out[column] = out[column].map(_map)
    return out


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce the given columns to numeric (invalid parsing becomes NaN)."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
