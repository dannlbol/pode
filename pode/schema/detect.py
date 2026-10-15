"""Automatic column-type detection.

Generalises the ad-hoc metadata heuristics from the original notebooks
(``metadata.detect_from_dataframe`` + manual ``update_column`` overrides) into a
data-agnostic routine driven entirely by :class:`~pode.config.SchemaConfig`.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

import pandas as pd
from pandas.api import types as ptypes

from ..config import SchemaConfig
from .column_types import ColumnType


@dataclass(frozen=True)
class SurvivalPair:
    """A matched (time, event) survival column pair, keyed by a shared base."""

    base: str
    time_col: str
    event_col: str


def _matches_suffix(name: str, suffixes: list[str]) -> str | None:
    """Return the base name if ``name`` ends with any suffix, else None."""
    for suffix in suffixes:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return None


def detect_survival_pairs(columns: list[str], config: SchemaConfig) -> list[SurvivalPair]:
    """Find (time, event) column pairs sharing a base name.

    e.g. ``OS_Time`` + ``OS_Status`` -> SurvivalPair(base="OS", ...).
    """
    pairs: list[SurvivalPair] = []
    seen: set[tuple[str, str]] = set()

    for pair_cfg in config.survival_pairs:
        time_col = pair_cfg.get("time_col")
        event_col = pair_cfg.get("event_col")
        if not time_col or not event_col:
            continue
        if time_col not in columns or event_col not in columns:
            continue
        key = (time_col, event_col)
        if key in seen:
            continue
        base = pair_cfg.get("base") or time_col
        pairs.append(SurvivalPair(base=base, time_col=time_col, event_col=event_col))
        seen.add(key)

    time_bases: dict[str, str] = {}
    event_bases: dict[str, str] = {}
    for col in columns:
        base_t = _matches_suffix(col, config.survival_time_suffixes)
        if base_t is not None:
            time_bases.setdefault(base_t, col)
        base_e = _matches_suffix(col, config.survival_event_suffixes)
        if base_e is not None:
            event_bases.setdefault(base_e, col)

    for base, time_col in time_bases.items():
        if base in event_bases:
            key = (time_col, event_bases[base])
            if key in seen:
                continue
            pairs.append(SurvivalPair(base=base, time_col=time_col, event_col=event_bases[base]))
            seen.add(key)
    return pairs


def detect_identifiers(df: pd.DataFrame, config: SchemaConfig | None = None) -> set[str]:
    """Detect identifier columns (unique row tags like Sample_ID, sentrix.id).

    A column is an identifier if its name matches an ``identifier_patterns`` glob
    (case-insensitive) **or** it is non-numeric and its unique fraction meets
    ``identifier_unique_fraction``. The non-numeric guard prevents continuous
    columns (often all-unique floats, e.g. Age) being misflagged. Survival
    columns are never identifiers.
    """
    config = config or SchemaConfig()
    if not config.detect_identifiers:
        return set()

    survival = set()
    for pair in detect_survival_pairs(list(df.columns), config):
        survival.update({pair.time_col, pair.event_col})

    patterns_lower = [p.lower() for p in config.identifier_patterns]
    n = len(df)
    identifiers: set[str] = set()
    for col in df.columns:
        if col in survival:
            continue
        if any(fnmatch.fnmatch(col.lower(), pat) for pat in patterns_lower):
            identifiers.add(col)
            continue
        series = df[col]
        if not ptypes.is_numeric_dtype(series) and n > 0:
            if series.dropna().nunique() / n >= config.identifier_unique_fraction:
                identifiers.add(col)
    return identifiers


def _infer_from_series(series: pd.Series, config: SchemaConfig) -> ColumnType:
    """Infer a column type from the data alone (no name-based hints)."""
    non_null = series.dropna()
    n_unique = non_null.nunique()

    if n_unique <= 2:
        return ColumnType.BINARY

    if ptypes.is_bool_dtype(series):
        return ColumnType.BINARY

    if ptypes.is_numeric_dtype(series):
        # Integer-valued with low cardinality -> ordinal; otherwise continuous.
        try:
            all_integer = bool((non_null == non_null.round()).all())
        except TypeError:
            all_integer = False
        if all_integer and n_unique <= config.max_cardinality_for_discrete:
            return ColumnType.ORDINAL
        return ColumnType.CONTINUOUS

    # Object / string / category dtype.
    return ColumnType.CATEGORICAL


def detect_column_types(
    df: pd.DataFrame, config: SchemaConfig | None = None
) -> dict[str, ColumnType]:
    """Detect a :class:`ColumnType` for every column in ``df``.

    Resolution order per column:
      1. explicit ``type_overrides`` from config,
      2. survival time/event pairing,
      3. ``continuous_prefixes`` name match,
      4. data-driven inference (dtype + cardinality).
    """
    config = config or SchemaConfig()
    columns = list(df.columns)
    result: dict[str, ColumnType] = {}

    survival_pairs = detect_survival_pairs(columns, config)
    survival_lookup: dict[str, ColumnType] = {}
    for pair in survival_pairs:
        survival_lookup[pair.time_col] = ColumnType.SURVIVAL_TIME
        survival_lookup[pair.event_col] = ColumnType.SURVIVAL_EVENT

    for col in columns:
        if col in config.type_overrides:
            result[col] = ColumnType(config.type_overrides[col])
            continue
        if col in survival_lookup:
            result[col] = survival_lookup[col]
            continue
        if any(col.startswith(prefix) for prefix in config.continuous_prefixes):
            result[col] = ColumnType.CONTINUOUS
            continue
        result[col] = _infer_from_series(df[col], config)

    return result
