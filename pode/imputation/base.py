"""Shared imputer interface and small dataframe helpers.

Every engine (classical baselines, GAIN, and both TVAE engines) implements
:class:`Imputer`: ``fit`` on a (possibly incomplete) table, then ``impute``
returns ``m`` completed copies of an incomplete table. Observed cells are
always passed through unchanged.
"""

from __future__ import annotations

import fnmatch
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

from ..schema.schema import DatasetSchema


OUTCOME_POLICIES = ("use", "hide", "exclude")


class Imputer(ABC):
    """Base class for all imputation engines.

    ``fit``/``impute`` are templates that apply the leakage controls set by
    :meth:`configure`, then delegate to the engine's ``_fit``/``_impute``:

    * ``outcome_policy='use'``     - outcome columns are ordinary predictors.
    * ``outcome_policy='hide'``    - outcomes stay in the training data, but the
      rows being imputed have them set to missing for the engine (and restored
      afterwards), so a row's own outcome never informs its covariates.
    * ``outcome_policy='exclude'`` - outcome columns never reach the engine.
    * ``exclude_columns``          - other columns the engine never sees (e.g.
      post-baseline variables); they are passed through and not imputed.
    """

    name: str = "imputer"
    # False for deterministic engines whose M draws are identical.
    multiple: bool = True
    # CPU cores fit/impute may use (BLAS/OpenMP and torch threads), so library
    # use is as polite on a shared server as the CLI. -1 = all cores.
    n_jobs: int = 4

    def __init__(self) -> None:
        self.schema: DatasetSchema | None = None
        self.columns: list[str] = []
        self._diagnostics: dict[str, Any] = {}
        self.outcome_policy = "use"
        self._outcome_patterns: list[str] = []
        self._exclude_patterns: list[str] = []
        self.outcome_columns: list[str] = []
        self.excluded_columns: list[str] = []

    # ---- leakage controls ----------------------------------------------------
    def configure(
        self,
        outcome_policy: str = "use",
        outcome_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
    ) -> "Imputer":
        """Set what the engine may see. Call before ``fit`` (or after loading)."""
        if outcome_policy not in OUTCOME_POLICIES:
            raise ValueError(f"outcome_policy must be one of {OUTCOME_POLICIES}, got '{outcome_policy}'.")
        self.outcome_policy = outcome_policy
        self._outcome_patterns = list(outcome_columns or [])
        self._exclude_patterns = list(exclude_columns or [])
        if self.columns and self.schema is not None:
            # Already fitted (e.g. a pre-trained model): columns cannot be
            # removed any more, so 'exclude' degrades to 'hide'.
            self.outcome_columns = self._resolve_outcomes(self.schema, self.columns)
            if outcome_policy == "exclude" and self.outcome_columns:
                warnings.warn(
                    f"{self.name}: outcome columns are part of the trained model and cannot be "
                    "excluded; hiding them from the rows being imputed instead.",
                    stacklevel=2,
                )
                self.outcome_policy = "hide"
        return self

    def _resolve_outcomes(self, schema: DatasetSchema, cols: list[str]) -> list[str]:
        if self._outcome_patterns:
            return resolve_columns(self._outcome_patterns, cols)
        return [c for c in survival_columns(schema) if c in cols]

    def leakage_settings(self) -> dict[str, Any]:
        return {
            "outcome_policy": self.outcome_policy,
            "outcome_columns": list(self.outcome_columns),
            "excluded_columns": list(self.excluded_columns),
            "fit_scope": getattr(self, "fit_scope", None),
        }

    # ---- templates -----------------------------------------------------------
    def fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "Imputer":
        """Fit on ``df`` (NaN = missing), honouring the leakage controls."""
        cols = [c for c in schema.columns if c in df.columns]
        self.outcome_columns = self._resolve_outcomes(schema, cols)
        drop = resolve_columns(self._exclude_patterns, cols)
        if self.outcome_policy == "exclude":
            drop += [c for c in self.outcome_columns if c not in drop]
        self.excluded_columns = drop
        keep = [c for c in cols if c not in drop]
        if not keep:
            raise ValueError("No columns left to model after exclusions.")
        with limit_cpu(self.n_jobs):
            return self._fit(df[keep], subset_schema(schema, keep))

    @abstractmethod
    def _fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "Imputer":
        """Engine-specific fit on the permitted columns. Returns self."""

    @abstractmethod
    def _impute(self, df: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
        """Return ``m`` imputed copies of ``df[self.columns]``."""

    def impute(self, df: pd.DataFrame, m: int = 1, seed: int = 0) -> list[pd.DataFrame]:
        """Return ``m`` completed copies of ``df``.

        Columns the imputer was not fitted on are passed through untouched;
        fitted columns absent from ``df`` are treated as entirely missing.
        """
        work = df.copy()
        for col in self.columns:
            if col not in work.columns:
                work[col] = np.nan
        engine_in = work[self.columns].copy()
        if self.outcome_policy == "hide":
            for col in self.outcome_columns:
                if col in engine_in.columns:
                    engine_in[col] = np.nan
        # Engines see a RangeIndex; the caller's index is restored afterwards.
        with limit_cpu(self.n_jobs):
            draws = self._impute(engine_in.reset_index(drop=True), m, seed)
        out = []
        for d in draws:
            d = d.set_axis(work.index, axis=0)
            full = work.copy()
            # Restore against the caller's data, so hidden outcomes come back.
            filled = restore_observed(work[self.columns], d)
            for col in self.columns:
                full[col] = filled[col]
            out.append(finalize_dtypes(full, self.schema))
        return out

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._diagnostics)


def subset_schema(schema: DatasetSchema, cols: list[str]) -> DatasetSchema:
    """A schema restricted to ``cols`` (survival pairs kept only if complete)."""
    return DatasetSchema(
        types={c: schema.types[c] for c in cols},
        roles={c: schema.roles[c] for c in cols},
        survival_pairs=[p for p in schema.survival_pairs if p.time_col in cols and p.event_col in cols],
    )


def restore_observed(original: pd.DataFrame, imputed: pd.DataFrame) -> pd.DataFrame:
    """Take ``imputed`` values only where ``original`` is missing."""
    imputed = imputed.reindex(index=original.index, columns=original.columns)
    out = original.astype(object).copy()
    miss = original.isna()
    for col in original.columns:
        if miss[col].any():
            out.loc[miss[col], col] = imputed.loc[miss[col], col].to_numpy()
    return out.infer_objects()


def finalize_dtypes(df: pd.DataFrame, schema: DatasetSchema | None) -> pd.DataFrame:
    """Coerce numeric columns back to float/int where the values allow it."""
    out = df.copy()
    if schema is None:
        return out
    for col in out.columns:
        if col not in schema.types:
            continue
        s = out[col]
        if schema.types[col].is_numeric:
            out[col] = pd.to_numeric(s, errors="coerce")
            continue
        num = pd.to_numeric(s, errors="coerce")
        if s.notna().all() and num.notna().all() and np.allclose(num, np.round(num)):
            out[col] = num.round().astype("int64")
    return out


def resolve_columns(patterns: list[str] | None, columns: list[str]) -> list[str]:
    """Expand glob ``patterns`` against ``columns`` (order of ``columns`` kept)."""
    if not patterns:
        return []
    return [c for c in columns if any(fnmatch.fnmatchcase(c, p) for p in patterns)]


def category_key(values) -> np.ndarray:
    """Normalise categorical values to strings so 1, 1.0 and '1' compare equal."""

    def _one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return str(int(v))
        if isinstance(v, (np.integer,)):
            return str(int(v))
        return str(v)

    return np.array([_one(v) for v in np.asarray(values, dtype=object)], dtype=object)


def survival_columns(schema: DatasetSchema) -> list[str]:
    from ..schema.column_types import ColumnType

    return schema.columns_of_type(ColumnType.SURVIVAL_TIME, ColumnType.SURVIVAL_EVENT)


def n_workers(n_jobs: int) -> int:
    """Resolve an ``n_jobs`` setting (-1/0 = all cores) to a core count."""
    import os

    cpus = os.cpu_count() or 1
    return cpus if n_jobs in (-1, 0) else max(1, min(int(n_jobs), cpus))


def parallel_map(tasks, n_jobs: int) -> list:
    """Run joblib ``delayed`` tasks on at most ``n_jobs`` processes.

    Each worker is limited to one BLAS/OpenMP thread so the total stays at
    ``n_jobs`` cores rather than ``n_jobs`` x threads-per-library.
    """
    from joblib import Parallel, parallel_config

    workers = n_workers(n_jobs)
    if workers == 1:
        return [fn(*args, **kwargs) for fn, args, kwargs in tasks]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=workers)(tasks)


@contextmanager
def limit_cpu(n_jobs: int):
    """Cap BLAS/OpenMP thread pools and torch CPU threads at ``n_jobs`` cores."""
    from threadpoolctl import threadpool_limits

    cores = n_workers(n_jobs)
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    prev = torch.get_num_threads() if torch is not None else None
    if torch is not None:
        torch.set_num_threads(cores)
    try:
        with threadpool_limits(limits=cores):
            yield cores
    finally:
        if torch is not None:
            torch.set_num_threads(prev)
