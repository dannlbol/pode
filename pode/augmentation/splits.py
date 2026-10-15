"""Split first, synthesise second.

Every split is defined on real development rows before any generator is
fitted, and the external cohort (if any) is removed up front, so no generator
or model ever sees a row it is scored on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AugmentationConfig


@dataclass(frozen=True)
class Split:
    repeat: int
    fold: int
    train: np.ndarray  # positional indices into the development table
    test: np.ndarray

    @property
    def key(self) -> str:
        return f"rep{self.repeat}_fold{self.fold}"


def split_external(df: pd.DataFrame, cfg: AugmentationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Development rows and the external cohort (empty when not configured)."""
    ext = cfg.external
    if not ext.column:
        return df.reset_index(drop=True), df.iloc[0:0].copy()
    if ext.column not in df.columns:
        raise ValueError(f"external.column '{ext.column}' is not in the data.")
    wanted = {str(v) for v in ext.values}
    is_ext = df[ext.column].astype(str).isin(wanted)
    if not is_ext.any():
        raise ValueError(f"No rows have {ext.column} in {sorted(wanted)}.")
    return df[~is_ext].reset_index(drop=True), df[is_ext].reset_index(drop=True)


def strat_labels(df: pd.DataFrame, event_col: str, columns: list[str], min_count: int) -> np.ndarray:
    """Event indicator joined with ``columns``; labels rarer than ``min_count``
    collapse to their event value so stratified K-fold stays feasible."""
    parts = [df[event_col].astype(str)] + [df[c].astype(str) for c in columns]
    full = pd.concat(parts, axis=1).agg("|".join, axis=1)
    counts = full.map(full.value_counts())
    labels = np.where(counts >= min_count, full, "rare|" + df[event_col].astype(str))
    return np.asarray(labels)


def make_splits(dev: pd.DataFrame, cfg: AugmentationConfig, event_col: str) -> list[Split]:
    """Repeated stratified (optionally grouped) K-fold over ``dev``."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    sc = cfg.splits
    missing = [c for c in sc.stratify if c not in dev.columns]
    if missing:
        raise ValueError(f"splits.stratify columns not in the data: {missing}")
    labels = strat_labels(dev, event_col, sc.stratify, sc.folds)
    out: list[Split] = []
    for rep in range(sc.repeats):
        seed = cfg.seed + 1000 * rep
        if sc.group_column:
            splitter = StratifiedGroupKFold(sc.folds, shuffle=True, random_state=seed)
            it = splitter.split(dev, labels, groups=dev[sc.group_column].astype(str))
        else:
            it = StratifiedKFold(sc.folds, shuffle=True, random_state=seed).split(dev, labels)
        for fold, (tr, te) in enumerate(it):
            out.append(Split(rep, fold, np.asarray(tr), np.asarray(te)))
    return out
