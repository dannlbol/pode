"""Seeded masking schemes for imputation benchmarks.

Each scheme hides observed cells of a complete (or partially observed) table so
imputers can be scored against the hidden truth. Masks are boolean arrays
aligned to the dataframe, with ``True`` meaning *hidden by the benchmark*.

Kinds (``MaskSpec.kind``):
  * ``mcar``          - each eligible cell hidden with probability ``rate``.
  * ``mar``           - row-level probability driven by observed ``drivers``,
                        calibrated so the mean rate over target cells is ``rate``.
  * ``mnar_self``     - a cell's probability depends on its own value
                        (``strength`` > 0 hides high values); a stress test.
  * ``block``         - a whole column ``group`` hidden in ``row_frac`` of rows.
  * ``tail``          - values beyond the ``quantile`` on ``side`` hidden
                        (with probability ``rate``).
  * ``rare_category`` - categories with frequency below ``max_freq`` hidden
                        (with probability ``rate``).
  * ``union``         - the OR of the sub-scenarios in ``parts``, e.g. scattered
                        gaps plus an absent modality block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..schema.schema import DatasetSchema
from .base import resolve_columns, survival_columns


@dataclass
class MaskSpec:
    name: str
    kind: str
    rate: float = 0.2
    # Target columns (globs); empty -> all eligible columns.
    columns: list[str] = field(default_factory=list)
    # mar: driver columns (kept observed).
    drivers: list[str] = field(default_factory=list)
    # block: name of a configured block group.
    group: str | None = None
    row_frac: float = 0.3
    # tail
    quantile: float = 0.9
    side: str = "upper"
    # mnar_self
    strength: float = 2.0
    # rare_category
    max_freq: float = 0.1
    # union: list of sub-scenario dicts
    parts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MaskSpec":
        d = dict(d)
        d.setdefault("name", d.get("kind", "scenario"))
        return cls(**d)


def eligible_columns(
    df: pd.DataFrame,
    schema: DatasetSchema,
    protect: list[str] | None = None,
    protect_survival: bool = True,
) -> list[str]:
    """Columns the benchmark may hide: schema columns minus protected ones."""
    cols = [c for c in df.columns if c in schema.types]
    blocked = set(resolve_columns(protect or [], cols))
    if protect_survival:
        blocked.update(survival_columns(schema))
    return [c for c in cols if c not in blocked]


def make_mask(
    df: pd.DataFrame,
    spec: MaskSpec,
    schema: DatasetSchema,
    rng: np.random.Generator,
    protect: list[str] | None = None,
    protect_survival: bool = True,
    groups: dict[str, list[str]] | None = None,
    atomic_blocks: bool = False,
) -> np.ndarray:
    """Build a boolean mask (True = hide) for ``df`` under ``spec``.

    Cells that are already missing are never marked: the mask only hides
    observed values, so the truth is known for every masked cell.

    With ``atomic_blocks`` the columns of every block group (e.g. all latents
    derived from one omics modality) are all-or-nothing, as in real cohorts: a
    patient has the whole modality or none of it. Cell-level kinds (mcar, mar,
    mnar_self, tail, rare_category) then never touch block columns, and only
    ``block`` hides them, a whole group at a time.
    """
    eligible = eligible_columns(df, schema, protect, protect_survival)
    in_block = _block_membership(df, groups) if atomic_blocks else {}
    if spec.kind not in ("block", "union"):
        if spec.columns:
            bad = [c for c in resolve_columns(spec.columns, eligible) if c in in_block]
            if bad:
                raise ValueError(
                    f"Scenario '{spec.name}': {bad[:3]} belong to block group "
                    f"'{in_block[bad[0]]}', which is all-or-nothing (atomic_blocks); "
                    "use kind: block to hide it."
                )
        eligible = [c for c in eligible if c not in in_block]
    targets = resolve_columns(spec.columns, eligible) if spec.columns else list(eligible)
    col_idx = {c: i for i, c in enumerate(df.columns)}
    n = len(df)
    mask = np.zeros(df.shape, dtype=bool)

    kind = spec.kind
    if kind == "mcar":
        for c in targets:
            mask[:, col_idx[c]] = rng.random(n) < spec.rate

    elif kind == "mar":
        drivers = [d for d in spec.drivers if d in df.columns]
        if not drivers:
            raise ValueError(f"Scenario '{spec.name}': mar needs observed 'drivers'.")
        targets = [c for c in targets if c not in drivers]
        score = _driver_score(df, drivers, schema, rng)
        p = _calibrated_probs(score, spec.rate)
        for c in targets:
            mask[:, col_idx[c]] = rng.random(n) < p

    elif kind == "mnar_self":
        for c in targets:
            z = _standardised(df[c], schema, c)
            p = _calibrated_probs(spec.strength * np.nan_to_num(z), spec.rate)
            mask[:, col_idx[c]] = rng.random(n) < p

    elif kind == "block":
        if not spec.group or spec.group not in (groups or {}):
            raise ValueError(f"Scenario '{spec.name}': unknown block group '{spec.group}'.")
        block_cols = [c for c in resolve_columns(groups[spec.group], list(df.columns)) if c in eligible]
        if not block_cols:
            raise ValueError(f"Scenario '{spec.name}': block group '{spec.group}' matched no columns.")
        rows = rng.random(n) < spec.row_frac
        for c in block_cols:
            mask[:, col_idx[c]] = rows

    elif kind == "tail":
        for c in targets:
            if not schema.types[c].is_numeric:
                continue
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            hi = np.nanquantile(x, spec.quantile)
            lo = np.nanquantile(x, 1 - spec.quantile)
            if spec.side == "upper":
                tail = x > hi
            elif spec.side == "lower":
                tail = x < lo
            else:
                tail = (x > hi) | (x < lo)
            mask[:, col_idx[c]] = tail & (rng.random(n) < spec.rate)

    elif kind == "rare_category":
        for c in targets:
            if not schema.types[c].is_discrete:
                continue
            freq = df[c].map(df[c].value_counts(normalize=True)).to_numpy(dtype=float)
            mask[:, col_idx[c]] = (freq < spec.max_freq) & (rng.random(n) < spec.rate)

    elif kind == "union":
        if not spec.parts:
            raise ValueError(f"Scenario '{spec.name}': union needs 'parts'.")
        for i, part in enumerate(spec.parts):
            sub = MaskSpec.from_dict({"name": f"{spec.name}[{i}]", **part})
            mask |= make_mask(df, sub, schema, rng, protect, protect_survival, groups, atomic_blocks)

    else:
        raise ValueError(f"Unknown mask kind '{kind}'.")

    mask &= df.notna().to_numpy()
    if atomic_blocks and groups:
        partial = partial_block_rows(apply_mask(df, mask), groups, observed_before=df)
        if partial:
            raise AssertionError(f"Scenario '{spec.name}' split a block group: {partial}")
    return mask


def _block_membership(df: pd.DataFrame, groups: dict[str, list[str]] | None) -> dict[str, str]:
    """{column: group name} for every column in a block group."""
    out: dict[str, str] = {}
    for name, globs in (groups or {}).items():
        for c in resolve_columns(globs, list(df.columns)):
            out.setdefault(c, name)
    return out


def partial_block_rows(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    observed_before: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Rows where a block group is partly missing, per group.

    Real modality-derived columns are all present or all absent, so partial
    rows point to a data-preparation problem (or, in a benchmark, a mask that
    split a block). With ``observed_before``, cells already missing there are
    ignored, so only newly introduced partial blocks count.
    """
    out = {}
    for name, cols in ((n, resolve_columns(g, list(df.columns))) for n, g in groups.items()):
        if not cols:
            continue
        miss = df[cols].isna().to_numpy()
        if observed_before is not None:
            before = observed_before[cols].isna().to_numpy()
            miss_new = miss & ~before
            partial = miss_new.any(axis=1) & ~(miss | before).all(axis=1)
        else:
            partial = miss.any(axis=1) & ~miss.all(axis=1)
        if partial.any():
            out[name] = int(partial.sum())
    return out


def apply_mask(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Return a copy of ``df`` with masked cells set to NaN."""
    return df.mask(pd.DataFrame(mask, index=df.index, columns=df.columns))


def _standardised(s: pd.Series, schema: DatasetSchema, col: str) -> np.ndarray:
    if schema.types[col].is_numeric:
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    else:
        # Order categories by frequency so "rank" has a stable meaning.
        order = {v: i for i, v in enumerate(s.value_counts().index)}
        x = s.map(order).to_numpy(dtype=float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)


def _driver_score(
    df: pd.DataFrame, drivers: list[str], schema: DatasetSchema, rng: np.random.Generator
) -> np.ndarray:
    """Linear score from observed drivers; categorical levels get random effects."""
    score = np.zeros(len(df))
    for d in drivers:
        if d in schema.types and schema.types[d].is_numeric:
            score += np.nan_to_num(_standardised(df[d], schema, d))
        else:
            levels = df[d].dropna().unique()
            effects = dict(zip(levels, rng.normal(0, 1, len(levels))))
            score += df[d].map(effects).fillna(0.0).to_numpy(dtype=float)
    sd = score.std()
    return score / sd if sd > 0 else score


def _calibrated_probs(score: np.ndarray, rate: float) -> np.ndarray:
    """sigmoid(a + score) with intercept ``a`` chosen so mean probability = rate."""
    rate = float(np.clip(rate, 1e-6, 1 - 1e-6))
    lo, hi = -50.0, 50.0
    for _ in range(100):
        a = 0.5 * (lo + hi)
        if np.mean(1 / (1 + np.exp(-(a + score)))) < rate:
            lo = a
        else:
            hi = a
    return 1 / (1 + np.exp(-(0.5 * (lo + hi) + score)))
