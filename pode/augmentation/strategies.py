"""Augmentation strategies: TVAE pools and cheap real-data baselines.

Each split builds one *pool* per strategy, then every (ratio, draw) takes a
random subsample of it, as in the original M+ augmentation prototype. Sizes
are multiples of the split's real training rows, so "1.0" means "as many
synthetic rows as real ones" for every strategy alike.

* ``tvae_natural``  - unconditional TVAE samples (the generator's own case mix);
* ``tvae_targeted`` - TVAE samples meeting ``target`` (e.g. Metastases=Yes, or
  OS_Status=1, optionally with time ranges), via conditional generation;
* ``tvae_subgroup`` - TVAE samples spread over the levels of ``params.columns``
  in proportion to each level's deficit against the largest level (or
  ``params.target_count``), to lift under-represented subgroups;
* ``duplicate``     - target rows resampled with replacement (random oversampling);
* ``jitter``        - duplicates with Gaussian noise (``params.sd_frac`` x the
  training SD) on continuous non-outcome features; outcomes and categories are
  copied, so it is the survival-safe analogue of SMOTE;
* ``weight``        - no new rows: target rows get case weights adding the same
  effective count (only for models that accept weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..imputation.base import category_key
from ..schema.schema import DatasetSchema
from .config import StrategySpec
from .generator import SplitGenerator, apply_constraints

SOURCE = "_source"


def target_config(target: dict) -> dict:
    """A strategy target in the ``--conditions-config`` shape ({groups: [...]})."""
    if not target:
        return {}
    if "groups" in target:
        return {"groups": target["groups"]}
    return {"groups": [{"proportion": 1.0, **target}]}


def target_mask(df: pd.DataFrame, target: dict) -> pd.Series:
    """Rows matching any group of ``target`` (all rows when there is no target)."""
    groups = target_config(target).get("groups")
    if not groups:
        return pd.Series(True, index=df.index)
    out = pd.Series(False, index=df.index)
    for g in groups:
        m = pd.Series(True, index=df.index)
        for col, val in (g.get("equals") or {}).items():
            # category_key makes 1, 1.0 and "1" compare equal.
            m &= pd.Series(category_key(df[col]) == category_key([val])[0], index=df.index)
        for col, b in (g.get("ranges") or {}).items():
            v = pd.to_numeric(df[col], errors="coerce")
            m &= (v >= float(b["min"])) & (v <= float(b["max"]))
        out |= m
    return out


def subgroup_groups(train: pd.DataFrame, columns: list[str], target_count="match_largest") -> list[dict]:
    """Condition groups weighting each level combination by its deficit."""
    key = train[columns].astype(str).agg("|".join, axis=1)
    counts = key.value_counts()
    goal = counts.max() if target_count in (None, "match_largest") else int(target_count)
    deficit = (goal - counts).clip(lower=0)
    if deficit.sum() == 0:  # already balanced: spread evenly
        deficit = pd.Series(1, index=counts.index)
    deficit = deficit[deficit > 0]
    groups = []
    for level, d in deficit.items():
        row = train.loc[key == level, columns].iloc[0]
        groups.append({"proportion": float(d / deficit.sum()),
                       "equals": {c: _native(row[c]) for c in columns}})
    # Proportions must sum to exactly 1 for the condition parser.
    groups[-1]["proportion"] = 1.0 - sum(g["proportion"] for g in groups[:-1])
    return groups


def _native(v):
    return v.item() if isinstance(v, np.generic) else v


@dataclass
class Pool:
    """Rows (or weights) one strategy can add for one split."""

    spec: StrategySpec
    rows: pd.DataFrame
    target_index: np.ndarray  # positions of the strategy's target rows in train
    diagnostics: dict = field(default_factory=dict)

    @property
    def is_weight(self) -> bool:
        return self.spec.kind == "weight"


def build_pool(
    spec: StrategySpec,
    train: pd.DataFrame,
    schema: DatasetSchema,
    generator: SplitGenerator | None,
    n_pool: int,
    seed: int,
    constraints: dict,
    outcome_columns: list[str],
) -> Pool:
    """Sample/assemble ``spec``'s pool from one split's real training rows."""
    from ..generation.conditional import parse_conditions_config

    rng = np.random.default_rng(seed)
    train = train.reset_index(drop=True)
    tmask = target_mask(train, spec.target).to_numpy()
    tidx = np.flatnonzero(tmask)
    diag = {"strategy": spec.key, "kind": spec.kind, "requested": n_pool, "target_rows": int(len(tidx))}

    if spec.uses_tvae:
        if generator is None:
            raise ValueError(f"Strategy '{spec.key}' needs a fitted generator.")
        conditions = None
        if spec.kind == "tvae_targeted":
            conditions = parse_conditions_config(target_config(spec.target), n_pool)
        elif spec.kind == "tvae_subgroup":
            groups = subgroup_groups(train, list(spec.params["columns"]), spec.params.get("target_count"))
            conditions = parse_conditions_config({"groups": groups}, n_pool)
            diag["subgroup_groups"] = groups
        rows, shortfall = generator.sample(n_pool, conditions=conditions, seed=seed)
        rows, dropped = apply_constraints(rows, constraints)
        diag.update(produced=int(len(rows)), shortfall=int(shortfall),
                    constraint_dropped=int(sum(dropped.values())), constraint_detail=dropped)
        rows[SOURCE] = f"synthetic:{spec.key}"
        return Pool(spec, rows, tidx, diag)

    if len(tidx) == 0:
        raise ValueError(f"Strategy '{spec.key}': no training rows match its target {spec.target}.")
    if spec.kind == "weight":
        diag["produced"] = 0
        return Pool(spec, train.iloc[0:0].assign(**{SOURCE: []}), tidx, diag)

    rows = train.iloc[rng.choice(tidx, n_pool, replace=True)].reset_index(drop=True)
    if spec.kind == "jitter":
        sd_frac = float(spec.params.get("sd_frac", 0.1))
        for col in schema.numerical_columns:
            if col in rows.columns and col not in outcome_columns:
                sd = pd.to_numeric(train[col], errors="coerce").std()
                if np.isfinite(sd) and sd > 0:
                    v = pd.to_numeric(rows[col], errors="coerce")
                    rows[col] = v + rng.normal(0.0, sd_frac * sd, len(rows))
    diag["produced"] = int(len(rows))
    rows[SOURCE] = f"{spec.kind}:{spec.key}"
    return Pool(spec, rows, tidx, diag)


def draw(pool: Pool, train: pd.DataFrame, n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Real training rows + ``n`` pool rows, and per-row case weights."""
    real = train.reset_index(drop=True).assign(**{SOURCE: "real"})
    if pool.is_weight:
        w = np.ones(len(real))
        if len(pool.target_index):
            w[pool.target_index] += n / len(pool.target_index)
        return real, w
    if n <= 0 or pool.rows.empty:
        return real, np.ones(len(real))
    replace = n > len(pool.rows)
    extra = pool.rows.iloc[rng.choice(len(pool.rows), n, replace=replace)]
    out = pd.concat([real, extra], ignore_index=True)
    return out, np.ones(len(out))
