"""Conditional synthetic-data generation.

Ported from ``parse_conditions`` / ``generate_conditional_data`` in
``standalone_synthetic_generator_cond.py``. The condition mini-language is:

    "prop1:col1=val1,col2:min:max; prop2:col3=val3; ..."

where each ``;``-separated group is ``proportion:constraints`` and proportions
must sum to 1.0. Categorical equality constraints (``col=val``) use SDV's
``Condition``; continuous range constraints (``col:min:max``) use rejection
sampling. ``parse_conditions`` is pure Python and unit-testable without SDV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..schema.schema import DatasetSchema


@dataclass
class ConditionGroup:
    """One parsed proportion group."""

    num_rows: int
    fixed_conditions: dict = field(default_factory=dict)  # categorical equality
    rejection_conditions: list = field(default_factory=list)  # [{column,min,max}]


def _cast_value(val: str):
    """Cast a string constraint value to int/float when possible."""
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def _validate_proportions(proportions: list[float]) -> None:
    total_proportion = sum(proportions)
    if not np.isclose(total_proportion, 1.0):
        raise ValueError(f"Proportions must sum to 1.0, got {total_proportion}")


def parse_conditions(
    conditions_str: str,
    total_samples: int,
    numeric_columns: set[str] | None = None,
) -> list[ConditionGroup]:
    """Parse a condition string into :class:`ConditionGroup` objects.

    ``numeric_columns`` is used only to warn when a range constraint targets a
    non-numeric column; it does not affect parsing.
    """
    numeric_columns = numeric_columns or set()
    groups = [c.strip() for c in conditions_str.split(";") if c.strip()]

    proportions = []
    for group in groups:
        try:
            prop_str, _ = group.split(":", 1)
            proportions.append(float(prop_str))
        except ValueError as exc:
            raise ValueError(
                f"Invalid condition group '{group}'. Expected 'proportion:constraints'."
            ) from exc
    _validate_proportions(proportions)

    parsed: list[ConditionGroup] = []
    for group in groups:
        prop_str, constraints_str = group.split(":", 1)
        num_rows = int(round(total_samples * float(prop_str)))
        if num_rows == 0:
            continue

        fixed: dict = {}
        rejection: list = []
        for constraint in (c.strip() for c in constraints_str.split(",") if c.strip()):
            if "=" in constraint:
                col, val = constraint.split("=", 1)
                fixed[col.strip()] = _cast_value(val.strip())
            elif constraint.count(":") == 2:
                col, min_val, max_val = constraint.split(":")
                rejection.append(
                    {"column": col.strip(), "min": float(min_val), "max": float(max_val)}
                )
            else:
                raise ValueError(
                    f"Invalid constraint '{constraint}'. Use 'col=val' or 'col:min:max'."
                )
        parsed.append(
            ConditionGroup(
                num_rows=num_rows, fixed_conditions=fixed, rejection_conditions=rejection
            )
        )
    return parsed


def load_conditions_config(path: str | Path, total_samples: int) -> list[ConditionGroup]:
    """Load a YAML condition config into condition groups.

    Expected shape::

        groups:
          - proportion: 0.5
            equals:
              ConsensusMYC: 1
            ranges:
              Age:
                min: 3
                max: 10
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return parse_conditions_config(data, total_samples)


def parse_conditions_config(data: dict[str, Any], total_samples: int) -> list[ConditionGroup]:
    """Parse YAML-style condition data into :class:`ConditionGroup` objects."""
    if not isinstance(data, dict):
        raise ValueError("Condition config must be a mapping.")
    unknown_top_level = set(data) - {"groups"}
    if unknown_top_level:
        raise ValueError(
            f"Unsupported condition config keys: {', '.join(sorted(unknown_top_level))}"
        )

    groups = data.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Condition config must contain a non-empty 'groups' list.")

    allowed_group_keys = {"proportion", "equals", "ranges"}
    proportions: list[float] = []
    for i, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"Condition group {i} must be a mapping.")
        unknown_group_keys = set(group) - allowed_group_keys
        if unknown_group_keys:
            raise ValueError(
                f"Unsupported condition group {i} keys: {', '.join(sorted(unknown_group_keys))}"
            )
        if "proportion" not in group:
            raise ValueError(f"Condition group {i} is missing 'proportion'.")
        proportions.append(float(group["proportion"]))
    _validate_proportions(proportions)

    parsed: list[ConditionGroup] = []
    for i, group in enumerate(groups):
        num_rows = int(round(total_samples * float(group["proportion"])))
        if num_rows == 0:
            continue

        equals = group.get("equals", {}) or {}
        if not isinstance(equals, dict):
            raise ValueError(f"Condition group {i} 'equals' must be a mapping.")

        ranges = group.get("ranges", {}) or {}
        if not isinstance(ranges, dict):
            raise ValueError(f"Condition group {i} 'ranges' must be a mapping.")

        rejection = []
        for column, bounds in ranges.items():
            if not isinstance(bounds, dict) or "min" not in bounds or "max" not in bounds:
                raise ValueError(
                    f"Range for '{column}' in condition group {i} must define 'min' and 'max'."
                )
            rejection.append(
                {"column": str(column), "min": float(bounds["min"]), "max": float(bounds["max"])}
            )

        parsed.append(
            ConditionGroup(
                num_rows=num_rows,
                fixed_conditions=dict(equals),
                rejection_conditions=rejection,
            )
        )
    return parsed


def generate_conditional_data(
    synthesizer,
    num_samples: int,
    conditions_str: str | list[ConditionGroup],
    schema: DatasetSchema,
    seed_value: int | None = None,
    max_attempts: int = 30,
    max_tries_per_batch: int = 100,
    max_stalls: int = 3,
) -> pd.DataFrame | None:
    """Generate synthetic data honouring a condition string.

    Combines SDV conditional sampling (fixed categorical values) with rejection
    sampling (continuous ranges). Returns a shuffled dataframe, or None if no
    rows could be produced.

    Robustness / fail-fast: an undertrained model may rarely (or never) emit a
    requested categorical value, in which case SDV's ``sample_from_conditions``
    raises. The first time that happens for a group we switch *permanently* to
    plain sampling filtered to the fixed conditions (the conditional path is
    expensive to retry). A stall detector stops a group after ``max_stalls``
    consecutive attempts that add no rows, so a value the model never produces
    degrades to fewer rows quickly instead of spinning. Under-generation emits a
    warning rather than an error.
    """
    import warnings

    from sdv.sampling import Condition

    from ..reproducibility.seeds import set_synthesizer_seed

    if seed_value is not None:
        set_synthesizer_seed(synthesizer, seed_value)

    numeric_cols = set(schema.numerical_columns)
    groups = (
        conditions_str
        if isinstance(conditions_str, list)
        else parse_conditions(conditions_str, num_samples, numeric_cols)
    )

    all_dfs: list[pd.DataFrame] = []
    for group in groups:
        collected = pd.DataFrame()
        has_fixed = bool(group.fixed_conditions)
        # Once the conditional path fails for this group, never retry it.
        use_fallback = not has_fixed
        sdv_condition = (
            Condition(num_rows=group.num_rows, column_values=group.fixed_conditions)
            if has_fixed
            else None
        )
        attempts = 0
        stalls = 0
        while len(collected) < group.num_rows and attempts < max_attempts:
            multiplier = 5 if group.rejection_conditions else 1.2
            batch_n = max(int((group.num_rows - len(collected)) * multiplier), 50)
            if seed_value is not None:
                set_synthesizer_seed(synthesizer, seed_value + attempts)

            if has_fixed and not use_fallback:
                sdv_condition.num_rows = batch_n
                try:
                    batch = synthesizer.sample_from_conditions(
                        conditions=[sdv_condition], max_tries_per_batch=max_tries_per_batch
                    )
                except (ValueError, RuntimeError):
                    # Model rarely emits the requested value: stop using the
                    # expensive conditional path and filter plain samples instead.
                    use_fallback = True
                    batch = _fallback_filter(synthesizer, batch_n, group.fixed_conditions)
            elif has_fixed:  # use_fallback for a group that has fixed conditions
                batch = _fallback_filter(synthesizer, batch_n, group.fixed_conditions)
            else:
                batch = synthesizer.sample(num_rows=batch_n)

            if group.rejection_conditions:
                mask = pd.Series(True, index=batch.index)
                for rc in group.rejection_conditions:
                    col = rc["column"]
                    if col in batch.columns:
                        numeric = pd.to_numeric(batch[col], errors="coerce")
                        mask &= (numeric >= rc["min"]) & (numeric <= rc["max"])
                batch = batch[mask]

            stalls = stalls + 1 if len(batch) == 0 else 0
            if stalls >= max_stalls:
                break
            collected = pd.concat([collected, batch], ignore_index=True)
            attempts += 1

        if len(collected) < group.num_rows:
            warnings.warn(
                f"Conditional group {group.fixed_conditions or group.rejection_conditions} "
                f"produced only {len(collected)}/{group.num_rows} rows; the conditions may "
                f"be too restrictive for this model.",
                stacklevel=2,
            )
        all_dfs.append(collected.head(group.num_rows))

    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)


def _fallback_filter(synthesizer, batch_n: int, fixed_conditions: dict) -> pd.DataFrame:
    """Plain-sample an oversized batch and keep rows matching fixed conditions."""
    batch = synthesizer.sample(num_rows=batch_n * 5)
    for col, val in fixed_conditions.items():
        if col in batch.columns:
            batch = batch[batch[col] == val]
    return batch
