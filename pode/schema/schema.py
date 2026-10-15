"""The :class:`DatasetSchema` - PODE's single source of truth about columns.

A schema records each column's :class:`ColumnType` and :class:`ColumnRole`, the
detected survival pairs, and knows how to build SDV metadata. It is consumed by
generation, tuning, evaluation and (later) feature-selection so none of those
need to re-derive column semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from ..config import SchemaConfig
from .column_types import ColumnRole, ColumnType
from .detect import (
    SurvivalPair,
    detect_column_types,
    detect_identifiers,
    detect_survival_pairs,
)


def _default_role(col_type: ColumnType) -> ColumnRole:
    """Assign a sensible default role from a column's type."""
    if col_type in (ColumnType.SURVIVAL_TIME, ColumnType.SURVIVAL_EVENT):
        return ColumnRole.SURVIVAL
    return ColumnRole.FEATURE


@dataclass
class DatasetSchema:
    """Typed description of a tabular dataset."""

    types: dict[str, ColumnType] = field(default_factory=dict)
    roles: dict[str, ColumnRole] = field(default_factory=dict)
    survival_pairs: list[SurvivalPair] = field(default_factory=list)

    # ---- construction ------------------------------------------------------
    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, config: SchemaConfig | None = None
    ) -> "DatasetSchema":
        """Auto-detect a schema from a dataframe, honouring config overrides."""
        config = config or SchemaConfig()
        types = detect_column_types(df, config)
        pairs = detect_survival_pairs(list(df.columns), config)
        roles = {col: _default_role(t) for col, t in types.items()}
        # Identifier columns (unique row tags) get the IDENTIFIER role so they
        # are excluded from features / HR analysis.
        for col in detect_identifiers(df, config):
            if col in roles:
                roles[col] = ColumnRole.IDENTIFIER
        for col, role in config.role_overrides.items():
            if col in roles:
                roles[col] = ColumnRole(role)
        return cls(types=types, roles=roles, survival_pairs=pairs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetSchema":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetSchema":
        types = {c: ColumnType(t) for c, t in data.get("types", {}).items()}
        roles = {c: ColumnRole(r) for c, r in data.get("roles", {}).items()}
        # Fill missing roles with defaults.
        for col, t in types.items():
            roles.setdefault(col, _default_role(t))
        pairs = [
            SurvivalPair(base=p["base"], time_col=p["time_col"], event_col=p["event_col"])
            for p in data.get("survival_pairs", [])
        ]
        return cls(types=types, roles=roles, survival_pairs=pairs)

    # ---- mutation ----------------------------------------------------------
    def override(
        self,
        column: str,
        type: ColumnType | str | None = None,
        role: ColumnRole | str | None = None,
    ) -> "DatasetSchema":
        """Override a column's type and/or role in place; returns self."""
        if column not in self.types:
            raise KeyError(f"Column '{column}' not in schema")
        if type is not None:
            self.types[column] = ColumnType(type)
            self.roles.setdefault(column, _default_role(self.types[column]))
        if role is not None:
            self.roles[column] = ColumnRole(role)
        return self

    # ---- queries -----------------------------------------------------------
    @property
    def columns(self) -> list[str]:
        return list(self.types.keys())

    def columns_of_type(self, *col_types: ColumnType) -> list[str]:
        wanted = set(col_types)
        return [c for c, t in self.types.items() if t in wanted]

    def columns_of_role(self, *col_roles: ColumnRole) -> list[str]:
        wanted = set(col_roles)
        return [c for c, r in self.roles.items() if r in wanted]

    @property
    def numerical_columns(self) -> list[str]:
        return [c for c, t in self.types.items() if t.is_numeric]

    @property
    def categorical_columns(self) -> list[str]:
        return [c for c, t in self.types.items() if t.is_discrete]

    @property
    def feature_columns(self) -> list[str]:
        """Non-survival, non-id covariates - the candidates for HR/analysis."""
        return [
            c
            for c, r in self.roles.items()
            if r in (ColumnRole.FEATURE, ColumnRole.TARGET)
            and self.types[c] not in (ColumnType.SURVIVAL_TIME, ColumnType.SURVIVAL_EVENT)
        ]

    def primary_survival_pair(self) -> SurvivalPair | None:
        """The first detected survival pair, if any (e.g. overall survival)."""
        return self.survival_pairs[0] if self.survival_pairs else None

    # ---- SDV bridge --------------------------------------------------------
    def to_sdv_metadata(self):
        """Build an SDV ``SingleTableMetadata`` matching this schema.

        Imported lazily so that schema/data utilities work without SDV present.
        """
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        for col, col_type in self.types.items():
            if col_type.is_numeric:
                metadata.add_column(col, sdtype="numerical", computer_representation="Float")
            else:
                metadata.add_column(col, sdtype="categorical")
        return metadata

    # ---- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "types": {c: t.value for c, t in self.types.items()},
            "roles": {c: r.value for c, r in self.roles.items()},
            "survival_pairs": [
                {"base": p.base, "time_col": p.time_col, "event_col": p.event_col}
                for p in self.survival_pairs
            ],
        }

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
