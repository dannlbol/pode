"""Column type and role taxonomy for PODE datasets.

These enums are the shared vocabulary that every downstream module (generation,
tuning, evaluation, and the future biomarker / feature-selection phase) uses to
reason about columns without re-deriving their nature each time.
"""

from __future__ import annotations

from enum import Enum


class ColumnType(str, Enum):
    """The statistical nature of a column.

    Inherits from ``str`` so values serialise cleanly to/from YAML and JSON.
    """

    CONTINUOUS = "continuous"
    BINARY = "binary"
    ORDINAL = "ordinal"
    CATEGORICAL = "categorical"
    SURVIVAL_TIME = "survival_time"
    SURVIVAL_EVENT = "survival_event"

    @property
    def is_numeric(self) -> bool:
        """Whether the type maps to a numerical SDV sdtype."""
        return self in (ColumnType.CONTINUOUS, ColumnType.SURVIVAL_TIME)

    @property
    def is_discrete(self) -> bool:
        """Whether the type is treated as discrete/categorical-like."""
        return self in (
            ColumnType.BINARY,
            ColumnType.ORDINAL,
            ColumnType.CATEGORICAL,
            ColumnType.SURVIVAL_EVENT,
        )

    def to_sdtype(self) -> str:
        """Map to an SDV ``sdtype`` ('numerical' or 'categorical')."""
        return "numerical" if self.is_numeric else "categorical"


class ColumnRole(str, Enum):
    """The semantic role a column plays in modelling.

    Roles let downstream feature-selection / biomarker code enumerate features
    vs targets vs survival columns generically.
    """

    FEATURE = "feature"
    TARGET = "target"
    SURVIVAL = "survival"
    IDENTIFIER = "identifier"
    METADATA = "metadata"


# Convenience sets for callers that want quick membership tests.
NUMERIC_TYPES = frozenset({ColumnType.CONTINUOUS, ColumnType.SURVIVAL_TIME})
DISCRETE_TYPES = frozenset(
    {
        ColumnType.BINARY,
        ColumnType.ORDINAL,
        ColumnType.CATEGORICAL,
        ColumnType.SURVIVAL_EVENT,
    }
)
SURVIVAL_TYPES = frozenset({ColumnType.SURVIVAL_TIME, ColumnType.SURVIVAL_EVENT})
