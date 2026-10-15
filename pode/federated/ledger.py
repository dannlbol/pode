"""A record of everything that crosses a site boundary.

Federated learning moves *something* out of every site: model weights, score
vectors, histograms, summary statistics, sometimes per-patient values. The
privacy argument is only as good as an honest list of those things, so every
site operation goes through :meth:`pode.federated.sites.Site.run`, which logs
here what was returned: the operation, the site, the payload's shape and size,
and whether it is **per-patient** (one value per patient, e.g. held-out risk
scores) rather than an aggregate. Reports summarise the ledger so a reader can
see at a glance what each protocol disclosed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def payload_size(x: Any) -> tuple[int, str]:
    """(bytes, short shape description) of a returned payload."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if x is None:
        return 0, "none"
    if torch is not None and isinstance(x, torch.Tensor):
        return int(x.numel() * x.element_size()), f"tensor{tuple(x.shape)}"
    if isinstance(x, np.ndarray):
        return int(x.nbytes), f"array{x.shape}"
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return int(x.memory_usage(deep=True).sum() if isinstance(x, pd.DataFrame) else x.memory_usage(deep=True)), \
            f"frame{x.shape}"
    if isinstance(x, dict):
        total = sum(payload_size(v)[0] for v in x.values())
        return int(total), f"dict[{len(x)}]"
    if isinstance(x, (list, tuple)):
        total = sum(payload_size(v)[0] for v in x)
        return int(total), f"list[{len(x)}]"
    if isinstance(x, str):
        return len(x.encode()), "str"
    return 8, type(x).__name__


@dataclass
class DisclosureLedger:
    rows: list[dict] = field(default_factory=list)
    round: int = 0

    def record(self, site: str, op: str, payload: Any, per_patient: bool = False, note: str = "") -> None:
        nbytes, shape = payload_size(payload)
        self.rows.append({"round": self.round, "site": site, "op": op, "bytes": nbytes, "shape": shape,
                          "per_patient": bool(per_patient), "note": note})

    def next_round(self) -> None:
        self.round += 1

    def extend(self, other: "DisclosureLedger", prefix: str = "") -> None:
        for r in other.rows:
            self.rows.append({**r, "op": f"{prefix}{r['op']}"})

    @property
    def frame(self) -> pd.DataFrame:
        cols = ["round", "site", "op", "bytes", "shape", "per_patient", "note"]
        return pd.DataFrame(self.rows, columns=cols)

    @property
    def total_bytes(self) -> int:
        return int(sum(r["bytes"] for r in self.rows))

    def summary(self) -> pd.DataFrame:
        """One row per operation: calls, bytes, sites and whether it is per-patient."""
        f = self.frame
        if f.empty:
            return pd.DataFrame(columns=["op", "calls", "bytes", "sites", "per_patient"])
        return (f.groupby("op", sort=False)
                .agg(calls=("op", "size"), bytes=("bytes", "sum"), sites=("site", "nunique"),
                     per_patient=("per_patient", "any"))
                .reset_index())

    def save(self, path) -> None:
        self.frame.to_csv(path, index=False)
