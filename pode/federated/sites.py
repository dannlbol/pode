"""Sites: the data-holding parties of a federation.

A :class:`Site` keeps its patient rows private and exposes one door,
:meth:`Site.run`, which executes a *site operation* (a plain function whose
first argument is the site's data frame) and returns only that operation's
result, logging it in the :class:`~pode.federated.ledger.DisclosureLedger`.

Coordinator code never reads ``site._data``. This is the transport seam: in
this in-process simulation ``run`` is a function call; in a real multi-host
deployment the same operations would execute inside a site client (e.g. a
Flower ``NumPyClient``) and ``run`` would be a network round-trip, without
changing any coordinator code.

Site operations are the functions named ``site_*`` in this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .ledger import DisclosureLedger


def stateful(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a site operation that also receives the site's private ``local`` state."""
    fn.stateful = True
    return fn


class Site:
    def __init__(self, name: str, data: pd.DataFrame, ledger: DisclosureLedger | None = None):
        self.name = str(name)
        # Tabular sites hold a data frame; signature sites hold matrix + phenotype.
        self._data = data.reset_index(drop=True) if isinstance(data, pd.DataFrame) else data
        self.ledger = ledger
        # Site-private persistent state (encoded tensors, optimiser state,
        # personalised models). Like ``_data``, it never leaves the site.
        self.local: dict[str, Any] = {}

    @property
    def n(self) -> int:
        """Number of patients (a count every federation discloses)."""
        return len(self._data)

    def run(self, op: Callable[..., Any], *args, per_patient: bool = False, label: str | None = None,
            **kwargs) -> Any:
        """Execute ``op`` at the site; log and return its result.

        ``op(site_data, *args, **kwargs)`` for a plain operation, or
        ``op(site_data, site_local_state, *args, **kwargs)`` for one decorated
        with :func:`stateful`.
        """
        if getattr(op, "stateful", False):
            out = op(self._data, self.local, *args, **kwargs)
        else:
            out = op(self._data, *args, **kwargs)
        if self.ledger is not None:
            self.ledger.record(self.name, label or op.__name__, out, per_patient)
        return out

    def subset(self, rows: np.ndarray, suffix: str = "") -> "Site":
        """A site holding only ``rows`` (e.g. its training share). Runs at the site."""
        return Site(self.name + suffix, self._data.iloc[np.asarray(rows)], self.ledger)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Site({self.name!r}, n={self.n})"


def partition_sites(
    df: pd.DataFrame,
    column: str,
    include: list[str] | None = None,
    min_site_n: int = 1,
    ledger: DisclosureLedger | None = None,
    drop_column: bool = True,
) -> list[Site]:
    """Split one table into sites by ``column`` (simulation mode)."""
    if column not in df.columns:
        raise ValueError(f"Site column '{column}' is not in the data. Columns: {list(df.columns)[:20]}")
    labels = df[column].astype(str)
    levels = [str(x) for x in include] if include else sorted(labels.dropna().unique().tolist())
    missing = [s for s in levels if s not in set(labels)]
    if missing:
        raise ValueError(f"sites.include names sites not present in '{column}': {missing}")
    sites = []
    for level in levels:
        rows = df[labels == level]
        if drop_column:
            rows = rows.drop(columns=[column])
        if len(rows) < min_site_n:
            raise ValueError(f"Site '{level}' has {len(rows)} patients, below sites.min_site_n={min_site_n}.")
        sites.append(Site(level, rows, ledger))
    if len(sites) < 1:
        raise ValueError("No sites found.")
    return sites


def load_site_manifest(path: str | Path, ledger: DisclosureLedger | None = None) -> list[Site]:
    """Sites from a YAML manifest: ``sites: [{name: A, data: a.csv}, ...]``.

    The future multi-host layout: each entry would become a remote client; here
    each CSV is loaded locally.
    """
    import yaml

    from ..data.io import load_dataset

    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    base = Path(path).parent
    sites = []
    for entry in spec.get("sites", []):
        p = Path(entry["data"]).expanduser()
        if not p.is_absolute():
            p = base / p
        sites.append(Site(entry["name"], load_dataset(p), ledger))
    if not sites:
        raise ValueError(f"Site manifest {path} lists no sites.")
    return sites


def stack_sites(sites: list[Site], column: str | None = None) -> pd.DataFrame:
    """Pool site rows. **Only** for centralised baselines and evaluation, never inside a protocol."""
    parts = []
    for s in sites:
        d = s._data
        parts.append(d.assign(**{column: s.name}) if column else d)
    return pd.concat(parts, ignore_index=True)
