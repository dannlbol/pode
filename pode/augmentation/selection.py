"""Feature-selection stability for models that choose their own features
(``metacpg_search``).

Augmentation can change *which* CpGs a signature uses even when the C-index
barely moves, so the selected sets are summarised directly:

* ``jaccard_vs_real_only`` - overlap (|A∩B| / |A∪B|) between an augmented fit's
  selection and the real-only selection **in the same split**: how far
  augmentation moves the signature (1 = unchanged);
* ``reference_overlap`` - share of a reference signature (e.g. a published
  signature's CpGs) that the fit selects;
* ``across_split_jaccard`` - mean pairwise overlap of one arm's selections
  across the internal splits: how *stable* the signature is to resampling.
  Higher than real-only means augmentation stabilised selection.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARM_KEYS = ["model", "arm", "strategy", "ratio"]


def load_reference(ref) -> list[str]:
    """A feature list, or the signature of a MetaCPG model.json."""
    if not ref:
        return []
    if isinstance(ref, (list, tuple)):
        return [str(f) for f in ref]
    blob = json.loads(Path(str(ref)).expanduser().read_text())
    return [s["feature"] for s in blob["signature"]]


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else np.nan


def selection_stability(results: pd.DataFrame, reference: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary per arm, selection frequency per arm and feature)."""
    if results.empty or "selected_features" not in results:
        return pd.DataFrame(), pd.DataFrame()
    r = results[results["selected_features"].notna() & (results["evaluation"] == "internal")
                & results["arm"].isin(["real_only", "augmented"])].copy()
    if r.empty:
        return pd.DataFrame(), pd.DataFrame()
    r["sel"] = r["selected_features"].map(lambda s: frozenset(str(s).split(";")))
    ref = set(reference)
    base = r[r["arm"] == "real_only"].set_index(["split", "model"])["sel"]
    r["jaccard_vs_real_only"] = [_jaccard(s, base.get((sp, m), frozenset())) for s, sp, m in
                                 zip(r["sel"], r["split"], r["model"])]
    r["reference_overlap"] = [len(s & ref) / len(ref) if ref else np.nan for s in r["sel"]]

    rows, freq = [], []
    for key, g in r.groupby(ARM_KEYS):
        first = g.sort_values("draw").groupby("split")["sel"].first()
        pairs = [_jaccard(a, b) for a, b in itertools.combinations(first.tolist(), 2)]
        rows.append({**dict(zip(ARM_KEYS, key)), "fits": len(g),
                     "jaccard_vs_real_only": float(g["jaccard_vs_real_only"].mean()),
                     "reference_overlap": float(g["reference_overlap"].mean()),
                     "across_split_jaccard": float(np.mean(pairs)) if pairs else np.nan})
        counts = pd.Series([f for s in g["sel"] for f in s]).value_counts()
        freq += [{**dict(zip(ARM_KEYS, key)), "feature": f, "frequency": c / len(g),
                  "in_reference": f in ref} for f, c in counts.items()]
    summary = pd.DataFrame(rows).sort_values(["model", "arm", "strategy", "ratio"]).reset_index(drop=True)
    return summary, pd.DataFrame(freq)
