"""Harrell's concordance index (NumPy, no dependencies).

Higher ``risk`` should correspond to shorter survival. A pair ``(i, j)`` is
*comparable* when the one with the shorter time had an event; it is *concordant*
when the model ranks that subject as higher risk. Ties in risk count as 0.5.
O(n^2) -- fine for evaluation-sized cohorts.
"""
from __future__ import annotations

import numpy as np


def concordance_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(bool)
    risk = np.asarray(risk, dtype=float)
    n = time.shape[0]

    num = 0.0
    den = 0.0
    for i in range(n):
        if not event[i]:
            continue
        # subject i had an event at time[i]; compare against all j with time[j] > time[i]
        later = time > time[i]
        # also compare with tied times where j censored? Standard Harrell excludes
        # ties in time unless one is event/other censored later; keep the common
        # definition: comparable if time[j] > time[i].
        if not np.any(later):
            continue
        rj = risk[later]
        ri = risk[i]
        den += later.sum()
        num += np.sum(ri > rj) + 0.5 * np.sum(ri == rj)
    if den == 0:
        return float("nan")
    return float(num / den)
