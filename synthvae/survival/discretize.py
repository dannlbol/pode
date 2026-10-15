"""Time discretization for discrete-time (logistic-hazard) survival models.

We map continuous follow-up times onto ``K`` contiguous intervals. Cut points are
placed at quantiles of the *event* times by default (equal number of events per
bin), which is the usual, well-behaved choice for logistic-hazard / nnet-survival
style models.

The only state that must persist from train to synthesis time is the array of cut
points, so this object is trivially serialisable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class TimeGrid:
    """Discretization grid derived from training data.

    Attributes
    ----------
    cuts : np.ndarray
        Strictly increasing interior cut points, shape ``(K-1,)``. Interval ``k``
        (``0 <= k < K``) covers ``(-inf, cuts[0]]`` for ``k == 0`` ... and
        ``(cuts[K-2], +inf)`` for ``k == K-1``. In practice all training times are
        finite and fall inside the outermost finite edges.
    n_bins : int
        Number of intervals ``K``.
    """

    cuts: np.ndarray
    n_bins: int = field(init=False)

    def __post_init__(self) -> None:
        self.cuts = np.asarray(self.cuts, dtype=float)
        if self.cuts.ndim != 1:
            raise ValueError("cuts must be 1-D")
        if np.any(np.diff(self.cuts) <= 0):
            raise ValueError("cuts must be strictly increasing")
        self.n_bins = int(self.cuts.shape[0] + 1)

    @classmethod
    def from_times(
        cls,
        time: np.ndarray,
        event: np.ndarray,
        n_bins: int = 20,
        scheme: str = "quantile",
    ) -> "TimeGrid":
        """Build a grid from observed follow-up.

        Parameters
        ----------
        time, event : np.ndarray
            Follow-up times and 1/0 event indicators.
        n_bins : int
            Target number of intervals ``K``.
        scheme : {"quantile", "equidistant"}
            ``quantile`` places interior cuts at equally spaced quantiles of the
            *event* times (roughly equal events per bin); ``equidistant`` spaces
            them uniformly across the observed time range.
        """
        time = np.asarray(time, dtype=float)
        event = np.asarray(event).astype(bool)
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")

        if scheme == "quantile":
            ev_times = time[event]
            if ev_times.size < n_bins:
                # Fall back to all-times quantiles if events are scarce.
                ev_times = time
            qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
            cuts = np.quantile(ev_times, qs)
        elif scheme == "equidistant":
            lo, hi = float(time.min()), float(time.max())
            cuts = np.linspace(lo, hi, n_bins + 1)[1:-1]
        else:
            raise ValueError(f"unknown scheme {scheme!r}")

        # De-duplicate cut points (heavy ties in event times can collapse bins).
        cuts = np.unique(cuts)
        return cls(cuts=cuts)

    def index(self, time: np.ndarray) -> np.ndarray:
        """Return the interval index ``j`` in ``[0, K-1]`` for each time."""
        time = np.asarray(time, dtype=float)
        # np.searchsorted with 'right' gives the count of cuts strictly below t.
        return np.searchsorted(self.cuts, time, side="left").astype(int)

    def to_dict(self) -> dict:
        return {"cuts": self.cuts.tolist(), "n_bins": self.n_bins}

    @classmethod
    def from_dict(cls, d: dict) -> "TimeGrid":
        return cls(cuts=np.asarray(d["cuts"], dtype=float))
