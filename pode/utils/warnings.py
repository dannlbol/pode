"""Central control of known-benign third-party warnings.

The synthesizer backend (SDV/ctgan) and lifelines emit deprecation and
approximation warnings that are noise for PODE users. ``configure_warnings``
silences only those specific, known-benign messages, leaving PODE's own
warnings (e.g. conditional under-generation) and any unexpected warnings
visible. The CLI calls this at startup; pass ``show=True`` (``--show-warnings``)
to keep everything.
"""

from __future__ import annotations

import warnings

# (category, message-regex) pairs for the dependency warnings we suppress.
# Note: warnings.filterwarnings matches the message regex with re.match (anchored
# at the start) and greedy ``.*``; keep phrases contiguous so a later identical
# token (e.g. "save" inside "save_to_json") cannot defeat the match.
_BENIGN: list[tuple[type[Warning], str]] = [
    (FutureWarning, r".*SingleTableMetadata.*deprecated.*"),
    (UserWarning, r".*recommend saving the metadata.*"),
    (FutureWarning, r".*cuda.*parameter is deprecated.*"),
]


def configure_warnings(show: bool = False) -> None:
    """Install or remove filters for benign dependency warnings.

    Called at ``import pode`` with ``show=False`` (suppress by default).
    The CLI's ``--show-warnings`` flag calls again with ``show=True`` to
    reset the filter stack so dependency warnings become visible.
    """
    if show:
        warnings.resetwarnings()
        return

    for category, message in _BENIGN:
        warnings.filterwarnings("ignore", message=message, category=category)

    try:
        from lifelines.exceptions import ApproximationWarning, ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=ApproximationWarning)
    except Exception:
        pass
