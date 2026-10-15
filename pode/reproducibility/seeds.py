"""Seed management for reproducible runs.

Merges the notebook ``SeedTracker`` class with the ``seed_manager.py`` CLI logic
into one API. A :class:`SeedTracker` tracks per-operation seeds (training,
sampling, ...), can persist them to / restore them from timestamped JSON logs,
and seeds Python, NumPy and (if present) PyTorch RNGs.
"""

from __future__ import annotations

import glob
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np

SEED_LOG_PATTERN = "seed_log_*.json"


def set_all_seeds(seed_value: int) -> None:
    """Seed Python ``random``, NumPy and (optionally) PyTorch."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    try:  # torch is optional for non-generation use
        import torch

        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
    except ImportError:
        pass


def generate_random_seed() -> int:
    """Generate a random seed in the inclusive range [1, 999999]."""
    return random.randint(1, 999999)


def set_synthesizer_seed(synthesizer, seed_value: int) -> None:
    """Set an SDV synthesizer's internal random state, if supported."""
    if hasattr(synthesizer, "_set_random_state"):
        synthesizer._set_random_state(seed_value)


class SeedTracker:
    """Track, persist and restore per-operation seeds.

    Modes:
      * ``'new'``    - always mint fresh seeds.
      * ``'specific'`` - load seeds from ``seed_file``.
      * ``'auto'``   - load the most recent seed log if one exists, else fresh.
    """

    def __init__(
        self,
        mode: str = "new",
        seed_file: str | Path | None = None,
        search_dir: str | Path = ".",
    ):
        self.mode = mode
        self.search_dir = Path(search_dir)
        self.session_start = datetime.now().isoformat()
        self.seeds: dict[str, dict] = {}

        if mode == "specific" and seed_file is not None:
            self.load_seeds(seed_file)
        elif mode == "auto":
            latest = self.latest_seed_file(self.search_dir)
            if latest is not None:
                self.load_seeds(latest)

    # ---- core API ----------------------------------------------------------
    def get_seed(self, operation: str, default: int | None = None) -> int:
        """Return the seed for ``operation``, minting one if absent."""
        if operation not in self.seeds:
            seed_value = default if default is not None else generate_random_seed()
            self.set_seed(operation, seed_value)
        return int(self.seeds[operation]["seed"])

    def set_seed(self, operation: str, seed_value: int, forced_new: bool = False) -> int:
        self.seeds[operation] = {
            "seed": int(seed_value),
            "timestamp": datetime.now().isoformat(),
            "forced_new": forced_new,
        }
        return int(seed_value)

    # ---- persistence -------------------------------------------------------
    def save_seeds(self, output_dir: str | Path = ".", filename: str | None = None) -> Path:
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename = f"seed_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = Path(output_dir) / filename
        payload = {
            "session_start": self.session_start,
            "save_time": datetime.now().isoformat(),
            "mode_used": self.mode,
            "total_operations": len(self.seeds),
            "seeds": self.seeds,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def load_seeds(self, seed_file: str | Path) -> None:
        with open(seed_file, "r") as f:
            data = json.load(f)
        self.seeds = data.get("seeds", {})

    # ---- discovery ---------------------------------------------------------
    @staticmethod
    def list_seed_files(search_dir: str | Path = ".") -> list[str]:
        return sorted(glob.glob(str(Path(search_dir) / SEED_LOG_PATTERN)), reverse=True)

    @classmethod
    def latest_seed_file(cls, search_dir: str | Path = ".") -> str | None:
        files = cls.list_seed_files(search_dir)
        return files[0] if files else None

    def summary(self) -> str:
        lines = [f"mode={self.mode}", f"operations={len(self.seeds)}"]
        for op, info in self.seeds.items():
            lines.append(f"  {op}: {info['seed']}")
        return "\n".join(lines)
