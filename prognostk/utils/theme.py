"""Shared figure styling: palette resolution, p-value formatting and output helpers.
Python mirror of the R utils_theme.R.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

from ..config import Config


def format_pvalue(p: float) -> str:
    """Format a p-value like ``"p = 0.032"`` / ``"p < 0.001"``."""
    if p is None or p != p:  # NaN
        return "p = NA"
    if p < 2.2e-16:
        return "p < 2.2e-16"
    if p < 0.001:
        return f"p = {p:.2e}"
    return f"p = {round(p, 3)}"


def _auto_colours(n: int) -> List[str]:
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [mpl.colors.to_hex(cmap(i % cmap.N)) for i in range(n)]


def prognostk_palette(cfg: Config, which: str = "risk",
                      levels: Optional[Sequence[str]] = None):
    """Resolve a palette family from the config, auto-filling unmapped levels."""
    if which == "heatmap":
        return list(cfg["palettes"]["heatmap"])
    pal = cfg["palettes"].get(which)
    if pal is None:
        pal = cfg["palettes"].get("categorical", {}).get(which)
    pal = dict(pal) if pal else {}
    if levels is not None:
        levels = [str(x) for x in levels]
        missing = [lv for lv in levels if lv not in pal]
        if missing:
            auto = _auto_colours(len(missing))
            pal.update(dict(zip(missing, auto)))
        return {lv: pal[lv] for lv in levels}
    return pal


def apply_base_style(cfg: Config) -> None:
    """Apply a minimal, publication-ish rcParams style keyed to the base font size."""
    bs = cfg["figure"]["page"]["base_size"]
    mpl.rcParams.update({
        "font.size": bs,
        "axes.titlesize": bs + 1,
        "axes.titleweight": "bold",
        "axes.labelsize": bs,
        "xtick.labelsize": bs - 1,
        "ytick.labelsize": bs - 1,
        "legend.fontsize": bs - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
    })


def out_path(cfg: Config, filename: str) -> str:
    outdir = cfg["figure"].get("outdir", ".")
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, filename)


def save_figure(fig, cfg: Config, filename: Optional[str],
                width: Optional[float] = None, height: Optional[float] = None):
    """Save a matplotlib figure to the configured outdir/device, or return it."""
    if filename is None:
        return fig
    w = width or cfg["figure"]["page"]["width_in"]
    h = height or cfg["figure"]["page"]["height_in"]
    fig.set_size_inches(w, h)
    path = out_path(cfg, filename)
    fig.savefig(path, dpi=cfg["figure"].get("dpi", 300), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path
