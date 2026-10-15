"""Kaplan-Meier curves by group with an optional number-at-risk table.
Python mirror of the R plot_km.R (built on lifelines).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from ..config import Config
from ..utils.theme import apply_base_style, prognostk_palette, save_figure


def plot_km(data: pd.DataFrame, cfg: Config, group_col: Optional[str] = None,
            time_col: Optional[str] = None, status_col: Optional[str] = None,
            palette=None, time_points: Optional[Sequence[float]] = None,
            title: Optional[str] = None, risk_table: bool = True,
            filename: Optional[str] = None):
    """Stratified KM curves coloured by ``group_col`` with the horizon marked and
    an optional number-at-risk table."""
    apply_base_style(cfg)
    group_col = cfg.col("group", group_col)
    time_col = cfg.col("time", time_col)
    status_col = cfg.col("status", status_col)

    d = data.dropna(subset=[group_col, time_col, status_col]).copy()
    d[status_col] = d[status_col].astype(float)
    labels = cfg["risk_groups"]["labels"]
    present = [g for g in labels if g in set(d[group_col].astype(str))]
    glevels = present if present else sorted(d[group_col].astype(str).unique())
    horizon = cfg.horizon
    if time_points is None:
        time_points = np.linspace(0, np.ceil(2 * horizon), 6)
    if palette is None:
        palette = prognostk_palette(cfg, "risk", glevels)

    if risk_table:
        fig, (ax, axt) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [4, 1]}, constrained_layout=True)
    else:
        fig, ax = plt.subplots(constrained_layout=True)
        axt = None

    at_risk = {}
    for g in glevels:
        m = d[group_col].astype(str) == g
        kmf = KaplanMeierFitter().fit(d.loc[m, time_col], d.loc[m, status_col], label=g)
        kmf.plot_survival_function(ax=ax, ci_show=False, color=palette.get(g), lw=1.6)
        at_risk[g] = [int((d.loc[m, time_col] >= t).sum()) for t in time_points]

    ax.axvline(horizon, ls="--", color="grey", alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Overall survival")
    ax.set_xlabel(f"Time ({cfg['survival']['time_unit']})")
    if title:
        ax.set_title(title, loc="left")
    ax.legend(loc="upper right", frameon=False, ncol=len(glevels))

    if axt is not None:
        axt.set_xlim(ax.get_xlim())
        axt.set_ylim(-0.5, len(glevels) - 0.5)
        for yi, g in enumerate(reversed(glevels)):
            for t, n in zip(time_points, at_risk[g]):
                axt.text(t, yi, str(n), ha="center", va="center",
                         color=palette.get(g), fontsize=cfg["figure"]["page"]["base_size"] - 1)
        axt.set_yticks(range(len(glevels)))
        axt.set_yticklabels(list(reversed(glevels)))
        axt.set_xticks([])
        for s in axt.spines.values():
            s.set_visible(False)
        axt.set_title("Number at risk", loc="left",
                      fontsize=cfg["figure"]["page"]["base_size"])

    return save_figure(fig, cfg, filename,
                       height=cfg["figure"]["page"]["width_in"] * 0.9)
