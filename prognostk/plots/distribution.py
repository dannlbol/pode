"""Distribution of a score across groups: violin / box / ridge, with an omnibus
statistic. Python mirror of the R plot_distribution.R.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from ..config import Config
from ..utils.theme import apply_base_style, format_pvalue, prognostk_palette, save_figure


def plot_distribution(data: pd.DataFrame, cfg: Config, group_col: Optional[str] = None,
                      score_col: Optional[str] = None, kind: str = "violin",
                      palette=None, add_points: bool = True,
                      title: Optional[str] = None, filename: Optional[str] = None):
    """Plot ``score_col`` grouped by ``group_col`` as violins, boxes or ridges,
    annotated with a t-test (2 groups) or one-way ANOVA (>2)."""
    apply_base_style(cfg)
    score_col = cfg.col("score", score_col)
    if group_col is None:
        cats = cfg["columns"].get("categorical") or []
        if not cats:
            raise ValueError("plot_distribution: no group_col and no columns.categorical.")
        group_col = cats[0]

    d = data.dropna(subset=[group_col, score_col]).copy()
    d[group_col] = d[group_col].astype(str)
    glevels = sorted(d[group_col].unique())
    groups = [d.loc[d[group_col] == g, score_col].to_numpy() for g in glevels]
    if palette is None:
        palette = prognostk_palette(cfg, group_col, glevels)

    # Omnibus statistic
    stat_lbl = ""
    try:
        if len(glevels) == 2:
            t, p = stats.ttest_ind(*groups, equal_var=False)
            stat_lbl = f"t = {t:.2f}, {format_pvalue(p)}"
        elif len(glevels) > 2:
            f, p = stats.f_oneway(*groups)
            stat_lbl = f"F = {f:.2f}, {format_pvalue(p)}"
    except Exception:
        pass

    fig, ax = plt.subplots(constrained_layout=True)
    colours = [palette.get(g) for g in glevels]

    if kind == "ridge":
        for i, (g, vals) in enumerate(zip(glevels, groups)):
            if len(vals) < 2:
                continue
            kde = stats.gaussian_kde(vals)
            xs = np.linspace(min(vals), max(vals), 200)
            ys = kde(xs); ys = ys / ys.max() * 0.9
            ax.fill_between(xs, i, i + ys, color=colours[i], alpha=0.7, lw=0.5,
                            edgecolor="black")
        ax.set_yticks(range(len(glevels)))
        ax.set_yticklabels(glevels)
        ax.set_xlabel(score_col)
    else:
        if kind == "box":
            bp = ax.boxplot(groups, patch_artist=True, showfliers=False,
                            positions=range(len(glevels)))
            for patch, c in zip(bp["boxes"], colours):
                patch.set_facecolor(c); patch.set_alpha(0.6)
        else:  # violin
            vp = ax.violinplot(groups, positions=range(len(glevels)), showmeans=False,
                               showextrema=False)
            for body, c in zip(vp["bodies"], colours):
                body.set_facecolor(c); body.set_alpha(0.6); body.set_edgecolor(c)
        if add_points:
            for i, vals in enumerate(groups):
                jitter = np.random.default_rng(0).normal(0, 0.05, len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals, s=4, alpha=0.4,
                           color=colours[i])
        ax.set_xticks(range(len(glevels)))
        ax.set_xticklabels(glevels)
        ax.set_ylabel(score_col)

    ttl = title or ""
    ax.set_title((ttl + ("\n" if ttl and stat_lbl else "") + stat_lbl), loc="left")
    return save_figure(fig, cfg, filename,
                       height=cfg["figure"]["page"]["width_in"] * 0.8)
