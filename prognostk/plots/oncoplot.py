"""Oncoplot: an annotated heatmap of samples with spec-driven annotation tracks.
Python mirror of the R plot_oncoplot.R using a matplotlib GridSpec (heatmap body
plus stacked point / categorical / line tracks). No hard-coded columns.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from ..config import Config
from ..utils.theme import apply_base_style, out_path, prognostk_palette


def _draw_points(ax, vals, track, data):
    x = np.arange(len(vals))
    colours = "grey"
    marker_mask = None
    if track.get("status_col") and track["status_col"] in data:
        st = data[track["status_col"]].astype(float).to_numpy()
        colours = np.where(st == 1, "firebrick", "grey")
        marker_mask = st == 1
    ax.scatter(x, vals, s=3, c=colours)
    if marker_mask is not None and marker_mask.any():
        ax.scatter(x[marker_mask], np.asarray(vals)[marker_mask], s=10,
                   marker="x", c="firebrick")
    if track.get("baseline") is not None:
        ax.axhline(track["baseline"], ls="--", color="grey", lw=0.6)
    if track.get("ylim"):
        ax.set_ylim(*track["ylim"])
    ax.set_xlim(-0.5, len(vals) - 0.5)
    ax.set_yticks([])


def _draw_line(ax, vals):
    ax.plot(np.arange(len(vals)), vals, color="darkred", lw=1.0)
    ax.set_xlim(-0.5, len(vals) - 0.5)
    ax.set_yticks([])


def _draw_categorical(ax, vals, cfg, column):
    svals = ["" if pd.isna(v) else str(v) for v in vals]
    levels = list(pd.unique(pd.Series([v for v in svals if v != ""])))
    cmap = prognostk_palette(cfg, column, levels)
    idx = {lv: i for i, lv in enumerate(levels)}
    row = np.array([[idx.get(v, -1) for v in svals]])
    colours = [cmap[lv] for lv in levels]
    listed = mpl.colors.ListedColormap([to_rgb(c) for c in colours] or ["white"])
    masked = np.ma.masked_less(row, 0)
    ax.imshow(masked, aspect="auto", cmap=listed,
              extent=[-0.5, len(vals) - 0.5, 0, 1], vmin=0, vmax=max(len(levels) - 1, 1))
    ax.set_xlim(-0.5, len(vals) - 0.5)
    ax.set_xticks([]); ax.set_yticks([])


def plot_oncoplot(data: pd.DataFrame, cfg: Config, matrix: Optional[pd.DataFrame] = None,
                  tracks: Optional[List[dict]] = None, order_by: Optional[str] = None,
                  id_col: Optional[str] = None, title: str = "Oncoplot",
                  filename: Optional[str] = None):
    """Draw an oncoplot: an optional feature heatmap with a stack of annotation
    tracks from ``cfg['oncoplot']['tracks']``. Samples are ordered by ``order_by``
    (default the score column); ``matrix`` columns are matched to ``data`` by id.
    """
    apply_base_style(cfg)
    tracks = tracks if tracks is not None else cfg["oncoplot"]["tracks"]
    if not tracks:
        raise ValueError("plot_oncoplot: no tracks configured.")
    id_col = id_col or cfg["columns"].get("id")
    order_by = order_by or cfg["columns"].get("score")

    d = data.copy()
    if order_by and order_by in d:
        d = d.sort_values(order_by)

    mat = None
    if matrix is not None:
        m = matrix.copy()
        m = m.set_index(m.columns[0])
        if not id_col:
            raise ValueError("plot_oncoplot: matrix supplied but no id column configured.")
        ids = d[id_col].astype(str)
        keep = [i for i in ids if i in m.columns]
        d = d[d[id_col].astype(str).isin(keep)]
        mat = m[d[id_col].astype(str).tolist()].to_numpy()

    n_tracks = len(tracks)
    heights = [1.2 if t["type"] == "points" else (0.9 if t["type"] == "line" else 0.35)
               for t in tracks]
    if mat is not None:
        heights.append(6.0)
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.15)

    axes = []
    for i, tr in enumerate(tracks):
        ax = fig.add_subplot(gs[i, 0])
        vals = d[tr["column"]].to_numpy()
        if tr["type"] == "points":
            _draw_points(ax, vals, tr, d)
        elif tr["type"] == "line":
            _draw_line(ax, vals)
        else:
            _draw_categorical(ax, vals, cfg, tr["column"])
        ax.set_ylabel(tr["name"], rotation=0, ha="right", va="center",
                      fontsize=cfg["figure"]["page"]["base_size"] - 1)
        ax.set_xticks([])
        axes.append(ax)
    axes[0].set_title(title, loc="left")

    if mat is not None:
        axh = fig.add_subplot(gs[-1, 0])
        ramp = prognostk_palette(cfg, "heatmap")
        cmap = LinearSegmentedColormap.from_list("prognostk", ramp)
        axh.imshow(mat, aspect="auto", cmap=cmap,
                   extent=[-0.5, mat.shape[1] - 0.5, 0, mat.shape[0]])
        axh.set_xticks([]); axh.set_yticks([])
        axh.set_ylabel("features", fontsize=cfg["figure"]["page"]["base_size"] - 1)

    if filename is None:
        return fig
    w = cfg["figure"]["page"]["width_in"]
    fig.set_size_inches(w, w * 0.85)
    path = out_path(cfg, filename)
    fig.savefig(path, dpi=cfg["figure"].get("dpi", 300), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path
