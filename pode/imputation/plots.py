"""Benchmark figures (static PNGs, matching pode.evaluation.plots).

* ``metrics_by_scenario.png`` - small multiples of key metrics, engines × scenarios;
* ``coverage_calibration.png`` - nominal vs empirical interval coverage per engine;
* ``delta_<scenario>.png``     - pooled Cox HR across the MNAR delta grid;
* ``imputed_vs_true_<scenario>.png`` - imputed draws vs hidden truth for the
  most-masked continuous column.

Engines keep one colour each across every figure (fixed categorical order),
with distinct markers as a secondary encoding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Reference categorical palette, fixed order (blue, orange, aqua, yellow,
# magenta, green, violet, red); never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE = "#fcfcfb"
INK = "#1f1f1e"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"

PANELS = [
    ("cont_crps_norm", "CRPS / SD (lower is better)"),
    ("cont_nrmse", "NRMSE (lower is better)"),
    ("disc_log_score", "Log score, discrete (lower is better)"),
    ("cox_median_abs_bias", "Cox median |log-HR bias| (lower is better)"),
]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.title.set_color(INK)


def _top_legend(fig, handles, labels, title: str | None = None) -> None:
    """Legend above all panels (title as legend title), clear of axes titles."""
    leg = fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 1.0),
                     ncol=min(len(labels), 7), fontsize=8, frameon=False, title=title,
                     alignment="left")
    if leg.get_title():
        leg.get_title().set_fontsize(10)


def _engine_colors(engines: list[str]) -> dict[str, tuple[str, str]]:
    if len(engines) > len(SERIES):
        raise ValueError("More engines than categorical slots; split the plot.")
    return {e: (SERIES[i], MARKERS[i]) for i, e in enumerate(engines)}


def write_plots(
    tables: dict[str, pd.DataFrame],
    out_dir: str | Path,
    distributions: dict[str, dict] | None = None,
    engine_order: list[str] | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = tables["summary"]
    if summary.empty:
        return []
    present = set(summary["engine"])
    engines = [e for e in (engine_order or []) if e in present]
    engines += [e for e in dict.fromkeys(summary["engine"]) if e not in engines]
    # The palette has len(SERIES) distinguishable hues; more engines are split
    # across figures (…_part1.png, …_part2.png) rather than recycling colours
    # within one figure.
    chunks = [engines[i : i + len(SERIES)] for i in range(0, len(engines), len(SERIES))]
    paths: list[Path] = []
    for k, chunk in enumerate(chunks, start=1):
        sfx = f"_part{k}" if len(chunks) > 1 else ""
        sub = summary[summary["engine"].isin(chunk)]
        colors = _engine_colors(chunk)
        paths.append(_metrics_by_scenario(sub, chunk, colors, out, sfx))
        cov = _coverage(sub, chunk, colors, out, sfx)
        if cov:
            paths.append(cov)
        paths += _delta(tables.get("delta_sensitivity", pd.DataFrame()), chunk, colors, out, sfx)
        paths += _imputed_vs_true(distributions or {}, chunk, colors, out, sfx)
    return paths


def _metrics_by_scenario(summary, engines, colors, out: Path, sfx: str = "") -> Path:
    import matplotlib.pyplot as plt

    panels = [(k, t) for k, t in PANELS if k in summary.columns and summary[k].notna().any()]
    scenarios = list(dict.fromkeys(summary["scenario"]))
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(7, 1.2 * len(scenarios) * len(engines) / 3), 2.6 * len(panels)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    width = 0.8 / len(engines)
    x = np.arange(len(scenarios))
    for ax, (key, title) in zip(axes[:, 0], panels):
        _style(ax)
        for i, e in enumerate(engines):
            vals = summary[summary["engine"] == e].set_index("scenario")[key].reindex(scenarios)
            ax.bar(x + (i - (len(engines) - 1) / 2) * width, vals.to_numpy(), width * 0.9,
                   color=colors[e][0], label=e, edgecolor=SURFACE, linewidth=1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=30, ha="right")
        ax.set_xlim(-0.5, len(scenarios) - 0.5)
    fig.tight_layout()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _top_legend(fig, handles, labels, "Imputation quality by masking scenario")
    path = out / f"metrics_by_scenario{sfx}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _coverage(summary, engines, colors, out: Path, sfx: str = "") -> Path | None:
    import matplotlib.pyplot as plt

    if "cont_coverage50" not in summary.columns:
        return None
    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1, linestyle="--", label="ideal")
    for e in engines:
        g = summary[summary["engine"] == e]
        y = [g["cont_coverage50"].mean(), g["cont_coverage90"].mean()]
        ax.plot([0.5, 0.9], y, color=colors[e][0], marker=colors[e][1], markersize=8, linewidth=2, label=e)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal interval coverage", color=INK_SECONDARY)
    ax.set_ylabel("Empirical coverage of hidden values", color=INK_SECONDARY)
    ax.set_title("Interval calibration (continuous columns)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    path = out / f"coverage_calibration{sfx}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _delta(delta: pd.DataFrame, engines, colors, out: Path, sfx: str = "", max_terms: int = 6) -> list[Path]:
    import matplotlib.pyplot as plt

    if delta.empty:
        return []
    paths = []
    for scen, g in delta.groupby("scenario", sort=False):
        g = g.groupby(["engine", "term", "delta"], as_index=False)["hr"].mean()
        spread = g.groupby("term")["hr"].agg(lambda s: np.log(s).max() - np.log(s).min())
        terms = list(spread.sort_values(ascending=False).index[:max_terms])
        ncol = min(3, len(terms))
        nrow = int(np.ceil(len(terms) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for ax, term in zip(axes.ravel(), terms):
            _style(ax)
            ax.axhline(1.0, color=INK_SECONDARY, linewidth=1)
            for e in engines:
                t = g[(g["engine"] == e) & (g["term"] == term)].sort_values("delta")
                if not t.empty:
                    ax.plot(t["delta"], t["hr"], color=colors[e][0], marker=colors[e][1], markersize=8, linewidth=2, label=e)
            ax.set_title(term, fontsize=9, loc="left")
            ax.set_xlabel("delta", color=INK_SECONDARY, fontsize=8)
        for ax in axes.ravel()[len(terms):]:
            ax.set_visible(False)
        axes[0, 0].set_ylabel("Pooled HR", color=INK_SECONDARY)
        fig.tight_layout()
        handles, labels = axes[0, 0].get_legend_handles_labels()
        _top_legend(fig, handles, labels, f"MNAR delta sensitivity (pooled HR): {scen}")
        path = out / f"delta_{scen}{sfx}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths


def _imputed_vs_true(distributions: dict[str, dict], engines, colors, out: Path, sfx: str = "") -> list[Path]:
    """``distributions[scenario] = {"column": c, "truth": arr, "draws": {engine: arr}}``."""
    import matplotlib.pyplot as plt

    paths = []
    for scen, d in distributions.items():
        present = [e for e in engines if e in d["draws"]]
        if not present:
            continue
        truth = np.asarray(d["truth"], float)
        allv = np.concatenate([truth, *[np.asarray(d["draws"][e], float) for e in present]])
        bins = np.histogram_bin_edges(allv[np.isfinite(allv)], bins=25)
        ncol = min(4, len(present))
        nrow = int(np.ceil(len(present) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False, sharex=True)
        fig.patch.set_facecolor(SURFACE)
        for ax, e in zip(axes.ravel(), present):
            _style(ax)
            ax.hist(truth, bins=bins, density=True, histtype="stepfilled", color=GRID, label="hidden truth")
            ax.hist(d["draws"][e], bins=bins, density=True, histtype="step", color=colors[e][0], linewidth=2, label=f"{e} draws")
            ax.set_title(e, fontsize=9, loc="left")
            ax.legend(fontsize=7, frameon=False)
        for ax in axes.ravel()[len(present):]:
            ax.set_visible(False)
        fig.suptitle(f"{d['column']}: imputed vs hidden values ({scen})", fontsize=10, x=0.01, ha="left")
        fig.tight_layout()
        path = out / f"imputed_vs_true_{scen}{sfx}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths
