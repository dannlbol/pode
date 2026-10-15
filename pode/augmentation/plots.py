"""Benchmark figures (static PNGs, styled like :mod:`pode.imputation.plots`).

* ``dose_response.png``    - paired ΔC (95% CI) against augmentation ratio, one
  panel per model, one series per strategy;
* ``learning_curve.png``   - real-only C-index against real training size, with
  each augmented arm placed at its real-data equivalent;
* ``gain_vs_calibration.png`` - ΔC against Δ calibration slope per configuration
  (the utility/calibration trade-off), coloured by strategy;
* ``fidelity_privacy.png`` - discriminator and membership-inference AUC per TVAE
  strategy and split, with the scorecard gates;
* ``subgroups.png``        - mean subgroup ΔC for the best configurations.

Strategies keep one colour each across every figure (fixed order, never cycled).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..imputation.plots import GRID, INK_SECONDARY, MARKERS, SERIES, _style, _top_legend
from .config import AugmentationConfig


def _colors(strategies: list[str]) -> dict[str, tuple[str, str]]:
    # More strategies than slots fold into grey rather than generating new hues.
    return {s: (SERIES[i], MARKERS[i]) if i < len(SERIES) else ("#8a8984", "o") for i, s in enumerate(strategies)}


def _zero(ax, y: float = 0.0, label: str | None = None) -> None:
    ax.axhline(y, color=INK_SECONDARY, linewidth=0.8, linestyle="--", zorder=1)
    if label:
        ax.annotate(label, (0.0, y), xycoords=("axes fraction", "data"), xytext=(2, 3),
                    textcoords="offset points", ha="left", fontsize=7, color=INK_SECONDARY)


def write_plots(tables: dict[str, pd.DataFrame], out_dir: Path, cfg: AugmentationConfig) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pm = cfg.gates.primary_metric
    strategies = [s.key for s in cfg.strategies]
    colors = _colors(strategies)
    written: list[Path] = []
    paired = tables.get("paired_differences", pd.DataFrame())

    def save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # ---- dose-response -----------------------------------------------------------
    if not paired.empty:
        p = paired[paired["metric"] == pm]
        models = list(dict.fromkeys(p["model"]))
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), squeeze=False, sharey=True)
        handles = {}
        for ax, model in zip(axes[0], models):
            _style(ax)
            _zero(ax, label="real only")
            g = p[p["model"] == model]
            for j, s in enumerate(strategies):
                h = g[g["strategy"] == s].sort_values("ratio")
                if h.empty:
                    continue
                c, mk = colors[s]
                x = h["ratio"].to_numpy() * (1 + 0.04 * (j - len(strategies) / 2))  # dodge
                ax.errorbar(x, h["delta"], yerr=[h["delta"] - h["ci_low"], h["ci_high"] - h["delta"]],
                            color=c, marker=mk, markersize=5, linewidth=2, capsize=0, elinewidth=1)
                handles[s] = ax.lines[-1]
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(p["ratio"].unique()))
            ax.set_xticklabels([f"{r:g}×" for r in sorted(p["ratio"].unique())])
            ax.set_title(model, fontsize=10, loc="left")
            ax.set_xlabel("synthetic rows / real training rows", fontsize=8, color=INK_SECONDARY)
        axes[0][0].set_ylabel(f"Δ {pm} vs real only (95% CI)", fontsize=8, color=INK_SECONDARY)
        _top_legend(fig, list(handles.values()), list(handles), "strategy")
        save(fig, "dose_response.png")

    # ---- learning curve + real-data equivalents ----------------------------------
    lc, eff = tables.get("learning_curve", pd.DataFrame()), tables.get("efficiency", pd.DataFrame())
    if not lc.empty:
        models = list(dict.fromkeys(lc["model"]))
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), squeeze=False)
        handles = {}
        for ax, model in zip(axes[0], models):
            _style(ax)
            g = lc[lc["model"] == model].sort_values("n_real")
            line = ax.plot(g["n_real"], g["value"], color=INK_SECONDARY, marker="o", markersize=5,
                           linewidth=2, label="real only")[0]
            handles["real only (learning curve)"] = line
            if not eff.empty and "extrapolation_x" in eff:
                # Equivalents far outside the observed curve are extrapolation artefacts: not drawn.
                e = eff[(eff["model"] == model) & (eff["metric"] == pm) & np.isfinite(eff["n_equiv"])
                        & eff["extrapolation_x"].between(0.25, 2.5)]
                real_c = g["value"].iloc[-1]
                pd_ = paired[(paired["model"] == model) & (paired["metric"] == pm)].set_index(["strategy", "ratio"])
                for s in strategies:
                    h = e[e["strategy"] == s]
                    if h.empty:
                        continue
                    c, mk = colors[s]
                    y = [real_c + pd_.loc[(s, r), "delta"] for r in h["ratio"]]
                    handles[s] = ax.scatter(h["n_equiv"], y, color=c, marker=mk, s=40, zorder=3,
                                            edgecolors="white", linewidths=1.5)
            slopes = tables.get("learning_curve_slopes", pd.DataFrame())
            flat = slopes[(slopes.get("model") == model) & (slopes.get("metric") == pm)] if not slopes.empty else slopes
            if not flat.empty and not bool(flat["improves"].iloc[0]):
                ax.annotate("flat learning curve:\nno real-data equivalent", (0.03, 0.95), xycoords="axes fraction",
                            va="top", fontsize=8, color=INK_SECONDARY)
            ax.set_title(model, fontsize=10, loc="left")
            ax.set_xlabel("real training patients (equivalent; shown within 0.25-2.5x)", fontsize=8,
                          color=INK_SECONDARY)
        axes[0][0].set_ylabel(pm, fontsize=8, color=INK_SECONDARY)
        _top_legend(fig, list(handles.values()), list(handles), None)
        save(fig, "learning_curve.png")

    # ---- utility vs calibration trade-off ----------------------------------------
    card = tables.get("scorecard", pd.DataFrame())
    if not card.empty and "cal_slope_err:delta" in card:
        models = list(dict.fromkeys(card["model"]))
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), squeeze=False, sharey=True)
        handles = {}
        for ax, model in zip(axes[0], models):
            _style(ax)
            ax.axvline(0, color=GRID, linewidth=0.8)
            _zero(ax)
            g = card[card["model"] == model]
            for s in strategies:
                h = g[g["strategy"] == s]
                if h.empty:
                    continue
                c, mk = colors[s]
                handles[s] = ax.scatter(h["cal_slope_err:delta"], h[f"{pm}:delta"], color=c, marker=mk,
                                        s=18 + 30 * h["ratio"], edgecolors="white", linewidths=1.5, zorder=3)
            ax.set_title(model, fontsize=10, loc="left")
            # Shade what passes the gate: calibration improved, unchanged, or worsened within the limit.
            lo, hi = ax.get_xlim()
            gate = cfg.gates.max_abs_delta_slope
            ax.axvspan(lo, gate, color=GRID, alpha=0.35, linewidth=0, zorder=0)
            ax.set_xlim(lo, max(hi, gate * 1.1))
            ax.set_xlabel("Δ |calibration slope − 1|  (< 0: better calibrated; shaded: passes gate)", fontsize=8,
                          color=INK_SECONDARY)
        axes[0][0].set_ylabel(f"Δ {pm}", fontsize=8, color=INK_SECONDARY)
        _top_legend(fig, list(handles.values()), list(handles), "strategy (marker size = ratio)")
        save(fig, "gain_vs_calibration.png")

    # ---- fidelity & privacy per TVAE strategy ------------------------------------
    fid, priv = tables.get("fidelity", pd.DataFrame()), tables.get("privacy", pd.DataFrame())
    panels = [(fid, "discriminator_auc", "Discriminator AUC (real vs synthetic)", cfg.gates.max_discriminator_auc),
              (priv, "membership_auc", "Membership-inference AUC", cfg.gates.max_membership_auc)]
    panels = [pp for pp in panels if not pp[0].empty and pp[1] in pp[0]]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.2), squeeze=False)
        for ax, (t, col, title, gate) in zip(axes[0], panels):
            _style(ax)
            _zero(ax, 0.5, "chance")
            ax.axhline(gate, color="#e34948", linewidth=1, zorder=1)
            ax.annotate("gate", (1.0, gate), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                        textcoords="offset points", ha="right", fontsize=7, color=INK_SECONDARY)
            t = t[t["evaluation"] == "internal"]
            ss = [s for s in strategies if s in set(t["strategy"])]
            for i, s in enumerate(ss):
                v = t.loc[t["strategy"] == s, col].dropna().to_numpy()
                c, mk = colors[s]
                jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(v))
                ax.scatter(i + jitter, v, color=c, marker=mk, s=22, edgecolors="white", linewidths=1, zorder=3)
            ax.set_xticks(range(len(ss)))
            ax.set_xticklabels(ss, fontsize=8, rotation=20, ha="right")
            ax.set_ylim(0.35, 1.02)
            ax.set_title(title, fontsize=10, loc="left")
        save(fig, "fidelity_privacy.png")

    # ---- subgroup audit -----------------------------------------------------------
    sub = tables.get("subgroups", pd.DataFrame())
    if not sub.empty and not card.empty:
        internal = sub[sub["evaluation"] == "internal"]
        keys = ["split", "model", "subgroup", "level"]
        base = internal[internal["arm"] == "real_only"].groupby(keys)["c_harrell"].mean().rename("base")
        aug = (internal[internal["arm"] == "augmented"].groupby([*keys, "strategy", "ratio"])["c_harrell"]
               .mean().reset_index().join(base, on=keys))
        aug["d"] = aug["c_harrell"] - aug["base"]
        per = aug.groupby(["model", "strategy", "ratio", "subgroup", "level"])["d"].mean().reset_index()
        top = card.groupby("model").head(3)[["model", "strategy", "ratio"]]
        per = per.merge(top, on=["model", "strategy", "ratio"])
        if not per.empty:
            per["label"] = per["subgroup"] + "=" + per["level"].astype(str)
            labels = list(dict.fromkeys(per["label"]))
            models = list(dict.fromkeys(per["model"]))
            fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 0.35 * len(labels) + 1.6),
                                     squeeze=False, sharey=True)
            handles = {}
            for ax, model in zip(axes[0], models):
                _style(ax)
                ax.grid(axis="x", color=GRID, linewidth=0.8)
                ax.axvline(0, color=INK_SECONDARY, linewidth=0.8, linestyle="--")
                ax.axvline(cfg.gates.min_subgroup_delta_c, color="#e34948", linewidth=1)
                g = per[per["model"] == model]
                ratios = sorted(per["ratio"].unique())
                for j, ((s, r), h) in enumerate(g.groupby(["strategy", "ratio"], sort=False)):
                    # Colour = strategy, marker = ratio, so two ratios of one strategy stay distinct.
                    c = colors.get(s, ("#8a8984", "o"))[0]
                    mk = MARKERS[ratios.index(r) % len(MARKERS)]
                    y = [labels.index(lb) + 0.18 * (j - 1) for lb in h["label"]]
                    handles[f"{s} {r:g}×"] = ax.scatter(h["d"], y, color=c, marker=mk, s=26, zorder=3,
                                                        edgecolors="white", linewidths=1)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_title(model, fontsize=10, loc="left")
                ax.set_xlabel("subgroup Δ C-index (red: gate)", fontsize=8, color=INK_SECONDARY)
            _top_legend(fig, list(handles.values()), list(handles), "top configurations")
            save(fig, "subgroups.png")
    return written
