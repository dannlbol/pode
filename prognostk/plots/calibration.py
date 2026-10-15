"""Calibration curve and per-patient survival curves.
Python mirror of the R plot_calibration.R.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import Config
from ..metrics.calibration import calibration_table
from ..metrics.survival import (CoxModel, _as_model, assign_risk_groups,
                                predict_survival_at_time)
from ..utils.theme import apply_base_style, prognostk_palette, save_figure


def plot_calibration(x, cfg: Config, pred: Union[np.ndarray, str, None] = None,
                     n_bins: int = 10, title: Optional[str] = None,
                     filename: Optional[str] = None):
    """Predicted vs observed (KM) survival with the identity line and E/O ratio.

    ``x`` is either a ``calibration_table()`` result dict or a data frame (with
    ``pred``).
    """
    apply_base_style(cfg)
    cal = x if isinstance(x, dict) and "table" in x else calibration_table(x, pred, cfg, n_bins)
    tab = cal["table"]
    colour = list(prognostk_palette(cfg, "cohort").values())[0]

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    yerr = np.vstack([tab["obs_km"] - tab["obs_lower"], tab["obs_upper"] - tab["obs_km"]])
    yerr = np.where(np.isnan(yerr), 0, yerr)
    ax.errorbar(tab["mean_pred"], tab["obs_km"], yerr=yerr, fmt="o-",
                color=colour, ecolor="grey", capsize=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Predicted {cfg.horizon:g}-{cfg['survival']['time_unit']} survival")
    ax.set_ylabel("Observed (KM) survival")
    ttl = (title or "") + (f"\nE/O = {cal['eo_ratio']:.2f}")
    ax.set_title(ttl, loc="left")
    return save_figure(fig, cfg, filename,
                       width=cfg["figure"]["page"]["width_in"] * 0.7,
                       height=cfg["figure"]["page"]["width_in"] * 0.7)


def plot_individual_curves(fit: CoxModel, data: pd.DataFrame, cfg: Config,
                           risk_groups=None, max_time: Optional[float] = None,
                           title: Optional[str] = None, filename: Optional[str] = None):
    """Per-patient predicted survival curves coloured by risk group."""
    apply_base_style(cfg)
    model = _as_model(fit)
    horizon = cfg.horizon
    max_time = max_time or 2 * horizon
    grid = np.linspace(0.01, max_time, 60)

    if risk_groups is None:
        ps = predict_survival_at_time(model, data, time=horizon)
        risk_groups = assign_risk_groups(ps, cfg)
    labels = list(pd.Categorical(risk_groups).categories)
    palette = prognostk_palette(cfg, "risk", labels)

    sf = model.fitter.predict_survival_function(data, times=grid)  # times x subjects
    fig, ax = plt.subplots(constrained_layout=True)
    rg = np.asarray(risk_groups)
    for j in range(sf.shape[1]):
        ax.plot(grid, sf.iloc[:, j].to_numpy(), lw=0.4, alpha=0.4,
                color=palette.get(str(rg[j])))
    ax.axvline(horizon, ls="--", color="grey")
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Time ({cfg['survival']['time_unit']})")
    ax.set_ylabel("Predicted survival")
    handles = [plt.Line2D([0], [0], color=palette.get(l), lw=2) for l in labels]
    ax.legend(handles, labels, frameon=False, ncol=len(labels), loc="upper right")
    if title:
        ax.set_title(title, loc="left")
    return save_figure(fig, cfg, filename,
                       height=cfg["figure"]["page"]["width_in"] * 0.8)
