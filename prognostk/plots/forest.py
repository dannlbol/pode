"""Forest plot of Cox hazard ratios. Python mirror of the R plot_forest.R."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import Config
from ..metrics.survival import fit_cox
from ..utils.theme import apply_base_style, prognostk_palette, save_figure


def _tidy(model, label):
    s = model.fitter.summary
    return pd.DataFrame({
        "term": s.index,
        "HR": np.exp(s["coef"].to_numpy()),
        "lower": np.exp(s["coef lower 95%"].to_numpy()),
        "upper": np.exp(s["coef upper 95%"].to_numpy()),
        "p": s["p"].to_numpy(),
        "model": label,
    })


def plot_forest(data_list: Union[pd.DataFrame, Dict[str, pd.DataFrame]], cfg: Config,
                covariates: Optional[Sequence[str]] = None,
                term_labels: Optional[Dict[str, str]] = None,
                palette=None, title: Optional[str] = None,
                filename: Optional[str] = None):
    """Fit a Cox model per named cohort and draw their hazard ratios on a shared
    log-scale forest plot, dodged by cohort."""
    apply_base_style(cfg)
    if isinstance(data_list, pd.DataFrame):
        data_list = {"cohort": data_list}
    cohorts = list(data_list.keys())

    tidy = pd.concat(
        [_tidy(fit_cox(df, cfg, covariates=covariates), name)
         for name, df in data_list.items()], ignore_index=True)
    if term_labels:
        tidy["term"] = tidy["term"].map(lambda t: term_labels.get(t, t))

    terms = list(dict.fromkeys(tidy["term"]))
    if palette is None:
        palette = prognostk_palette(cfg, "cohort", cohorts)

    fig, ax = plt.subplots(constrained_layout=True)
    n = len(cohorts)
    offsets = np.linspace(-0.2, 0.2, n) if n > 1 else [0.0]
    for ci, coh in enumerate(cohorts):
        sub = tidy[tidy["model"] == coh]
        for _, row in sub.iterrows():
            y = terms.index(row["term"]) + offsets[ci]
            ax.plot([row["lower"], row["upper"]], [y, y], color=palette.get(coh), lw=1.4)
            ax.plot(row["HR"], y, "o", color=palette.get(coh), ms=5,
                    label=coh if row["term"] == sub.iloc[0]["term"] else None)
    ax.axvline(1, ls="--", color="grey")
    ax.set_xscale("log")
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms)
    ax.set_xlabel("Hazard ratio (95% CI)")
    if title:
        ax.set_title(title, loc="left")
    if n > 1:
        ax.legend(frameon=False, loc="best")
    return save_figure(fig, cfg, filename,
                       height=cfg["figure"]["page"]["width_in"] * 0.6)
