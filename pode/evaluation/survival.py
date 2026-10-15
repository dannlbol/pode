"""Survival-analysis visual diagnostics.

Ported from the notebook ``plot_kaplan_meier_robust`` (cell 43) and the
univariate Cox association table (cell 47), generalised to arbitrary
(time, event) columns via the schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..schema.schema import DatasetSchema


def plot_kaplan_meier(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    time_col: str,
    event_col: str,
    title: str = "Kaplan-Meier: real vs synthetic",
    output_path: str | Path | None = None,
):
    """Overlay real vs synthetic KM curves. Returns the matplotlib Figure or None."""
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    needed = [time_col, event_col]
    if not all(c in real_df.columns for c in needed) or not all(
        c in synthetic_df.columns for c in needed
    ):
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for df, label in ((real_df, "Real"), (synthetic_df, "Synthetic")):
        sub = df[needed].dropna()
        if sub.empty:
            continue
        try:
            kmf = KaplanMeierFitter()
            kmf.fit(sub[time_col], sub[event_col].astype(int))
            kmf.plot_survival_function(
                ax=ax,
                label=f"{label} (n={len(sub)}, median={kmf.median_survival_time_:.2f})",
            )
        except Exception:
            continue
    ax.set_title(title)
    ax.set_xlabel(time_col)
    ax.set_ylabel("Survival probability")
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def univariate_cox_table(
    df: pd.DataFrame,
    schema: DatasetSchema,
    time_col: str | None = None,
    event_col: str | None = None,
    min_rows: int = 15,
) -> pd.DataFrame:
    """Univariate Cox per feature column. Returns a tidy results table."""
    import warnings

    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError

    pair = schema.primary_survival_pair()
    time_col = time_col or (pair.time_col if pair else None)
    event_col = event_col or (pair.event_col if pair else None)
    if time_col is None or event_col is None:
        return pd.DataFrame()

    rows = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for cov in schema.feature_columns:
            if cov not in df.columns or df[cov].nunique(dropna=True) < 2:
                continue
            formula = f"C(`{cov}`)" if cov in schema.categorical_columns else f"`{cov}`"
            sub = df[[cov, time_col, event_col]].dropna()
            if len(sub) < min_rows:
                continue
            try:
                cph = CoxPHFitter()
                cph.fit(sub, time_col, event_col, formula=formula)
                s = cph.summary
                idx = s.index[s.index.str.contains(cov, regex=False)]
                if len(idx) == 0:
                    continue
                row = s.loc[idx[0]]
                rows.append(
                    {
                        "covariate": cov,
                        "hr": float(row["exp(coef)"]),
                        "coef": float(row["coef"]),
                        "p": float(row["p"]),
                    }
                )
            except (ConvergenceError, ValueError, np.linalg.LinAlgError, KeyError):
                continue
    return pd.DataFrame(rows)
