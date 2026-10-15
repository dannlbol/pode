"""Distribution and correlation diagnostics for synthetic vs real data.

Generalises notebook cells 39 (marginal distributions) and 41 (correlation
heatmaps). Plots are written to an output directory; nothing cohort-specific is
assumed - columns come from the schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.io import ensure_dir
from ..schema.schema import DatasetSchema


def plot_marginals(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    schema: DatasetSchema,
    output_dir: str | Path,
) -> list[Path]:
    """One real-vs-synthetic marginal comparison per column. Returns file paths."""
    import matplotlib.pyplot as plt

    out = ensure_dir(output_dir)
    paths: list[Path] = []
    for col in schema.columns:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        try:
            if schema.types[col].is_numeric:
                ax.hist(
                    pd.to_numeric(real_df[col], errors="coerce").dropna(),
                    bins=30, alpha=0.5, density=True, label="Real",
                )
                ax.hist(
                    pd.to_numeric(synth_df[col], errors="coerce").dropna(),
                    bins=30, alpha=0.5, density=True, label="Synthetic",
                )
            else:
                real_p = real_df[col].value_counts(normalize=True)
                synth_p = synth_df[col].value_counts(normalize=True)
                cats = real_p.index.union(synth_p.index)
                x = np.arange(len(cats))
                ax.bar(x - 0.2, real_p.reindex(cats, fill_value=0), 0.4, label="Real")
                ax.bar(x + 0.2, synth_p.reindex(cats, fill_value=0), 0.4, label="Synthetic")
                ax.set_xticks(x)
                ax.set_xticklabels([str(c) for c in cats], rotation=45, ha="right")
            ax.set_title(col)
            ax.legend()
            path = out / f"marginal_{col}.png"
            fig.savefig(path, bbox_inches="tight")
            paths.append(path)
        except Exception:
            pass
        finally:
            plt.close(fig)
    return paths


def plot_correlation_comparison(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    schema: DatasetSchema,
    output_dir: str | Path,
) -> Path | None:
    """Side-by-side correlation heatmaps of numeric columns; returns the path.

    Also annotates the Frobenius norm of the difference, a scalar summary of how
    well pairwise correlations are preserved.
    """
    import matplotlib.pyplot as plt

    num_cols = [
        c for c in schema.numerical_columns if c in real_df.columns and c in synth_df.columns
    ]
    if len(num_cols) < 2:
        return None

    real_corr = real_df[num_cols].apply(pd.to_numeric, errors="coerce").corr()
    synth_corr = synth_df[num_cols].apply(pd.to_numeric, errors="coerce").corr()
    frob = float(np.linalg.norm((real_corr - synth_corr).fillna(0).values))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, corr, title in (
        (axes[0], real_corr, "Real"),
        (axes[1], synth_corr, "Synthetic"),
    ):
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Correlation comparison (Frobenius diff = {frob:.3f})")

    out = ensure_dir(output_dir)
    path = out / "correlation_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
