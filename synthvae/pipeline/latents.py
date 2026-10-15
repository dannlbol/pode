"""Assemble and disassemble the table the synthesizer sees.

The synthesizer is trained on a single wide table::

    [ clinical columns | expr_latent_0 ... expr_latent_{k-1} | time | event ]

These helpers build that table (and its column-type metadata), and split a
sampled table back into (clinical, latents, survival) so latents can be decoded
to full expression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_table(clinical: pd.DataFrame, latents: np.ndarray, survival: pd.DataFrame,
                latent_prefix: str = "expr_latent_") -> pd.DataFrame:
    lat_cols = [f"{latent_prefix}{i}" for i in range(latents.shape[1])]
    lat_df = pd.DataFrame(latents, columns=lat_cols, index=clinical.index)
    survival_df = survival.reset_index(drop=True).copy()
    if "event" in survival_df.columns:
        # SDV boolean columns must contain actual booleans, not 0/1 numerics.
        survival_df["event"] = survival_df["event"].astype(bool)
    return pd.concat(
        [clinical.reset_index(drop=True),
         lat_df.reset_index(drop=True),
         survival_df],
        axis=1,
    )


def build_metadata(clinical: pd.DataFrame, n_latent: int, survival: pd.DataFrame,
                   categorical: list[str] | None = None,
                   latent_prefix: str = "expr_latent_") -> dict:
    """Column-type metadata consumed by the synthesizers.

    ``categorical`` lists clinical columns to treat as categorical; everything
    else clinical is numerical. Latents are numerical. ``time`` numerical,
    ``event`` boolean.
    """
    categorical = {str(c).casefold() for c in (categorical or [])}
    cols = {}
    for c in clinical.columns:
        col = clinical[c]
        is_named_categorical = str(c).casefold() in categorical
        is_dtype_categorical = (
            col.dtype == object
            or str(col.dtype).startswith("category")
            or str(col.dtype) == "bool"
        )
        is_low_cardinality_integer = (
            pd.api.types.is_integer_dtype(col)
            and col.nunique(dropna=True) <= 10
        )
        cols[c] = {
            "sdtype": "categorical"
            if (is_named_categorical or is_dtype_categorical or is_low_cardinality_integer)
            else "numerical"
        }
    for i in range(n_latent):
        cols[f"{latent_prefix}{i}"] = {"sdtype": "numerical"}
    cols["time"] = {"sdtype": "numerical"}
    cols["event"] = {"sdtype": "boolean"}
    return {"columns": cols}


def split_table(table: pd.DataFrame, clinical_cols: list[str], n_latent: int,
                latent_prefix: str = "expr_latent_"):
    lat_cols = [f"{latent_prefix}{i}" for i in range(n_latent)]
    clinical = table[clinical_cols].reset_index(drop=True)
    latents = table[lat_cols].to_numpy(dtype=float)
    survival = table[["time", "event"]].reset_index(drop=True)
    # normalise event back to int in case a synthesizer emitted bool/float
    survival["event"] = survival["event"].astype(float).round().astype(int)
    survival["time"] = np.clip(survival["time"].astype(float), 1e-6, None)
    return clinical, latents, survival
