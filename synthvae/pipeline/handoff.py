"""File-based latent handoff utilities.

These helpers make the high-dimensional matrix compressor usable independently
of the bundled synthesizer: export real matrix latents for an external TVAE,
then later decode externally sampled latents back into matrix space.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.preprocessing import ExpressionPreprocessor


LATENT_PREFIX = "expr_latent_"
METHYLATION_LATENT_PREFIX = "methylation_latent_"

_LATENT_PREFIXES = {
    "expression": LATENT_PREFIX,
    "methylation": METHYLATION_LATENT_PREFIX,
    "copy_number": "cnv_latent_",
    "copy_number_discrete": "cnv_call_latent_",
}


def latent_prefix_for_matrix_kind(matrix_kind: str) -> str:
    try:
        return _LATENT_PREFIXES[matrix_kind]
    except KeyError:
        raise ValueError(
            "matrix_kind must be one of " + ", ".join(sorted(_LATENT_PREFIXES))
        )


def input_matrix_name(matrix_kind: str) -> str:
    return f"{matrix_kind}_input.csv"


def decoded_matrix_name(matrix_kind: str) -> str:
    return f"synthetic_{matrix_kind}.csv"


@dataclass
class HandoffBundle:
    out_dir: Path
    latents_path: Path
    checkpoint_path: Path
    metadata_path: Path


def latent_frame(latents: np.ndarray, prefix: str = LATENT_PREFIX) -> pd.DataFrame:
    cols = [f"{prefix}{i}" for i in range(latents.shape[1])]
    return pd.DataFrame(np.asarray(latents, dtype=float), columns=cols)


def checkpoint_name(compressor_kind: str) -> str:
    return "compressor.pt" if compressor_kind == "survival_vae" else "compressor.pkl"


def write_metadata(
    path: str | Path,
    *,
    compressor_kind: str,
    checkpoint_path: str | Path,
    preprocessor: ExpressionPreprocessor,
    expression_columns: list[str] | None = None,
    matrix_columns: list[str] | None = None,
    matrix_kind: str = "expression",
    latent_prefix: str | None = None,
    config: dict | None = None,
    metrics: dict | None = None,
) -> None:
    columns = list(matrix_columns if matrix_columns is not None else expression_columns)
    latent_prefix = latent_prefix or latent_prefix_for_matrix_kind(matrix_kind)
    selected = [int(i) for i in preprocessor.selected_]
    payload = {
        "matrix_kind": matrix_kind,
        "compressor_kind": compressor_kind,
        "checkpoint_path": str(Path(checkpoint_path).name),
        "latent_prefix": latent_prefix,
        "n_latent": int((config or {}).get("n_latent", 0)),
        "preprocessor": {
            "n_top_genes": preprocessor.n_top_genes,
            "log1p": preprocessor.log1p,
            "matrix_kind": preprocessor.matrix_kind,
            "beta_epsilon": preprocessor.beta_epsilon,
            "selected_indices": selected,
            "selected_columns": [columns[i] for i in selected],
            "fill_values": [float(v) for v in preprocessor.fill_values_],
        },
        "matrix_columns": columns,
        # Backward-compatible key for existing expression bundles/readers.
        "expression_columns": columns,
        "config": config or {},
        "metrics": metrics or {},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def read_metadata(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_handoff_bundle(
    *,
    compressor,
    compressor_kind: str,
    preprocessor: ExpressionPreprocessor,
    out_dir: str | Path,
    expression: pd.DataFrame | None = None,
    matrix: pd.DataFrame | None = None,
    matrix_kind: str = "expression",
    latent_prefix: str | None = None,
    config: dict | None = None,
    metrics: dict | None = None,
    include_input: bool = False,
) -> HandoffBundle:
    if matrix is None:
        if expression is None:
            raise ValueError("save_handoff_bundle requires `matrix` or `expression`")
        matrix = expression
    latent_prefix = latent_prefix or latent_prefix_for_matrix_kind(matrix_kind)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X = preprocessor.transform(matrix.to_numpy(dtype=float))
    latents = compressor.transform(X)
    latents_path = out / "latents_full.csv"
    latent_frame(latents, prefix=latent_prefix).to_csv(latents_path, index=False)

    if include_input:
        matrix.reset_index(drop=True).to_csv(out / input_matrix_name(matrix_kind), index=False)

    checkpoint_path = out / checkpoint_name(compressor_kind)
    compressor.save(checkpoint_path)

    metadata_path = out / "handoff_metadata.json"
    final_config = dict(config or {})
    final_config["n_latent"] = int(latents.shape[1])
    final_config["matrix_kind"] = matrix_kind
    final_config["latent_prefix"] = latent_prefix
    write_metadata(
        metadata_path,
        compressor_kind=compressor_kind,
        checkpoint_path=checkpoint_path,
        preprocessor=preprocessor,
        matrix_columns=list(matrix.columns),
        matrix_kind=matrix_kind,
        latent_prefix=latent_prefix,
        config=final_config,
        metrics=metrics,
    )
    return HandoffBundle(out, latents_path, checkpoint_path, metadata_path)


def load_compressor_from_bundle(bundle_dir: str | Path):
    bundle = Path(bundle_dir)
    metadata = read_metadata(bundle / "handoff_metadata.json")
    checkpoint = bundle / metadata["checkpoint_path"]
    kind = metadata["compressor_kind"]
    if kind == "pca":
        from ..compressors.pca import PCACompressor

        return PCACompressor.load(checkpoint), metadata
    if kind == "survival_vae":
        from ..compressors.survival_vae import SurvivalVAE

        return SurvivalVAE.load(checkpoint), metadata
    raise ValueError(f"unknown compressor kind in metadata: {kind!r}")


def decode_latent_csv(
    *,
    bundle_dir: str | Path,
    latents_csv: str | Path,
    out_expression_csv: str | Path | None = None,
    out_matrix_csv: str | Path | None = None,
) -> pd.DataFrame:
    if out_matrix_csv is None:
        if out_expression_csv is None:
            raise ValueError("decode_latent_csv requires `out_matrix_csv`")
        out_matrix_csv = out_expression_csv
    compressor, metadata = load_compressor_from_bundle(bundle_dir)
    latents = pd.read_csv(latents_csv)
    latent_cols = [
        f"{metadata.get('latent_prefix', LATENT_PREFIX)}{i}"
        for i in range(int(metadata["config"]["n_latent"]))
    ]
    missing = [c for c in latent_cols if c not in latents.columns]
    if missing:
        raise ValueError(f"latent CSV is missing expected columns: {missing}")

    X_selected = compressor.inverse_transform(latents[latent_cols].to_numpy(dtype=float))
    prep = metadata["preprocessor"]
    fill_values = np.asarray(prep.get("fill_values", []), dtype=float)
    if fill_values.size:
        X = np.tile(fill_values, (X_selected.shape[0], 1))
        X[:, np.asarray(prep["selected_indices"], dtype=int)] = X_selected
        matrix_cols = metadata.get("matrix_columns", metadata["expression_columns"])
    else:
        X = X_selected
        matrix_cols = prep["selected_columns"]
    pre = ExpressionPreprocessor(
        n_top_genes=prep.get("n_top_genes"),
        log1p=prep.get("log1p", False),
        matrix_kind=prep.get("matrix_kind", metadata.get("matrix_kind", "expression")),
        beta_epsilon=prep.get("beta_epsilon", 1e-6),
    )
    decoded = pd.DataFrame(pre._inverse_transform(X), columns=matrix_cols)
    out_path = Path(out_matrix_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    decoded.to_csv(out_path, index=False)
    return decoded
