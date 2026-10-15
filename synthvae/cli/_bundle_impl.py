"""Prepare a PODE-ready training table from a synthvae-hybrid handoff bundle.

This script reads the aligned files produced by

    python scripts/run_hybrid.py export-latents --out-dir <bundle_dir> ...

and writes:

1. a single wide CSV combining
   [clinical_full.csv | latents_full.csv | survival_full.csv]
2. a starter PODE YAML config that marks the survival pair explicitly and gives
   you a clean place to tune TVAE hyperparameters.

The default survival output names are ``OS_Time`` and ``OS_Status`` because
those match PODE's built-in detection rules. Latent columns are read from the
bundle metadata and are treated as continuous automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_OBJECTIVE_WEIGHTS = {
    "hr_mae": 1.0,
    "tvd_mean": 1.0,
    "surv_prob_diff_5yr": 1.0,
}

DEFAULT_TUNING_GRID = {
    "batch_size": [128, 256, 500],
    "embedding_dim": [64, 128],
    "compress_dims": [[128, 128], [256, 256]],
    "decompress_dims": [[128, 128], [256, 256]],
    "l2scale": [1.0e-5, 1.0e-6],
    "loss_factor": [1, 2],
}


def add_args(ap):
    ap.add_argument(
        "--bundle-dir",
        required=True,
        nargs="+",
        help="one or more export-latents output directories. Pass a single dir for the "
        "usual single-modality handoff, or several (e.g. an expression bundle and a "
        "methylation bundle) to concatenate their latent blocks into one joint table.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="directory to receive the concatenated CSV and generated PODE YAML",
    )
    ap.add_argument(
        "--table-name",
        default="training_table.csv",
        help="output CSV filename inside --out-dir",
    )
    ap.add_argument(
        "--config-name",
        default="pode_config.yaml",
        help="output YAML filename inside --out-dir",
    )
    ap.add_argument(
        "--survival-base",
        default="OS",
        help="base name for the exported survival pair, e.g. OS -> OS_Time/OS_Status",
    )
    ap.add_argument(
        "--time-col",
        default="time",
        help="column name in survival_full.csv containing survival time",
    )
    ap.add_argument(
        "--event-col",
        default="event",
        help="column name in survival_full.csv containing event indicator",
    )
    ap.add_argument(
        "--continuous-cols",
        nargs="*",
        default=[],
        help="clinical columns to force to continuous in the generated PODE YAML",
    )
    ap.add_argument(
        "--categorical-cols",
        nargs="*",
        default=[],
        help="clinical columns to force to categorical in the generated PODE YAML",
    )
    ap.add_argument(
        "--drop-columns",
        nargs="*",
        default=[],
        help="columns to list under data.drop_columns in the generated PODE YAML",
    )
    ap.add_argument(
        "--drop-patterns",
        nargs="*",
        default=[],
        help="glob patterns to list under data.drop_patterns in the generated PODE YAML",
    )
    ap.add_argument(
        "--require-columns",
        nargs="*",
        default=[],
        help="extra columns required by PODE before training",
    )
    ap.add_argument(
        "--survival-time-scale",
        type=float,
        default=1.0,
        help="factor applied by PODE to survival-time columns during preparation",
    )
    ap.add_argument(
        "--censor-time-threshold",
        type=float,
        default=None,
        help="administrative censoring threshold to place into the generated PODE YAML",
    )
    ap.add_argument("--epochs", type=int, default=1500, help="PODE TVAE training epochs")
    ap.add_argument("--batch-size", type=int, default=500, help="PODE TVAE batch size")
    ap.add_argument("--embedding-dim", type=int, default=128, help="PODE TVAE embedding dim")
    ap.add_argument(
        "--compress-dims",
        nargs="*",
        type=int,
        default=[256, 256],
        help="PODE TVAE encoder widths",
    )
    ap.add_argument(
        "--decompress-dims",
        nargs="*",
        type=int,
        default=[256, 256],
        help="PODE TVAE decoder widths",
    )
    ap.add_argument("--l2scale", type=float, default=1.0e-6, help="PODE TVAE L2 scale")
    ap.add_argument("--loss-factor", type=int, default=2, help="PODE TVAE loss factor")
    ap.add_argument(
        "--tuning-epochs",
        type=int,
        default=500,
        help="epochs field for the generated tuning block",
    )
    ap.add_argument(
        "--stability-runs",
        type=int,
        default=3,
        help="stability_runs field for the generated tuning block",
    )
    ap.add_argument(
        "--objective-timepoint",
        type=float,
        default=5.0,
        help="survival timepoint for the generated objective block",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="set synthesizer.verbose: true in the generated PODE YAML",
    )
    ap.add_argument(
        "--singleton-tuning-grid",
        action="store_true",
        help="emit a 1-combination tuning grid matching the main synthesizer settings",
    )
    return ap


def _read_metadata(bundle_dir: Path) -> dict:
    metadata_path = bundle_dir / "handoff_metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_bundles(
    bundle_dirs: list[Path], time_col: str, event_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[Path, dict, list[str]]]]:
    """Read one or more handoff bundles into a single wide table.

    Clinical and survival are taken from the first bundle and required to be
    identical across all bundles (the latent rows must be aligned to the same
    patients in the same order). Each bundle's ``latents_full.csv`` is appended as
    its own block; the blocks must use distinct latent prefixes so they never
    collide. Returns ``(combined, clinical, blocks)`` where ``blocks`` is a list of
    ``(bundle_dir, metadata, latent_columns)`` in the given order.
    """
    first = bundle_dirs[0]
    clinical = pd.read_csv(first / "clinical_full.csv")
    survival = pd.read_csv(first / "survival_full.csv")
    if time_col not in survival.columns or event_col not in survival.columns:
        raise ValueError(
            f"survival_full.csv must contain {time_col!r} and {event_col!r}; "
            f"found {list(survival.columns)!r}"
        )
    n = len(clinical)

    latent_frames: list[pd.DataFrame] = []
    blocks: list[tuple[Path, dict, list[str]]] = []
    seen_prefixes: dict[str, Path] = {}
    for bundle_dir in bundle_dirs:
        metadata = _read_metadata(bundle_dir)
        latents = pd.read_csv(bundle_dir / "latents_full.csv")
        if len(latents) != n:
            raise ValueError(
                f"row count of {bundle_dir}/latents_full.csv ({len(latents)}) does not "
                f"match the first bundle ({n}); bundles must be row-aligned."
            )
        # Clinical/survival must match the first bundle so the concatenated latent
        # blocks all describe the same patients in the same order.
        if bundle_dir != first:
            other_clinical = pd.read_csv(bundle_dir / "clinical_full.csv")
            other_survival = pd.read_csv(bundle_dir / "survival_full.csv")
            if not clinical.equals(other_clinical):
                raise ValueError(
                    f"clinical_full.csv in {bundle_dir} differs from {first}; joint "
                    "handoff requires identical, row-aligned clinical tables."
                )
            if not survival[[time_col, event_col]].equals(
                other_survival[[time_col, event_col]]
            ):
                raise ValueError(
                    f"survival_full.csv in {bundle_dir} differs from {first}; joint "
                    "handoff requires identical, row-aligned survival tables."
                )
        prefix = metadata.get("latent_prefix", "expr_latent_")
        latent_cols = [c for c in latents.columns if c.startswith(prefix)]
        if not latent_cols:
            raise ValueError(
                f"no latent columns with prefix {prefix!r} found in "
                f"{bundle_dir}/latents_full.csv"
            )
        if prefix in seen_prefixes:
            raise ValueError(
                f"latent prefix {prefix!r} appears in both {seen_prefixes[prefix]} and "
                f"{bundle_dir}; joint bundles must use distinct prefixes."
            )
        seen_prefixes[prefix] = bundle_dir
        latent_frames.append(latents[latent_cols].reset_index(drop=True))
        blocks.append((bundle_dir, metadata, latent_cols))

    combined = pd.concat(
        [clinical.reset_index(drop=True)]
        + latent_frames
        + [survival[[time_col, event_col]].reset_index(drop=True)],
        axis=1,
    )
    assert len(combined) == n
    return combined, clinical, blocks


def load_bundle(
    bundle_dir: Path, time_col: str, event_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Backward-compatible wrapper for single-bundle callers.

    Returns the same 3-tuple shape older tests and scripts expect:
    ``(combined, clinical, metadata)``.
    """
    combined, clinical, blocks = load_bundles([Path(bundle_dir)], time_col, event_col)
    _, metadata, _ = blocks[0]
    return combined, clinical, metadata


def rename_survival_columns(table: pd.DataFrame, survival_base: str, time_col: str, event_col: str) -> tuple[pd.DataFrame, str, str]:
    out = table.copy()
    time_name = f"{survival_base}_Time"
    event_name = f"{survival_base}_Status"
    out = out.rename(columns={time_col: time_name, event_col: event_name})
    return out, time_name, event_name


def build_type_overrides(
    *,
    clinical: pd.DataFrame,
    latent_columns: list[str],
    continuous_cols: list[str],
    categorical_cols: list[str],
    time_name: str,
    event_name: str,
) -> dict[str, str]:
    overrides: dict[str, str] = {}

    for col in latent_columns:
        overrides[col] = "continuous"

    for col in continuous_cols:
        if col not in clinical.columns:
            raise ValueError(f"--continuous-cols includes unknown clinical column {col!r}")
        overrides[col] = "continuous"

    for col in categorical_cols:
        if col not in clinical.columns:
            raise ValueError(f"--categorical-cols includes unknown clinical column {col!r}")
        overrides[col] = "categorical"

    overrides[time_name] = "survival_time"
    overrides[event_name] = "survival_event"
    return overrides


def build_config(
    *,
    type_overrides: dict[str, str],
    time_name: str,
    event_name: str,
    args: argparse.Namespace,
) -> dict:
    if args.singleton_tuning_grid:
        tuning_grid = {
            "batch_size": [args.batch_size],
            "embedding_dim": [args.embedding_dim],
            "compress_dims": [list(args.compress_dims)],
            "decompress_dims": [list(args.decompress_dims)],
            "l2scale": [args.l2scale],
            "loss_factor": [args.loss_factor],
        }
    else:
        tuning_grid = dict(DEFAULT_TUNING_GRID)

    return {
        "schema": {
            "max_cardinality_for_discrete": 20,
            "survival_pairs": [
                {
                    "base": args.survival_base,
                    "time_col": time_name,
                    "event_col": event_name,
                }
            ],
            "type_overrides": type_overrides,
        },
        "data": {
            "drop_patterns": list(args.drop_patterns),
            "drop_columns": list(args.drop_columns),
            "censor_time_threshold": args.censor_time_threshold,
            "survival_time_scale": args.survival_time_scale,
            "require_columns": list(args.require_columns),
            "max_na_fraction": 1.0,
            "min_category_count": 0,
        },
        "synthesizer": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "embedding_dim": args.embedding_dim,
            "compress_dims": list(args.compress_dims),
            "decompress_dims": list(args.decompress_dims),
            "l2scale": args.l2scale,
            "loss_factor": args.loss_factor,
            "verbose": bool(args.verbose),
        },
        "objective": {
            "weights": dict(DEFAULT_OBJECTIVE_WEIGHTS),
            "survival_timepoint": args.objective_timepoint,
        },
        "tuning": {
            "epochs": args.tuning_epochs,
            "stability_runs": args.stability_runs,
            "param_grid": tuning_grid,
        },
        "adjustment": {
            "enabled": False,
            "method": "event_conversion",
            "timepoint": args.objective_timepoint,
            "stratify_col": None,
        },
    }


def run(args) -> None:
    bundle_dirs = [Path(b) for b in args.bundle_dir]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined, clinical, blocks = load_bundles(bundle_dirs, args.time_col, args.event_col)
    combined, time_name, event_name = rename_survival_columns(
        combined, args.survival_base, args.time_col, args.event_col
    )

    # Union of every bundle's latent columns; all are decoded downstream and so are
    # all continuous for the TVAE.
    latent_columns = [col for _, _, cols in blocks for col in cols]

    type_overrides = build_type_overrides(
        clinical=clinical,
        latent_columns=latent_columns,
        continuous_cols=list(args.continuous_cols),
        categorical_cols=list(args.categorical_cols),
        time_name=time_name,
        event_name=event_name,
    )
    config = build_config(
        type_overrides=type_overrides,
        time_name=time_name,
        event_name=event_name,
        args=args,
    )

    csv_path = out_dir / args.table_name
    yaml_path = out_dir / args.config_name
    combined.to_csv(csv_path, index=False)
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    print(f"Wrote PODE training table: {csv_path}")
    print(f"Wrote PODE config: {yaml_path}")
    print(f"Rows: {len(combined)}")
    print(f"Clinical columns: {clinical.shape[1]}")
    print(f"Latent columns (total): {len(latent_columns)}")
    for bundle_dir, metadata, cols in blocks:
        prefix = metadata.get("latent_prefix", "expr_latent_")
        kind = metadata.get("matrix_kind") or "expression"
        print(f"  - {kind}: {len(cols)} cols (prefix {prefix!r}) from {bundle_dir}")
    print(f"Survival columns: {time_name}, {event_name}")


def main(argv=None) -> int:
    ap = add_args(argparse.ArgumentParser(description=__doc__))
    run(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
