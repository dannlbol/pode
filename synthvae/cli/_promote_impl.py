"""Promote a PODE tuning result into a final training YAML.

Typical use:

    python scripts/promote_pode_best_config.py \
        --base-config /path/to/pode_config.yaml \
        --best-config /path/to/best_configuration.json \
        --out-config /path/to/pode_config_tuned.yaml \
        --train-epochs 1500

This takes the hyperparameters chosen by ``pode tune`` and merges them into the
``synthesizer:`` block of a base PODE YAML so the result can be passed directly
to ``pode train``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


SYNTH_KEYS = {
    "batch_size",
    "embedding_dim",
    "compress_dims",
    "decompress_dims",
    "l2scale",
    "loss_factor",
    "epochs",
    "verbose",
}

INT_KEYS = {"batch_size", "embedding_dim", "loss_factor", "epochs"}
FLOAT_KEYS = {"l2scale"}
LIST_OF_INT_KEYS = {"compress_dims", "decompress_dims"}


def add_args(ap):
    ap.add_argument("--base-config", required=True, help="input PODE YAML used for tuning")
    ap.add_argument("--best-config", required=True, help="best_configuration.json from `pode tune`")
    ap.add_argument("--out-config", required=True, help="output YAML for final `pode train`")
    ap.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help="optional final value for synthesizer.epochs; defaults to the base YAML",
    )
    ap.add_argument(
        "--clear-tuning-grid",
        action="store_true",
        help="replace tuning.param_grid with singleton lists matching the promoted config",
    )
    ap.add_argument(
        "--set-tuning-epochs",
        type=int,
        default=None,
        help="optional replacement for tuning.epochs in the emitted YAML",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the summary printout",
    )
    return ap


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_best(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError("best configuration JSON must be a JSON object")
    return payload


def normalize_value(value):
    if isinstance(value, tuple):
        return [normalize_value(v) for v in value]
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def coerce_synth_value(key: str, value):
    value = normalize_value(value)
    if key in INT_KEYS:
        return int(value)
    if key in FLOAT_KEYS:
        return float(value)
    if key in LIST_OF_INT_KEYS:
        return [int(v) for v in value]
    return value


def run(args) -> None:
    config = load_yaml(args.base_config)
    best = load_best(args.best_config)

    synth = dict(config.get("synthesizer", {}))
    promoted = {}
    skipped = {}

    for key, value in best.items():
        if key == "mean_composite":
            skipped[key] = value
            continue
        if key in SYNTH_KEYS:
            synth[key] = coerce_synth_value(key, value)
            promoted[key] = synth[key]
        else:
            skipped[key] = value

    if args.train_epochs is not None:
        synth["epochs"] = int(args.train_epochs)
        promoted["epochs"] = synth["epochs"]

    config["synthesizer"] = synth

    tuning = dict(config.get("tuning", {}))
    if args.set_tuning_epochs is not None:
        tuning["epochs"] = int(args.set_tuning_epochs)
    if args.clear_tuning_grid:
        tuning["param_grid"] = {
            k: [normalize_value(v)]
            for k, v in promoted.items()
            if k in {"batch_size", "embedding_dim", "compress_dims", "decompress_dims", "l2scale", "loss_factor"}
        }
    if tuning:
        config["tuning"] = tuning

    out_path = Path(args.out_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    if not args.quiet:
        print(f"Wrote tuned training config: {out_path}")
        if promoted:
            print("Promoted synthesizer settings:")
            for key in sorted(promoted):
                print(f"  {key}: {promoted[key]}")
        if skipped:
            print("Ignored non-synthesizer keys from best configuration:")
            for key in sorted(skipped):
                print(f"  {key}: {skipped[key]}")


def main(argv=None) -> int:
    ap = add_args(argparse.ArgumentParser(description=__doc__))
    run(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
