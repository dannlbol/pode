#!/usr/bin/env python3
"""Run, tune, export, and decode the hybrid latent workflow.

Legacy end-to-end invocation still works::

    python scripts/run_hybrid.py --compressor pca --synthesizer gmm

Controlled workflow::

    python scripts/run_hybrid.py tune --sweep config/hybrid_sweep.yaml --out-dir runs/tune
    python scripts/run_hybrid.py export-latents --config runs/tune/selected_config.json \
        --clinical clinical.csv --expression expression.csv --survival survival.csv \
        --out-dir runs/final_handoff
    python scripts/run_hybrid.py export-latents \
        --clinical clinical.csv --methylation methylation.csv --survival survival.csv \
        --out-dir runs/methylation_handoff
    python scripts/run_hybrid.py sample-decode --bundle-dir runs/final_handoff \
        --latents sampled_latents.csv --out-matrix synthetic_expression.csv
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from synthvae.cli._io import read_table
from synthvae.data.toy import make_cohort, ToyConfig
from synthvae.data.preprocessing import ExpressionPreprocessor
from synthvae.evaluation.report import evaluate, format_report
from synthvae.pipeline.handoff import (
    decoded_matrix_name,
    decode_latent_csv,
    latent_frame,
    latent_prefix_for_matrix_kind,
    save_handoff_bundle,
)
from synthvae.pipeline.hybrid import HybridPipeline


BASE_COMPRESSOR_CONFIG = {
    "compressor": "pca",
    "n_latent": 16,
    "epochs": 100,
    "n_top_genes": 0,
}

TUNING_OBJECTIVES = ("full_joint", "latents_survival", "reconstruction_only")


def build_compressor(kind: str, params: dict):
    n_latent = int(params.get("n_latent", 16))
    if kind == "pca":
        from synthvae.compressors.pca import PCACompressor

        return PCACompressor(n_latent=n_latent, random_state=int(params.get("seed", 0)))
    if kind == "survival_vae":
        from synthvae.compressors.survival_vae import SurvivalVAE, SurvivalVAEConfig

        cfg_names = {f.name for f in fields(SurvivalVAEConfig)}
        cfg_kwargs = {k: v for k, v in params.items() if k in cfg_names}
        cfg_kwargs["n_latent"] = n_latent
        cfg_kwargs["max_epochs"] = int(params.get("epochs", params.get("max_epochs", 100)))
        return SurvivalVAE(SurvivalVAEConfig(**cfg_kwargs))
    raise ValueError(f"unknown compressor {kind!r}")


def build_synthesizer(kind: str, params: dict):
    if kind == "gmm":
        from synthvae.synthesizers.gmm_fallback import GaussianMixtureFallback

        return GaussianMixtureFallback(random_state=int(params.get("seed", 0)))
    if kind == "sdv":
        from synthvae.synthesizers.sdv_tvae import SdvTvaeSynthesizer

        return SdvTvaeSynthesizer(epochs=int(params.get("tvae_epochs", params.get("epochs", 300))))
    raise ValueError(f"unknown synthesizer {kind!r}")


def build_preprocessor(params: dict) -> ExpressionPreprocessor:
    n_top_genes = int(params.get("n_top_genes", 0) or 0)
    return ExpressionPreprocessor(
        n_top_genes=(n_top_genes or None),
        log1p=bool(params.get("log1p", False)),
        matrix_kind=params.get("matrix_kind", "expression"),
        beta_epsilon=float(params.get("beta_epsilon", 1e-6)),
    )


def load_data(args):
    methylation = getattr(args, "methylation", None)
    copy_number = getattr(args, "copy_number", None)
    if args.expression or methylation or copy_number or args.survival or args.clinical:
        matrix_inputs = [p for p in (args.expression, methylation, copy_number) if p]
        if not (len(matrix_inputs) == 1 and args.survival):
            raise ValueError(
                "BYO runs require --survival and exactly one of --expression, "
                "--methylation, or --copy-number. --clinical is optional."
            )
        if copy_number:
            representation = getattr(args, "copy_number_representation", "continuous")
            matrix_kind = (
                "copy_number_discrete" if representation == "discrete" else "copy_number"
            )
            matrix = read_table(copy_number)
        else:
            matrix_kind = "methylation" if methylation else "expression"
            matrix = read_table(methylation or args.expression)
        survival = read_table(args.survival)
        if args.clinical:
            clinical = read_table(args.clinical)
        else:
            clinical = pd.DataFrame(index=np.arange(len(matrix)))
        if not (len(clinical) == len(matrix) == len(survival)):
            raise ValueError("clinical/matrix/survival CSVs must have the same row count.")
        return clinical, matrix, survival, matrix_kind
    clinical, expression, survival, _ = make_cohort(
        ToyConfig(n=args.n, n_genes=args.n_genes, seed=args.seed)
    )
    return clinical, expression, survival, "expression"


def split_data(clinical, expression, survival, *, seed: int, test_fraction: float):
    rng = np.random.default_rng(seed)
    n = len(clinical)
    perm = rng.permutation(n)
    n_test = int(test_fraction * n)
    test_ix, train_ix = perm[:n_test], perm[n_test:]

    def subset(df, ix):
        return df.iloc[ix].reset_index(drop=True)

    return (
        subset(clinical, train_ix),
        subset(expression, train_ix),
        subset(survival, train_ix),
        subset(clinical, test_ix),
        subset(expression, test_ix),
        subset(survival, test_ix),
    )


def prepare_params(args, overrides: dict | None = None) -> dict:
    params = dict(BASE_COMPRESSOR_CONFIG)
    for key in (
        "compressor", "synthesizer", "n_latent", "epochs", "n_top_genes",
        "seed", "beta_epsilon",
    ):
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                params[key] = value
    params["synthesizer"] = params.get("synthesizer", "gmm")
    if overrides:
        params.update(overrides)
    return params


def run_once(args, params: dict | None = None, *, write_outputs: bool = True) -> dict:
    params = prepare_params(args, params)
    clinical, matrix, survival, matrix_kind = load_data(args)
    params["matrix_kind"] = matrix_kind
    latent_prefix = latent_prefix_for_matrix_kind(matrix_kind)
    cl_tr, ex_tr, sv_tr, cl_te, ex_te, sv_te = split_data(
        clinical,
        matrix,
        survival,
        seed=int(params.get("seed", args.seed)),
        test_fraction=float(args.test_fraction),
    )

    pre = build_preprocessor(params)
    compressor = build_compressor(params["compressor"], params)
    synthesizer = build_synthesizer(params["synthesizer"], params)

    print(
        f"Fitting hybrid: compressor={params['compressor']} "
        f"synthesizer={params['synthesizer']} n_latent={params['n_latent']}"
    )
    pipe = HybridPipeline(
        compressor, synthesizer, preprocessor=pre, categorical=args.categorical,
        latent_prefix=latent_prefix,
    )
    pipe.fit(cl_tr, ex_tr, sv_tr)

    X_te = pre.transform(ex_te.to_numpy(dtype=float))
    recon_mse = compressor.reconstruction_error(X_te)
    print(f"Held-out {matrix_kind} reconstruction MSE: {recon_mse:.4f}")

    result = pipe.sample(len(cl_te))
    has_clinical = len(cl_te.columns) > 0
    real = {
        "clinical": cl_te,
        "latents": pipe.encode_real(ex_te),
        "time": sv_te["time"].to_numpy(dtype=float),
        "event": sv_te["event"].to_numpy(dtype=float),
    }
    train_real = {
        "clinical": cl_tr,
        "latents": pipe.encode_real(ex_tr),
        "time": sv_tr["time"].to_numpy(dtype=float),
        "event": sv_tr["event"].to_numpy(dtype=float),
    }
    synth = {
        "clinical": result.clinical,
        "latents": result.latents,
        "time": result.survival["time"].to_numpy(dtype=float),
        "event": result.survival["event"].to_numpy(dtype=float),
    }
    full_metrics = evaluate(real, synth, train_real=train_real, prefer_lifelines=not args.no_lifelines)
    metrics = dict(full_metrics)
    metrics["reconstruction_mse"] = recon_mse
    metrics["has_clinical"] = has_clinical
    if has_clinical:
        empty_real = dict(real, clinical=pd.DataFrame(index=real["clinical"].index))
        empty_train_real = dict(train_real, clinical=pd.DataFrame(index=train_real["clinical"].index))
        empty_synth = dict(synth, clinical=pd.DataFrame(index=synth["clinical"].index))
        latent_metrics = evaluate(
            empty_real,
            empty_synth,
            train_real=empty_train_real,
            prefer_lifelines=not args.no_lifelines,
        )
        metrics.update({f"latent_only_{k}": v for k, v in latent_metrics.items()})
    print()
    print(format_report(full_metrics))

    if write_outputs and args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.clinical.to_csv(out / "synthetic_clinical.csv", index=False)
        result.expression.to_csv(out / decoded_matrix_name(matrix_kind), index=False)
        result.survival.to_csv(out / "synthetic_survival.csv", index=False)
        latent_frame(result.latents, prefix=latent_prefix).to_csv(out / "sampled_latents.csv", index=False)
        with open(out / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(jsonable(metrics), fh, indent=2, sort_keys=True)
    return metrics


def metric_score(
    metrics: dict,
    weights: dict | None = None,
    objective: str = "full_joint",
) -> float:
    if objective not in TUNING_OBJECTIVES:
        raise ValueError(
            f"unknown tuning objective {objective!r}; expected one of {TUNING_OBJECTIVES}"
        )
    if objective == "reconstruction_only":
        recon = metrics.get("reconstruction_mse", np.nan)
        return float(recon) if np.isfinite(recon) else 1e6

    weights = weights or {
        "reconstruction_mse": 1.0,
        "concordance_gap": 1.0,
        "km_iad": 1.0,
        "latent_cov_relgap": 1.0,
        "hr_logcorr_penalty": 1.0,
    }
    prefix = ""
    if objective == "latents_survival" and metrics.get("has_clinical"):
        prefix = "latent_only_"
    hr = metrics.get(f"{prefix}hr_logcorr", np.nan)
    hr_penalty = 1.0 - hr if np.isfinite(hr) else 1.0
    terms = {
        "reconstruction_mse": metrics.get("reconstruction_mse", np.nan),
        "concordance_gap": abs(metrics.get(f"{prefix}concordance_gap", np.nan)),
        "km_iad": metrics.get(f"{prefix}km_iad", np.nan),
        "latent_cov_relgap": metrics.get(f"{prefix}latent_cov_relgap", np.nan),
        "hr_logcorr_penalty": hr_penalty,
    }
    score = 0.0
    for key, weight in weights.items():
        value = terms.get(key, metrics.get(key, np.nan))
        score += float(weight) * (float(value) if np.isfinite(value) else 1e6)
    return score


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if not isinstance(value, (dict, list, tuple)) and pd.isna(value):
        return None
    return value


def aggregate_seed_results(results: pd.DataFrame, config_keys: list[str]) -> pd.DataFrame:
    """Collapse per-seed sweep rows into per-config mean/std summaries."""
    if "seed" not in results.columns or results["seed"].nunique() <= 1:
        return pd.DataFrame()

    group_cols = [k for k in config_keys if k in results.columns and k != "seed"]
    numeric_cols = [
        c for c in results.columns
        if c not in set(group_cols) | {"candidate", "tuning_objective", "has_clinical"}
        and pd.api.types.is_numeric_dtype(results[c])
    ]
    rows = []
    for key, frame in results.groupby(group_cols, dropna=False, sort=False):
        key = (key,) if len(group_cols) == 1 else tuple(key)
        row = dict(zip(group_cols, key))
        seeds = sorted(int(s) for s in pd.unique(frame["seed"]))
        row["seed_values"] = json.dumps(seeds)
        row["n_seeds"] = len(seeds)
        row["has_clinical"] = bool(frame["has_clinical"].iloc[0])
        row["tuning_objective"] = frame["tuning_objective"].iloc[0]
        for col in numeric_cols:
            row[f"{col}_mean"] = float(frame[col].mean())
            row[f"{col}_std"] = float(frame[col].std(ddof=0))
        rows.append(row)
    aggregated = pd.DataFrame(rows)
    return aggregated.sort_values("rank_score_mean").reset_index(drop=True)


def expand_grid(sweep: dict) -> list[dict]:
    fixed = dict(sweep.get("fixed", {}))
    grid = sweep.get("grid", {})
    if not grid:
        return [fixed]
    keys = list(grid)
    values = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [dict(fixed, **dict(zip(keys, combo))) for combo in itertools.product(*values)]


def load_sweep(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        sweep = yaml.safe_load(fh) or {}
    if "grid" not in sweep and "fixed" not in sweep:
        raise ValueError("sweep YAML must define at least `grid` or `fixed`")
    return sweep


def command_tune(args) -> None:
    sweep = load_sweep(args.sweep)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = {
        "compressor": sweep.get("compressor", args.compressor),
        "synthesizer": sweep.get("synthesizer", args.synthesizer),
        "seed": args.seed,
    }
    weights = sweep.get("metric_weights")
    objective = sweep.get("tuning_objective", args.tuning_objective)
    rows = []
    candidates = expand_grid(sweep)
    config_keys = sorted(set(BASE_COMPRESSOR_CONFIG) | {
        "compressor",
        "synthesizer",
        "seed",
        "w_surv",
        "w_recon",
        "beta_max",
        "free_bits",
        "kl_warmup_epochs",
        "batch_size",
        "lr",
        "dropout",
        "tvae_epochs",
        "log1p",
        "matrix_kind",
        "beta_epsilon",
    })
    for i, candidate in enumerate(candidates):
        params = prepare_params(args, dict(base, **candidate))
        print(f"\nSweep {i + 1}/{len(candidates)}: {params}")
        metrics = run_once(args, params, write_outputs=False)
        score = metric_score(metrics, weights, objective=objective)
        rows.append({
            "rank_score": score,
            "tuning_objective": objective,
            "candidate": i,
            **params,
            **metrics,
        })

    results = pd.DataFrame(rows).sort_values("rank_score").reset_index(drop=True)
    results.to_csv(out / "sweep_results.csv", index=False)
    aggregated = aggregate_seed_results(results, config_keys)
    if not aggregated.empty:
        aggregated.to_csv(out / "sweep_results_aggregated.csv", index=False)
        best = aggregated.iloc[0].to_dict()
        selected = {
            k: best[k]
            for k in config_keys
            if k in best and pd.notna(best[k]) and k != "seed"
        }
        selected["seed"] = int(args.seed)
        best_metrics = {k: best[k] for k in best if k not in selected}
        aggregation = {
            "over": "seed",
            "seed_values": json.loads(best["seed_values"]),
            "refit_seed": int(args.seed),
        }
    else:
        best = results.iloc[0].to_dict()
        selected = {k: best[k] for k in config_keys if k in best and pd.notna(best[k])}
        best_metrics = {k: best[k] for k in best if k not in selected}
        aggregation = None
    with open(out / "selected_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            jsonable({
                "best_config": selected,
                "best_metrics": best_metrics,
                "metric_weights": weights,
                "tuning_objective": objective,
                "aggregation": aggregation,
            }),
            fh,
            indent=2,
            sort_keys=True,
        )
    print(f"\nBest config written to {out / 'selected_config.json'}")
    print(f"Ranked sweep results written to {out / 'sweep_results.csv'}")


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        if str(path).endswith((".yaml", ".yml")):
            payload = yaml.safe_load(fh) or {}
        else:
            payload = json.load(fh)
    return payload.get("best_config", payload.get("config", payload))


def command_export_latents(args) -> None:
    config = load_config(args.config)
    params = prepare_params(args, config)
    clinical, matrix, survival, matrix_kind = load_data(args)
    params["matrix_kind"] = matrix_kind
    params["latent_prefix"] = latent_prefix_for_matrix_kind(matrix_kind)

    pre = build_preprocessor(params)
    X = pre.fit_transform(matrix.to_numpy(dtype=float))
    compressor = build_compressor(params["compressor"], params)
    compressor.fit(
        X,
        survival["time"].to_numpy(dtype=float),
        survival["event"].to_numpy(dtype=float),
    )
    recon_mse = compressor.reconstruction_error(X)
    metrics = {"full_reconstruction_mse": recon_mse}
    bundle = save_handoff_bundle(
        compressor=compressor,
        compressor_kind=params["compressor"],
        preprocessor=pre,
        matrix=matrix,
        matrix_kind=matrix_kind,
        latent_prefix=params["latent_prefix"],
        out_dir=args.out_dir,
        config=params,
        metrics=metrics,
        include_input=args.include_input,
    )
    clinical.reset_index(drop=True).to_csv(Path(args.out_dir) / "clinical_full.csv", index=False)
    survival.reset_index(drop=True).to_csv(Path(args.out_dir) / "survival_full.csv", index=False)
    print(f"Full-data latent table written to {bundle.latents_path}")
    print(f"Compressor checkpoint written to {bundle.checkpoint_path}")
    print(f"Handoff metadata written to {bundle.metadata_path}")
    print(f"Full-data reconstruction MSE: {recon_mse:.4f}")


def command_sample_decode(args) -> None:
    from synthvae.pipeline.handoff import read_metadata

    metadata = read_metadata(Path(args.bundle_dir) / "handoff_metadata.json")
    matrix_kind = metadata.get("matrix_kind", "expression")
    out_matrix = args.out_matrix or decoded_matrix_name(matrix_kind)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_matrix = out / Path(out_matrix).name
        latents = pd.read_csv(args.latents)
        latents.to_csv(out / "sampled_latents.csv", index=False)
        for src, name in (
            (args.clinical, "synthetic_clinical.csv"),
            (args.survival, "synthetic_survival.csv"),
        ):
            if src:
                pd.read_csv(src).to_csv(out / name, index=False)
    decoded = decode_latent_csv(
        bundle_dir=args.bundle_dir,
        latents_csv=args.latents,
        out_matrix_csv=out_matrix,
    )
    print(f"Decoded {matrix_kind} matrix written to {out_matrix} ({decoded.shape[0]} rows)")


def add_data_args(ap):
    ap.add_argument("--clinical", default=None, help="clinical CSV (BYO data)")
    ap.add_argument("--expression", default=None, help="expression CSV, genes as columns")
    ap.add_argument("--methylation", default=None, help="methylation beta-value CSV, features as columns")
    ap.add_argument(
        "--copy-number",
        dest="copy_number",
        default=None,
        help="copy-number matrix CSV, genomic bins as columns "
        "(produced by synthvae.data.copy_number.segments_to_matrix)",
    )
    ap.add_argument(
        "--copy-number-representation",
        dest="copy_number_representation",
        choices=["continuous", "discrete"],
        default="continuous",
        help="continuous log2-ratio matrix (Gaussian) or ordinal {-1,0,1} calls",
    )
    ap.add_argument("--survival", default=None, help="survival CSV with time,event columns")
    ap.add_argument("--n", type=int, default=800, help="toy cohort size")
    ap.add_argument("--n-genes", type=int, default=300, help="toy gene count")
    ap.add_argument("--seed", type=int, default=0)


def add_common_model_args(ap):
    ap.add_argument("--compressor", choices=["pca", "survival_vae"], default="pca")
    ap.add_argument("--synthesizer", choices=["gmm", "sdv"], default="gmm")
    ap.add_argument("--n-latent", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-top-genes", type=int, default=0, help="HVG selection; 0 = keep all genes")
    ap.add_argument("--beta-epsilon", type=float, default=1e-6, help="epsilon for methylation beta-value logit clipping")
    ap.add_argument("--categorical", nargs="*", default=["stage", "sex"])
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--no-lifelines", action="store_true", help="force the numpy Cox fallback")
    ap.add_argument(
        "--tuning-objective",
        choices=TUNING_OBJECTIVES,
        default="full_joint",
        help="how tuning should rank configurations",
    )


def _add_subcommands(sub) -> None:
    """Add run/tune/export-latents/sample-decode onto an ``add_subparsers`` object."""
    run = sub.add_parser("run", help="fit compressor+synthesizer, sample, and evaluate")
    add_common_model_args(run)
    add_data_args(run)
    run.add_argument("--out-dir", default=None, help="optional directory for sampled outputs")
    run.set_defaults(func=run_once)

    tune = sub.add_parser("tune", help="run a YAML-defined compressor/synthesizer sweep")
    add_common_model_args(tune)
    add_data_args(tune)
    tune.add_argument("--sweep", required=True, help="YAML sweep spec")
    tune.add_argument("--out-dir", required=True, help="directory for sweep results")
    tune.set_defaults(func=command_tune)

    export = sub.add_parser("export-latents", help="fit on all rows and export latents")
    add_common_model_args(export)
    add_data_args(export)
    export.add_argument("--config", default=None, help="selected_config.json or YAML config")
    export.add_argument("--out-dir", required=True, help="handoff artifact directory")
    export.add_argument("--include-input", action="store_true", help="also write the input matrix CSV")
    export.set_defaults(func=command_export_latents)

    decode = sub.add_parser("sample-decode", help="decode externally sampled latents")
    decode.add_argument("--bundle-dir", required=True, help="directory from export-latents")
    decode.add_argument("--latents", required=True, help="sampled latent CSV")
    decode.add_argument("--out-matrix", default=None, help="decoded matrix CSV; defaults from bundle modality")
    decode.add_argument("--out-expression", dest="out_matrix", help=argparse.SUPPRESS)
    decode.add_argument("--out-dir", default=None, help="optional directory for standard outputs")
    decode.add_argument("--clinical", default=None, help="optional sampled clinical CSV to copy")
    decode.add_argument("--survival", default=None, help="optional sampled survival CSV to copy")
    decode.set_defaults(func=command_sample_decode)


def register(subparsers) -> None:
    """Mount the ``hybrid`` namespace under a parent ``add_subparsers`` (pode synthesis)."""
    hybrid = subparsers.add_parser(
        "hybrid", help="two-stage compressor + synthesizer hybrid pipeline.")
    _add_subcommands(hybrid.add_subparsers(dest="hybrid_command", required=True))


def main(argv=None) -> int:
    """Standalone entry: ``python -m synthvae.cli.hybrid <run|tune|...>``."""
    parser = argparse.ArgumentParser(prog="synthvae-hybrid", description=__doc__)
    _add_subcommands(parser.add_subparsers(dest="hybrid_command", required=True))
    args = parser.parse_args(argv)
    rc = args.func(args)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
