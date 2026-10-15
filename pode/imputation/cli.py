"""``pode synthesis impute ...`` subcommands.

  * ``train``     - train a mask-aware TVAE (``tvae-masked``) on incomplete data.
  * ``apply``     - multiply impute an incomplete table with a trained model
                    (an SDV ``model.pkl`` -> ``tvae-decoder``, or a ``train``
                    output directory -> ``tvae-masked``).
  * ``benchmark`` - hide known values under masking scenarios and score engines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .base import limit_cpu

N_JOBS_HELP = "Max CPU cores to use (default: config n_jobs, 4; -1 = all cores)."


def _latent_overrides(icfg, args) -> None:
    if getattr(args, "k", None) is not None:
        icfg.latent.k = args.k
    if getattr(args, "method", None) is not None:
        icfg.latent.method = args.method


def cmd_train(args) -> int:
    from ..cli.main import _load_schema, _log_torch_device
    from ..config import load_config
    from ..data.io import ensure_dir, load_dataset
    from ..pipeline import prepare_data
    from ..reproducibility.seeds import SeedTracker, set_all_seeds
    from .config import load_imputation_config
    from .masked_tvae import MaskedTVAEImputer

    config = load_config(args.config)
    icfg = load_imputation_config(args.impute_config)
    df = load_dataset(args.data)
    schema = _load_schema(df, config, args.schema)
    data, schema = prepare_data(df, config, schema)

    tracker = SeedTracker(mode="new")
    seed = args.seed if args.seed is not None else tracker.get_seed("imputation_training")
    set_all_seeds(seed)
    _log_torch_device()

    _leakage_overrides(icfg, args)
    with limit_cpu(args.n_jobs if args.n_jobs is not None else icfg.n_jobs):
        imp = MaskedTVAEImputer(
            config.synthesizer, icfg.masked_tvae, icfg.latent, icfg.block_groups,
            epochs=args.epochs or icfg.masked_tvae.epochs, fit_seed=seed, device=icfg.device,
            atomic_blocks=icfg.atomic_blocks,
        ).configure(icfg.outcome_policy, icfg.outcome_columns, icfg.exclude_columns).fit(data, schema)
    out_dir = ensure_dir(args.output_dir)
    imp.save(out_dir)
    tracker.set_seed("imputation_training", seed)
    tracker.save_seeds(out_dir)
    print(f"Wrote mask-aware TVAE to {out_dir} ({len(data)} rows, "
          f"{int(data.isna().sum().sum())} missing cells, final loss {imp.diagnostics()['final_loss']:.3f})")
    return 0


def _leakage_overrides(icfg, args) -> None:
    """Apply --outcome-policy / --outcome-columns / --exclude-columns / --fit-scope."""
    if getattr(args, "outcome_policy", None):
        icfg.outcome_policy = args.outcome_policy
    if getattr(args, "outcome_columns", None):
        icfg.outcome_columns = _split(args.outcome_columns)
    if getattr(args, "exclude_columns", None):
        icfg.exclude_columns = _split(args.exclude_columns)
    if getattr(args, "fit_scope", None):
        icfg.fit_scope = args.fit_scope
    if getattr(args, "device", None):
        icfg.device = args.device


def _split(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def _cli_schema(df, config, schema_path):
    """Schema from --schema, else detected; identifier columns are left out."""
    from ..schema.column_types import ColumnRole
    from ..schema.schema import DatasetSchema
    from .base import subset_schema

    schema = DatasetSchema.from_yaml(schema_path) if schema_path else DatasetSchema.from_dataframe(df, config.schema)
    ids = set(schema.columns_of_role(ColumnRole.IDENTIFIER))
    return subset_schema(schema, [c for c in schema.columns if c not in ids and c in df.columns])


def cmd_apply(args) -> int:
    import pandas as pd

    from ..config import load_config
    from ..data.io import ensure_dir, load_dataset, save_dataset
    from ..reproducibility.seeds import SeedTracker, set_all_seeds
    from ..schema.schema import DatasetSchema
    from .config import load_imputation_config

    if bool(args.model) == bool(args.engine):
        print("error: give exactly one of --model (pre-trained TVAE) or --engine (fit an engine)", file=sys.stderr)
        return 2
    if args.model and args.reference:
        print("error: --reference only applies with --engine (a --model is already trained)", file=sys.stderr)
        return 2

    icfg = load_imputation_config(args.impute_config)
    _latent_overrides(icfg, args)
    _leakage_overrides(icfg, args)
    df = load_dataset(args.data)
    m = args.m or icfg.m
    tracker = SeedTracker(mode="new")
    seed = args.seed if args.seed is not None else tracker.get_seed("imputation")
    set_all_seeds(seed)
    cores = args.n_jobs if args.n_jobs is not None else icfg.n_jobs
    if args.n_jobs is not None:
        icfg.n_jobs = args.n_jobs
    _leakage_overrides(icfg, args)

    if args.model:
        from .masked_tvae import MaskedTVAEImputer, is_masked_model_dir

        model = Path(args.model)
        if is_masked_model_dir(model):
            imp = MaskedTVAEImputer.load(model, latent=icfg.latent, device=icfg.device)
        else:
            from .decoder_imputer import TVAEDecoderImputer

            schema_path = Path(args.schema) if args.schema else model.parent / "schema.yaml"
            if not schema_path.is_file():
                print("error: --schema is required (no schema.yaml next to the model)", file=sys.stderr)
                return 2
            imp = TVAEDecoderImputer.from_model(model, DatasetSchema.from_yaml(schema_path), icfg.latent,
                                                device=icfg.device)
        from .base import resolve_columns

        clash = resolve_columns(icfg.exclude_columns, imp.columns)
        if clash:
            print(f"error: {clash} are part of the trained model and cannot be excluded; "
                  "retrain without them or use --engine", file=sys.stderr)
            return 2
        if args.outcome_policy or args.outcome_columns:
            imp.configure(icfg.outcome_policy, icfg.outcome_columns)
        label = imp.name
    else:
        from .engines import build_imputer, find_spec

        config = load_config(args.config)
        reference = load_dataset(args.reference) if args.reference else df
        if not args.reference and icfg.outcome_policy == "hide":
            # The engine is fitted on these very rows, so their outcomes would
            # reach it through training: the only leak-free option is exclude.
            print("note: without --reference the engine is fitted on the rows being imputed; "
                  "'hide' is upgraded to 'exclude' so their outcomes never reach it.", file=sys.stderr)
            icfg.outcome_policy = "exclude"
        spec = find_spec(args.engine, icfg)
        if "outcome_policy" in spec.params and not args.reference and spec.params["outcome_policy"] == "hide":
            spec.params["outcome_policy"] = "exclude"
        schema = _cli_schema(reference, config, args.schema)
        imp = build_imputer(spec, config, icfg, seed)
        with limit_cpu(cores):
            imp.fit(reference, schema)
        label = spec.key

    from .masking import partial_block_rows

    groups = icfg.block_groups or getattr(imp, "block_groups", {}) or {}
    partial = partial_block_rows(df, groups) if (groups and icfg.atomic_blocks) else {}
    if partial:
        print(f"warning: rows with a partly missing block group {partial}. Modality-derived "
              "columns are normally all present or all absent; check the data preparation.",
              file=sys.stderr)

    model_cols = [c for c in imp.columns if c in df.columns]
    absent = [c for c in imp.columns if c not in df.columns]
    n_missing = int(df[model_cols].isna().sum().sum())
    print(f"Engine {label}: {len(model_cols)} model columns present, {len(absent)} absent "
          f"(imputed as fully missing), {n_missing} missing cells in {len(df)} rows.", file=sys.stderr)
    print(f"Leakage settings: {json.dumps(imp.leakage_settings())}", file=sys.stderr)

    with limit_cpu(cores):
        draws = imp.impute(df, m=m, seed=seed)

    out_dir = ensure_dir(args.output_dir)
    width = max(3, len(str(m)))
    for i, d in enumerate(draws, start=1):
        save_dataset(d, out_dir / f"imputed_{i:0{width}d}.csv")
    if not args.no_long:
        long = pd.concat(
            [df.assign(**{".imp": 0, ".id": range(len(df))})]
            + [d.assign(**{".imp": i, ".id": range(len(d))}) for i, d in enumerate(draws, start=1)],
            ignore_index=True,
        )
        save_dataset(long, out_dir / "imputed_long.csv")
    diag = {
        "engine": label, "m": m, "seed": seed,
        "model": str(args.model) if args.model else None,
        "reference": str(args.reference) if args.reference else None,
        "absent_columns": absent, "missing_cells": n_missing,
        "partial_block_rows": partial,
        **imp.leakage_settings(), **imp.diagnostics(),
    }
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2, default=float)
    rows = getattr(imp, "_row_diagnostics", None)
    if rows is not None:
        save_dataset(rows, out_dir / "row_diagnostics.csv")
    tracker.set_seed("imputation", seed)
    tracker.save_seeds(out_dir)
    print(f"Wrote {m} imputed datasets to {out_dir}")
    print("Diagnostics:", json.dumps(diag, indent=2, default=float))
    return 0


def cmd_benchmark(args) -> int:
    from ..cli.main import _load_schema, _log_torch_device
    from ..config import load_config
    from ..data.io import ensure_dir, load_dataset
    from ..pipeline import prepare_data
    from .benchmark import run_benchmark
    from .config import EngineSpec, load_imputation_config

    config = load_config(args.config)
    icfg = load_imputation_config(args.impute_config)
    if args.quick:
        icfg = icfg.quick()
    if args.seed is not None:
        icfg.seed = args.seed
    if args.mode:
        icfg.mode = args.mode
    if args.n_jobs is not None:
        icfg.n_jobs = args.n_jobs
    _leakage_overrides(icfg, args)
    if args.engines:
        wanted = [e.strip() for e in args.engines.split(",") if e.strip()]
        known = {e.key: e for e in icfg.engines}
        icfg.engines = [known.get(w, EngineSpec(w)) for w in wanted]

    df = load_dataset(args.data)
    schema = _load_schema(df, config, args.schema)
    data, schema = prepare_data(df, config, schema)
    out_dir = ensure_dir(args.output_dir)
    icfg.to_yaml(out_dir / "imputation_config_used.yaml")
    _log_torch_device()
    result = run_benchmark(data, schema, config, icfg, out_dir)
    summary = result["tables"]["summary"]
    print(f"Benchmark complete: {len(summary)} scenario×engine results in {out_dir}")
    print(f"See {out_dir / 'report.md'}")
    return 0


def register(syn) -> None:
    """Mount ``impute`` under the ``pode synthesis`` subparsers."""
    from .engines import ENGINES

    p = syn.add_parser("impute", help="generative (multiple) imputation with TVAE + benchmarking.")
    sub = p.add_subparsers(dest="impute_command", required=True)

    warn = argparse.ArgumentParser(add_help=False)
    warn.add_argument("--show-warnings", action="store_true",
                      help="Show benign third-party warnings, suppressed by default.")

    leak = argparse.ArgumentParser(add_help=False)
    leak.add_argument("--outcome-policy", choices=["use", "hide", "exclude"], default=None,
                      help="Outcome columns as predictors: use (default; inference), hide (rows being "
                           "imputed don't see their own outcome; prediction), exclude (never seen).")
    leak.add_argument("--outcome-columns", default=None,
                      help="Comma-separated outcome columns/globs (default: survival time/event).")
    leak.add_argument("--exclude-columns", default=None,
                      help="Comma-separated columns/globs the imputer never sees, e.g. post-baseline variables.")
    leak.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None,
                      help="Torch device for TVAE engines/GAIN (default auto: GPU if usable, else CPU). "
                           "Use cpu when the shared GPU is busy.")
    leak.add_argument("--fit-scope", choices=["stacked", "reference"], default=None,
                      help="knn/mice/missforest: learn from reference rows only (reference) or also "
                           "from the rows being imputed (stacked, default).")

    t = sub.add_parser("train", parents=[warn, leak], help="Train a mask-aware TVAE on (incomplete) data.")
    t.add_argument("--data", required=True, help="Training CSV (may contain missing values).")
    t.add_argument("--config", default=None, help="PODE YAML config (schema + synthesizer hyperparameters).")
    t.add_argument("--schema", default=None, help="Optional schema YAML.")
    t.add_argument("--impute-config", default=None, help="Imputation YAML (masked_tvae, block_groups, latent).")
    t.add_argument("--output-dir", default="./pode_imputer")
    t.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    t.add_argument("--n-jobs", type=int, default=None, help=N_JOBS_HELP)
    t.add_argument("--seed", type=int, default=None)
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("apply", parents=[warn, leak], help="Multiply impute an incomplete table.")
    a.add_argument("--model", default=None,
                   help="Pre-trained TVAE: SDV model.pkl from `pode synthesis train` (tvae-decoder) "
                        "or a directory from `pode synthesis impute train` (tvae-masked).")
    a.add_argument("--engine", default=None,
                   help=f"Fit an engine instead of loading a model ({', '.join(ENGINES)}, or an "
                        "engine label from --impute-config).")
    a.add_argument("--reference", default=None,
                   help="With --engine: fit on this reference CSV and impute --data "
                        "(default: fit on --data itself).")
    a.add_argument("--config", default=None, help="PODE YAML config (schema + synthesizer hyperparameters).")
    a.add_argument("--schema", default=None, help="Schema YAML for an SDV model (default: next to it).")
    a.add_argument("--data", required=True, help="Incomplete CSV, in the same units as the training table.")
    a.add_argument("--impute-config", default=None, help="Imputation YAML (latent sampler settings, m).")
    a.add_argument("--m", type=int, default=None, help="Number of imputations (default: config m).")
    a.add_argument("--k", type=int, default=None, help="Importance samples per row.")
    a.add_argument("--method", choices=["sir", "pcn", "smc", "auto"], default=None, help="Latent sampler.")
    a.add_argument("--output-dir", default="./pode_imputed")
    a.add_argument("--no-long", action="store_true", help="Skip the stacked imputed_long.csv.")
    a.add_argument("--n-jobs", type=int, default=None, help=N_JOBS_HELP)
    a.add_argument("--seed", type=int, default=None)
    a.set_defaults(func=cmd_apply)

    b = sub.add_parser("benchmark", parents=[warn, leak], help="Mask known values and score imputation engines.")
    b.add_argument("--data", required=True, help="Reference CSV (ideally complete).")
    b.add_argument("--config", default=None, help="PODE YAML config.")
    b.add_argument("--schema", default=None, help="Optional schema YAML.")
    b.add_argument("--impute-config", default=None, help="Imputation/benchmark YAML.")
    b.add_argument("--output-dir", default="./pode_impute_benchmark")
    b.add_argument("--engines", default=None, help=f"Comma-separated engines to run ({', '.join(ENGINES)}).")
    b.add_argument("--mode", choices=["kfold", "transductive"], default=None)
    b.add_argument("--n-jobs", type=int, default=None, help=N_JOBS_HELP)
    b.add_argument("--quick", action="store_true", help="Cheap smoke-test settings.")
    b.add_argument("--seed", type=int, default=None)
    b.set_defaults(func=cmd_benchmark)
