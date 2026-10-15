"""Command-line interface for PODE.

Subcommands:
  * ``train``    - train a TVAE and save a reusable model.
  * ``generate`` - sample synthetic data, optionally training first.
  * ``tune``     - run a hyperparameter grid search.
  * ``evaluate`` - score an existing synthetic dataset against real data.
  * ``schema``   - print the auto-detected schema for a dataset.
  * ``seeds``    - inspect seed-log files.

Heavy imports (SDV/torch) are deferred into the handlers so ``--help`` and the
``schema``/``seeds`` commands stay fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_common_io(parser: argparse.ArgumentParser, data_required: bool = True) -> None:
    parser.add_argument("--data", required=data_required, help="Path to the real/training CSV.")
    parser.add_argument("--config", default=None, help="Path to a PODE YAML config.")
    parser.add_argument("--schema", default=None, help="Optional schema YAML to load.")


def _load_schema(df, config, schema_path):
    from ..schema.schema import DatasetSchema

    if schema_path:
        return DatasetSchema.from_yaml(schema_path)
    return DatasetSchema.from_dataframe(df, config.schema)


def cmd_schema(args) -> int:
    from ..config import load_config
    from ..data.io import load_dataset
    from ..pipeline import prepare_data

    config = load_config(args.config)
    df = load_dataset(args.data)
    schema = _load_schema(df, config, args.schema)
    if args.prepared:
        _, schema = prepare_data(df, config, schema)
    print(json.dumps(schema.to_dict(), indent=2))
    if args.out:
        schema.to_yaml(args.out)
        print(f"Schema written to {args.out}", file=sys.stderr)
    return 0


def _load_conditions(args, num_samples):
    if getattr(args, "conditions_config", None):
        from ..generation.conditional import load_conditions_config

        return load_conditions_config(args.conditions_config, num_samples)
    return getattr(args, "conditions", None)


def _log_torch_device() -> None:
    """Print the PyTorch runtime device status for generation backends."""
    try:
        import torch
    except ImportError:
        print("PyTorch runtime: not installed", file=sys.stderr)
        return

    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda or "none"
    print(
        f"PyTorch runtime: cuda_available={cuda_available} "
        f"cuda_version={cuda_version} device_count={torch.cuda.device_count()}",
        file=sys.stderr,
    )
    if cuda_available:
        print(f"Using CUDA device 0: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    else:
        print("Using CPU backend", file=sys.stderr)


def cmd_train(args) -> int:
    from ..config import load_config
    from ..data.io import ensure_dir, load_dataset
    from ..generation.synthesizer import TVAEGenerator
    from ..pipeline import prepare_data
    from ..reproducibility.seeds import SeedTracker, set_all_seeds

    config = load_config(args.config)
    df = load_dataset(args.data)
    schema = _load_schema(df, config, args.schema)
    real, schema = prepare_data(df, config, schema)

    out_dir = ensure_dir(args.output_dir)
    model_path = Path(args.model_out) if args.model_out else out_dir / "model.pkl"
    schema_path = Path(args.schema_out) if args.schema_out else out_dir / "schema.yaml"
    if model_path.parent:
        ensure_dir(model_path.parent)
    if schema_path.parent:
        ensure_dir(schema_path.parent)

    tracker = SeedTracker(mode="new")
    seed = args.seed if args.seed is not None else tracker.get_seed("training")
    set_all_seeds(seed)
    _log_torch_device()

    generator = TVAEGenerator(schema, config.synthesizer).fit(real, seed=seed)
    generator.save(model_path)
    schema.to_yaml(schema_path)
    tracker.set_seed("training", seed)
    tracker.save_seeds(out_dir)
    print(f"Wrote trained model to {model_path}")
    print(f"Wrote prepared schema to {schema_path}")
    return 0


def cmd_generate(args) -> int:
    from ..config import load_config
    from ..data.io import ensure_dir, load_dataset, save_dataset
    from ..pipeline import generate_synthetic, prepare_data
    from ..reproducibility.seeds import SeedTracker, set_all_seeds
    from ..schema.schema import DatasetSchema

    config = load_config(args.config)
    if args.model:
        if not args.schema:
            print("error: --schema is required when using --model", file=sys.stderr)
            return 2
        if args.evaluate and not args.data:
            print("error: --data is required with --evaluate when using --model", file=sys.stderr)
            return 2
        if args.impute and not args.data:
            print("error: --data is required with --impute when using --model", file=sys.stderr)
            return 2
    elif not args.data:
        print("error: --data is required unless --model is provided", file=sys.stderr)
        return 2

    out_dir = ensure_dir(args.output_dir)
    tracker = SeedTracker(mode="new")
    seed = args.seed if args.seed is not None else tracker.get_seed("generation")
    set_all_seeds(seed)

    conditions = _load_conditions(args, args.num_samples)

    if args.model:
        from ..generation.synthesizer import TVAEGenerator

        schema = DatasetSchema.from_yaml(args.schema)
        generator = TVAEGenerator.load(args.model, schema=schema)
        synthetic = generator.sample(
            args.num_samples,
            conditions=conditions,
            seed=seed,
            censor_threshold=config.data.censor_time_threshold,
        )
        if args.impute:
            from ..data.imputation import impute_synthetic_nans

            real, schema = prepare_data(load_dataset(args.data), config, schema)
            synthetic = impute_synthetic_nans(synthetic, real, schema)
        elif args.data:
            real, schema = prepare_data(load_dataset(args.data), config, schema)
        else:
            real = None
    else:
        df = load_dataset(args.data)
        schema = _load_schema(df, config, args.schema)
        real, schema = prepare_data(df, config, schema)
        synthetic, _ = generate_synthetic(
            real, config, args.num_samples, schema=schema,
            conditions=conditions, seed=seed, impute=args.impute,
        )

    save_dataset(synthetic, out_dir / "synthetic_data.csv")
    tracker.set_seed("generation", seed)
    tracker.save_seeds(out_dir)
    print(f"Wrote {len(synthetic)} synthetic rows to {out_dir/'synthetic_data.csv'}")

    if args.evaluate:
        if real is None:
            print("error: --data is required with --evaluate", file=sys.stderr)
            return 2
        _run_evaluation(real, synthetic, schema, config, out_dir)
    return 0


def cmd_evaluate(args) -> int:
    from ..config import load_config
    from ..data.io import ensure_dir, load_dataset
    from ..pipeline import prepare_data

    config = load_config(args.config)
    df = load_dataset(args.data)
    schema = _load_schema(df, config, args.schema)
    real, schema = prepare_data(df, config, schema)
    synthetic = load_dataset(args.synthetic)
    _run_evaluation(real, synthetic, schema, config, ensure_dir(args.output_dir))
    return 0


def _run_evaluation(real, synthetic, schema, config, out_dir) -> None:
    from ..evaluation.objective import compute_composite_score
    from ..evaluation.plots import plot_correlation_comparison, plot_marginals
    from ..evaluation.survival import plot_kaplan_meier

    scored = compute_composite_score(real, synthetic, schema, config.objective)
    with open(out_dir / "fidelity_scores.json", "w") as f:
        json.dump(scored, f, indent=2, default=float)
    print("Fidelity:", json.dumps(scored, indent=2, default=float))

    plot_marginals(real, synthetic, schema, out_dir / "plots" / "marginals")
    plot_correlation_comparison(real, synthetic, schema, out_dir / "plots")
    pair = schema.primary_survival_pair()
    if pair is not None:
        plot_kaplan_meier(
            real, synthetic, pair.time_col, pair.event_col,
            output_path=out_dir / "plots" / "kaplan_meier.png",
        )


def cmd_tune(args) -> int:
    from ..config import load_config
    from ..data.io import ensure_dir, load_dataset
    from ..pipeline import prepare_data
    from ..tuning.grid_search import run_grid_search

    config = load_config(args.config)
    df = load_dataset(args.data)
    schema = _load_schema(df, config, args.schema)
    real, schema = prepare_data(df, config, schema)
    out_dir = ensure_dir(args.output_dir)
    _log_torch_device()
    results = run_grid_search(
        real, schema, config.tuning, config.objective, output_dir=out_dir
    )
    print(f"Grid search complete: {len(results)} combinations. Results in {out_dir}")
    return 0


def cmd_seeds(args) -> int:
    from ..reproducibility.seeds import SeedTracker

    files = SeedTracker.list_seed_files(args.dir)
    if not files:
        print("No seed-log files found.")
        return 0
    for f in files:
        print(f)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pode", description="Pediatric Oncology Discovery Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    _build_synthesis(sub)
    _mount_theme(sub, "signatures", "metacpg.cli",
                 "prognostic CpG signature extraction (metacpg).")
    _mount_theme(sub, "prognosis", "prognostk.cli",
                 "prognostic metrics and figures (prognostk).")
    _mount_theme(sub, "mantis", "pode_mantis.cli",
                 "MANTIS-MB survival-model inference wrapper (pode_mantis).")
    _mount_theme(sub, "foundation", "pode_foundation.cli",
                 "methylation-array masked-autoencoder pretraining (pode_foundation).")
    _mount_theme(sub, "federated", "pode.federated.cli",
                 "federated learning across sites: TVAE, gated exchange, DP, MetaCPG signatures.")

    return parser


def _build_synthesis(subparsers) -> None:
    """Mount the Synthesis theme: the core TVAE commands plus the synthvae omics
    front end, all under ``pode synthesis ...``.

    Core commands (schema/train/generate/evaluate/tune/seeds) live here because
    they *are* synthetic-data generation; the omics hybrid/mmvae/handoff commands
    come from ``synthvae.cli``.
    """
    theme = subparsers.add_parser("synthesis", help="synthetic clinical & omics data generation.")
    syn = theme.add_subparsers(dest="synthesis_command", required=True)

    # Shared flag available on the core synthesis commands (e.g. `--show-warnings`).
    warn = argparse.ArgumentParser(add_help=False)
    warn.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show benign third-party (SDV/ctgan/lifelines) warnings, suppressed by default.",
    )

    p_schema = syn.add_parser("schema", parents=[warn], help="Print the detected schema.")
    _add_common_io(p_schema)
    p_schema.add_argument("--out", default=None, help="Write schema YAML here.")
    p_schema.add_argument(
        "--prepared",
        action="store_true",
        help="Print/save the schema after PODE's data preparation pipeline.",
    )
    p_schema.set_defaults(func=cmd_schema)

    p_train = syn.add_parser("train", parents=[warn], help="Train and save a reusable model.")
    _add_common_io(p_train)
    p_train.add_argument("--output-dir", default="./pode_model")
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--model-out", default=None, help="Model path; defaults to output-dir/model.pkl.")
    p_train.add_argument("--schema-out", default=None, help="Schema path; defaults to output-dir/schema.yaml.")
    p_train.set_defaults(func=cmd_train)

    p_gen = syn.add_parser("generate", parents=[warn], help="Generate synthetic data.")
    _add_common_io(p_gen, data_required=False)
    p_gen.add_argument("--model", default=None, help="Reusable model file from `pode synthesis train`.")
    p_gen.add_argument("--num-samples", type=int, default=1000)
    p_gen.add_argument("--output-dir", default="./pode_output")
    conditions = p_gen.add_mutually_exclusive_group()
    conditions.add_argument("--conditions", default=None, help="Condition string, e.g. '0.5:COL=1;0.5:COL=0'.")
    conditions.add_argument("--conditions-config", default=None, help="YAML condition config.")
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.add_argument("--impute", action="store_true", help="Impute NaNs in synthetic output.")
    p_gen.add_argument("--evaluate", action="store_true", help="Also run fidelity evaluation.")
    p_gen.set_defaults(func=cmd_generate)

    p_eval = syn.add_parser("evaluate", parents=[warn], help="Evaluate an existing synthetic dataset.")
    _add_common_io(p_eval)
    p_eval.add_argument("--synthetic", required=True, help="Path to the synthetic CSV.")
    p_eval.add_argument("--output-dir", default="./pode_eval")
    p_eval.set_defaults(func=cmd_evaluate)

    p_tune = syn.add_parser("tune", parents=[warn], help="Run a hyperparameter grid search.")
    _add_common_io(p_tune)
    p_tune.add_argument("--output-dir", default="./pode_tuning")
    p_tune.set_defaults(func=cmd_tune)

    p_seeds = syn.add_parser("seeds", parents=[warn], help="List seed-log files.")
    p_seeds.add_argument("--dir", default=".", help="Directory to search.")
    p_seeds.set_defaults(func=cmd_seeds)

    # Generative imputation (train / apply / benchmark). Degrade gracefully so
    # a problem here never takes down the core synthesis commands.
    try:
        from ..imputation import cli as _impute_cli
        _impute_cli.register(syn)
    except Exception as exc:  # pragma: no cover - only when optional deps missing
        print(f"warning: synthesis impute commands unavailable: {exc}", file=sys.stderr)

    # Generative data augmentation for prognostic models (benchmark / generate /
    # report). Same graceful degradation as imputation.
    try:
        from ..augmentation import cli as _augment_cli
        _augment_cli.register(syn)
    except Exception as exc:  # pragma: no cover - only when optional deps missing
        print(f"warning: synthesis augment commands unavailable: {exc}", file=sys.stderr)

    # Privacy-preservation assessment of synthetic data (assess / audit). Same
    # graceful degradation as imputation.
    try:
        from ..privacy import cli as _privacy_cli
        _privacy_cli.register(syn)
    except Exception as exc:  # pragma: no cover - only when optional deps missing
        print(f"warning: synthesis privacy commands unavailable: {exc}", file=sys.stderr)

    # Omics front end (hybrid / mmvae / bundle / promote / decode). Degrade
    # gracefully if its optional backends are missing.
    try:
        import synthvae.cli as _synthvae_cli
        _synthvae_cli.register(syn)
    except Exception as exc:  # pragma: no cover - only when optional deps missing
        print(f"warning: synthesis omics commands unavailable: {exc}", file=sys.stderr)


def _mount_theme(subparsers, name: str, module_path: str, help_text: str) -> None:
    """Mount a suite theme's subcommands under ``pode <name> ...``.

    The component module's ``register(add_subparsers)`` populates the namespace.
    Import failures (e.g. an uninstalled optional backend) degrade gracefully to a
    stub whose handler explains what to install, rather than breaking the whole CLI.
    """
    import importlib

    theme = subparsers.add_parser(name, help=help_text)
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # pragma: no cover - only when optional deps missing
        theme.add_subparsers(dest=f"{name}_command", required=False)

        def _stub(args, _name=name, _exc=exc) -> int:
            print(f"error: the '{_name}' theme is unavailable: {_exc}", file=sys.stderr)
            print(f"       install its extra with: pip install 'pode[{_name}]'", file=sys.stderr)
            return 1

        theme.set_defaults(func=_stub)
        return

    theme_sub = theme.add_subparsers(dest=f"{name}_command", required=True)
    module.register(theme_sub)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from ..utils.warnings import configure_warnings

    configure_warnings(show=getattr(args, "show_warnings", False))
    rc = args.func(args)
    # Core handlers return int exit codes; the mounted theme handlers may return
    # a result object or None, which we treat as success.
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
