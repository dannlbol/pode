"""Command-line interface for the metacpg (Signatures) theme.

The subcommands are exposed two ways:
  * standalone via ``main()`` (the ``metacpg-py`` console-script alias), and
  * mounted under the unified ``pode signatures ...`` surface through
    :func:`register` (see ``pode.cli.main``).

Dispatch keys off a ``metacpg_command`` default stashed on each subparser rather
than the subparser's ``dest`` name, so it behaves the same whether it owns the
top-level parser or is nested under ``pode signatures``.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml

from .core import (ProgressReporter, aggregate_run, apply_signature, compare_results, extract_signature,
                   fit_metacpg, format_parity_report, read_config, heartbeat_seconds_from_config,
                   parse_memory_budget, preflight_resources, progress_enabled_from_config, run_pipeline,
                   run_worker, search_features, search_only, submit_slurm, validate_inputs,
                   verify_engine_parity)


def _positive_worker_count(value: str) -> int:
    try:
        workers = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workers must be an integer") from exc
    if workers < 1:
        raise argparse.ArgumentTypeError("workers must be at least 1")
    return workers


def _memory_budget(value: str) -> int:
    try:
        return parse_memory_budget(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _nonnegative_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("heartbeat seconds must be a number") from exc
    if seconds < 0:
        raise argparse.ArgumentTypeError("heartbeat seconds must be non-negative")
    return seconds


def _add_local_resource_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", dest="engine", choices=("exact", "batched", "gpu", "auto"),
                        help="Cox solver engine: exact (default, per-feature scipy), batched (vectorized "
                             "CPU), gpu (CUDA), or auto (gpu if available else batched).")
    parser.add_argument("--workers", "--cores", dest="workers", type=_positive_worker_count,
                        help="Local legacy worker count for this invocation (default: execution.workers).")
    parser.add_argument("--memory-budget", type=_memory_budget,
                        help="Advisory local process-memory budget, for example 8G (default: execution.memory_budget).")
    heartbeat = parser.add_mutually_exclusive_group()
    heartbeat.add_argument("--heartbeat-seconds", type=_nonnegative_seconds,
                           help="Seconds between periodic progress/memory updates (default: 300).")
    heartbeat.add_argument("--no-heartbeat", dest="heartbeat_seconds", action="store_const", const=0,
                           help="Disable periodic progress and memory updates.")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", default=None,
                        help="Suppress Cox progress bars and periodic fit-count updates.")


def register(subparsers) -> None:
    """Attach metacpg subcommands to an ``add_subparsers`` object."""
    for name in ("run", "search", "extract", "validate", "submit"):
        p = subparsers.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--backend", choices=("local", "slurm"), default="local")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--feature-count", type=int)
        p.add_argument("--method")
        if name in {"run", "search", "extract"}:
            _add_local_resource_options(p)
            p.add_argument("--preflight", action="store_true",
                           help="Validate inputs and print local resource estimates without fitting Cox models.")
        p.set_defaults(func=_dispatch, metacpg_command=name)
    p = subparsers.add_parser("worker")
    p.add_argument("--config", required=True)
    p.add_argument("--phase", choices=("search", "extraction"))
    p.add_argument("--feature-count", type=int)
    p.add_argument("--run-id")
    p.add_argument("--task-manifest")
    p.add_argument("--task-index", type=int)
    _add_local_resource_options(p)
    p.add_argument("--preflight", action="store_true",
                   help="Validate inputs and print local resource estimates without fitting Cox models.")
    p.set_defaults(func=_dispatch, metacpg_command="worker")
    p = subparsers.add_parser("aggregate")
    p.add_argument("--config", required=True)
    p.add_argument("--feature-count", type=int)
    p.add_argument("--run-id", required=True)
    p.add_argument("--task-manifest")
    p.set_defaults(func=_dispatch, metacpg_command="aggregate")
    p = subparsers.add_parser("compare")
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--engine", choices=("r", "python"), default="python")
    p.add_argument("--output")
    p.set_defaults(func=_dispatch, metacpg_command="compare")
    p = subparsers.add_parser("verify-parity",
                              help="Run the signatures workflow under several Cox engines and "
                                   "report whether they agree (proves CPU/GPU give the same result).")
    p.add_argument("--config", required=True)
    p.add_argument("--feature-count", type=int)
    p.add_argument("--engines", default="exact,batched,gpu",
                   help="Comma-separated engines to compare; the first is the reference (default: exact,batched,gpu).")
    p.add_argument("--tolerance", type=float, default=1e-6,
                   help="Max allowed absolute coefficient difference vs the reference engine (default: 1e-6).")
    p.add_argument("--workers", "--cores", dest="workers", type=_positive_worker_count,
                   help="Worker count for the exact reference run (default: execution.workers).")
    p.add_argument("--output", help="Optional path to write the JSON parity report.")
    p.set_defaults(func=_dispatch, metacpg_command="verify-parity")
    p = subparsers.add_parser("apply", help="Apply a saved legacy meta-CpG model to a new "
                                            "methylation matrix and write per-patient scores.")
    p.add_argument("--model", required=True, help="Path to a saved model.json.")
    p.add_argument("--matrix", required=True, help="New methylation matrix (CSV/TSV) with the signature CpGs.")
    p.add_argument("--orientation", choices=("features_by_samples", "samples_by_features"),
                   help="Override the matrix orientation (default: the model's recorded input).")
    p.add_argument("--scale", choices=("beta", "m_value"),
                   help="Override the input scale (default: the model's recorded input).")
    p.add_argument("--feature-id-column", help="Override the feature-id column (default: model's recorded input).")
    p.add_argument("--separator", help="Override the field separator (default: inferred from extension).")
    p.add_argument("--allow-missing", action="store_true",
                   help="Score over the present signature CpGs instead of failing when some are absent.")
    p.add_argument("--output", help="Destination CSV (default: applied_scores.csv beside the matrix).")
    p.set_defaults(func=_dispatch, metacpg_command="apply")
    p = subparsers.add_parser("methods", help="List schema-v2 signature methods and availability.")
    p.set_defaults(func=_dispatch, metacpg_command="methods")
    p = subparsers.add_parser("benchmark", help="Benchmark configured schema-v2 signature methods.")
    p.add_argument("--config", required=True)
    p.add_argument("--preset", choices=("quick", "balanced", "exhaustive"), default="quick")
    p.set_defaults(func=_dispatch, metacpg_command="benchmark")
    p = subparsers.add_parser("predict", help="Apply a persisted portable schema-v2 signature.")
    p.add_argument("--config", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir")
    p.set_defaults(func=_dispatch, metacpg_command="predict")


def _schema_version(path: str) -> int:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    return int(value.get("schema_version", 1))


def _advanced_dispatch(args) -> int:
    from .advanced import (benchmark_methods, load_advanced_data, method_catalog,
                           predict_signature, read_advanced_config, run_advanced_pipeline)

    def with_progress(operation):
        reporter = ProgressReporter(heartbeat_seconds=0)
        try:
            return operation(reporter.stage)
        finally:
            reporter.close()

    command = args.metacpg_command
    if command == "methods":
        result = method_catalog()
    elif command == "benchmark":
        value = with_progress(lambda progress: benchmark_methods(args.config, args.preset, progress=progress))
        result = {"output_dir": value["output_dir"],
                  "primary_metric": value["primary_metric"],
                  "ranking": value["ranking"],
                  "artifacts": value["artifacts"],
                  "leaderboard": value["leaderboard"].to_dict(orient="records")}
    elif command == "predict":
        scores = with_progress(lambda progress: predict_signature(args.model_dir, args.config, args.output_dir,
                                                                  progress=progress))
        result = {"rows": len(scores), "endpoints": sorted(scores["endpoint"].unique().tolist()),
                  "output_dir": str(Path(args.output_dir) if args.output_dir else Path(read_advanced_config(args.config)["output_dir"]) / "python" / "predictions")}
    elif command == "worker":
        if args.task_manifest is None or args.task_index is None:
            raise ValueError("schema-v2 worker requires --task-manifest and --task-index")
        from .advanced.slurm import run_advanced_worker
        result = with_progress(lambda progress: run_advanced_worker(
            args.config, args.task_manifest, args.task_index, progress=progress))
    elif command == "aggregate":
        if args.task_manifest is None:
            raise ValueError("schema-v2 aggregate requires --task-manifest")
        from .advanced.slurm import aggregate_advanced_tasks
        result = with_progress(lambda progress: aggregate_advanced_tasks(
            args.config, args.task_manifest, args.run_id, progress=progress))
    else:
        cfg = read_advanced_config(args.config)
        requested_method = getattr(args, "method", None)
        if requested_method:
            if requested_method not in cfg["methods"]["enabled"]:
                raise ValueError(f"method {requested_method!r} is not listed in methods.enabled")
            if getattr(args, "backend", "local") == "slurm" or command == "submit":
                cfg["methods"]["enabled"] = [requested_method]
        if getattr(args, "feature_count", None) is not None:
            cfg.setdefault("selection", {})["feature_counts"] = [int(args.feature_count)]
        if command == "validate" or (getattr(args, "dry_run", False)
                                     and getattr(args, "backend", "local") == "local"
                                     and command in {"run", "search", "extract"}):
            data = with_progress(lambda progress: load_advanced_data(cfg))
            result = {"valid": True, "samples": len(data.sample_ids), "blocks": list(data.blocks)}
        elif getattr(args, "backend", "local") == "slurm" or command == "submit":
            from .advanced.slurm import submit_advanced_slurm
            result = with_progress(lambda progress: submit_advanced_slurm(
                cfg, getattr(args, "dry_run", False), progress=progress))
        elif command in {"run", "search", "extract"}:
            value = with_progress(lambda progress: run_advanced_pipeline(cfg, requested_method,
                                                                          progress=progress))
            result = {"output_dir": value.get("output_dir"), "method": value["method"],
                      "maturity": value["maturity"], "elapsed_seconds": value["elapsed_seconds"]}
        else:
            raise ValueError(f"command {command} is not available for schema-v2 signatures")
    print(json.dumps(result, indent=2, default=str))
    return 0


def _dispatch(args) -> int:
    command = args.metacpg_command
    if command == "methods":
        return _advanced_dispatch(args)
    if command in {"benchmark", "predict"}:
        return _advanced_dispatch(args)
    if command == "apply":
        summary = apply_signature(args.model, args.matrix, orientation=args.orientation,
                                  scale=args.scale, feature_id_column=args.feature_id_column,
                                  separator=args.separator, allow_missing=args.allow_missing,
                                  output=args.output)
        print(json.dumps(summary, indent=2, default=str))
        return 0
    if command == "verify-parity":
        engines = [e.strip() for e in args.engines.split(",") if e.strip()]
        report = verify_engine_parity(args.config, feature_count=args.feature_count,
                                      engines=engines, tolerance=args.tolerance, workers=args.workers)
        print(format_parity_report(report))
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2, default=str) + "\n")
        return 0 if report["overall_pass"] else 1
    if command != "compare" and getattr(args, "config", None) and _schema_version(args.config) == 2:
        if (getattr(args, "workers", None) is not None or getattr(args, "memory_budget", None) is not None
                or getattr(args, "heartbeat_seconds", None) is not None
                or getattr(args, "show_progress", None) is not None
                or getattr(args, "engine", None) is not None
                or getattr(args, "preflight", False)):
            raise ValueError("local resource options and --preflight are available only for schema-v1 legacy metaCpG runs")
        return _advanced_dispatch(args)
    if command == "compare":
        result = compare_results(args.reference, args.candidate, args.engine)
        text = json.dumps(result, indent=2)
        if args.output:
            output = Path(args.output); output.write_text(text + "\n")
            rows = ["# metaCpG parity report", "", f"Overall: **{'PASS' if result['passed'] else 'FAIL'}**", "",
                    "| Metric | Observed | Threshold | Pass |", "|---|---:|---:|:---:|"]
            for name, passed in result["checks"].items():
                observed = result["metrics"][name]; threshold = result["thresholds"][name]
                relation = ">=" if name in ("feature_overlap", "risk_spearman") else "<="
                rows.append(f"| {name} | {observed:.4g} | {relation} {threshold:.4g} | {'yes' if passed else 'no'} |")
            rows += ["", "Additional coefficient diagnostics are recorded in the JSON report."]
            output.with_suffix(".md").write_text("\n".join(rows) + "\n")
            try:
                os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "metacpg-matplotlib"))
                import matplotlib.pyplot as plt
                import pandas as pd
                rs = pd.read_csv(Path(args.reference) / "risk_scores.csv")
                cs = pd.read_csv(Path(args.candidate) / "risk_scores.csv")
                merged = rs.merge(cs, on="sample_id", suffixes=("_reference", "_candidate"))
                fig, ax = plt.subplots(figsize=(5, 5)); ax.scatter(merged.risk_score_reference, merged.risk_score_candidate, s=12, alpha=.7)
                ax.set(xlabel="Reference risk score", ylabel="Candidate risk score", title="Risk-score agreement"); fig.tight_layout()
                fig.savefig(output.with_suffix(".png"), dpi=150); plt.close(fig)
            except Exception:
                pass
        print(text)
        return 0 if result["passed"] else 1
    cfg = read_config(args.config)
    requested_workers = getattr(args, "workers", None)
    requested_memory_budget = getattr(args, "memory_budget", None)
    requested_heartbeat = getattr(args, "heartbeat_seconds", None)
    requested_progress = getattr(args, "show_progress", None)
    requested_engine = getattr(args, "engine", None)
    preflight = getattr(args, "preflight", False)
    if (requested_workers is not None or requested_memory_budget is not None
            or requested_heartbeat is not None or requested_progress is not None
            or requested_engine is not None):
        if getattr(args, "backend", "local") == "slurm":
            raise ValueError("local resource options are local-only; use execution.slurm resources for SLURM allocation")
        execution = cfg.setdefault("execution", {})
        if requested_workers is not None:
            execution["workers"] = requested_workers
        if requested_memory_budget is not None:
            execution["memory_budget"] = requested_memory_budget
        if requested_heartbeat is not None:
            execution["heartbeat_seconds"] = requested_heartbeat
        if requested_progress is not None:
            execution["progress"] = requested_progress
        if requested_engine is not None:
            execution["engine"] = requested_engine
    if preflight and getattr(args, "backend", "local") == "slurm":
        raise ValueError("--preflight is local-only; SLURM resource estimates come from execution.slurm")
    progress = ProgressReporter(heartbeat_seconds=heartbeat_seconds_from_config(cfg),
                                show_progress=progress_enabled_from_config(cfg))
    try:
        if preflight:
            progress.stage("Preflight: validating inputs and estimating resources")
            result = preflight_resources(cfg, feature_count=getattr(args, "feature_count", None))
            progress.stage("Preflight complete")
        elif command == "worker":
            result = run_worker(cfg, args.phase, args.feature_count, args.run_id, progress)
        elif command == "aggregate":
            if args.feature_count is None:
                raise ValueError("legacy aggregate requires --feature-count")
            result = aggregate_run(cfg, args.run_id, args.feature_count, progress)
            result = {"output_dir": result.get("output_dir"), "performance": result.get("performance"),
                      "weight_refit": result.get("weight_refit")}
        elif command == "validate":
            progress.stage("Validating inputs")
            validate_inputs(cfg); progress.stage("Validation complete"); result = {"valid": True}
        elif command == "submit":
            result = submit_slurm(cfg, args.dry_run)
        elif command == "search":
            outcome = search_only(cfg, progress=progress)
            result = {"output_dir": outcome.get("output_dir"), "phase": outcome.get("phase"),
                      "next_step": outcome.get("next_step"), "message": outcome.get("message")}
        elif command == "extract":
            outcome = extract_signature(cfg, feature_count=args.feature_count, progress=progress)
            result = {"output_dir": outcome.get("output_dir"), "performance": outcome.get("performance"),
                      "weight_refit": outcome.get("weight_refit")}
        else:
            result = run_pipeline(cfg, args.backend, args.dry_run, args.feature_count, progress)
            result = {"output_dir": result.get("output_dir"), "performance": result.get("performance"),
                      "weight_refit": result.get("weight_refit"),
                      "phase": result.get("phase"), "message": result.get("message"),
                      "next_step": result.get("next_step"),
                      "scripts": result.get("scripts"), "submitted": result.get("submitted"),
                      "run_id": result.get("run_id"), "job_ids": result.get("job_ids")}
    finally:
        progress.close()
    print(json.dumps(result, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="metacpg-py")
    sub = parser.add_subparsers(dest="command", required=True)
    register(sub)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
