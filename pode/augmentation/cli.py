"""``pode synthesis augment ...`` subcommands.

  * ``benchmark`` - fit a TVAE inside every split, augment, refit prognostic
                    models, score on real held-out (and external) patients.
  * ``generate``  - write one augmented training table (real + synthetic rows,
                    tagged by ``_source``) for a chosen strategy and ratio.
  * ``report``    - re-render report.md and plots from a benchmark directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

N_JOBS_HELP = "Max CPU cores to use (default: config n_jobs, 4; -1 = all cores)."


def _prepare(args):
    from ..cli.main import _load_schema
    from ..config import load_config
    from ..data.io import load_dataset
    from ..pipeline import prepare_data
    from .config import load_augmentation_config

    config = load_config(args.config)
    acfg = load_augmentation_config(args.augment_config)
    if getattr(args, "quick", False):
        acfg = acfg.quick()
    _apply_overrides(acfg, args)
    acfg.validate()
    df = load_dataset(args.data)
    if getattr(args, "external", None):
        ext = load_dataset(args.external)
        flag = "_external"
        df = _concat_external(df, ext, flag)
        acfg.external.column, acfg.external.values = flag, ["1"]
    schema = _load_schema(df, config, args.schema)
    # Columns used only for splitting/strata/external must survive prepare_data,
    # which would otherwise drop high-cardinality text columns as identifiers.
    keep = [c for c in [acfg.id_column, acfg.external.column, *acfg.strata, *acfg.subgroups,
                        *acfg.splits.stratify, acfg.splits.group_column] if c]
    data, schema = prepare_data(df, config, schema)
    for c in keep:
        if c in df.columns and c not in data.columns:
            data[c] = df.loc[data.index, c]
    return config, acfg, data.reset_index(drop=True), schema


def _csv(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def _subset(items: list, wanted: list[str], what: str) -> list:
    known = {i.key: i for i in items}
    missing = [w for w in wanted if w not in known]
    if missing:
        raise SystemExit(f"error: unknown {what} {missing}; the config defines {sorted(known)}")
    return [known[w] for w in wanted]


def _apply_overrides(acfg, args) -> None:
    """Command-line overrides, so a scenario can be varied without editing YAML."""
    for name in ("seed", "n_jobs", "device", "draws"):
        if getattr(args, name, None) is not None:
            setattr(acfg, name, getattr(args, name))
    if getattr(args, "folds", None) is not None:
        acfg.splits.folds = args.folds
    if getattr(args, "repeats", None) is not None:
        acfg.splits.repeats = args.repeats
    if getattr(args, "ratios", None):
        acfg.ratios = [float(r) for r in _csv(args.ratios)]
    if getattr(args, "strategies", None):
        acfg.strategies = _subset(acfg.strategies, _csv(args.strategies), "strategies")
    if getattr(args, "models", None):
        acfg.models = _subset(acfg.models, _csv(args.models), "models")


def _concat_external(dev, ext, flag: str):
    """Stack an external-cohort table under the development rows, flagged by ``flag``."""
    import pandas as pd

    return pd.concat([dev.assign(**{flag: "0"}), ext.assign(**{flag: "1"})], ignore_index=True)


def cmd_benchmark(args) -> int:
    from ..cli.main import _log_torch_device
    from ..data.io import ensure_dir
    from .benchmark import run_benchmark

    config, acfg, data, schema = _prepare(args)
    if args.dry_run:
        from .benchmark import plan

        print(plan(data, schema, acfg, args.generator_model))
        return 0
    out_dir = ensure_dir(args.output_dir)
    acfg.to_yaml(out_dir / "augmentation_config_used.yaml")
    _log_torch_device()
    result = run_benchmark(data, schema, config, acfg, out_dir, generator_model=args.generator_model)
    card = result["tables"].get("scorecard")
    if card is not None and not card.empty:
        counts = card["verdict"].value_counts().to_dict()
        print(f"Benchmark complete: {len(card)} model×strategy×ratio configurations {counts} in {out_dir}")
    else:
        print(f"Benchmark complete (no scorecard; see failures) in {out_dir}")
    print(f"See {out_dir / 'report.md'}")
    return 0


def cmd_generate(args) -> int:
    import numpy as np

    from ..data.io import ensure_dir, save_dataset
    from ..imputation.base import limit_cpu
    from .benchmark import Columns, strata_labels
    from .generator import HybridSplitGenerator, SplitGenerator, with_survival_rules
    from .privacy import privacy_metrics
    from .strategies import SOURCE, build_pool, draw
    from .splits import split_external

    config, acfg, data, schema = _prepare(args)
    spec = next((s for s in acfg.strategies if s.key == args.strategy), None)
    if spec is None:
        print(f"error: strategy '{args.strategy}' is not in the config "
              f"({', '.join(s.key for s in acfg.strategies)})", file=sys.stderr)
        return 2
    if args.benchmark_dir:
        _print_verdict(Path(args.benchmark_dir), spec.key, args.ratio)

    dev, ext = split_external(data, acfg)
    if len(ext):
        print(f"note: {len(ext)} external rows are left out of generation and of the output.", file=sys.stderr)
    cols = Columns(dev, schema, acfg)
    out_dir = ensure_dir(args.output_dir)
    with limit_cpu(acfg.n_jobs):
        gen = None
        if spec.uses_tvae and acfg.generator.backend == "hybrid":
            if args.generator_model:
                raise SystemExit("error: --generator-model is a plain TVAE; it cannot stand in for the hybrid backend")
            gen = HybridSplitGenerator.fit(dev, schema, config, acfg, acfg.seed, out_dir / "generator",
                                           endpoint=(cols.time, cols.event))
        elif spec.uses_tvae:
            gen = (SplitGenerator.from_model(args.generator_model, schema, acfg) if args.generator_model else
                   SplitGenerator.fit(dev, schema, config, acfg, acfg.seed, out_dir / "generator"))
        n = int(round(args.ratio * len(dev)))
        pool = build_pool(spec, dev, schema, gen, n, acfg.seed, with_survival_rules(schema, acfg.generator.constraints),
                          cols.outcomes)
        aug, w = draw(pool, dev, n, np.random.default_rng(acfg.seed))
    if acfg.strata:
        aug["_stratum"] = strata_labels(aug, acfg)
    if spec.kind == "weight":
        aug["_weight"] = w
    save_dataset(aug, out_dir / "augmented_training_table.csv")
    synth = aug[aug[SOURCE] != "real"]
    diag = {"strategy": spec.key, "ratio": args.ratio, "real_rows": int((aug[SOURCE] == "real").sum()),
            "added_rows": int(len(synth)), **{k: v for k, v in pool.diagnostics.items()}}
    if spec.uses_tvae and not synth.empty:
        pcols = [c for c in gen.columns if c not in cols.outcomes]
        num = [c for c in pcols if c in gen.schema.types and gen.schema.types[c].is_numeric]
        cat = [c for c in pcols if c not in num]
        diag["privacy_vs_training"] = privacy_metrics(gen.view(dev), dev.iloc[0:0], gen.view(synth) if
                                                      acfg.generator.backend == "hybrid" else synth, num, cat,
                                                      acfg.privacy.rare_k, acfg.privacy.max_synthetic, acfg.seed)
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2, default=float)
    print(f"Wrote {len(aug)} rows ({len(synth)} added by '{spec.key}') to "
          f"{out_dir / 'augmented_training_table.csv'}. The `{SOURCE}` column marks synthetic rows; keep them "
          "out of any evaluation set.")
    return 0


def _print_verdict(bench: Path, strategy: str, ratio: float) -> None:
    import pandas as pd

    path = bench / "scorecard.csv"
    if not path.is_file():
        print(f"warning: no scorecard.csv in {bench}", file=sys.stderr)
        return
    card = pd.read_csv(path)
    hit = card[(card["strategy"] == strategy) & (abs(card["ratio"] - ratio) < 1e-9)]
    if hit.empty:
        print(f"warning: {strategy} at {ratio:g}x was not benchmarked in {bench}", file=sys.stderr)
    for _, r in hit.iterrows():
        msg = f"Benchmark verdict for {r['model']} / {strategy} at {ratio:g}x: {r['verdict']} ({r['reasons']})"
        print(("warning: " if r["verdict"] != "RECOMMENDED" else "") + msg, file=sys.stderr)


def cmd_report(args) -> int:
    """Recompute the derived tables from a run's raw tables, then re-render."""
    import pandas as pd

    from .benchmark import summarise_tables
    from .config import AugmentationConfig
    from .report import write_markdown_report

    d = Path(args.output_dir)
    report = json.loads((d / "report.json").read_text())
    cfg = AugmentationConfig.from_dict(report["config"])
    tables = {p.stem: pd.read_csv(p) for p in d.glob("*.csv")}
    for name in ("failures", "scorecard", "external_paired", "timings", "subgroups", "fidelity", "privacy",
                 "selection_stability", "selection_frequency"):
        tables.setdefault(name, pd.DataFrame())
    tables = summarise_tables(tables, cfg, {s.key for s in cfg.strategies if s.uses_tvae})
    for name, t in tables.items():
        if isinstance(t, pd.DataFrame) and not t.empty:
            t.to_csv(d / f"{name}.csv", index=False)
    write_markdown_report(tables, report, d / "report.md")
    if cfg.plots:
        from .plots import write_plots

        write_plots(tables, d / "plots", cfg)
    print(f"Re-summarised and re-rendered {d / 'report.md'}")
    return 0


def register(syn) -> None:
    """Mount ``augment`` under the ``pode synthesis`` subparsers."""
    p = syn.add_parser("augment", help="TVAE data augmentation for prognostic models + benchmarking.")
    sub = p.add_subparsers(dest="augment_command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--show-warnings", action="store_true",
                        help="Show benign third-party warnings, suppressed by default.")
    common.add_argument("--data", required=True, help="Training table CSV (development rows; may include "
                                                       "external rows flagged by external.column).")
    common.add_argument("--config", default=None,
                        help="PODE YAML config (schema + TVAE hyperparameters, e.g. a promoted pode_config_final.yaml).")
    common.add_argument("--schema", default=None, help="Optional schema YAML.")
    common.add_argument("--augment-config", default=None, help="Augmentation YAML (strategies, models, splits, gates).")
    common.add_argument("--external", default=None,
                        help="Optional external-cohort CSV (same columns), used only as a test set.")
    common.add_argument("--generator-model", default=None,
                        help="Pre-trained SDV model.pkl. benchmark: used ONLY for the external cohort (internal "
                             "splits always fit their own). generate: used instead of fitting.")
    common.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None,
                        help="Torch device for TVAE fits (default auto). Use cpu when the shared GPU is busy.")
    common.add_argument("--n-jobs", type=int, default=None, help=N_JOBS_HELP)
    common.add_argument("--seed", type=int, default=None)
    common.add_argument("--strategies", default=None,
                        help="Comma-separated strategy labels from the config to run (default: all).")
    common.add_argument("--models", default=None,
                        help="Comma-separated model labels from the config to run (default: all).")

    b = sub.add_parser("benchmark", parents=[common], help="Benchmark augmentation strategies on real held-out patients.")
    b.add_argument("--output-dir", default="./pode_augment_benchmark")
    b.add_argument("--quick", action="store_true", help="Cheap smoke-test settings.")
    b.add_argument("--ratios", default=None, help="Comma-separated synthetic/real ratios, e.g. 0.25,0.5,1.")
    b.add_argument("--draws", type=int, default=None, help="Random draws per (split, strategy, ratio).")
    b.add_argument("--folds", type=int, default=None, help="Cross-validation folds.")
    b.add_argument("--repeats", type=int, default=None, help="Cross-validation repeats.")
    b.add_argument("--dry-run", action="store_true",
                   help="Resolve columns, splits and the workload, print them, and exit without fitting.")
    b.set_defaults(func=cmd_benchmark)

    g = sub.add_parser("generate", parents=[common], help="Write an augmented training table.")
    g.add_argument("--strategy", required=True, help="Strategy label from the augmentation config.")
    g.add_argument("--ratio", type=float, required=True, help="Synthetic rows as a multiple of the real rows.")
    g.add_argument("--benchmark-dir", default=None, help="Benchmark output; prints that configuration's verdict.")
    g.add_argument("--output-dir", default="./pode_augmented")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("report", help="Re-render report.md and plots from a benchmark directory.")
    r.add_argument("--output-dir", required=True, help="Benchmark output directory.")
    r.set_defaults(func=cmd_report)
