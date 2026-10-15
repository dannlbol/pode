"""``pode federated benchmark`` and ``pode federated report``."""

from __future__ import annotations

from .cli import _common, _configs, _federation_flags, _log, load_federation


def cmd_benchmark(args) -> int:
    from ..cli.main import _log_torch_device
    from ..data.io import ensure_dir
    from ..privacy.config import load_privacy_config
    from .benchmark import Benchmark, run_benchmark

    config, fcfg = _configs(args)
    if args.conditions:
        fcfg.evaluation.conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if args.repeats is not None:
        fcfg.evaluation.repeats = args.repeats
    if args.loso:
        fcfg.sites.loso = [s.strip() for s in args.loso.split(",") if s.strip()]
    fcfg.validate()
    pcfg = load_privacy_config(args.privacy_config or fcfg.evaluation.privacy_config)
    pcfg.n_jobs = fcfg.n_jobs
    if args.quick:
        pcfg = pcfg.quick()
    _, schema, cols, _, data = load_federation(args, config, fcfg)
    if args.dry_run:
        b = Benchmark(data, schema, cols, config.synthesizer, fcfg, pcfg, config.data.censor_time_threshold, _log)
        print(f"sites: {', '.join(f'{s}={len(b.frames[s])}' for s in b.site_names)}")
        print(f"generator columns ({len(cols)}): {', '.join(cols)}")
        print(f"endpoint: {b.ep.time} / {b.ep.event}, horizon {b.ep.horizon:.4g}")
        print(f"model features: numeric {b.ep.numeric}; categorical {b.ep.categorical}")
        print(f"conditions: {fcfg.evaluation.conditions}; repeats {fcfg.evaluation.repeats}; LOSO {fcfg.sites.loso}")
        return 0
    _log_torch_device()
    tables = run_benchmark(data, schema, cols, config.synthesizer, fcfg, pcfg, ensure_dir(args.output_dir),
                           config.data.censor_time_threshold, _log)
    sc = tables.get("scorecard")
    if sc is not None and len(sc):
        print(sc[["condition", "verdict", "c_harrell", "delta_vs_central_tvae", "membership_auc_max"]]
              .to_string(index=False))
    print(f"See {args.output_dir}/report.md")
    return 0


def cmd_report(args) -> int:
    import json
    from pathlib import Path

    import pandas as pd

    from .config import load_federated_config
    from .report import summarise, write_report

    out = Path(args.output_dir)
    cfg = load_federated_config(out / "federated_config_used.yaml")
    tables = {n: pd.read_csv(out / f"{n}.csv") if (out / f"{n}.csv").is_file() and (out / f"{n}.csv").stat().st_size > 1
              else pd.DataFrame() for n in ("results", "pairs", "generators", "loso")}
    tables.update(summarise(tables, cfg))
    for name in ("summary", "by_site", "deltas", "privacy", "scorecard"):
        if name in tables:
            tables[name].to_csv(out / f"{name}.csv", index=False)
    meta = json.loads((out / "benchmark_meta.json").read_text())
    print(f"Wrote {write_report(tables, cfg, meta, out)}")
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("benchmark", help="Benchmark federated vs local vs central generators "
                                                "(fidelity, utility on real held-out patients, privacy).")
    _common(p)
    _federation_flags(p)
    p.add_argument("--privacy-config", default=None, help="Privacy YAML for the per-site assessments.")
    p.add_argument("--conditions", default=None, help="Comma-separated conditions (see configs/federated/).")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--loso", default=None, help="Comma-separated sites for the leave-one-site-out arm.")
    p.add_argument("--dry-run", action="store_true", help="Show what the config resolves to, fit nothing.")
    p.add_argument("--quick", action="store_true", help="Tiny settings for a smoke test.")
    p.add_argument("--output-dir", default="./pode_federated_benchmark")
    p.set_defaults(func=cmd_benchmark)

    r = subparsers.add_parser("report", help="Re-render a finished benchmark's report.")
    r.add_argument("--output-dir", required=True)
    r.set_defaults(func=cmd_report)
