"""Command-line interface for the mantis theme.

Exposes pode_mantis's config-driven pipeline as subcommands. Mounted under
the unified ``pode mantis ...`` surface (see ``pode.cli.main``) and also
runnable standalone via ``python -m pode_mantis.cli`` / the ``register``
hook.

Subcommands:
  * ``prepare-beta``     - reindex a cohort's beta matrix to the 357,692-CpG order.
  * ``prepare-cnv``      - R/conumee2 CNV feature calling (subprocess to Rscript).
  * ``prepare-survival`` - build the MANTIS-format survival/clinical CSV.
  * ``generate-config``  - assemble a reference-repo config.yaml for a new cohort.
  * ``infer``            - pre-flight checks, then run the reference inference.py.
  * ``evaluate``         - run the reference evaluate.py, then post-flight checks.
  * ``plot``              - optional: run the reference plot_results.py.

Heavy work (pandas/subprocess dispatch) is deferred into the handlers so
``--help`` stays fast.
"""
from __future__ import annotations

import argparse


def _load_cfg(args):
    from .config import load_config

    return load_config(args.config)


def cmd_prepare_beta(args) -> int:
    from .core.prepare_beta import run_prepare_beta

    cfg = _load_cfg(args)
    result = run_prepare_beta(cfg, betas=args.betas, out_dir=args.out_dir)
    print(f"\nWrote {result.n_samples} samples x {result.n_cpgs} CpGs to {result.numeric_path}")
    return 0


def cmd_prepare_cnv(args) -> int:
    from .core.prepare_cnv import run_prepare_cnv

    cfg = _load_cfg(args)
    result = run_prepare_cnv(cfg, out_dir=args.out_dir, n_test=args.n_test)
    print(f"\nWrote CNV features for {len(result.numeric_paths)} cohort(s) to {result.out_dir}")
    return 0


def cmd_prepare_survival(args) -> int:
    from .core.prepare_survival import run_prepare_survival

    cfg = _load_cfg(args)
    result = run_prepare_survival(cfg, clinical=args.clinical, sample_order_path=args.sample_order, out_dir=args.out_dir)
    print(f"\nWrote {result.n_samples} samples to {result.out_path}")
    return 0


def cmd_generate_config(args) -> int:
    from .core.generate_config import run_generate_config

    cfg = _load_cfg(args)
    result = run_generate_config(
        cfg,
        cohort_name=args.cohort,
        beta_path=args.beta,
        cnv_path=args.cnv,
        surv_path=args.surv,
        endpoint=args.endpoint,
        out_path=args.out,
    )
    print(f"\nWrote reference config to {result.out_path}")
    return 0


def cmd_infer(args) -> int:
    from .core.assertions import check_pre_flight
    from .core.run_reference import run_inference

    cfg = _load_cfg(args)
    if args.beta_sample_order and args.survival:
        report = check_pre_flight(args.beta_sample_order, args.survival, cnv_sample_order_path=args.cnv_sample_order)
        if not report.ok:
            report.raise_if_bad()
        print("[pre-flight] beta/CNV/survival sample order agree.")
    else:
        print("[pre-flight] skipped (pass --beta-sample-order and --survival to enable it -- recommended).")

    pred_path = run_inference(cfg, args.mantis_config, mantis_python=args.mantis_python, reference_repo=args.reference_repo)
    print(f"\nPredictions written to {pred_path}")
    return 0


def cmd_evaluate(args) -> int:
    from .core.assertions import check_post_flight
    from .core.run_reference import run_evaluate

    cfg = _load_cfg(args)
    metrics_path = run_evaluate(
        cfg, args.mantis_config, args.pred_path,
        output_metrics=args.output_metrics, mantis_python=args.mantis_python, reference_repo=args.reference_repo,
    )
    if args.survival:
        report = check_post_flight(args.pred_path, args.survival)
        if not report.ok:
            report.raise_if_bad()
        print("[post-flight] predictions row count/order/shape/NaN-explanation all pass.")
    print(f"\nMetrics written to {metrics_path}")
    return 0


def cmd_plot(args) -> int:
    from .core.run_reference import run_plot

    cfg = _load_cfg(args)
    out_dir = run_plot(
        cfg, args.mantis_config, args.metrics_file, args.pred_path, args.out_dir,
        mantis_python=args.mantis_python, reference_repo=args.reference_repo,
    )
    print(f"\nPlots written to {out_dir}")
    return 0


def _add_config(p) -> None:
    p.add_argument("--config", required=True, help="Path to a pode_mantis YAML config.")


def register(subparsers) -> None:
    """Attach pode_mantis subcommands to an ``add_subparsers`` object."""
    pb = subparsers.add_parser("prepare-beta", help="Reindex a cohort's beta matrix to the 357,692-CpG order.")
    _add_config(pb)
    pb.add_argument("--betas", default=None, help="Override config beta.cohort_path.")
    pb.add_argument("--out-dir", default=None, help="Override config beta.output_dir.")
    pb.set_defaults(func=cmd_prepare_beta)

    pc = subparsers.add_parser("prepare-cnv", help="R/conumee2 CNV feature calling (requires Rscript + Bioconductor).")
    _add_config(pc)
    pc.add_argument("--out-dir", default=None, help="Override config cnv.output_dir.")
    pc.add_argument("--n-test", type=int, default=None, help="Subset every Mset to its first N samples (dry run).")
    pc.set_defaults(func=cmd_prepare_cnv)

    ps = subparsers.add_parser("prepare-survival", help="Build the MANTIS-format survival/clinical CSV.")
    _add_config(ps)
    ps.add_argument("--clinical", default=None, help="Override config survival.clinical_path.")
    ps.add_argument("--sample-order", default=None, help="Beta sample-order sidecar to assert row order against.")
    ps.add_argument("--out-dir", default=None, help="Override config survival.output_dir.")
    ps.set_defaults(func=cmd_prepare_survival)

    gc = subparsers.add_parser("generate-config", help="Assemble a reference-repo config.yaml for a new cohort.")
    _add_config(gc)
    gc.add_argument("--cohort", required=True, help="Cohort name (used as the config's test_cohort and path prefix).")
    gc.add_argument("--beta", required=True, help="This cohort's beta_numeric.csv (from prepare-beta).")
    gc.add_argument("--cnv", required=True, help="This cohort's cnv_numeric.csv (from prepare-cnv).")
    gc.add_argument("--surv", required=True, help="This cohort's survival.csv (from prepare-survival).")
    gc.add_argument("--endpoint", choices=("os", "pfs"), default=None, help="Override config reference_config.endpoint.")
    gc.add_argument("--out", default=None, help="Override config reference_config.output_path.")
    gc.set_defaults(func=cmd_generate_config)

    inf = subparsers.add_parser("infer", help="Pre-flight checks, then run the reference inference.py.")
    _add_config(inf)
    inf.add_argument("--mantis-config", required=True, help="Reference-repo config.yaml (from generate-config).")
    inf.add_argument("--beta-sample-order", default=None, help="Beta sample-order sidecar, for the pre-flight check.")
    inf.add_argument("--cnv-sample-order", default=None, help="CNV sample-order sidecar, for the pre-flight check.")
    inf.add_argument("--survival", default=None, help="Survival CSV, for the pre-flight check.")
    inf.add_argument("--mantis-python", default=None, help="Override config paths.mantis_python.")
    inf.add_argument("--reference-repo", default=None, help="Override config paths.reference_repo.")
    inf.set_defaults(func=cmd_infer)

    ev = subparsers.add_parser("evaluate", help="Run the reference evaluate.py, then post-flight checks.")
    _add_config(ev)
    ev.add_argument("--mantis-config", required=True, help="Reference-repo config.yaml.")
    ev.add_argument("--pred-path", required=True, help="Predictions CSV (from infer).")
    ev.add_argument("--output-metrics", default=None, help="Where to write the metrics JSON.")
    ev.add_argument("--survival", default=None, help="Survival CSV, for the post-flight check.")
    ev.add_argument("--mantis-python", default=None, help="Override config paths.mantis_python.")
    ev.add_argument("--reference-repo", default=None, help="Override config paths.reference_repo.")
    ev.set_defaults(func=cmd_evaluate)

    pl = subparsers.add_parser("plot", help="Optional: run the reference plot_results.py.")
    _add_config(pl)
    pl.add_argument("--mantis-config", required=True)
    pl.add_argument("--metrics-file", required=True)
    pl.add_argument("--pred-path", required=True)
    pl.add_argument("--out-dir", required=True)
    pl.add_argument("--mantis-python", default=None)
    pl.add_argument("--reference-repo", default=None)
    pl.set_defaults(func=cmd_plot)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pode_mantis", description="MANTIS-MB survival-model inference wrapper (pode_mantis)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    register(sub)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
