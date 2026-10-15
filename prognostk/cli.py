"""Command-line interface for the prognostk (Prognosis) theme.

Exposes prognostk's config-driven metrics and figures as subcommands. Mounted
under the unified ``pode prognosis ...`` surface (see ``pode.cli.main``) and also
runnable standalone via ``python -m prognostk.cli`` / the ``register`` hook.

Subcommands:
  * ``metrics``      - fit a Cox model, report discrimination, write risk scores.
  * ``cutoffs``      - optimise descending risk thresholds on a predictor.
  * ``validate``     - external-validate a frozen Cox model on a second cohort.
  * ``km``           - Kaplan-Meier figure (with number-at-risk).
  * ``distribution`` - violin/box/ridge distribution figure.
  * ``forest``       - hazard-ratio forest figure.
  * ``calibration``  - calibration figure at the config horizon.
  * ``oncoplot``     - annotated oncoplot/heatmap figure.

Heavy imports (pandas, lifelines, matplotlib) are deferred into the handlers so
``--help`` stays fast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_cfg_data(args):
    """Load config + primary data CSV, applying a --outdir override if given."""
    import pandas as pd

    from .config import load_config

    cfg = load_config(args.config)
    if getattr(args, "outdir", None):
        cfg["figure"]["outdir"] = args.outdir
    data = pd.read_csv(args.data)
    return cfg, data


def _covariates(args):
    return getattr(args, "covariates", None) or None


def _ensure_group_column(data, cfg, group):
    """Derive a missing risk-group column from a fitted Cox model.

    Mirrors the canonical prognostk workflow (fit -> predict survival at horizon ->
    assign_risk_groups) so figures whose configured grouping column is a *derived*
    risk tier (e.g. ``riskGrpBoot``) work directly on a raw cohort.
    """
    if not group or group in data.columns:
        return data
    from .metrics.survival import (assign_risk_groups, fit_cox,
                                   predict_survival_at_time)

    fit = fit_cox(data, cfg)
    pred = predict_survival_at_time(fit, data, cfg=cfg)
    return data.assign(**{group: assign_risk_groups(pred, cfg)})


def cmd_metrics(args) -> int:
    from .metrics.survival import (assign_risk_groups, fit_cox,
                                   predict_survival_at_time)

    cfg, data = _load_cfg_data(args)
    fit = fit_cox(data, cfg, covariates=_covariates(args))
    print(fit.fitter.summary.to_string())
    print(f"\nConcordance (in-sample): {fit.fitter.concordance_index_:.4f}")

    pred = predict_survival_at_time(fit, data, cfg=cfg)
    groups = assign_risk_groups(pred, cfg)
    if args.out:
        import pandas as pd

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = {"pred_survival": pred, "risk_group": groups}
        id_col = cfg["columns"].get("id")
        if id_col and id_col in data.columns:
            cols = {id_col: data[id_col].to_numpy(), **cols}
        pd.DataFrame(cols).to_csv(out, index=False)
        print(f"\nWrote per-patient risk scores to {out}")
    return 0


def cmd_cutoffs(args) -> int:
    from .metrics.cutoffs import optimise_cutoffs

    cfg, data = _load_cfg_data(args)
    pred = args.pred or cfg["columns"].get("score")
    if not pred:
        print("error: --pred (a column name) is required unless columns.score is "
              "set in the config.")
        return 2
    result = optimise_cutoffs(data, pred, cfg)
    print(json.dumps(result, indent=2, default=float))
    return 0


def cmd_validate(args) -> int:
    from .metrics.calibration import external_validate
    from .metrics.survival import fit_cox

    cfg, data = _load_cfg_data(args)
    import pandas as pd

    valid = pd.read_csv(args.valid_data)
    fit = fit_cox(data, cfg, covariates=_covariates(args))
    result = external_validate(valid, fit, cfg)
    print(json.dumps(result, indent=2, default=float))
    return 0


def _figure_handler(plot_name):
    """Build a handler that loads cfg+data and calls a prognostk plot function."""

    def handler(args) -> int:
        cfg, data = _load_cfg_data(args)
        filename = args.out or f"{plot_name}.{cfg['figure']['device']}"
        kwargs = dict(title=getattr(args, "title", None), filename=filename)

        if plot_name == "km":
            from .plots.km import plot_km
            group = args.group or cfg["columns"].get("group")
            data = _ensure_group_column(data, cfg, group)
            plot_km(data, cfg, group_col=group, **kwargs)
        elif plot_name == "distribution":
            from .plots.distribution import plot_distribution
            plot_distribution(data, cfg, group_col=args.group, kind=args.kind, **kwargs)
        elif plot_name == "forest":
            from .plots.forest import plot_forest
            plot_forest(data, cfg, covariates=_covariates(args), **kwargs)
        elif plot_name == "calibration":
            from .metrics.survival import fit_cox, predict_survival_at_time
            from .plots.calibration import plot_calibration
            fit = fit_cox(data, cfg, covariates=_covariates(args))
            pred = predict_survival_at_time(fit, data, cfg=cfg)
            plot_calibration(data, cfg, pred=pred, n_bins=args.bins, **kwargs)
        elif plot_name == "oncoplot":
            import pandas as pd
            from .plots.oncoplot import plot_oncoplot
            matrix = pd.read_csv(args.matrix, index_col=0) if args.matrix else None
            plot_oncoplot(data, cfg, matrix=matrix, order_by=args.order_by, **kwargs)
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(plot_name)

        print(f"Wrote {plot_name} figure to {Path(cfg['figure']['outdir']) / filename}")
        return 0

    return handler


def _add_common_io(p, needs_data=True):
    p.add_argument("--config", required=True, help="Path to a prognostk YAML config.")
    if needs_data:
        p.add_argument("--data", required=True, help="Path to the cohort CSV.")
    p.add_argument("--outdir", default=None, help="Override figure.outdir / output dir.")


def register(subparsers) -> None:
    """Attach prognostk subcommands to an ``add_subparsers`` object."""
    m = subparsers.add_parser("metrics", help="Fit a Cox model; report discrimination.")
    _add_common_io(m)
    m.add_argument("--covariates", nargs="*", help="Override config covariate terms.")
    m.add_argument("--out", default=None, help="Write per-patient risk scores CSV here.")
    m.set_defaults(func=cmd_metrics)

    c = subparsers.add_parser("cutoffs", help="Optimise descending risk thresholds.")
    _add_common_io(c)
    c.add_argument("--pred", default=None, help="Predictor column (default: columns.score).")
    c.set_defaults(func=cmd_cutoffs)

    v = subparsers.add_parser("validate", help="External-validate a frozen Cox model.")
    _add_common_io(v)
    v.add_argument("--valid-data", required=True, help="Validation cohort CSV.")
    v.add_argument("--covariates", nargs="*", help="Override config covariate terms.")
    v.set_defaults(func=cmd_validate)

    km = subparsers.add_parser("km", help="Kaplan-Meier figure.")
    _add_common_io(km)
    km.add_argument("--group", default=None, help="Grouping column (default: config).")
    km.add_argument("--title", default=None)
    km.add_argument("--out", default=None, help="Output filename (in outdir).")
    km.set_defaults(func=_figure_handler("km"))

    dist = subparsers.add_parser("distribution", help="Distribution figure.")
    _add_common_io(dist)
    dist.add_argument("--group", default=None, help="Grouping column.")
    dist.add_argument("--kind", default="violin", choices=("violin", "box", "ridge"))
    dist.add_argument("--title", default=None)
    dist.add_argument("--out", default=None)
    dist.set_defaults(func=_figure_handler("distribution"))

    fo = subparsers.add_parser("forest", help="Hazard-ratio forest figure.")
    _add_common_io(fo)
    fo.add_argument("--covariates", nargs="*", help="Override config covariate terms.")
    fo.add_argument("--title", default=None)
    fo.add_argument("--out", default=None)
    fo.set_defaults(func=_figure_handler("forest"))

    ca = subparsers.add_parser("calibration", help="Calibration figure at horizon.")
    _add_common_io(ca)
    ca.add_argument("--covariates", nargs="*", help="Override config covariate terms.")
    ca.add_argument("--bins", type=int, default=10)
    ca.add_argument("--title", default=None)
    ca.add_argument("--out", default=None)
    ca.set_defaults(func=_figure_handler("calibration"))

    on = subparsers.add_parser("oncoplot", help="Annotated oncoplot/heatmap figure.")
    _add_common_io(on)
    on.add_argument("--matrix", default=None, help="Optional feature matrix CSV (rows=features).")
    on.add_argument("--order-by", default=None, help="Column to order samples by.")
    on.add_argument("--title", default="Oncoplot")
    on.add_argument("--out", default=None)
    on.set_defaults(func=_figure_handler("oncoplot"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prognostk", description="Prognostic metrics and figures (prognostk).")
    sub = parser.add_subparsers(dest="command", required=True)
    register(sub)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
