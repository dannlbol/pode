"""``pode federated ...`` subcommands.

  * ``tvae train``  - fit a federated TVAE across sites (a table + ``--site-column``,
                      or a ``--site-manifest`` of per-site CSVs) and save it;
  * ``tvae sample`` - sample synthetic rows from a saved federated TVAE.

Further commands (``exchange``, ``benchmark``, ``report``, ``signature``) are
registered by :func:`register` as they become available.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

N_JOBS_HELP = "Max CPU cores to use (default: config n_jobs, 4; -1 = all cores)."
ROW = "_pode_row_"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---- shared loading --------------------------------------------------------------
def _configs(args):
    from ..config import load_config
    from .config import load_federated_config

    config = load_config(getattr(args, "config", None))
    fcfg = load_federated_config(getattr(args, "federated_config", None))
    if getattr(args, "quick", False):
        fcfg = fcfg.quick()
    for name in ("seed", "n_jobs", "device"):
        if getattr(args, name, None) is not None:
            setattr(fcfg, name, getattr(args, name))
    g = fcfg.generator
    for name, attr in (("rounds", "rounds"), ("local_epochs", "local_epochs"), ("algorithm", "algorithm"),
                       ("weighting", "weighting"), ("personalise_epochs", "personalise_epochs")):
        if getattr(args, name, None) is not None:
            setattr(g, attr, getattr(args, name))
    if getattr(args, "site_column", None):
        fcfg.sites.column = args.site_column
    if getattr(args, "no_secure_aggregation", False):
        g.secure_aggregation = False
    if getattr(args, "dp", False) or getattr(args, "target_epsilon", None) is not None:
        fcfg.dp.enabled = True
        if args.target_epsilon is not None:
            fcfg.dp.target_epsilon = args.target_epsilon
    return config, fcfg.validate()


def prepare_table(df, config, schema=None, keep: list[str] | None = None):
    """PODE's ``prepare_data`` (the agreed data model), keeping ``keep`` columns and row identity."""
    from ..imputation.base import subset_schema
    from ..pipeline import prepare_data

    keep = [c for c in (keep or []) if c in df.columns]
    side = df[keep].copy()
    tagged = df.drop(columns=keep).assign(**{ROW: np.arange(len(df), dtype=float)})
    data, schema = prepare_data(tagged, config, schema)
    rows = data.pop(ROW).astype(int).to_numpy()
    schema = subset_schema(schema, [c for c in schema.columns if c != ROW and c in data.columns])
    data = data.reset_index(drop=True)
    for c in keep:
        data[c] = side[c].to_numpy()[rows]
    return data, schema, rows


def model_columns(data, schema, fcfg, exclude_extra: list[str] | None = None) -> list[str]:
    """Generator columns: the schema's minus identifiers, the site column and ``exclude_columns``."""
    from ..imputation.base import resolve_columns
    from ..schema.column_types import ColumnRole

    cols = [c for c in schema.columns if c in data.columns]
    drop = set(schema.columns_of_role(ColumnRole.IDENTIFIER)) | {fcfg.sites.column} | set(exclude_extra or [])
    drop |= set(resolve_columns(fcfg.generator.exclude_columns, cols))
    return [c for c in cols if c not in drop]


def load_federation(args, config, fcfg):
    """(sites, schema, columns, ledger, pooled prepared table) from CLI arguments."""
    from ..cli.main import _load_schema
    from ..data.io import load_dataset
    from .ledger import DisclosureLedger
    from .sites import load_site_manifest, partition_sites, stack_sites

    ledger = DisclosureLedger()
    if getattr(args, "site_manifest", None):
        raw_sites = load_site_manifest(args.site_manifest)
        df = stack_sites(raw_sites, fcfg.sites.column)
    else:
        if not args.data:
            raise SystemExit("error: give --data (with --site-column) or --site-manifest.")
        df = load_dataset(args.data)
    if fcfg.sites.column not in df.columns:
        raise SystemExit(f"error: site column '{fcfg.sites.column}' not in the data (use --site-column).")
    # The schema is the federation's agreed data model: from --schema, else
    # detected (in simulation) on the pooled table.
    schema = _load_schema(df.drop(columns=[fcfg.sites.column]), config, getattr(args, "schema", None))
    data, schema, _ = prepare_table(df, config, schema, keep=[fcfg.sites.column])
    sites = partition_sites(data, fcfg.sites.column, fcfg.sites.include, fcfg.sites.min_site_n, ledger)
    return sites, schema, model_columns(data, schema, fcfg), ledger, data


# ---- tvae ------------------------------------------------------------------------
def cmd_tvae_train(args) -> int:
    from ..cli.main import _log_torch_device
    from ..data.io import ensure_dir
    from ..imputation.base import limit_cpu
    from .tvae import FederatedTVAE

    config, fcfg = _configs(args)
    sites, schema, cols, ledger, _ = load_federation(args, config, fcfg)
    _log_torch_device()
    _log(f"federated TVAE: {len(sites)} sites ({', '.join(f'{s.name}={s.n}' for s in sites)}), "
         f"{len(cols)} columns, {fcfg.generator.rounds} rounds x {fcfg.generator.local_epochs} local epochs, "
         f"{fcfg.generator.algorithm}{', DP' if fcfg.dp.enabled else ''}")
    with limit_cpu(fcfg.n_jobs):
        model = FederatedTVAE(config.synthesizer, fcfg, schema, config.data.censor_time_threshold)
        model.fit(sites, cols, log=_log)
    out = ensure_dir(args.output_dir)
    model.save(out)
    ledger.save(out / "disclosure_ledger.csv")
    ledger.summary().to_csv(out / "disclosure_summary.csv", index=False)
    if args.num_samples:
        from ..data.io import save_dataset

        save_dataset(model.sample(args.num_samples, seed=fcfg.seed), out / "synthetic.csv")
    for name, p in model.record.privacy.items():
        if p.get("dp"):
            _log(f"  site {name}: epsilon {p['epsilon_total']:.2f} (SGD {p['epsilon_sgd']:.2f} + encoder "
                 f"{p['epsilon_transformer']:.2f}), delta {p['delta']:.1e}")
    print(f"Saved federated TVAE to {out}")
    return 0


def cmd_tvae_sample(args) -> int:
    from ..config import load_config
    from ..data.io import ensure_dir, save_dataset
    from .tvae import FederatedTVAE

    config = load_config(args.config)
    model = FederatedTVAE.load(args.model, config.synthesizer)
    out = ensure_dir(args.output_dir)
    df = model.sample(args.num_samples, seed=args.seed, site=args.site)
    save_dataset(df, out / "synthetic.csv")
    print(f"Wrote {len(df)} rows to {out / 'synthetic.csv'}")
    return 0


# ---- registration ----------------------------------------------------------------
def _common(p, data=True) -> None:
    if data:
        p.add_argument("--data", default=None, help="One table with a site column (simulation mode).")
        p.add_argument("--site-column", default=None, help="Column whose levels are the sites.")
        p.add_argument("--site-manifest", default=None, help="YAML listing per-site CSVs (instead of --data).")
        p.add_argument("--schema", default=None, help="Schema YAML (the agreed data model).")
    p.add_argument("--config", default=None, help="PODE config (TVAE architecture, data preparation).")
    p.add_argument("--federated-config", default=None, help="Federated YAML (see configs/federated/).")
    p.add_argument("--n-jobs", type=int, default=None, help=N_JOBS_HELP)
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None)


def _federation_flags(p) -> None:
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--local-epochs", type=int, default=None)
    p.add_argument("--algorithm", default=None, choices=["fedavg", "fedprox"])
    p.add_argument("--weighting", default=None, choices=["n", "sqrt_n", "uniform", "capped"])
    p.add_argument("--personalise-epochs", type=int, default=None)
    p.add_argument("--no-secure-aggregation", action="store_true")
    p.add_argument("--dp", action="store_true", help="Train with DP-SGD (needs a noise multiplier or target).")
    p.add_argument("--target-epsilon", type=float, default=None, help="DP target epsilon per site (implies --dp).")


def register(subparsers) -> None:
    p_tvae = subparsers.add_parser("tvae", help="Federated TVAE: train / sample.")
    tsub = p_tvae.add_subparsers(dest="tvae_command", required=True)

    p_train = tsub.add_parser("train", help="Fit a federated TVAE across sites and save it.")
    _common(p_train)
    _federation_flags(p_train)
    p_train.add_argument("--output-dir", default="./pode_federated_tvae")
    p_train.add_argument("--num-samples", type=int, default=0, help="Also write this many synthetic rows.")
    p_train.add_argument("--quick", action="store_true", help="Tiny settings for a smoke test.")
    p_train.set_defaults(func=cmd_tvae_train)

    p_sample = tsub.add_parser("sample", help="Sample from a saved federated TVAE.")
    p_sample.add_argument("--model", required=True, help="Directory written by `tvae train`.")
    p_sample.add_argument("--config", default=None, help="PODE config used for training (architecture).")
    p_sample.add_argument("--num-samples", type=int, default=1000)
    p_sample.add_argument("--site", default=None, help="Only rows for this site (needs site_in_model).")
    p_sample.add_argument("--seed", type=int, default=0)
    p_sample.add_argument("--output-dir", default="./pode_federated_sample")
    p_sample.set_defaults(func=cmd_tvae_sample)

    for extra in ("exchange", "benchmark", "signatures"):
        try:
            module = __import__(f"pode.federated.cli_{extra}", fromlist=["register"])
        except ImportError:
            continue
        module.register(subparsers)


__all__ = ["register", "load_federation", "prepare_table", "model_columns"]
