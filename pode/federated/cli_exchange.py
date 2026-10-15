"""``pode federated exchange``: local TVAEs + privacy-gated pooled release."""

from __future__ import annotations

from .cli import _common, _configs, _log, load_federation


def cmd_exchange(args) -> int:
    from ..data.io import ensure_dir
    from ..imputation.base import limit_cpu
    from ..privacy.config import load_privacy_config
    from .exchange import local_exchange

    config, fcfg = _configs(args)
    if args.generator:
        fcfg.exchange.generator = args.generator
    if args.allow_review:
        fcfg.exchange.release_verdicts = ["PASS", "REVIEW"]
    pcfg = load_privacy_config(args.privacy_config or fcfg.exchange.privacy_config)
    if args.quick:
        pcfg = pcfg.quick()
    sites, schema, cols, ledger, _ = load_federation(args, config, fcfg)
    _log(f"local exchange: {len(sites)} sites, generator {fcfg.exchange.generator}"
         f"{' + DP' if fcfg.dp.enabled else ''}, release on {fcfg.exchange.release_verdicts}")
    with limit_cpu(fcfg.n_jobs):
        result = local_exchange(sites, cols, schema, config.synthesizer, fcfg, pcfg,
                                config.data.censor_time_threshold, log=_log)
    out = result.save(ensure_dir(args.output_dir))
    ledger.save(out / "disclosure_ledger.csv")
    print(result.sites.to_string(index=False))
    print(f"Released {len(result.released)} rows from {len(result.card['sites_released'])}/{len(sites)} sites. "
          f"See {out}")
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("exchange", help="Local TVAEs per site; release only privacy-gated synthetic rows.")
    _common(p)
    p.add_argument("--privacy-config", default=None, help="Privacy YAML for the release gate.")
    p.add_argument("--generator", default=None, choices=["fed_tvae", "sdv_tvae"])
    p.add_argument("--allow-review", action="store_true", help="Also release REVIEW verdicts.")
    p.add_argument("--rounds", type=int, default=None, help="Local training budget is rounds x local epochs.")
    p.add_argument("--local-epochs", type=int, default=None)
    p.add_argument("--dp", action="store_true")
    p.add_argument("--target-epsilon", type=float, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", default="./pode_federated_exchange")
    p.set_defaults(func=cmd_exchange)
