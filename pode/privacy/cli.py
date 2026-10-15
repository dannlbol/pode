"""``pode synthesis privacy ...`` subcommands.

  * ``assess`` - assess an existing synthetic table against the real patients
                 its generator was trained on (and, ideally, held-out patients
                 it never saw).
  * ``audit``  - split the real data, fit a TVAE on the training share with the
                 PODE config, generate, and assess against the held-out share.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

N_JOBS_HELP = "Max CPU cores to use (default: config n_jobs, 4; -1 = all cores)."
ROW = "_pode_row_"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _configs(args):
    from ..config import load_config
    from .config import load_privacy_config

    config = load_config(args.config)
    pcfg = load_privacy_config(args.privacy_config)
    if getattr(args, "quick", False):
        pcfg = pcfg.quick()
    for name in ("seed", "n_jobs"):
        if getattr(args, name, None) is not None:
            setattr(pcfg, name, getattr(args, name))
    return config, pcfg


def _prepare(df, config, schema, id_column):
    """``prepare_data`` as generation saw it, keeping track of which raw row each prepared row is."""
    from ..imputation.base import subset_schema
    from ..pipeline import prepare_data

    tagged = df.assign(**{ROW: np.arange(len(df), dtype=float)})
    data, schema = prepare_data(tagged, config, schema)
    if ROW in data.columns:
        rows = data.pop(ROW).astype(int).to_numpy()
    else:  # pragma: no cover - only if a quality filter dropped the tag
        rows = np.arange(len(data)) if len(data) == len(df) else None
    schema = subset_schema(schema, [c for c in schema.columns if c != ROW and c in data.columns])
    ids = df[id_column].to_numpy()[rows] if id_column and id_column in df.columns and rows is not None else None
    return data.reset_index(drop=True), schema, ids


def cmd_assess(args) -> int:
    from ..cli.main import _load_schema
    from ..data.io import ensure_dir, load_dataset
    from .assess import assess_privacy

    config, pcfg = _configs(args)
    raw_train, raw_synth = load_dataset(args.train), load_dataset(args.synthetic)
    schema = _load_schema(raw_train, config, args.schema)
    train, schema, ids = _prepare(raw_train, config, schema, pcfg.id_column)
    holdout = None
    if args.holdout:
        holdout, _, _ = _prepare(load_dataset(args.holdout), config, schema, None)
    else:
        _log("warning: no --holdout given. Membership inference needs real patients the generator never saw; "
             "consider `pode synthesis privacy audit`.")
    report = assess_privacy(train, raw_synth, holdout, schema, pcfg, raw_train=raw_train, raw_synthetic=raw_synth,
                            train_ids=ids, log=_log)
    out = report.save(ensure_dir(args.output_dir))
    _print_gate(report.gate)
    print(f"Privacy verdict: {report.verdict}. See {out / 'report.md'}")
    return 0 if report.verdict != "FAIL" or not args.fail_on_fail else 3


def cmd_audit(args) -> int:
    from ..cli.main import _load_schema, _log_torch_device
    from ..data.io import ensure_dir, load_dataset
    from .audit import audit_tvae

    config, pcfg = _configs(args)
    for name, attr in (("holdout_frac", "holdout_frac"), ("num_samples", "num_samples"), ("repeats", "repeats"),
                       ("epochs", "epochs"), ("device", "device")):
        if getattr(args, name, None) is not None:
            setattr(pcfg.audit, attr, getattr(args, name))
    pcfg.validate()
    raw = load_dataset(args.data)
    schema = _load_schema(raw, config, args.schema)
    data, schema, ids = _prepare(raw, config, schema, pcfg.id_column)
    out = ensure_dir(args.output_dir)
    _log_torch_device()
    result = audit_tvae(data, schema, config, pcfg, out, ids=ids, log=_log)
    print(f"Privacy audit verdict: {result['verdict']} over {len(result['repeats'])} repeat(s). "
          f"See {out / 'report.md'}")
    return 0 if result["verdict"] != "FAIL" or not args.fail_on_fail else 3


def _print_gate(gate) -> None:
    for r in gate[gate["status"] != "MANUAL"].itertuples():
        print(f"  {r.status:<12} {r.check:<20} {r.metric} = {r.value} ({r.reason})", file=sys.stderr)


def register(syn) -> None:
    """Mount ``privacy`` under the ``pode synthesis`` subparsers."""
    p = syn.add_parser("privacy", help="Privacy-preservation assessment of TVAE synthetic data.")
    sub = p.add_subparsers(dest="privacy_command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--show-warnings", action="store_true",
                        help="Show benign third-party warnings, suppressed by default.")
    common.add_argument("--config", default=None, help="PODE YAML config (schema, data preparation, TVAE settings).")
    common.add_argument("--schema", default=None, help="Optional schema YAML.")
    common.add_argument("--privacy-config", default=None,
                        help="Privacy YAML: attacker tiers, sensitive attributes, strata, gates "
                             "(see configs/privacy/small_cohort.example.yaml).")
    common.add_argument("--n-jobs", type=int, default=None, help=N_JOBS_HELP)
    common.add_argument("--seed", type=int, default=None)
    common.add_argument("--fail-on-fail", action="store_true", help="Exit with code 3 when the verdict is FAIL.")

    a = sub.add_parser("assess", parents=[common], help="Assess an existing synthetic table.")
    a.add_argument("--train", required=True, help="Real CSV the generator was trained on (raw; prepared as PODE does).")
    a.add_argument("--synthetic", required=True, help="Synthetic CSV to assess.")
    a.add_argument("--holdout", default=None,
                   help="Real CSV of patients from the same population the generator never saw (strongly "
                        "recommended; never an external cohort).")
    a.add_argument("--output-dir", default="./pode_privacy")
    a.set_defaults(func=cmd_assess)

    u = sub.add_parser("audit", parents=[common], help="Split, fit a TVAE on the training share, generate, assess.")
    u.add_argument("--data", required=True, help="Real CSV (all patients).")
    u.add_argument("--holdout-frac", type=float, default=None, help="Share of patients held out (default 0.3).")
    u.add_argument("--num-samples", type=int, default=None, help="Synthetic rows per repeat (default: n training).")
    u.add_argument("--repeats", type=int, default=None, help="Independent split/fit/assess repeats (default 1).")
    u.add_argument("--epochs", type=int, default=None, help="TVAE epoch override (default: PODE config).")
    u.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None,
                   help="Torch device for the TVAE fit. Use cpu when the shared GPU is busy.")
    u.add_argument("--quick", action="store_true", help="Cheap smoke-test settings.")
    u.add_argument("--output-dir", default="./pode_privacy_audit")
    u.set_defaults(func=cmd_audit)
