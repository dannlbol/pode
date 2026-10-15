"""Command-line interface for the foundation (methylation-array MAE) theme.

Exposes pode_foundation's config-driven pipeline as subcommands. Mounted
under the unified ``pode foundation ...`` surface (see ``pode.cli.main``) and
also runnable standalone via ``python -m pode_foundation.cli`` / the
``register`` hook.

Subcommands:
  * ``preprocess`` - QC filter + genomic LD-prune a beta matrix to a fixed feature set.
  * ``combine``    - EXPERIMENTAL: build a broad-reference Stage-1 pretraining corpus.
  * ``pretrain``   - train a MethylationMAE from scratch (also the Stage-1 trainer).
  * ``finetune``   - Stage-2 domain-adaptive fine-tune from a Stage-1 checkpoint.
  * ``embed``      - apply a checkpoint's frozen encoder to a beta matrix.
  * ``survival``   - optional: benchmark embeddings (or fine-tune) against survival.

Heavy imports (torch, lifelines, sklearn) are deferred into the handlers so
``--help`` stays fast even without torch installed.
"""
from __future__ import annotations

import argparse
import json


def _load_cfg(args):
    from .config import load_config

    return load_config(args.config)


def cmd_preprocess(args) -> int:
    from .core.preprocess import run_preprocess

    cfg = _load_cfg(args)
    result = run_preprocess(cfg, beta_path=args.beta, anno_path=args.anno, out_dir=args.out_dir)
    print(f"\nWrote {result.n_samples} x {result.n_features} matrix to {result.out_dir}")
    return 0


def cmd_combine(args) -> int:
    from .core.combine import run_combine

    cfg = _load_cfg(args)
    result = run_combine(cfg, out_dir=args.out_dir)
    print(f"\nWrote combined corpus ({result.n_samples} x {result.n_features}) to {result.x_path}")
    return 0


def cmd_pretrain(args) -> int:
    from .core.train import run_pretrain

    cfg = _load_cfg(args)
    result = run_pretrain(cfg, x_path=args.x_path, out_dir=args.out_dir, stage_tag=args.stage_tag)
    print(f"\nBest checkpoint: {result.ckpt_path}  (val MSE {result.best_val_mse:.5f}, epoch {result.epoch})")
    return 0


def cmd_finetune(args) -> int:
    from .core.finetune import run_finetune

    cfg = _load_cfg(args)
    result = run_finetune(cfg, args.init_ckpt, x_path=args.x_path, out_dir=args.out_dir)
    print(f"\nBest checkpoint: {result.ckpt_path}  (val MSE {result.best_val_mse:.5f}, epoch {result.epoch})")
    return 0


def cmd_embed(args) -> int:
    from .core.embed import run_embed

    cfg = _load_cfg(args)
    result = run_embed(cfg, args.ckpt, betas=args.betas, out=args.out)
    print(f"\nWrote {result.n_samples} x {result.n_latent} embeddings to {result.out_path}")
    return 0


def cmd_survival(args) -> int:
    from .core.survival import run_survival

    cfg = _load_cfg(args)
    result = run_survival(
        cfg,
        mode=args.mode,
        embeddings_path=args.embeddings,
        ckpt=args.ckpt,
        betas_path=args.betas,
        clinical_path=args.clinical,
        out_dir=args.out_dir,
    )
    print(json.dumps({"c_index": result.c_index, "method": result.method, "n_samples": result.n_samples,
                       "n_events": result.n_events, "risk_path": str(result.risk_path)}, indent=2))
    return 0


def _add_config(p) -> None:
    p.add_argument("--config", required=True, help="Path to a pode_foundation YAML config.")


def register(subparsers) -> None:
    """Attach pode_foundation subcommands to an ``add_subparsers`` object."""
    pp = subparsers.add_parser("preprocess", help="QC filter + genomic LD-prune a beta matrix.")
    _add_config(pp)
    pp.add_argument("--beta", default=None, help="Override config matrix.path.")
    pp.add_argument("--anno", default=None, help="Override config preprocess.annotation_path.")
    pp.add_argument("--out-dir", default=None, help="Override config preprocess.output_dir.")
    pp.set_defaults(func=cmd_preprocess)

    cb = subparsers.add_parser(
        "combine", help="EXPERIMENTAL: build a broad-reference Stage-1 corpus (written, not run upstream)."
    )
    _add_config(cb)
    cb.add_argument("--out-dir", default=None, help="Override config pretrain_stage1.output_dir.")
    cb.set_defaults(func=cmd_combine)

    pt = subparsers.add_parser("pretrain", help="Train a MethylationMAE from scratch on a beta matrix.")
    _add_config(pt)
    pt.add_argument("--x-path", default=None, help="Preprocessed X.npy (default: config matrix.path).")
    pt.add_argument("--out-dir", default=None, help="Override config training.output_dir.")
    pt.add_argument("--stage-tag", type=int, default=None, help="Tag the checkpoint 'stage' (e.g. 1 for Stage-1).")
    pt.set_defaults(func=cmd_pretrain)

    ft = subparsers.add_parser("finetune", help="Stage-2 domain-adaptive fine-tune from a Stage-1 checkpoint.")
    _add_config(ft)
    ft.add_argument("--init-ckpt", default=None, help="Stage-1 checkpoint (default: config finetune_stage2.init_ckpt).")
    ft.add_argument("--x-path", default=None, help="Fine-tuning X.npy (default: config matrix.path).")
    ft.add_argument("--out-dir", default=None, help="Override config finetune_stage2.output_dir.")
    ft.set_defaults(func=cmd_finetune)

    em = subparsers.add_parser("embed", help="Apply a checkpoint's frozen encoder to a beta matrix.")
    _add_config(em)
    em.add_argument("--ckpt", required=True, help="Pretrain/finetune checkpoint (mae_best.pt).")
    em.add_argument("--betas", default=None, help="Beta matrix to embed (default: config matrix.path).")
    em.add_argument("--out", default=None, help="Override config embed.output_path.")
    em.set_defaults(func=cmd_embed)

    sv = subparsers.add_parser("survival", help="Benchmark embeddings (or fine-tune) against survival.")
    _add_config(sv)
    sv.add_argument("--mode", choices=("frozen", "finetune"), default=None, help="Override config survival.mode.")
    sv.add_argument("--embeddings", default=None, help="Embeddings CSV (required for --mode frozen).")
    sv.add_argument("--ckpt", default=None, help="Checkpoint (required for --mode finetune).")
    sv.add_argument("--betas", default=None, help="Beta matrix (--mode finetune; default: config matrix.path).")
    sv.add_argument("--clinical", default=None, help="Clinical/phenotype CSV (default: config survival.clinical_path).")
    sv.add_argument("--out-dir", default=None, help="Override config survival.output_dir.")
    sv.set_defaults(func=cmd_survival)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pode_foundation", description="Methylation-array MAE pretraining (pode_foundation)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    register(sub)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
