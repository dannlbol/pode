#!/usr/bin/env python3
"""Head-to-head benchmark: the joint multimodal VAE vs the hybrid pipeline.

Runs both architectures across their sweep grids on the *same* cohort and the
*same* train/test split (per seed), scores them with the shared survival-preserving
evaluation harness, and reports which architecture wins on each metric.

Because the full grids are large (the mmvae sweep is 3x2x2x4x3x3 = 432 combos; the
hybrid sweep is 432, each an SDV TVAE fit), the default caps the number of configs
per architecture. Use ``--full`` to run the entire grid, or ``--quick`` for a fast
smoke.

Examples::

    python scripts/benchmark.py --toy --quick
    python scripts/benchmark.py --toy --seeds 0 1 2 --max-configs 6
    python scripts/benchmark.py --toy --full            # the whole grid (slow!)
    python scripts/benchmark.py --clinical c.csv --expression e.csv --survival s.csv

The hybrid arm defaults to the production path (``survival_vae`` compressor +
``sdv`` TVAE); switch with ``--hybrid-compressor`` / ``--hybrid-synthesizer``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0]))

import run_hybrid          # noqa: E402  (reused, read-only)
import run_mmvae           # noqa: E402
from synthvae.data.toy import make_cohort, ToyConfig  # noqa: E402
from synthvae.mmvae import MMVAEConfig                 # noqa: E402
from dataclasses import fields as dc_fields            # noqa: E402

KEY_METRICS = [
    "tstr_concordance", "trtr_concordance", "concordance_gap",
    "hr_logcorr", "km_iad", "km_iad_late", "latent_cov_relgap",
]

# Small, fast grids for --quick (hybrid uses the torch-free GMM synthesizer).
QUICK_JOINT_GRID = {
    "fusion.strategy": ["poe", "mopoe"],
    "objective.weighting_scheme": ["fixed", "uncertainty"],
}
QUICK_HYBRID_SWEEP = {
    "compressor": "survival_vae",
    "synthesizer": "gmm",
    "grid": {"n_latent": [16, 32], "w_surv": [1.0]},
}


# --------------------------------------------------------------------------- #
# cohort + composite score
# --------------------------------------------------------------------------- #
def load_cohort(args, seed):
    if args.clinical or args.expression or args.survival:
        if not (args.expression and args.survival):
            raise ValueError("BYO runs require --expression and --survival.")
        expression = pd.read_csv(args.expression).reset_index(drop=True)
        survival = pd.read_csv(args.survival).reset_index(drop=True)
        clinical = (pd.read_csv(args.clinical).reset_index(drop=True)
                    if args.clinical else pd.DataFrame(index=np.arange(len(expression))))
        return clinical, expression, survival
    clinical, expression, survival, _ = make_cohort(
        ToyConfig(n=args.n, n_genes=args.n_genes, seed=seed)
    )
    return clinical, expression, survival


def composite(metrics: dict) -> float:
    """Lower is better: |concordance gap| + late-KM drift + (1 - HR log-corr).

    Delegates to run_mmvae.composite_score so the benchmark and `run_mmvae.py tune`
    rank joint configs identically.
    """
    return run_mmvae.composite_score(metrics)


# --------------------------------------------------------------------------- #
# grid expansion
# --------------------------------------------------------------------------- #
def expand_joint_grid(base_kwargs: dict, sweep: dict) -> list[dict]:
    """Product over a dotted-key ``sweeps`` dict -> list of MMVAEConfig kwargs."""
    if not sweep:
        return [dict(base_kwargs)]
    keys = list(sweep)
    valuesets = [v if isinstance(v, list) else [v] for v in sweep.values()]
    valid = {f.name for f in dc_fields(MMVAEConfig)}
    combos = []
    for combo in itertools.product(*valuesets):
        kw = dict(base_kwargs)
        for dotted, val in zip(keys, combo):
            field = run_mmvae.SWEEP_KEY_MAP.get(dotted, dotted.split(".")[-1])
            if field in valid and field != "seed":
                kw[field] = val
        combos.append(kw)
    # de-duplicate while preserving order
    seen, out = set(), []
    for kw in combos:
        key = json.dumps(kw, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(kw)
    return out


def cap(configs: list, max_configs: int | None) -> list:
    if max_configs is None or len(configs) <= max_configs:
        return configs
    # deterministic evenly-spaced subset across the grid
    idx = np.linspace(0, len(configs) - 1, max_configs).round().astype(int)
    return [configs[i] for i in sorted(set(idx.tolist()))]


# --------------------------------------------------------------------------- #
# per-arch runners
# --------------------------------------------------------------------------- #
def run_joint_arm(args, joint_configs, seeds):
    rows = []
    for seed in seeds:
        clinical, expression, survival = load_cohort(args, seed)
        splits = run_mmvae.split_frames(clinical, expression, survival, seed,
                                        args.test_fraction)
        for ci, kw in enumerate(joint_configs):
            kw = dict(kw, seed=seed, device=args.device)
            if args.epochs:
                kw["max_epochs"] = args.epochs
            valid = {f.name for f in dc_fields(MMVAEConfig)}
            cfg = MMVAEConfig(**{k: v for k, v in kw.items() if k in valid})
            label = f"fusion={cfg.fusion},gate={cfg.gating},weight={cfg.weighting_scheme},beta={cfg.beta_max},fb={cfg.free_bits},ld={cfg.latent_dim}"
            print(f"[joint {ci+1}/{len(joint_configs)} seed={seed}] {label}")
            metrics, _, _ = run_mmvae.run_joint(cfg, splits)
            rows.append({"arch": "joint", "config": label, "seed": seed, **metrics})
    return rows


def _hybrid_args(args, params_seed):
    """Namespace with the fields run_hybrid.run_once / load_data / prepare_params read."""
    return SimpleNamespace(
        clinical=args.clinical, expression=args.expression, survival=args.survival,
        methylation=None, n=args.n, n_genes=args.n_genes,
        seed=params_seed, test_fraction=args.test_fraction,
        categorical=[], no_lifelines=False, out_dir=None,
        compressor=args.hybrid_compressor, synthesizer=args.hybrid_synthesizer,
        n_latent=16, epochs=(args.epochs or 100), n_top_genes=0, beta_epsilon=1e-6,
    )


def run_hybrid_arm(args, hybrid_sweep, seeds, max_configs):
    combos = run_hybrid.expand_grid(hybrid_sweep)
    for c in combos:
        c.pop("seed", None)  # benchmark seeds drive the split
    combos = cap(combos, max_configs)
    # CLI choice wins (the sweep file's compressor/synthesizer are ignored so the
    # benchmark's chosen baseline is honoured).
    compressor = args.hybrid_compressor
    synthesizer = args.hybrid_synthesizer
    rows = []
    for seed in seeds:
        for ci, combo in enumerate(combos):
            params = dict(combo, compressor=compressor, synthesizer=synthesizer,
                          seed=seed)
            if args.epochs:
                params["epochs"] = args.epochs
                params["tvae_epochs"] = args.epochs
            hargs = _hybrid_args(args, seed)
            label = ",".join(f"{k}={v}" for k, v in sorted(combo.items())) or "default"
            label = f"{compressor}+{synthesizer},{label}"
            print(f"[hybrid {ci+1}/{len(combos)} seed={seed}] {label}")
            metrics = run_hybrid.run_once(hargs, params, write_outputs=False)
            rows.append({"arch": "hybrid", "config": label, "seed": seed,
                         **{k: v for k, v in metrics.items() if not isinstance(v, bool)}})
    return rows, len(combos)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    metric_cols = [c for c in KEY_METRICS if c in results.columns]
    agg = (results.groupby(["arch", "config"], sort=False)[metric_cols]
           .mean().reset_index())
    agg["composite"] = agg.apply(lambda r: composite(r.to_dict()), axis=1)

    lines = ["# Joint vs Hybrid benchmark\n"]
    lines.append(f"Configs evaluated: {len(agg)} "
                 f"(joint={sum(agg.arch=='joint')}, hybrid={sum(agg.arch=='hybrid')}); "
                 f"seeds={sorted(results['seed'].unique().tolist())}\n")

    best = {}
    for arch in ("joint", "hybrid"):
        sub = agg[agg.arch == arch]
        if len(sub):
            best[arch] = sub.sort_values("composite").iloc[0]

    if len(best) == 2:
        header = "| metric | best joint | best hybrid | winner |"
        sep = "|---|---|---|---|"
        rows = [header, sep]
        # for gap/km/latent lower is better; for concordance/hr higher is better
        higher_better = {"tstr_concordance", "trtr_concordance", "hr_logcorr"}
        for m in metric_cols + ["composite"]:
            jv, hv = float(best["joint"][m]), float(best["hybrid"][m])
            if m in higher_better:
                win = "joint" if jv > hv else "hybrid"
            else:
                win = "joint" if jv < hv else "hybrid"
            rows.append(f"| {m} | {jv:.4f} | {hv:.4f} | {win} |")
        lines.append("\n".join(rows))
        lines.append(f"\n- Best joint config: `{best['joint']['config']}`")
        lines.append(f"- Best hybrid config: `{best['hybrid']['config']}`")
        lines.append("\n> Note: on the toy cohort the hybrid is expected to lead "
                     "(see docs/ARCHITECTURE.md — the hybrid is the stronger default; "
                     "the joint model earns its keep on cross-modal / missing-modality "
                     "generation, not raw survival preservation here).")
    else:
        for arch, row in best.items():
            lines.append(f"\nBest {arch}: `{row['config']}` "
                         f"(composite={row['composite']:.4f})")
    return agg, "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--toy", action="store_true", help="use the synthetic cohort")
    p.add_argument("--clinical", default=None)
    p.add_argument("--expression", default=None)
    p.add_argument("--survival", default=None)
    p.add_argument("--n", type=int, default=400, help="toy cohort size")
    p.add_argument("--n-genes", type=int, default=150, help="toy gene count")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--joint-sweep",
                   default=str(_SCRIPTS.parents[0] / "configs" / "synthvae" / "mmvae.yaml"))
    p.add_argument("--hybrid-sweep",
                   default=str(_SCRIPTS.parents[0] / "configs" / "synthvae" / "hybrid_sweep.yaml"))
    p.add_argument("--hybrid-compressor", default="survival_vae",
                   choices=["pca", "survival_vae"])
    p.add_argument("--hybrid-synthesizer", default="sdv", choices=["gmm", "sdv"])
    p.add_argument("--max-configs", type=int, default=8,
                   help="cap configs per architecture (default 8; use --full to remove)")
    p.add_argument("--full", action="store_true", help="run the entire grid (slow!)")
    p.add_argument("--quick", action="store_true", help="tiny built-in grid + gmm")
    p.add_argument("--joint-only", action="store_true")
    p.add_argument("--hybrid-only", action="store_true")
    p.add_argument("--epochs", type=int, default=None, help="epoch override for both arms")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="runs/benchmark")
    return p


def main(argv=None):
    import yaml
    args = build_parser().parse_args(argv)
    max_configs = None if args.full else args.max_configs

    run_joint = not args.hybrid_only
    run_hyb = not args.joint_only

    # ---- assemble grids ----
    if args.quick:
        joint_base = run_mmvae.load_config(args.joint_sweep)
        joint_configs = expand_joint_grid(joint_base, QUICK_JOINT_GRID)
        hybrid_sweep = QUICK_HYBRID_SWEEP
        # quick mode is a fast smoke: force the torch-free GMM synthesizer
        args.hybrid_compressor = "survival_vae"
        args.hybrid_synthesizer = "gmm"
        if args.epochs is None:
            args.epochs = 15
    else:
        joint_base = run_mmvae.load_config(args.joint_sweep)
        with open(args.joint_sweep) as fh:
            joint_sweep_block = (yaml.safe_load(fh) or {}).get("sweeps", {})
        joint_configs = expand_joint_grid(joint_base, joint_sweep_block)
        hybrid_sweep = run_hybrid.load_sweep(args.hybrid_sweep)

    joint_configs = cap(joint_configs, max_configs) if run_joint else []

    # ---- plan / warn ----
    n_hyb_est = len(cap([c for c in run_hybrid.expand_grid(hybrid_sweep)], max_configs)) if run_hyb else 0
    n_planned = (len(joint_configs) + n_hyb_est) * len(args.seeds)
    print(f"Planned model fits: {n_planned} "
          f"(joint configs={len(joint_configs)}, hybrid configs~={n_hyb_est}, "
          f"seeds={len(args.seeds)}).")
    if run_hyb and args.hybrid_synthesizer == "sdv":
        print("  NOTE: hybrid arm uses the SDV TVAE — each fit is slow. Consider "
              "--quick, --max-configs, --epochs, or --hybrid-synthesizer gmm.")

    rows = []
    if run_joint:
        rows += run_joint_arm(args, joint_configs, args.seeds)
    if run_hyb:
        hrows, _ = run_hybrid_arm(args, hybrid_sweep, args.seeds, max_configs)
        rows += hrows

    results = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "benchmark_results.csv", index=False)

    agg, summary = summarize(results)
    if results["seed"].nunique() > 1:
        agg.to_csv(out / "benchmark_aggregated.csv", index=False)
    (out / "benchmark_summary.md").write_text(summary + "\n")
    print("\n" + summary)
    print(f"\nWrote {out/'benchmark_results.csv'} and {out/'benchmark_summary.md'}")


if __name__ == "__main__":
    main()
