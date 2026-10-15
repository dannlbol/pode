"""Split a PODE-generated table and decode hybrid latents back to matrices.

This is the reverse of ``prepare_pode_training.py``. It expects a wide table
produced by a downstream TVAE such as PODE, containing:

    [ clinical columns | *_latent_* columns | survival columns ]

It writes:

* ``sampled_latents.csv``
* ``synthetic_clinical.csv``
* ``synthetic_survival.csv``
* ``synthetic_expression.csv`` and/or ``synthetic_methylation.csv``

using the decoder(s) saved in one or more synthvae-hybrid ``export-latents``
bundles. Pass a single ``--bundle-dir`` for the usual single-modality decode, or
several (e.g. an expression bundle and a methylation bundle) to split a jointly
handed-off table and decode every latent block back to its own matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

def add_args(ap):
    ap.add_argument(
        "--bundle-dir",
        required=True,
        nargs="+",
        help="one or more export-latents output directories (one per latent block)",
    )
    ap.add_argument("--pode-table", required=True, help="wide generated table from PODE or another TVAE")
    ap.add_argument("--out-dir", required=True, help="directory for split CSVs and decoded matrices")
    ap.add_argument(
        "--hybrid-root",
        default=None,
        help="path to a checkout containing the synthvae package; usually inferred automatically",
    )
    ap.add_argument(
        "--time-col",
        default="OS_Time",
        help="survival time column in the generated wide table",
    )
    ap.add_argument(
        "--event-col",
        default="OS_Status",
        help="survival event column in the generated wide table",
    )
    ap.add_argument(
        "--latents-name",
        default="sampled_latents.csv",
        help="output filename for the split latent table (all blocks)",
    )
    ap.add_argument(
        "--clinical-name",
        default="synthetic_clinical.csv",
        help="output filename for the split clinical table",
    )
    ap.add_argument(
        "--survival-name",
        default="synthetic_survival.csv",
        help="output filename for the split survival table",
    )
    ap.add_argument(
        "--decoded-name",
        default=None,
        help="output filename for the decoded matrix; only honored for a single "
        "--bundle-dir. With multiple bundles each block names itself "
        "(synthetic_<matrix_kind>.csv).",
    )
    ap.add_argument("--expression-name", dest="decoded_name", help=argparse.SUPPRESS)
    ap.add_argument(
        "--copy-input",
        action="store_true",
        help="also copy the original PODE-wide table into the output directory",
    )
    return ap


def add_hybrid_src_to_path(hybrid_root: str | Path | None) -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]

    # synthvae is now a first-class package in this repo (and normally pip
    # installed); prefer importing it directly rather than reaching into an
    # external checkout.
    if hybrid_root is None:
        try:
            import synthvae  # noqa: F401
            return repo_root
        except ImportError:
            pass

    candidates = []
    if hybrid_root is not None:
        candidates.append(Path(hybrid_root))
    candidates.extend(
        [
            repo_root,                                     # merged layout: <root>/synthvae
            Path.cwd(),
            here.parents[2] / "synth.vae" / "synthvae-hybrid",  # legacy external checkout
        ]
    )

    for root in candidates:
        # merged/flat layout keeps the package at <root>/synthvae; the legacy
        # synthvae-hybrid checkout used a src-layout at <root>/src/synthvae.
        for cand in (root, root / "src"):
            if (cand / "synthvae").is_dir():
                sys.path.insert(0, str(cand))
                return root
    raise RuntimeError(
        "could not locate the synthvae package; pass --hybrid-root explicitly"
    )


def latent_block_for_bundle(bundle_dir: str | Path, read_metadata) -> tuple[list[str], dict]:
    """Return ``(latent_columns, metadata)`` for one bundle."""
    metadata = read_metadata(Path(bundle_dir) / "handoff_metadata.json")
    prefix = metadata.get("latent_prefix", "expr_latent_")
    n_latent = int(metadata["config"]["n_latent"])
    cols = [f"{prefix}{i}" for i in range(n_latent)]
    return cols, metadata


def split_generated_table(
    table: pd.DataFrame,
    *,
    latent_cols: list[str],
    time_col: str,
    event_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carve the wide table into clinical / latents / survival.

    ``latent_cols`` is the union of every bundle's latent columns, so a second
    modality's latents are never mistaken for clinical columns.
    """
    missing = [c for c in [*latent_cols, time_col, event_col] if c not in table.columns]
    if missing:
        raise ValueError(f"generated table is missing required columns: {missing}")

    clinical_cols = [c for c in table.columns if c not in set(latent_cols + [time_col, event_col])]
    clinical = table[clinical_cols].reset_index(drop=True)
    latents = table[latent_cols].reset_index(drop=True)
    survival = table[[time_col, event_col]].reset_index(drop=True).rename(
        columns={time_col: "time", event_col: "event"}
    )
    return clinical, latents, survival


def run(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    add_hybrid_src_to_path(args.hybrid_root)
    from synthvae.pipeline.handoff import decode_latent_csv, decoded_matrix_name, read_metadata

    bundle_dirs = [Path(b) for b in args.bundle_dir]
    # Gather each bundle's latent block; the union defines what is NOT clinical.
    blocks: list[tuple[Path, list[str], dict]] = []
    all_latent_cols: list[str] = []
    for bundle_dir in bundle_dirs:
        cols, metadata = latent_block_for_bundle(bundle_dir, read_metadata)
        blocks.append((bundle_dir, cols, metadata))
        all_latent_cols.extend(cols)

    table = pd.read_csv(args.pode_table)

    clinical, latents, survival = split_generated_table(
        table,
        latent_cols=all_latent_cols,
        time_col=args.time_col,
        event_col=args.event_col,
    )

    latents_path = out_dir / args.latents_name
    clinical_path = out_dir / args.clinical_name
    survival_path = out_dir / args.survival_name

    latents.to_csv(latents_path, index=False)
    clinical.to_csv(clinical_path, index=False)
    survival.to_csv(survival_path, index=False)
    if args.copy_input:
        table.to_csv(out_dir / Path(args.pode_table).name, index=False)

    # ``--decoded-name`` only makes sense for a single block; with several bundles
    # each block names itself (synthetic_<matrix_kind>.csv) to avoid collisions.
    single = len(blocks) == 1
    if args.decoded_name and not single:
        print("note: --decoded-name is ignored for a multi-bundle decode; each block "
              "writes synthetic_<matrix_kind>.csv")

    decoded_blocks = []
    for bundle_dir, cols, metadata in blocks:
        matrix_kind = metadata.get("matrix_kind") or "expression"
        if single and args.decoded_name:
            decoded_path = out_dir / args.decoded_name
        else:
            decoded_path = out_dir / decoded_matrix_name(matrix_kind)
        decoded = decode_latent_csv(
            bundle_dir=bundle_dir,
            latents_csv=latents_path,
            out_matrix_csv=decoded_path,
        )
        decoded_blocks.append({
            "bundle_dir": str(bundle_dir),
            "matrix_kind": matrix_kind,
            "latent_prefix": metadata.get("latent_prefix"),
            "latent_columns": cols,
            "decoded_matrix": str(decoded_path),
            "decoded_matrix_columns": list(decoded.columns),
            "compressor_kind": metadata.get("compressor_kind"),
        })
        print(f"Wrote decoded {matrix_kind}: {decoded_path} ({len(decoded)} rows)")

    summary = {
        "rows": len(table),
        "clinical_columns": list(clinical.columns),
        "latent_columns": all_latent_cols,
        "survival_columns_in_input": [args.time_col, args.event_col],
        "blocks": decoded_blocks,
    }
    with open(out_dir / "decode_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(f"Wrote latent samples: {latents_path}")
    print(f"Wrote clinical table: {clinical_path}")
    print(f"Wrote survival table: {survival_path}")


def main(argv=None) -> int:
    ap = add_args(argparse.ArgumentParser(description=__doc__))
    run(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
