#!/usr/bin/env python3
"""Write a toy cohort to CSVs (clinical.csv, expression.csv, survival.csv)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthvae.data.toy import make_cohort, ToyConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_toy")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--n-genes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clinical, expression, survival, _ = make_cohort(
        ToyConfig(n=args.n, n_genes=args.n_genes, seed=args.seed)
    )
    clinical.to_csv(out / "clinical.csv", index=False)
    expression.to_csv(out / "expression.csv", index=False)
    survival.to_csv(out / "survival.csv", index=False)
    print(f"Wrote {out}/clinical.csv, expression.csv, survival.csv")


if __name__ == "__main__":
    main()
