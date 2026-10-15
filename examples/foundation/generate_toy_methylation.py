"""Generate a small synthetic methylation-array fixture for pode_foundation.

Writes, beside this script:
  toy_beta.csv        40 samples x 2000 CpGs, beta values in [0.05, 0.95]
                       (samples_by_features orientation: rows=samples).
  toy_annotation.csv  Name/chr/pos for every CpG in toy_beta.csv, spread
                       across a few chromosomes with real-scale genomic
                       positions and a handful of intentionally-clustered
                       probes, so `pode foundation preprocess`'s LD-pruning
                       step has something to prune.
  toy_phenotype.csv   sample_id/time/event/fold survival sidecar so
                       `pode foundation survival` is exercisable end to end.

This is a smoke-test fixture, not realistic biology -- it exists to exercise
every stage of the pipeline (preprocess -> pretrain -> finetune -> embed ->
survival) quickly and deterministically, not to produce meaningful results.
"""
from pathlib import Path

import numpy as np
import pandas as pd

N_SAMPLES = 40
N_CPGS = 2000
SEED = 42


def main() -> None:
    out = Path(__file__).resolve().parent
    rng = np.random.default_rng(SEED)

    samples = np.array([f"S{i:03d}" for i in range(N_SAMPLES)])
    cpgs = np.array([f"cg{i:07d}" for i in range(N_CPGS)])

    beta = rng.uniform(0.05, 0.95, size=(N_SAMPLES, N_CPGS)).astype(np.float32)

    # Spread probes across 3 chromosomes; cluster every 5th probe near its
    # predecessor (within 500bp) and make them near-duplicate columns, so LD
    # pruning has real correlated neighbours to remove.
    chrom = np.resize(["chr1", "chr2", "chr3"], N_CPGS)
    pos = (np.arange(N_CPGS) * 2000 + rng.integers(0, 500, N_CPGS)).astype(np.int64)
    for i in range(4, N_CPGS, 5):
        pos[i] = pos[i - 1] + rng.integers(50, 400)
        beta[:, i] = np.clip(beta[:, i - 1] + rng.normal(0, 0.02, N_SAMPLES), 0.01, 0.99)
    beta_df = pd.DataFrame(beta, columns=cpgs)
    beta_df.insert(0, "sample_id", samples)
    beta_df.to_csv(out / "toy_beta.csv", index=False)

    anno = pd.DataFrame({"Name": cpgs, "chr": chrom, "pos": pos})
    anno.to_csv(out / "toy_annotation.csv", index=False)

    # A mild synthetic risk signal from a handful of probes, so the survival
    # smoke test has *something* non-random to (weakly) recover.
    risk = 2.0 * (beta[:, 0] - 0.5) - 1.5 * (beta[:, 10] - 0.5)
    event_time = np.maximum(0.1, rng.exponential(np.exp(-risk / 2) * 5))
    censor_time = rng.uniform(1, 10, N_SAMPLES)
    phenotype = pd.DataFrame(
        {
            "sample_id": samples,
            "time": np.round(np.minimum(event_time, censor_time), 2),
            "event": (event_time <= censor_time).astype(int),
            "fold": rng.integers(0, 5, N_SAMPLES),
        }
    )
    phenotype.to_csv(out / "toy_phenotype.csv", index=False)

    print(f"Wrote toy fixtures to {out}")
    print(f"  toy_beta.csv        {beta_df.shape[0]} samples x {N_CPGS} CpGs")
    print(f"  toy_annotation.csv  {anno.shape[0]} probes")
    print(f"  toy_phenotype.csv   {phenotype.shape[0]} samples, {phenotype['event'].sum()} events")


if __name__ == "__main__":
    main()
