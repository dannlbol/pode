"""Generate small synthetic fixtures for pode_mantis, sized to exercise the
wrapper's own logic (reindexing, imputation, config generation, alignment
assertions) quickly and deterministically.

This does NOT fake conumee2/CNV calling -- that has no meaningful synthetic
substitute (real intensity data + a real control reference are required) and
isn't unit-tested here; see docs/mantis/README.md for real-data verification.
Instead this writes an already-CNV-shaped toy numeric matrix directly, in
exactly the on-disk format the reference implementation expects, so the
*downstream* plumbing (generate-config, assertions, run_reference dispatch)
is exercisable without R/Bioconductor or the external MANTIS environment.

Writes, beside this script:
  toy_cpg_order.txt        200 CpG IDs (stand-in for MANTIS's real 357,692)
  toy_beta.csv              30 samples x 200 CpGs, samples_by_features,
                             ~15% of probes deliberately absent per sample
  toy_training_means.csv    per-probe means for a subset of the 200 CpGs
  toy_gene_order.txt        50 gene symbols (stand-in for MANTIS's real 16,067)
  toy_cnv_numeric.csv        50 genes x 30 samples, bare numeric, no header/index
                             (the CNV block's exact on-disk format)
  toy_cnv_sample_order.txt  the 30 sample IDs in that column order
  toy_clinical.csv           30 samples: id/time/event/pfs_time/pfs_event/m
  toy_reference_config.yaml a minimal stand-in for the reference repo's own
                             config.yaml (fixed sections only: model/time_bins/
                             feature_types/metastasis_recoding/conn_mat_path)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

N_SAMPLES = 30
N_CPGS = 200
N_GENES = 50
SEED = 42


def main() -> None:
    out = Path(__file__).resolve().parent
    rng = np.random.default_rng(SEED)

    samples = np.array([f"SAMPLE_{i:03d}" for i in range(N_SAMPLES)])
    cpgs = np.array([f"cg{i:07d}" for i in range(N_CPGS)])
    (out / "toy_cpg_order.txt").write_text("\n".join(cpgs) + "\n")

    # Beta: samples_by_features (rows=samples), first column sample_id, deliberately
    # missing ~15% of probes per sample (to exercise imputation) plus a few probes
    # this cohort's array simply doesn't have at all (to exercise the two-fallback path).
    beta = rng.uniform(0.05, 0.95, size=(N_SAMPLES, N_CPGS))
    missing_mask = rng.random((N_SAMPLES, N_CPGS)) < 0.15
    beta_df = pd.DataFrame(beta, columns=cpgs)
    beta_df[missing_mask] = np.nan
    beta_df.insert(0, "sample_id", samples)
    beta_df.to_csv(out / "toy_beta.csv", index=False)

    # Training means: only for 3/4 of the CpGs, so the own-cohort-median fallback
    # path (for the remaining 1/4) is actually exercised too.
    n_with_training_mean = int(N_CPGS * 0.75)
    train_cpgs = rng.choice(cpgs, size=n_with_training_mean, replace=False)
    train_means = pd.DataFrame({"probe_id": train_cpgs, "mean_beta": rng.uniform(0.2, 0.8, n_with_training_mean)})
    train_means.to_csv(out / "toy_training_means.csv", index=False)

    # CNV: already in the exact on-disk shape the reference implementation needs
    # (genes as rows, samples as columns, bare numeric) -- no header, no index.
    genes = np.array([f"GENE{i:03d}" for i in range(N_GENES)])
    (out / "toy_gene_order.txt").write_text("\n".join(genes) + "\n")
    cnv = rng.normal(0, 0.3, size=(N_GENES, N_SAMPLES))
    pd.DataFrame(cnv).to_csv(out / "toy_cnv_numeric.csv", header=False, index=False, float_format="%.6f")
    (out / "toy_cnv_sample_order.txt").write_text("\n".join(samples) + "\n")

    # Clinical: mild synthetic risk signal, occasional missing metastasis status.
    risk = 2.0 * (beta[:, 0] - 0.5)
    os_time = np.maximum(0.1, rng.exponential(np.exp(-risk / 2) * 5))
    os_censor = rng.uniform(1, 10, N_SAMPLES)
    pfs_time = np.minimum(os_time, rng.exponential(np.exp(-risk / 2) * 3))
    m = rng.integers(0, 2, N_SAMPLES).astype(float)
    m[rng.random(N_SAMPLES) < 0.1] = np.nan  # a few missing, to exercise the warning path
    clinical = pd.DataFrame(
        {
            "patient_id": samples,
            "os_time": np.round(np.minimum(os_time, os_censor), 3),
            "os_event": (os_time <= os_censor).astype(int),
            "pfs_time": np.round(pfs_time, 3),
            "pfs_event": rng.integers(0, 2, N_SAMPLES),
            "metastasis": m,
        }
    )
    clinical.to_csv(out / "toy_clinical.csv", index=False)

    # Minimal reference-repo config.yaml stand-in: only the FIXED sections
    # generate_config.py copies verbatim -- enough to test that logic without
    # needing a real checkout of sstefan01/methylation-survival.
    reference_config = {
        "data": {
            "conn_mat_path": "data/conn_mat_beta.csv",
            "conn_mat_is_one_based": True,
            "time_bins": [0.02, 0.33, 0.58, 0.75, 0.94, 1.17, 1.5, 1.83, 2.08, 2.58, 3.25, 3.9, 5.67, 8.29, 22.0],
        },
        "run_setup": {
            "training_cohorts": ["toy_train_a", "toy_train_b"],
            "feature_types": ["beta", "cnv"],
            "metastasis_recoding": {"from": 0, "to": -1},
        },
        "model": {
            "part1": {"input_dim": N_CPGS + N_GENES, "layer_dims": [N_GENES, 16, 8, 4], "dropout_rate": 0.2, "c2_input_offset": N_CPGS},
            "part2": {"num_time_bins": 15, "dropout_rate": 0.2},
            "combined": {"num_clinical_features": 1, "clinical_feature_weight": -4.4},
        },
        "toy_train_a_surv_path": "data/surv_toy_train_a.csv",
        "toy_train_b_surv_path": "data/surv_toy_train_b.csv",
    }
    # Fold the two training-cohort paths under data:, matching the real shape.
    reference_config["data"]["toy_train_a_surv_path"] = reference_config.pop("toy_train_a_surv_path")
    reference_config["data"]["toy_train_b_surv_path"] = reference_config.pop("toy_train_b_surv_path")
    with open(out / "toy_reference_config.yaml", "w") as f:
        yaml.safe_dump(reference_config, f, sort_keys=False)

    print(f"Wrote toy MANTIS fixtures to {out}")
    print(f"  toy_beta.csv               {N_SAMPLES} samples x {N_CPGS} CpGs (~15% missing per sample)")
    print(f"  toy_cnv_numeric.csv        {N_GENES} genes x {N_SAMPLES} samples")
    print(f"  toy_clinical.csv           {N_SAMPLES} samples, {clinical['os_event'].sum()} OS events")
    print(f"  toy_reference_config.yaml  fixed sections only (model/time_bins/feature_types/metastasis_recoding)")


if __name__ == "__main__":
    main()
