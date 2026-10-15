"""Generate a fully synthetic toy cohort for trying out PODE.

This data is entirely made up (drawn from random distributions) - it contains no
real patient data. It is shaped like a medulloblastoma clinical table so it
exercises every column type PODE detects and every option in
``configs/example_cohort.yaml``:

  * continuous            -> Age
  * binary                -> Sex, ConsensusMYC, ConsensusMYCN
  * ordinal               -> CSI_dose, Radiotherapy_Type, M_stage
  * categorical           -> Histology, MNP_Subgroup
  * survival pair         -> OS_Time / OS_Status
  * dropped by patterns   -> PFS_Time / PFS_Status (drop_patterns: ["PFS_*"])
  * dropped by name       -> MethGroup (drop_columns)
  * continuous via prefix -> cg0001 .. cg0010 (schema.continuous_prefixes: ["cg"])
  * high-missingness col  -> SubtotalResection (filtered out by max_na_fraction)

Survival is made dependent on ConsensusMYC / MNP_Subgroup so the hazard-ratio
fidelity metrics produce meaningful (non-degenerate) numbers.

Run:  python examples/make_toy_cohort.py   ->  writes examples/cohort.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N = 200
SEED = 20260629


def main() -> None:
    rng = np.random.default_rng(SEED)

    subgroup = rng.choice(["G3", "G4", "SHH", "WNT"], N, p=[0.30, 0.30, 0.25, 0.15])
    # Kept reasonably balanced (~45%) so a TVAE learns both classes and the
    # conditional-generation example produces both halves; still enriched in G3.
    myc = (rng.random(N) < np.where(subgroup == "G3", 0.60, 0.38)).astype(int)
    mycn = (rng.random(N) < np.where(subgroup == "SHH", 0.45, 0.30)).astype(int)

    # Baseline hazard rises with MYC amplification and Group 3 membership.
    risk = 0.6 + 0.9 * myc + 0.5 * (subgroup == "G3") + 0.3 * mycn
    true_time = rng.exponential(scale=6.0 / risk)        # latent event time (yrs)
    censor_time = rng.uniform(3.0, 15.0, N)              # random follow-up
    os_time = np.minimum(true_time, censor_time)
    os_status = (true_time <= censor_time).astype(int)

    # Progression-free survival (will be dropped by the example config).
    pfs_time = np.maximum(0.1, os_time - rng.uniform(0, 1.5, N))
    pfs_status = os_status.copy()

    df = pd.DataFrame(
        {
            "Sample_ID": [f"TOY_{i:04d}" for i in range(N)],
            "Age": np.round(rng.uniform(0.5, 18.0, N), 2),
            "Sex": rng.choice(["M", "F"], N),
            "Histology": rng.choice(["CLA", "DN", "MBEN", "LCA"], N, p=[0.5, 0.25, 0.1, 0.15]),
            "MNP_Subgroup": subgroup,
            "ConsensusMYC": myc,
            "ConsensusMYCN": mycn,
            "M_stage": rng.choice([0, 1, 2, 3], N, p=[0.55, 0.15, 0.2, 0.1]),
            "CSI_dose": rng.choice([0, 1, 2, 3], N, p=[0.1, 0.45, 0.35, 0.1]),
            "Radiotherapy_Type": rng.choice([0, 1, 2], N, p=[0.3, 0.5, 0.2]),
            "OS_Time": np.round(os_time, 3),
            "OS_Status": os_status,
            "PFS_Time": np.round(pfs_time, 3),
            "PFS_Status": pfs_status,
            "MethGroup": rng.choice(["grpA", "grpB", "grpC"], N),
        }
    )

    # Methylation beta values (continuous in [0, 1]), detected via the "cg" prefix.
    for j in range(1, 11):
        df[f"cg{j:04d}"] = np.round(rng.beta(2, 2, N), 4)

    # A column with deliberately high missingness, to demonstrate quality
    # filtering (dropped when max_na_fraction < ~0.8).
    subtotal = rng.choice([0, 1], N).astype(float)
    subtotal[rng.random(N) < 0.85] = np.nan
    df["SubtotalResection"] = subtotal

    out = Path(__file__).resolve().parent / "cohort.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols to {out}")


if __name__ == "__main__":
    main()
