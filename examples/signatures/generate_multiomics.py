"""Generate a small multi-cohort, partially observed multi-omics signature fixture."""
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    out = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    samples = np.array([f"S{i:03}" for i in range(60)])
    expression = rng.normal(size=(60, 30))
    methylation = rng.uniform(.05, .95, size=(60, 24))
    risk = expression[:, 0] - .8 * expression[:, 1] + 1.2 * (methylation[:, 0] - .5)
    os_time = np.maximum(.2, rng.exponential(np.exp(-risk / 3) * 4))
    pfs_time = np.maximum(.2, rng.exponential(np.exp(-risk / 4) * 3))
    phenotype = pd.DataFrame({
        "sample_id": samples, "cohort": np.repeat(["A", "B", "C"], 20),
        "age": rng.uniform(2, 16, 60), "subgroup": np.resize(["III", "IV"], 60),
        "os_time": os_time, "os_event": rng.binomial(1, .7, 60),
        "pfs_time": pfs_time, "pfs_event": rng.binomial(1, .75, 60),
    })
    exp = pd.DataFrame(expression, columns=[f"gene_{i}" for i in range(expression.shape[1])])
    exp.insert(0, "sample_id", samples)
    meth = pd.DataFrame(methylation[5:], columns=[f"cpg_{i}" for i in range(methylation.shape[1])])
    meth.insert(0, "sample_id", samples[5:])  # first five samples demonstrate masked missing blocks
    # Small deterministic sidecars let the advanced config exercise the two
    # methods that require prior biological structure.  Graph nodes use original
    # feature names; pathway indices refer to the concatenated omics inputs.
    feature_names = [*exp.columns[1:], *meth.columns[1:]]
    graph = pd.DataFrame({
        "source": feature_names[:-1],
        "target": feature_names[1:],
        "weight": 1.0,
    })
    pathways = pd.DataFrame({
        "feature_index": np.arange(len(feature_names)),
        "pathway_index": np.arange(len(feature_names)) % 8,
    })
    phenotype.to_csv(out / "phenotype.csv", index=False)
    exp.to_csv(out / "expression.csv", index=False)
    meth.to_csv(out / "methylation.csv", index=False)
    graph.to_csv(out / "example_feature_graph.csv", index=False)
    pathways.to_csv(out / "example_pathways.csv", index=False)
    print(f"Wrote example inputs to {out}")


if __name__ == "__main__":
    main()
