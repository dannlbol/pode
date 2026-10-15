"""Combine is EXPERIMENTAL (upstream: "written, not run") -- this test only
checks the alignment logic doesn't crash and produces sane shapes on a tiny
synthetic source, not that broad-reference pretraining itself is validated.
"""
import numpy as np
import pandas as pd

from pode_foundation.core.combine import run_combine

from conftest import toy_cfg


def test_combine_aligns_wide_tsv_to_fixed_feature_set(toy_paths):
    # Feature set the "broad reference" must align to: 5 known probes plus
    # 2 the source file won't have (to exercise the 0.5 fill-in path).
    feature_ids = np.array([f"cgKNOWN{i}" for i in range(5)] + ["cgMISSING0", "cgMISSING1"])
    feature_ids_path = toy_paths["out_dir"] / "feature_ids.npy"
    feature_ids_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_ids_path, feature_ids)

    tcga_dir = toy_paths["out_dir"] / "tcga"
    tcga_dir.mkdir(parents=True, exist_ok=True)
    wide = pd.DataFrame(
        np.random.default_rng(0).uniform(0.05, 0.95, size=(5, 6)),
        index=[f"cgKNOWN{i}" for i in range(5)],
        columns=[f"TCGA-{j:02d}" for j in range(6)],
    )
    wide.index.name = "probe"
    wide.to_csv(tcga_dir / "TCGA-XX.methylation450.tsv.gz", sep="\t")

    cfg = toy_cfg(
        toy_paths,
        pretrain_stage1={
            "combine": {
                "tcga_dir": str(tcga_dir),
                "tcga_glob": "TCGA-*.methylation450.tsv.gz",
                "feature_ids_path": str(feature_ids_path),
            },
            "output_dir": str(toy_paths["out_dir"] / "stage1"),
        },
    )

    result = run_combine(cfg)
    x = np.load(result.x_path)
    assert x.shape == (6, 7)  # 6 samples x 7 feature_ids
    # the two probes absent from the source file are filled with the neutral 0.5 prior
    assert np.allclose(x[:, 5:], 0.5)
    assert result.source_labels_path.exists()
    assert result.probe_overlap_path.exists()
