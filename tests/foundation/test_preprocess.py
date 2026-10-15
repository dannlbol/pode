import numpy as np

from pode_foundation.core.preprocess import run_preprocess

from conftest import toy_cfg


def test_run_preprocess_shapes(toy_paths):
    cfg = toy_cfg(toy_paths, preprocess={"n_keep": 30, "n_candidates": 100})
    result = run_preprocess(cfg)

    assert result.n_features == 30
    x = np.load(result.x_path)
    assert x.shape == (result.n_samples, 30)
    assert x.dtype == np.float32
    assert np.all((x >= 0) & (x <= 1))

    feature_ids = np.load(result.feature_ids_path, allow_pickle=True)
    assert len(feature_ids) == 30
    assert len(set(feature_ids)) == 30  # no duplicate probes kept


def test_run_preprocess_drops_sex_chromosomes(toy_paths):
    import pandas as pd

    anno = pd.read_csv(toy_paths["anno"])
    anno.loc[0, "chr"] = "chrX"
    anno.to_csv(toy_paths["anno"], index=False)

    cfg = toy_cfg(toy_paths, preprocess={"n_keep": 30, "n_candidates": 100})
    result = run_preprocess(cfg)
    feature_ids = set(np.load(result.feature_ids_path, allow_pickle=True))
    assert anno.loc[0, "Name"] not in feature_ids


def test_run_preprocess_reads_parquet_with_probe_id_column(toy_paths):
    """Real cohort exports (e.g. Mset_noob.parquet-style) are probes x samples
    parquet with a `probe_id` column rather than an index -- same convention
    core/embed.py already handles for the embed step."""
    import pandas as pd

    beta_wide = pd.read_csv(toy_paths["beta"])  # samples_by_features, sample_id first col
    beta_features_by_samples = beta_wide.set_index("sample_id").T
    beta_features_by_samples.index.name = "probe_id"
    parquet_path = toy_paths["beta"].with_suffix(".parquet")
    beta_features_by_samples.reset_index().to_parquet(parquet_path, index=False)

    cfg = toy_cfg(
        toy_paths,
        matrix={"path": str(parquet_path), "orientation": "features_by_samples"},
        preprocess={"n_keep": 30, "n_candidates": 100},
    )
    result = run_preprocess(cfg)
    assert result.n_features == 30


def test_run_preprocess_requires_beta_path():
    import pytest
    import pode_foundation as pf

    cfg = pf.load_config(None, preprocess={"annotation_path": "somewhere.csv"})
    with pytest.raises(ValueError, match="beta matrix"):
        run_preprocess(cfg)
