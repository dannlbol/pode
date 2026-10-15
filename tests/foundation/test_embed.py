import numpy as np
import pandas as pd

from pode_foundation.core.preprocess import run_preprocess
from pode_foundation.core.train import run_pretrain
from pode_foundation.core.embed import run_embed

from conftest import toy_cfg


def _pretrained(toy_paths):
    cfg = toy_cfg(toy_paths)
    pp = run_preprocess(cfg)
    ckpt = run_pretrain(cfg, x_path=pp.x_path)
    cfg["embed"]["feature_ids_path"] = str(pp.feature_ids_path)
    kept_feature_ids = np.load(pp.feature_ids_path, allow_pickle=True)
    return cfg, ckpt, kept_feature_ids


def test_embed_shapes_and_no_nans(toy_paths):
    cfg, ckpt, _ = _pretrained(toy_paths)
    result = run_embed(cfg, ckpt.ckpt_path, betas=toy_paths["beta"])
    emb = pd.read_csv(result.out_path, index_col=0)
    assert emb.shape == (result.n_samples, result.n_latent)
    assert not emb.isna().values.any()


def test_embed_imputes_partially_missing_probe(toy_paths):
    cfg, ckpt, kept_feature_ids = _pretrained(toy_paths)
    beta = pd.read_csv(toy_paths["beta"])
    # blank one sample's value for a probe the model actually uses -> partial missingness
    col = str(kept_feature_ids[0])
    beta.loc[0, col] = np.nan
    partial_path = toy_paths["out_dir"] / "beta_partial.csv"
    beta.to_csv(partial_path, index=False)

    result = run_embed(cfg, ckpt.ckpt_path, betas=partial_path, out=toy_paths["out_dir"] / "emb_partial.csv")
    emb = pd.read_csv(result.out_path, index_col=0)
    assert not emb.isna().values.any()


def test_embed_handles_probe_entirely_absent_from_cohort(toy_paths):
    """Regression test: a probe with zero support in the target cohort (the
    whole column is missing, not just some samples) used to produce a
    NaN per-feature mean that propagated through the whole dense network,
    NaN-ing every sample's embedding. Should fall back to the neutral 0.5
    prior instead."""
    cfg, ckpt, kept_feature_ids = _pretrained(toy_paths)
    beta = pd.read_csv(toy_paths["beta"])
    # drop an entire probe column the model actually uses -> zero support
    col = str(kept_feature_ids[0])
    beta = beta.drop(columns=[col])
    missing_path = toy_paths["out_dir"] / "beta_missing_probe.csv"
    beta.to_csv(missing_path, index=False)

    result = run_embed(cfg, ckpt.ckpt_path, betas=missing_path, out=toy_paths["out_dir"] / "emb_missing.csv")
    emb = pd.read_csv(result.out_path, index_col=0)
    assert not emb.isna().values.any()
    assert emb.shape[0] == result.n_samples
