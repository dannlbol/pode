import numpy as np
import pandas as pd

from pode_foundation.core.embed import run_embed
from pode_foundation.core.finetune import run_finetune
from pode_foundation.core.preprocess import run_preprocess
from pode_foundation.core.survival import harrell_c, run_survival, run_survival_frozen
from pode_foundation.core.train import run_pretrain

from conftest import toy_cfg


def test_harrell_c_perfect_concordance_is_one():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 1, 1])
    risk = np.array([4.0, 3.0, 2.0, 1.0])  # perfectly anti-correlated with time = concordant
    assert harrell_c(risk, time, event) == 1.0


def test_harrell_c_handles_nan_risk():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 1, 0])
    risk = np.array([np.nan, 1.0, 2.0])
    c = harrell_c(risk, time, event)
    assert 0.0 <= c <= 1.0


def test_frozen_survival_end_to_end(toy_paths):
    cfg = toy_cfg(toy_paths)
    pp = run_preprocess(cfg)
    ckpt = run_pretrain(cfg, x_path=pp.x_path)
    cfg["embed"]["feature_ids_path"] = str(pp.feature_ids_path)
    emb_result = run_embed(cfg, ckpt.ckpt_path, betas=toy_paths["beta"])

    result = run_survival_frozen(cfg, emb_result.out_path)
    assert 0.0 <= result.c_index <= 1.0
    assert result.n_samples > 0
    assert result.risk_path.exists()

    risk_df = pd.read_csv(result.risk_path)
    assert {"risk", "time", "event", "fold"}.issubset(risk_df.columns)


def test_survival_frozen_auto_generates_folds_when_unset(toy_paths):
    cfg = toy_cfg(toy_paths, survival={"fold_column": None})
    pp = run_preprocess(cfg)
    ckpt = run_pretrain(cfg, x_path=pp.x_path)
    cfg["embed"]["feature_ids_path"] = str(pp.feature_ids_path)
    emb_result = run_embed(cfg, ckpt.ckpt_path, betas=toy_paths["beta"])

    result = run_survival_frozen(cfg, emb_result.out_path)
    assert 0.0 <= result.c_index <= 1.0


def test_run_survival_dispatches_on_mode(toy_paths):
    import pytest

    cfg = toy_cfg(toy_paths)
    with pytest.raises(ValueError, match="unknown mode"):
        run_survival(cfg, mode="telepathic")
    with pytest.raises(ValueError, match="embeddings"):
        run_survival(cfg, mode="frozen")
    with pytest.raises(ValueError, match="ckpt"):
        run_survival(cfg, mode="finetune")


def test_finetune_survival_end_to_end(toy_paths):
    cfg = toy_cfg(toy_paths, survival={"finetune_epochs": 2})
    pp = run_preprocess(cfg)
    ckpt = run_pretrain(cfg, x_path=pp.x_path)
    cfg["embed"]["feature_ids_path"] = str(pp.feature_ids_path)

    result = run_survival(cfg, mode="finetune", ckpt=ckpt.ckpt_path, betas_path=toy_paths["beta"])
    assert 0.0 <= result.c_index <= 1.0 or np.isnan(result.c_index)
    assert result.risk_path.exists()
