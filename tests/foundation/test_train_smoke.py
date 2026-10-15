import numpy as np
import torch

from pode_foundation.core.preprocess import run_preprocess
from pode_foundation.core.train import run_pretrain
from pode_foundation.core.finetune import run_finetune

from conftest import toy_cfg


def test_pretrain_smoke(toy_paths):
    cfg = toy_cfg(toy_paths)
    pp = run_preprocess(cfg)

    result = run_pretrain(cfg, x_path=pp.x_path)
    assert result.ckpt_path.exists()
    assert result.history_path.exists()
    assert np.isfinite(result.best_val_mse)

    ckpt = torch.load(result.ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["n_features"] == pp.n_features
    assert set(ckpt["model_state"].keys())  # non-empty state dict
    assert ckpt["args"]["latent_dim"] == cfg["model"]["latent_dim"]


def test_pretrain_stage_tag_recorded(toy_paths):
    cfg = toy_cfg(toy_paths)
    pp = run_preprocess(cfg)
    result = run_pretrain(cfg, x_path=pp.x_path, stage_tag=1)
    ckpt = torch.load(result.ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["stage"] == 1


def test_finetune_warm_starts_from_pretrain_checkpoint(toy_paths):
    cfg = toy_cfg(toy_paths)
    pp = run_preprocess(cfg)
    stage1 = run_pretrain(cfg, x_path=pp.x_path, stage_tag=1)

    result = run_finetune(cfg, init_ckpt=stage1.ckpt_path, x_path=pp.x_path)
    assert result.ckpt_path.exists()
    ckpt = torch.load(result.ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["stage"] == 2
    assert ckpt["stage1_ckpt"] == str(stage1.ckpt_path)
    # architecture is inherited from Stage 1, not re-read from cfg["model"]
    assert ckpt["args"]["latent_dim"] == cfg["model"]["latent_dim"]


def test_finetune_rejects_feature_count_mismatch(toy_paths):
    import pytest

    cfg = toy_cfg(toy_paths)
    pp = run_preprocess(cfg)
    stage1 = run_pretrain(cfg, x_path=pp.x_path)

    mismatched = toy_paths["out_dir"] / "mismatched_X.npy"
    np.save(mismatched, np.random.rand(pp.n_samples, pp.n_features + 5).astype(np.float32))

    with pytest.raises(ValueError, match="[Ff]eature"):
        run_finetune(cfg, init_ckpt=stage1.ckpt_path, x_path=mismatched)
