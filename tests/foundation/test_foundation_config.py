import pytest

import pode_foundation as pf


def test_defaults_load():
    cfg = pf.load_config(None)
    assert cfg["preprocess"]["n_keep"] == 50_000
    assert cfg["model"]["latent_dim"] == 256
    assert cfg["survival"]["mode"] == "frozen"


def test_overrides_win():
    cfg = pf.load_config(None, training={"epochs": 5})
    assert cfg["training"]["epochs"] == 5
    # sibling defaults survive a partial override of the same nested dict
    assert cfg["training"]["mask_ratio"] == 0.15


def test_bad_orientation_rejected():
    with pytest.raises(ValueError, match="orientation"):
        pf.load_config(None, matrix={"orientation": "sideways"})


def test_bad_scale_rejected():
    with pytest.raises(ValueError, match="scale"):
        pf.load_config(None, matrix={"scale": "z_score"})


def test_n_candidates_must_exceed_n_keep():
    with pytest.raises(ValueError, match="n_candidates"):
        pf.load_config(None, preprocess={"n_keep": 100, "n_candidates": 10})


def test_mask_ratio_must_be_fraction():
    with pytest.raises(ValueError, match="mask_ratio"):
        pf.load_config(None, training={"mask_ratio": 1.5})


def test_bad_survival_mode_rejected():
    with pytest.raises(ValueError, match="survival.mode"):
        pf.load_config(None, survival={"mode": "psychic"})


def test_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        pf.load_config("does/not/exist.yaml")
