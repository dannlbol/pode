import pytest

import pode_mantis as pm


def test_defaults_load():
    cfg = pm.load_config(None)
    assert cfg["beta"]["orientation"] == "samples_by_features"
    assert cfg["reference_config"]["endpoint"] == "os"
    assert cfg["cnv"]["control_mset"] is None
    assert cfg["cnv"]["noise_threshold"] is None


def test_overrides_win():
    cfg = pm.load_config(None, reference_config={"endpoint": "pfs"})
    assert cfg["reference_config"]["endpoint"] == "pfs"
    # sibling defaults survive a partial override of the same nested dict
    assert cfg["beta"]["orientation"] == "samples_by_features"


def test_bad_orientation_rejected():
    with pytest.raises(ValueError, match="orientation"):
        pm.load_config(None, beta={"orientation": "sideways"})


def test_bad_endpoint_rejected():
    with pytest.raises(ValueError, match="endpoint"):
        pm.load_config(None, reference_config={"endpoint": "efs"})


def test_implausible_noise_threshold_rejected():
    with pytest.raises(ValueError, match="noise_threshold"):
        pm.load_config(None, cnv={"noise_threshold": 30.0})


def test_plausible_noise_threshold_accepted():
    cfg = pm.load_config(None, cnv={"noise_threshold": 0.77})
    assert cfg["cnv"]["noise_threshold"] == 0.77


def test_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        pm.load_config("does/not/exist.yaml")


def test_no_default_control_mset_or_noise_threshold():
    """Confirms the deliberate design decision: these two fields must never
    have a shipped default (see config.schema.md)."""
    from pode_mantis.config import _defaults

    d = _defaults()
    assert d["cnv"]["control_mset"] is None
    assert d["cnv"]["noise_threshold"] is None
