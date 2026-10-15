import pytest
import prognostk as pk
from conftest import toy_cfg


def test_defaults_load():
    cfg = pk.load_config(None)
    assert cfg["survival"]["horizon"] == 5
    assert len(cfg["risk_groups"]["labels"]) == len(cfg["risk_groups"]["thresholds"]) + 1


def test_overrides_win():
    cfg = pk.load_config(None, survival={"horizon": 10})
    assert cfg.horizon == 10


def test_missing_required_columns():
    with pytest.raises(ValueError, match="columns.time"):
        pk.validate_config({"columns": {"time": None, "status": "s"}})


def test_bad_thresholds():
    with pytest.raises(ValueError, match="descending"):
        pk.load_config(None, risk_groups={"thresholds": [0.5, 0.7],
                                          "labels": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="length"):
        pk.load_config(None, risk_groups={"thresholds": [0.7, 0.5],
                                          "labels": ["a", "b"]})


def test_col_resolution():
    cfg = toy_cfg()
    assert cfg.col("time") == "OS_Time"
    assert cfg.col("time", "custom") == "custom"
