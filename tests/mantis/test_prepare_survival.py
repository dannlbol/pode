import pandas as pd
import pytest

from pode_mantis.core.prepare_beta import run_prepare_beta
from pode_mantis.core.prepare_survival import run_prepare_survival

from conftest import toy_cfg


def test_prepare_survival_columns_and_shape(toy_paths):
    cfg = toy_cfg(toy_paths)
    result = run_prepare_survival(cfg)
    df = pd.read_csv(result.out_path)
    assert list(df.columns) == ["Patient_ID", "time", "event", "m", "pfs_time", "pfs_event"]
    assert len(df) == len(toy_paths["samples"])
    assert result.n_events_os == int(df["event"].sum())
    assert result.n_events_pfs == int(df["pfs_event"].sum())


def test_prepare_survival_asserts_row_order_against_beta(toy_paths):
    cfg = toy_cfg(toy_paths)
    beta_result = run_prepare_beta(cfg)
    # Positive case: same underlying sample set/order -> should pass silently.
    run_prepare_survival(cfg, sample_order_path=beta_result.sample_order_path)


def test_prepare_survival_rejects_misaligned_sample_order(toy_paths, tmp_path):
    cfg = toy_cfg(toy_paths)
    scrambled = list(reversed(toy_paths["samples"]))
    bad_order_path = tmp_path / "scrambled_order.txt"
    bad_order_path.write_text("\n".join(scrambled) + "\n")

    with pytest.raises(ValueError, match="row order"):
        run_prepare_survival(cfg, sample_order_path=bad_order_path)


def test_prepare_survival_requires_metastasis_column(toy_paths):
    import pode_mantis as pm

    cfg = pm.load_config(
        None,
        survival={
            "clinical_path": str(toy_paths["clinical_path"]),
            "id_column": "pid",
            "time_column": "t",
            "event_column": "e",
            "output_dir": str(toy_paths["out_dir"] / "survival"),
        },
    )
    with pytest.raises(ValueError, match="metastasis_column"):
        run_prepare_survival(cfg)


def test_prepare_survival_does_not_recode_metastasis(toy_paths):
    """MANTIS's generated reference config recodes 0->-1 itself; prepare-survival
    must pass metastasis through unrecoded (0/1), per the documented warning."""
    cfg = toy_cfg(toy_paths)
    result = run_prepare_survival(cfg)
    df = pd.read_csv(result.out_path)
    assert set(df["m"].dropna().unique()) <= {0.0, 1.0}
