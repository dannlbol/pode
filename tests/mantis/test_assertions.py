import numpy as np
import pandas as pd
import pytest

from pode_mantis.core.assertions import check_post_flight, check_pre_flight


def _write_ids(path, ids):
    path.write_text("\n".join(ids) + "\n")


def test_pre_flight_passes_when_aligned(tmp_path):
    ids = [f"S{i}" for i in range(5)]
    beta_order = tmp_path / "beta_order.txt"
    _write_ids(beta_order, ids)
    surv_path = tmp_path / "surv.csv"
    pd.DataFrame({"Patient_ID": ids, "time": range(5), "event": [0, 1] * 2 + [0]}).to_csv(surv_path, index=False)

    report = check_pre_flight(beta_order, surv_path)
    assert report.ok
    report.raise_if_bad()  # should not raise


def test_pre_flight_catches_scrambled_order(tmp_path):
    ids = [f"S{i}" for i in range(5)]
    beta_order = tmp_path / "beta_order.txt"
    _write_ids(beta_order, ids)
    surv_path = tmp_path / "surv.csv"
    pd.DataFrame({"Patient_ID": list(reversed(ids)), "time": range(5), "event": [0] * 5}).to_csv(
        surv_path, index=False
    )

    report = check_pre_flight(beta_order, surv_path)
    assert not report.ok
    with pytest.raises(ValueError, match="Alignment check failed"):
        report.raise_if_bad()


def test_pre_flight_catches_cnv_mismatch(tmp_path):
    ids = [f"S{i}" for i in range(5)]
    beta_order = tmp_path / "beta_order.txt"
    _write_ids(beta_order, ids)
    cnv_order = tmp_path / "cnv_order.txt"
    _write_ids(cnv_order, list(reversed(ids)))
    surv_path = tmp_path / "surv.csv"
    pd.DataFrame({"Patient_ID": ids, "time": range(5), "event": [0] * 5}).to_csv(surv_path, index=False)

    report = check_pre_flight(beta_order, surv_path, cnv_sample_order_path=cnv_order)
    assert not report.ok
    assert any("CNV" in p for p in report.problems)


def test_post_flight_passes_for_well_formed_predictions(tmp_path):
    ids = [f"S{i}" for i in range(4)]
    pred_path = tmp_path / "pred.csv"
    # 16 monotone-non-increasing columns
    curves = np.sort(np.random.default_rng(0).uniform(0, 1, size=(4, 16)), axis=1)[:, ::-1]
    pred_df = pd.DataFrame(curves, index=pd.Index(ids, name="Patient_ID"))
    pred_df.to_csv(pred_path)

    surv_path = tmp_path / "surv.csv"
    pd.DataFrame({"Patient_ID": ids, "time": range(4), "event": [0, 1, 0, 1], "m": [0, 1, 0, 1]}).to_csv(
        surv_path, index=False
    )

    report = check_post_flight(pred_path, surv_path)
    assert report.ok


def test_post_flight_catches_row_count_mismatch(tmp_path):
    pred_path = tmp_path / "pred.csv"
    pd.DataFrame(np.zeros((3, 16)), index=pd.Index([f"S{i}" for i in range(3)], name="Patient_ID")).to_csv(pred_path)
    surv_path = tmp_path / "surv.csv"
    pd.DataFrame({"Patient_ID": [f"S{i}" for i in range(4)], "time": range(4), "event": [0] * 4}).to_csv(
        surv_path, index=False
    )

    report = check_post_flight(pred_path, surv_path)
    assert not report.ok
    assert any("row count" in p for p in report.problems)


def test_post_flight_catches_id_order_mismatch(tmp_path):
    ids = [f"S{i}" for i in range(4)]
    pred_path = tmp_path / "pred.csv"
    pd.DataFrame(np.zeros((4, 16)), index=pd.Index(ids, name="Patient_ID")).to_csv(pred_path)
    surv_path = tmp_path / "surv.csv"
    pd.DataFrame({"Patient_ID": list(reversed(ids)), "time": range(4), "event": [0] * 4}).to_csv(
        surv_path, index=False
    )

    report = check_post_flight(pred_path, surv_path)
    assert not report.ok
    assert any("elementwise" in p for p in report.problems)


def test_post_flight_wrong_column_count(tmp_path):
    ids = [f"S{i}" for i in range(3)]
    pred_path = tmp_path / "pred.csv"
    pd.DataFrame(np.zeros((3, 10)), index=pd.Index(ids, name="Patient_ID")).to_csv(pred_path)
    surv_path = tmp_path / "surv.csv"
    pd.DataFrame({"Patient_ID": ids, "time": range(3), "event": [0] * 3}).to_csv(surv_path, index=False)

    report = check_post_flight(pred_path, surv_path)
    assert not report.ok
    assert any("prediction columns" in p for p in report.problems)


def test_post_flight_nan_rows_explained_by_missing_metastasis(tmp_path):
    ids = [f"S{i}" for i in range(4)]
    curves = np.ones((4, 16)) * 0.5
    curves[1, :] = np.nan  # sample 1's row is fully NaN
    pred_path = tmp_path / "pred.csv"
    pd.DataFrame(curves, index=pd.Index(ids, name="Patient_ID")).to_csv(pred_path)

    surv_path = tmp_path / "surv.csv"
    m = [0, np.nan, 1, 0]  # sample 1 is exactly the one missing metastasis status
    pd.DataFrame({"Patient_ID": ids, "time": range(4), "event": [0] * 4, "m": m}).to_csv(surv_path, index=False)

    report = check_post_flight(pred_path, surv_path)
    assert report.ok  # NaN row is fully explained -> not a problem


def test_post_flight_unexplained_nan_rows_flagged(tmp_path):
    ids = [f"S{i}" for i in range(4)]
    curves = np.ones((4, 16)) * 0.5
    curves[1, :] = np.nan  # sample 1's row is fully NaN, but m is NOT missing for it
    pred_path = tmp_path / "pred.csv"
    pd.DataFrame(curves, index=pd.Index(ids, name="Patient_ID")).to_csv(pred_path)

    surv_path = tmp_path / "surv.csv"
    m = [0, 1, 1, 0]  # nothing missing -> the NaN row is unexplained
    pd.DataFrame({"Patient_ID": ids, "time": range(4), "event": [0] * 4, "m": m}).to_csv(surv_path, index=False)

    report = check_post_flight(pred_path, surv_path)
    assert not report.ok
    assert any("do not exactly match" in p for p in report.problems)
