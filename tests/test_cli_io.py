"""Tests for the shared CLI CSV loader that strips the unifying sample_id."""

from __future__ import annotations

import pandas as pd

from synthvae.cli._io import read_table


def test_read_table_drops_sample_id(tmp_path):
    path = tmp_path / "matrix.csv"
    pd.DataFrame(
        {"sample_id": ["GSM1", "GSM2"], "chr1:0-10": [0.1, -0.2], "chr1:10-20": [0.0, 0.3]}
    ).to_csv(path, index=False)

    out = read_table(path)

    assert "sample_id" not in out.columns
    assert list(out.columns) == ["chr1:0-10", "chr1:10-20"]
    assert out.select_dtypes("number").shape[1] == out.shape[1]  # no phantom feature
    assert len(out) == 2


def test_read_table_noop_without_sample_id(tmp_path):
    """Backward compatible with id-less matrices written before the id column."""
    path = tmp_path / "matrix.csv"
    frame = pd.DataFrame({"gene_a": [1.0, 2.0], "gene_b": [3.0, 4.0]})
    frame.to_csv(path, index=False)

    out = read_table(path)

    pd.testing.assert_frame_equal(out, frame)
