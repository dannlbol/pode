"""Tests for the DNA copy-number modality.

Split into three groups:
* converter unit tests (pure numpy/pandas, always run);
* a hybrid encode->latent->decode round-trip (skipped without torch);
* an MMVAE smoke test feeding converter output through a continuous head
  (skipped without torch).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthvae.data.copy_number import (
    ALTERATION_TO_ORDINAL,
    CHROM_ORDER,
    gain_loss_binary_to_calls,
    infer_chrom_lengths,
    make_bin_grid,
    ordinal_to_calls,
    segments_to_matrix,
)

# Optional check against a real segment table; set PODE_CNV_SEGMENTS_CSV to run it.
SEGMENTS_CSV = Path(os.environ.get("PODE_CNV_SEGMENTS_CSV", "/nonexistent"))


def _segments(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "Sample", "Chromosome", "Start_Position",
            "End_Position", "Seg.CN", "Alteration",
        ],
    )


# --------------------------------------------------------------------------- #
# converter unit tests                                                        #
# --------------------------------------------------------------------------- #
def test_length_weighted_mean_across_bin_boundary():
    # chr1:0-15Mb @ +0.5 and chr1:15-20Mb @ -1.0, 10 Mb bins.
    seg = _segments([
        ["s1", "chr1", 0, 15_000_000, 0.5, "gain"],
        ["s1", "chr1", 15_000_000, 20_000_000, -1.0, "loss"],
    ])
    res = segments_to_matrix(
        seg, bin_width=10_000_000, chrom_lengths={"chr1": 20_000_000}
    )
    values = res.continuous.loc["s1"].to_numpy()
    assert res.continuous.shape == (1, 2)
    # bin0 fully covered by the +0.5 segment
    assert values[0] == pytest.approx(0.5)
    # bin1 = (5Mb*0.5 + 5Mb*-1.0) / 10Mb = -0.25
    assert values[1] == pytest.approx(-0.25)


def test_bin_grid_and_inferred_lengths():
    seg = _segments([
        ["s1", "chr1", 0, 25_000_000, 0.1, "balanced"],
        ["s1", "chr2", 0, 8_000_000, 0.1, "balanced"],
    ])
    lengths = infer_chrom_lengths(seg)
    assert lengths == {"chr1": 25_000_000, "chr2": 8_000_000}
    grid = make_bin_grid(lengths, bin_width=10_000_000)
    # chr1 -> 3 bins (last truncated at 25Mb), chr2 -> 1 bin
    assert grid.columns == [
        "chr1:0-10000000", "chr1:10000000-20000000", "chr1:20000000-25000000",
        "chr2:0-8000000",
    ]
    assert grid.spans[2] == ("chr1", 20_000_000, 25_000_000)


def test_discrete_alteration_majority_and_encodings():
    # bin0: more "loss" length than balanced -> ordinal -1
    seg = _segments([
        ["s1", "chr1", 0, 7_000_000, -0.8, "loss"],
        ["s1", "chr1", 7_000_000, 10_000_000, 0.0, "balanced"],
    ])
    ordinal = segments_to_matrix(
        seg, representation="discrete", chrom_lengths={"chr1": 10_000_000},
        discrete_encoding="ordinal",
    ).discrete
    assert ordinal.shape == (1, 1)
    assert ordinal.iloc[0, 0] == ALTERATION_TO_ORDINAL["loss"]

    binary = segments_to_matrix(
        seg, representation="discrete", chrom_lengths={"chr1": 10_000_000},
        discrete_encoding="gain_loss_binary",
    ).discrete
    assert list(binary.columns) == ["chr1:0-10000000::gain", "chr1:0-10000000::loss"]
    assert binary.iloc[0].tolist() == [0.0, 1.0]
    # gain and loss are mutually exclusive per bin
    gain = binary.filter(like="::gain").to_numpy()
    loss = binary.filter(like="::loss").to_numpy()
    assert not np.any((gain == 1) & (loss == 1))


def test_discrete_threshold_mode_without_calls():
    seg = _segments([
        ["s1", "chr1", 0, 10_000_000, 0.5, "x"],     # > gain_threshold -> +1
        ["s1", "chr1", 10_000_000, 20_000_000, -0.5, "x"],   # < loss_threshold -> -1
        ["s1", "chr1", 20_000_000, 30_000_000, 0.05, "x"],   # neutral -> 0
    ])
    res = segments_to_matrix(
        seg, representation="discrete", discrete_from="threshold",
        chrom_lengths={"chr1": 30_000_000},
        gain_threshold=0.2, loss_threshold=-0.2,
    )
    assert res.discrete.iloc[0].tolist() == [1.0, -1.0, 0.0]


def test_both_alignment_and_sample_order():
    seg = _segments([
        ["s1", "chr1", 0, 10_000_000, 0.5, "gain"],
        ["s2", "chr1", 0, 10_000_000, -0.5, "loss"],
    ])
    order = ["s2", "s1", "s3"]  # s3 absent -> neutral-filled row
    res = segments_to_matrix(
        seg, representation="both", chrom_lengths={"chr1": 10_000_000},
        sample_order=order,
    )
    assert res.samples == order
    assert list(res.continuous.index) == order
    assert list(res.discrete.index) == order
    assert res.continuous.loc["s3"].tolist() == [0.0]     # neutral fill
    assert res.discrete.loc["s3"].tolist() == [0.0]       # balanced
    assert res.continuous.loc["s1"].iloc[0] == pytest.approx(0.5)


def test_round_trip_helpers():
    disc = pd.DataFrame(
        {"chr1:0-10000000": [-1.0, 0.0, 1.0]}, index=["a", "b", "c"]
    )
    calls = ordinal_to_calls(disc)
    assert calls["chr1:0-10000000"].tolist() == ["loss", "balanced", "gain"]

    binary = pd.DataFrame(
        {"b::gain": [1.0, 0.0, 0.0], "b::loss": [0.0, 1.0, 0.0]},
        index=["a", "b", "c"],
    )
    back = gain_loss_binary_to_calls(binary)
    assert back["b"].tolist() == ["gain", "loss", "balanced"]


@pytest.mark.skipif(not SEGMENTS_CSV.exists(), reason="PODE_CNV_SEGMENTS_CSV not set")
def test_real_segment_file():
    seg = pd.read_csv(SEGMENTS_CSV).drop(columns=["Unnamed: 0"], errors="ignore")
    res = segments_to_matrix(seg, bin_width=10_000_000, representation="both")
    assert res.continuous.shape[0] == seg["Sample"].nunique()
    assert np.isfinite(res.continuous.to_numpy()).all()
    assert set(np.unique(res.discrete.to_numpy())) <= {-1.0, 0.0, 1.0}
    # columns are all autosomes, in order
    chroms = {c.split(":")[0] for c in res.continuous.columns}
    assert chroms <= set(CHROM_ORDER)


# --------------------------------------------------------------------------- #
# torch-dependent round-trips                                                 #
# --------------------------------------------------------------------------- #
def _tiny_cnv_matrix(n=48, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"chr1:{i}-{i+1}" for i in range(6)]
    return pd.DataFrame(rng.normal(scale=0.4, size=(n, len(cols))), columns=cols)


def test_hybrid_continuous_roundtrip():
    pytest.importorskip("torch")
    from synthvae.compressors.survival_vae import SurvivalVAE, SurvivalVAEConfig
    from synthvae.data.preprocessing import ExpressionPreprocessor

    matrix = _tiny_cnv_matrix()
    rng = np.random.default_rng(1)
    time = rng.gamma(2.0, 300.0, len(matrix))
    event = rng.integers(0, 2, len(matrix)).astype(float)

    pre = ExpressionPreprocessor(matrix_kind="copy_number", n_top_genes=None)
    X = pre.fit_transform(matrix.to_numpy(dtype=float))
    vae = SurvivalVAE(SurvivalVAEConfig(
        n_latent=4, encoder_hidden=(16,), decoder_hidden=(16,),
        survival_hidden=(8,), n_time_bins=6, max_epochs=3, batch_size=16,
        device="cpu", seed=0,
    ))
    vae.fit(X, time, event)
    latents = vae.transform(X)
    decoded = pre.inverse_transform(vae.inverse_transform(latents))
    assert latents.shape == (len(matrix), 4)
    assert decoded.shape == matrix.shape
    assert np.isfinite(decoded).all()


def test_hybrid_discrete_decode_snaps_to_calls():
    pytest.importorskip("torch")
    from synthvae.compressors.survival_vae import SurvivalVAE, SurvivalVAEConfig
    from synthvae.data.preprocessing import ExpressionPreprocessor

    rng = np.random.default_rng(2)
    cols = [f"chr1:{i}-{i+1}" for i in range(6)]
    matrix = pd.DataFrame(
        rng.integers(-1, 2, size=(48, len(cols))).astype(float), columns=cols
    )
    time = rng.gamma(2.0, 300.0, len(matrix))
    event = rng.integers(0, 2, len(matrix)).astype(float)

    pre = ExpressionPreprocessor(
        matrix_kind="copy_number_discrete", n_top_genes=None
    )
    X = pre.fit_transform(matrix.to_numpy(dtype=float))
    vae = SurvivalVAE(SurvivalVAEConfig(
        n_latent=4, encoder_hidden=(16,), decoder_hidden=(16,),
        survival_hidden=(8,), n_time_bins=6, max_epochs=3, batch_size=16,
        device="cpu", seed=0,
    ))
    vae.fit(X, time, event)
    decoded = pre.inverse_transform(vae.inverse_transform(vae.transform(X)))
    assert set(np.unique(decoded)) <= {-1.0, 0.0, 1.0}


def test_mmvae_smoke_continuous_cnv_head():
    pytest.importorskip("torch")
    from synthvae.data.toy import make_cohort, ToyConfig
    from synthvae.mmvae import MMVAEConfig, JointVAESynthesizer

    clinical, expression, survival, _ = make_cohort(
        ToyConfig(n=64, n_genes=20, seed=0)
    )
    cnv = _tiny_cnv_matrix(n=64)
    cfg = MMVAEConfig(
        latent_dim=8, encoder_hidden=(32,), tabular_head_hidden=(16,),
        expression_head_hidden=(16,), survival_head_hidden=(16,),
        n_time_bins=6, batch_size=32, max_epochs=2, patience=5,
        device="cpu", seed=0,
        modalities={"cnv": {"kind": "continuous", "head_hidden": [16]}},
    )
    synth = JointVAESynthesizer(cfg=cfg).fit(
        clinical, expression, survival, omics={"cnv": cnv}
    )
    result = synth.sample(10)
    assert result.modalities["cnv"].shape == (10, cnv.shape[1])
    assert np.isfinite(result.modalities["cnv"].to_numpy()).all()
