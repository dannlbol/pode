"""SDV bridge, decoder-only engine, and mask-aware TVAE (few epochs, CPU-safe)."""

import numpy as np
import pandas as pd
import pytest
import torch

from pode.config import SynthesizerConfig
from pode.imputation.base import category_key
from pode.imputation.config import LatentConfig, MaskedTVAEConfig
from pode.imputation.decoder_imputer import TVAEDecoderImputer
from pode.imputation.masked_tvae import MaskedTVAEImputer, is_masked_model_dir
from pode.imputation.sdv_bridge import BridgeError, SDVDecoderBridge
from pode.imputation.spans import column_loglik

SMALL = SynthesizerConfig(epochs=5, batch_size=100, embedding_dim=8, compress_dims=(32,), decompress_dims=(32,))
LATENT = LatentConfig(k=200, method="auto", pcn_steps=10)


def _holes(df, seed=0, rate=0.3):
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in ["Age", "bio", "subgroup", "treated", "lat_0"]:
        out.loc[rng.random(len(out)) < rate, c] = np.nan
    return out


def _assert_valid(draws, masked, reference):
    for d in draws:
        assert not d.isna().any().any()
        obs = masked.notna()
        for c in masked.columns:
            assert (category_key(d.loc[obs[c], c]) == category_key(masked.loc[obs[c], c])).all(), c
        assert set(d["subgroup"]) <= set(reference["subgroup"])
        assert set(d["treated"]) <= {0, 1}


@pytest.fixture(scope="module")
def decoder_imputer(module_cohort):
    from pode.schema.schema import DatasetSchema

    df = module_cohort
    return TVAEDecoderImputer(SMALL, LATENT, fit_seed=0).fit(df.iloc[:150], DatasetSchema.from_dataframe(df))


def test_bridge_roundtrip_and_finite_loglik(decoder_imputer, cohort):
    bridge = decoder_imputer.bridge
    x, obs = bridge.encode(cohort.iloc[:20])
    assert obs.all() and x.shape[1] == bridge.spans[-1].end
    back = bridge.to_frame(x)
    assert np.allclose(back["bio"], cohort["bio"].iloc[:20], atol=1e-4)
    assert (category_key(back["subgroup"]) == category_key(cohort["subgroup"].iloc[:20])).all()
    z = torch.zeros(20, bridge.emb_dim, device=bridge.device)
    rec, sig = bridge.decode(z)
    ll = column_loglik(rec, sig, torch.as_tensor(x, device=bridge.device), bridge.spans)
    assert torch.isfinite(ll).all() and ll.shape == (20, len(bridge.spans))


def test_bridge_marks_missing_and_unknown_categories(decoder_imputer, cohort):
    df = cohort.iloc[:5].copy()
    df.loc[df.index[0], "bio"] = np.nan
    df.loc[df.index[1], "subgroup"] = "NEVER_SEEN"
    x, obs = decoder_imputer.bridge.encode(df)
    cols = decoder_imputer.bridge.columns
    assert not obs[0, cols.index("bio")] and not obs[1, cols.index("subgroup")]
    assert np.isfinite(x).all()


def test_bridge_rejects_unfitted():
    with pytest.raises(BridgeError):
        SDVDecoderBridge(object())


def test_decoder_imputer_multiple_draws(decoder_imputer, cohort):
    masked = _holes(cohort.iloc[150:])
    draws = decoder_imputer.impute(masked, m=4, seed=1)
    _assert_valid(draws, masked, cohort)
    hole = masked["bio"].isna()
    assert not np.allclose(draws[0].loc[hole, "bio"], draws[1].loc[hole, "bio"])
    diag = decoder_imputer.diagnostics()
    assert diag["rows_conditioned"] == int(masked.isna().any(axis=1).sum())
    again = decoder_imputer.impute(masked, m=4, seed=1)
    assert np.allclose(draws[2]["bio"], again[2]["bio"])


def test_decoder_imputer_from_saved_model(tmp_path, decoder_imputer, cohort):
    decoder_imputer.bridge.synth.save(str(tmp_path / "model.pkl"))
    decoder_imputer.schema.to_yaml(tmp_path / "schema.yaml")
    from pode.schema.schema import DatasetSchema

    loaded = TVAEDecoderImputer.from_model(tmp_path / "model.pkl", DatasetSchema.from_yaml(tmp_path / "schema.yaml"), LATENT)
    masked = _holes(cohort.iloc[150:170])
    _assert_valid(loaded.impute(masked, m=2, seed=0), masked, cohort)


def test_decoder_imputer_needs_complete_rows(cohort, cohort_schema):
    df = cohort.copy()
    df.loc[:, "bio"] = np.nan
    with pytest.raises(ValueError, match="complete rows"):
        TVAEDecoderImputer(SMALL, LATENT).fit(df, cohort_schema)


def test_masked_tvae_trains_on_incomplete_data_and_roundtrips(tmp_path, cohort, cohort_schema):
    train = _holes(cohort.iloc[:150], seed=1)
    imp = MaskedTVAEImputer(
        SMALL, MaskedTVAEConfig(), LATENT, block_groups={"lat": ["lat_*"]}, epochs=5, fit_seed=0, device="cpu"
    ).fit(train, cohort_schema)
    assert len(imp.loss_history) == 5 and np.isfinite(imp.loss_history).all()
    masked = _holes(cohort.iloc[150:], seed=2)
    draws = imp.impute(masked, m=3, seed=0)
    _assert_valid(draws, masked, cohort)
    lo, hi = imp.transformer._ranges["Age"]
    assert all(d["Age"].between(lo, hi).all() for d in draws)

    imp.save(tmp_path / "masked")
    assert is_masked_model_dir(tmp_path / "masked")
    loaded = MaskedTVAEImputer.load(tmp_path / "masked", device="cpu")
    again = loaded.impute(masked, m=3, seed=0)
    assert np.allclose(pd.to_numeric(again[1]["bio"]), pd.to_numeric(draws[1]["bio"]), atol=1e-4)


def test_masked_tvae_ignores_values_in_missing_cells(monkeypatch, cohort, cohort_schema):
    """Garbage in missing spans must not change training: they are zeroed for
    the encoder and carry zero weight in the reconstruction loss."""
    from pode.imputation import masked_tvae as mt
    from pode.imputation.spans import expand_mask

    train = _holes(cohort.iloc[:120], seed=3)

    def fit_losses():
        imp = MaskedTVAEImputer(SMALL, MaskedTVAEConfig(), LATENT, epochs=3, fit_seed=0, device="cpu")
        return imp.fit(train, cohort_schema).loss_history

    clean = fit_losses()
    original = mt.MaskedDataTransformer.encode

    def poisoned(self, df):
        x, obs = original(self, df)
        spans = mt.spans_from_transformer(self)
        x[~expand_mask(obs, spans)] = 1e3
        return x, obs

    monkeypatch.setattr(mt.MaskedDataTransformer, "encode", poisoned)
    assert fit_losses() == pytest.approx(clean, rel=1e-5)


def test_from_model_on_cpu_and_fallback(tmp_path, monkeypatch, decoder_imputer, cohort):
    from pode.generation import synthesizer as synth_mod
    from pode.schema.schema import DatasetSchema

    decoder_imputer.bridge.synth.save(str(tmp_path / "model.pkl"))
    schema = decoder_imputer.schema
    masked = _holes(cohort.iloc[150:170])

    cpu = TVAEDecoderImputer.from_model(tmp_path / "model.pkl", schema, LATENT, device="cpu")
    assert str(cpu.bridge.device) == "cpu"
    _assert_valid(cpu.impute(masked, m=2, seed=0), masked, cohort)

    # Simulate a full shared GPU: the first (default-device) load fails.
    real_load = synth_mod.TVAEGenerator.load.__func__
    calls = {"n": 0}

    def flaky(cls, path, schema=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA error: out of memory")
        return real_load(cls, path, schema)

    monkeypatch.setattr(synth_mod.TVAEGenerator, "load", classmethod(flaky))
    with pytest.warns(UserWarning, match="using CPU"):
        auto = TVAEDecoderImputer.from_model(tmp_path / "model.pkl", schema, LATENT)
    assert str(auto.bridge.device) == "cpu"
    with pytest.raises(RuntimeError):
        calls["n"] = 0
        TVAEDecoderImputer.from_model(tmp_path / "model.pkl", schema, LATENT, device="cuda")


def test_decoder_imputer_trains_on_cpu(cohort, cohort_schema):
    imp = TVAEDecoderImputer(SMALL, LATENT, fit_seed=0, device="cpu").fit(cohort.iloc[:150], cohort_schema)
    assert str(imp.bridge.device) == "cpu"
    masked = _holes(cohort.iloc[150:170])
    _assert_valid(imp.impute(masked, m=2, seed=0), masked, cohort)
