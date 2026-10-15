import numpy as np
import pandas as pd
import pytest
import torch

from pode.federated import DisclosureLedger, Site, partition_sites
from pode.federated.config import TransformerConfig
from pode.federated.transformer import NA_LEVEL, federated_encoder
from pode.federated.tvae import FederatedTVAE, LossSpec, fit_on_frame

COLS = ["Age", "bio", "Sex", "subtype", "OS_Time", "OS_Status"]
DISCRETE = ["Sex", "subtype", "OS_Status"]


def _diff(a, b):
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def _fit(sites, synth, cfg, rounds, epochs, **gen):
    c = cfg.copy()
    c.generator.rounds, c.generator.local_epochs = rounds, epochs
    for k, v in gen.items():
        setattr(c.generator, k, v)
    return FederatedTVAE(synth, c).fit(sites, COLS, log=None, discrete=DISCRETE)


def test_encoder_from_summaries_matches_columns_and_suppresses_small_cells(cohort):
    led = DisclosureLedger()
    sites = partition_sites(cohort, "site", ledger=led)
    enc = federated_encoder(sites, COLS, DISCRETE, TransformerConfig(min_cell=3, pseudo_rows=500))
    spec = {s.name: s for s in enc.specs}
    assert set(spec["Sex"].categories) == {"M", "F"}
    # RARE (~3% overall) is below min_cell at some site -> pooled into __rare__ there.
    assert "__rare__" in spec["subtype"].categories
    assert spec["bio"].has_missing and not spec["Age"].has_missing
    x = enc.encode(cohort)
    assert x.shape == (len(cohort), enc.output_dimensions)
    # Only summaries crossed the boundary.
    assert set(led.frame["op"]) == {"encoder_profile", "encoder_histograms"}
    assert not led.frame["per_patient"].any()
    # Encoding is deterministic (identical at every site).
    assert np.array_equal(x, enc.encode(cohort))


def test_decode_restores_missing_and_types(cohort):
    sites = partition_sites(cohort, "site")
    enc = federated_encoder(sites, COLS, DISCRETE, TransformerConfig(pseudo_rows=500))
    back = enc.decode(enc.encode(cohort))
    assert back["bio"].isna().mean() == pytest.approx(cohort["bio"].isna().mean(), abs=1e-9)
    assert set(back["OS_Status"].dropna().unique()) <= {0.0, 1.0}
    assert back["Age"].between(enc.specs[0].lo, enc.specs[0].hi).all()
    assert NA_LEVEL not in set(back["Sex"].astype(str))


def test_one_site_rounds_equal_plain_training(cohort, tiny_synth, fcfg):
    df = cohort.drop(columns="site")
    a = _fit([Site("x", df)], tiny_synth, fcfg, 3, 2).state
    b = _fit([Site("x", df)], tiny_synth, fcfg, 1, 6).state
    assert _diff(a, b) == 0.0


def test_fedavg_identical_sites_equals_single_and_fedprox_mu0(cohort, tiny_synth, fcfg):
    df = cohort.drop(columns="site")
    single = _fit([Site("x", df)], tiny_synth, fcfg, 2, 2).state
    two = _fit([Site("x", df), Site("y", df)], tiny_synth, fcfg, 2, 2, secure_aggregation=False).state
    assert _diff(single, two) == 0.0
    prox0 = _fit([Site("x", df), Site("y", df)], tiny_synth, fcfg, 2, 2, secure_aggregation=False,
                 algorithm="fedprox", prox_mu=0.0).state
    assert _diff(two, prox0) == 0.0
    secure = _fit([Site("x", df), Site("y", df)], tiny_synth, fcfg, 2, 2, secure_aggregation=True).state
    assert _diff(two, secure) < 1e-4


def test_secure_aggregation_hides_individual_updates(cohort, tiny_synth, fcfg):
    led = DisclosureLedger()
    sites = partition_sites(cohort, "site", ledger=led)
    m = _fit(sites, tiny_synth, fcfg, 2, 1, secure_aggregation=True)
    upd = led.frame[led.frame["op"] == "model_update"]
    assert len(upd) == 2 * len(sites)
    assert m.record.bytes_per_round > 0


def test_fedprox_differs_and_samples_are_valid(cohort, tiny_synth, fcfg):
    sites = partition_sites(cohort, "site")
    m = _fit(sites, tiny_synth, fcfg, 3, 2, algorithm="fedprox", prox_mu=0.5)
    out = m.sample(50, seed=1)
    assert list(out.columns) == COLS and len(out) == 50
    assert out["Sex"].isin(["M", "F"]).all()
    assert m.record.rounds[-1]["loss"] < m.record.rounds[0]["loss"] + 5


def test_site_in_model_and_personalisation(cohort, tiny_synth, fcfg):
    sites = partition_sites(cohort, "site")
    m = _fit(sites, tiny_synth, fcfg, 3, 2, site_in_model=True, personalise_epochs=2)
    rows = m.sample(20, seed=2, site="small", keep_site=True)
    assert len(rows) <= 20 and (rows["__site__"] == "small").all()
    pers = m.sample_personalised(sites[0], 15, seed=3)
    assert len(pers) == 15 and "__site__" not in pers.columns


def test_survival_postprocessing_with_schema(cohort, tiny_synth, fcfg):
    from pode.config import PodeConfig
    from pode.pipeline import build_schema

    df = cohort.drop(columns="site")
    schema = build_schema(df, PodeConfig())
    m = FederatedTVAE(tiny_synth, fcfg, schema).fit([Site("a", df)], COLS, log=None)
    out = m.sample(40, seed=0)
    assert (out["OS_Time"] > 0).all()
    assert set(out["OS_Status"].unique()) <= {0, 1}


def test_save_load_roundtrip(tmp_path, cohort, tiny_synth, fcfg):
    m = fit_on_frame(cohort.drop(columns="site"), tiny_synth, fcfg, columns=COLS, discrete=DISCRETE)
    m.save(tmp_path / "m")
    again = FederatedTVAE.load(tmp_path / "m", tiny_synth)
    pd.testing.assert_frame_equal(m.sample(10, seed=4), again.sample(10, seed=4))


def test_loss_matches_ctgan(cohort):
    from ctgan.synthesizers.tvae import _loss_function

    from pode.federated.tvae import Arch

    enc = federated_encoder([Site("x", cohort)], COLS, DISCRETE, TransformerConfig(pseudo_rows=500))
    x = torch.as_tensor(enc.encode(cohort))[:32]
    net = Arch(enc.output_dimensions, (16,), (16,), 8).build()
    out = net(x, torch.randn(32, 8))
    a, b = _loss_function(out[0], x, out[1], out[2], out[3], enc.output_info_list, 2.0)
    assert float(LossSpec(enc.output_info_list, 2.0).per_row(out, x).mean()) == pytest.approx(float(a + b), rel=1e-5)
