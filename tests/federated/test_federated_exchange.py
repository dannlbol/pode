import numpy as np
import pandas as pd
import pytest

from pode.config import PodeConfig
from pode.federated import DisclosureLedger, partition_sites
from pode.federated import exchange as X
from pode.federated import generators as G
from pode.pipeline import build_schema
from pode.privacy import PrivacyConfig

from .conftest import make_cohort

COLS = ["Age", "bio", "Sex", "subtype", "OS_Time", "OS_Status"]


class _Copy:
    """A 'generator' that memorises: it releases its training rows."""

    def __init__(self, train):
        self.train = train

    def sample(self, n, seed=0):
        return self.train.sample(n, replace=True, random_state=seed).reset_index(drop=True)


class _Fresh:
    """A 'generator' with no memorisation: fresh draws from the population."""

    def __init__(self, columns):
        self.columns = columns

    def sample(self, n, seed=0):
        return make_cohort(n, seed=1000 + seed)[self.columns]


@pytest.fixture
def setup(cohort):
    df = cohort
    schema = build_schema(df.drop(columns="site"), PodeConfig())
    pcfg = PrivacyConfig(quasi_identifiers={"basic": ["Sex", "subtype", "Age"]}, n_bootstrap=50, n_jobs=2)
    return df, schema, pcfg


def _fake(kind_cls):
    def fit(kind, train, columns, *a, **k):
        model = kind_cls(train[columns].reset_index(drop=True)) if kind_cls is _Copy else kind_cls(columns)
        return G.FittedGenerator("fed_tvae", model, columns)
    return fit


def test_memorising_generator_is_withheld(monkeypatch, setup, fcfg):
    df, schema, pcfg = setup
    monkeypatch.setattr(G, "fit_generator", _fake(_Copy))
    led = DisclosureLedger()
    res = X.local_exchange(partition_sites(df, "site", ledger=led), COLS, schema, None, fcfg, pcfg, log=None)
    assert not res.sites["released"].any()
    assert res.released.empty and res.card["sites_withheld"]
    assert set(led.frame["op"]) == {"local_release"}


def test_fresh_generator_is_released(monkeypatch, setup, fcfg):
    df, schema, pcfg = setup
    monkeypatch.setattr(G, "fit_generator", _fake(_Fresh))
    cfg = fcfg.copy()
    cfg.exchange.release_verdicts = ["PASS", "REVIEW"]
    res = X.local_exchange(partition_sites(df, "site"), COLS, schema, None, cfg, pcfg, log=None)
    # Fresh draws from the pooled population are not memorised, but small-sample
    # checks (and the pool/site distribution gap) can still trip one site.
    assert res.sites["released"].sum() >= 2
    assert res.sites.loc[res.sites["released"], "verdict"].isin(["PASS", "REVIEW"]).all()
    assert set(res.released["_site"]) == set(res.sites.loc[res.sites["released"], "site"])
    assert res.sites["membership_auc"].max() < 0.7


def test_real_local_tvae_exchange_saves(tmp_path, setup, tiny_synth, fcfg):
    df, schema, pcfg = setup
    cfg = fcfg.copy()
    cfg.exchange.release_verdicts = ["PASS", "REVIEW", "FAIL"]  # exercise the save path
    res = X.local_exchange(partition_sites(df, "site"), COLS, schema, tiny_synth, cfg, pcfg, log=None)
    out = res.save(tmp_path / "ex")
    assert (out / "release_card.json").is_file() and (out / "released_synthetic.csv").is_file()
    assert (out / "sites" / "big" / "report.md").is_file()
    assert len(res.released) == res.sites["rows_released"].sum()


def test_exchange_cli(tmp_path, fcfg):
    from pode.cli.main import main

    path = tmp_path / "cohort.csv"
    make_cohort(240).to_csv(path, index=False)
    rc = main(["federated", "exchange", "--data", str(path), "--site-column", "site", "--quick",
               "--device", "cpu", "--n-jobs", "2", "--allow-review", "--output-dir", str(tmp_path / "out")])
    assert rc == 0
    card = pd.read_csv(tmp_path / "out" / "release_sites.csv")
    assert len(card) == 3
