import numpy as np
import pytest
import torch

from pode.federated import DisclosureLedger, FederatedConfig, Site, partition_sites
from pode.federated.aggregate import SecureSum, flatten_state, secure_weighted_average, site_weights, \
    weighted_average
from pode.federated.config import dp_epsilon


def test_site_weights_schemes():
    ns = [100, 30, 10]
    assert np.allclose(site_weights(ns, "n"), np.array(ns) / 140)
    assert np.allclose(site_weights(ns, "uniform"), 1 / 3)
    w = site_weights(ns, "sqrt_n")
    assert w[0] > w[1] > w[2] and np.isclose(w.sum(), 1)
    capped = site_weights(ns, "capped", cap_fraction=0.5)
    assert np.isclose(capped.sum(), 1) and capped.max() <= 0.5 + 1e-9
    # Excess goes to the others in proportion to their size.
    assert np.isclose(capped[1] / capped[2], 3.0)


def test_secure_sum_masks_cancel_and_hide():
    rng = np.random.default_rng(0)
    vecs = [rng.normal(0, 1, 1000) for _ in range(4)]
    ss = SecureSum(4, secret=7)
    masked = [ss.mask(i, 3, v) for i, v in enumerate(vecs)]
    assert np.allclose(ss.aggregate(masked), np.sum(vecs, axis=0), atol=1e-5)
    # A single masked vector carries no usable signal about its site's vector.
    as_float = masked[0].view(np.int64).astype(float)
    assert abs(np.corrcoef(as_float, vecs[0])[0, 1]) < 0.1
    # A different round uses different masks.
    assert not np.array_equal(masked[0], ss.mask(0, 4, vecs[0]))


def test_secure_weighted_average_matches_plain():
    states = [{"a": torch.randn(3, 2), "b": torch.randn(5)} for _ in range(3)]
    w = site_weights([10, 20, 30], "n")
    plain = weighted_average(states, w)
    sec = secure_weighted_average(states, w, SecureSum(3, 1), 0)
    for k in plain:
        assert torch.allclose(plain[k], sec[k], atol=1e-5)
    vec, layout = flatten_state(states[0])
    assert vec.size == 11 and [k for k, *_ in layout] == ["a", "b"]


def test_partition_and_ledger(cohort):
    led = DisclosureLedger()
    sites = partition_sites(cohort, "site", ledger=led)
    assert sorted(s.name for s in sites) == ["big", "mid", "small"]
    assert sum(s.n for s in sites) == len(cohort)
    assert "site" not in sites[0]._data.columns
    sites[0].run(lambda d: d["Age"].mean(), label="mean_age")
    sites[1].run(lambda d: d["Age"].to_numpy(), label="ages", per_patient=True)
    s = led.summary().set_index("op")
    assert s.loc["mean_age", "per_patient"] == False  # noqa: E712
    assert s.loc["ages", "per_patient"] == True  # noqa: E712
    assert s.loc["ages", "bytes"] == sites[1].n * 8
    with pytest.raises(ValueError, match="min_site_n"):
        partition_sites(cohort, "site", min_site_n=10_000)


def test_stateful_site_ops_keep_private_state(cohort):
    from pode.federated.sites import stateful

    @stateful
    def remember(data, local, value):
        local["v"] = value
        return len(data)

    site = Site("x", cohort)
    assert site.run(remember, 5) == len(cohort)
    assert site.local["v"] == 5


def test_config_validation_and_roundtrip(tmp_path):
    cfg = FederatedConfig()
    cfg.evaluation.conditions = ["fedavg", "fed_dp_eps3", "fed_dp_eps0p5"]
    cfg.validate()
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    again = FederatedConfig.from_yaml(path)
    assert again.to_dict() == cfg.to_dict()
    assert dp_epsilon("fed_dp_eps3") == 3 and dp_epsilon("fed_dp_eps0p5") == 0.5 and dp_epsilon("fedavg") is None
    bad = FederatedConfig()
    bad.evaluation.conditions = ["nonsense"]
    with pytest.raises(ValueError, match="Unknown condition"):
        bad.validate()
    bad = FederatedConfig()
    bad.dp.enabled = True
    with pytest.raises(ValueError, match="noise_multiplier or dp.target_epsilon"):
        bad.validate()
