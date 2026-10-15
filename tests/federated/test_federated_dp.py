import numpy as np
import pytest
import torch

from pode.federated import Site, partition_sites
from pode.federated.config import TransformerConfig
from pode.federated.transformer import federated_encoder
from pode.federated.tvae import Arch, FederatedTVAE, LossSpec

COLS = ["Age", "bio", "Sex", "subtype", "OS_Time", "OS_Status"]
DISCRETE = ["Sex", "subtype", "OS_Status"]


def _setup(cohort):
    enc = federated_encoder([Site("x", cohort)], COLS, DISCRETE, TransformerConfig(pseudo_rows=500))
    net = Arch(enc.output_dimensions, (16,), (16,), 8).build()
    return enc, net, LossSpec(enc.output_info_list, 2.0)


def test_per_sample_grads_match_row_loop(cohort):
    from pode.federated.dp import per_sample_grads

    enc, net, loss = _setup(cohort)
    x = torch.as_tensor(enc.encode(cohort))[:6]
    eps = torch.randn(6, 8)
    g = per_sample_grads(net, loss.per_row, x, eps)
    for i in range(6):
        net.zero_grad()
        loss.per_row(net(x[i:i + 1], eps[i:i + 1]), x[i:i + 1])[0].backward()
        for k, p in net.named_parameters():
            # sigma's gradient (used only in the loss) is captured too.
            assert torch.allclose(p.grad, g[k][i], atol=1e-3, rtol=1e-3), k


def test_clipping_bounds_each_contribution():
    from pode.federated.dp import clip_and_noise

    grads = {"w": torch.randn(10, 4, 3) * 50, "b": torch.randn(10, 3) * 50}
    out, norms = clip_and_noise(grads, max_norm=1.0, noise_multiplier=0.0, expected_batch=1.0)
    total = torch.sqrt(out["w"].pow(2).sum() + out["b"].pow(2).sum())
    assert norms.min() > 1.0 and float(total) <= 10.0 + 1e-4


def test_accounting_monotone_and_target():
    pytest.importorskip("opacus")
    from pode.federated.dp import epsilon_spent, noise_for_target

    e1 = epsilon_spent(1.0, 0.1, 100, 1e-5)
    e2 = epsilon_spent(1.0, 0.1, 400, 1e-5)
    e3 = epsilon_spent(2.0, 0.1, 400, 1e-5)
    assert e2 > e1 and e3 < e2
    sigma = noise_for_target(3.0, 1e-5, 0.1, 400)
    assert epsilon_spent(sigma, 0.1, 400, 1e-5) == pytest.approx(3.0, abs=0.1)


def test_dp_federated_fit_meets_target(cohort, tiny_synth, fcfg):
    pytest.importorskip("opacus")
    cfg = fcfg.copy()
    cfg.dp.enabled, cfg.dp.target_epsilon, cfg.dp.batch_size = True, 4.0, 32
    cfg.transformer.bounds = {"Age": [0, 18], "bio": [-5, 12], "OS_Time": [0, 30]}
    sites = partition_sites(cohort, "site")
    m = FederatedTVAE(tiny_synth, cfg).fit(sites, COLS, log=None, discrete=DISCRETE)
    for name, p in m.record.privacy.items():
        assert p["dp"] and p["epsilon_sgd"] == pytest.approx(4.0, abs=0.15)
        assert p["epsilon_total"] == pytest.approx(p["epsilon_sgd"] + cfg.dp.transformer_epsilon)
        assert p["steps"] == p["planned_steps"]
        # Public bounds remove the range notes; vocabularies remain data-derived.
        assert not any("range of" in n for n in p["transformer_notes"])
    assert len(m.sample(20, seed=0)) == 20


def test_dp_without_opacus_raises(monkeypatch, cohort, tiny_synth, fcfg):
    import builtins

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "opacus" or name.startswith("opacus."):
            raise ImportError("no opacus")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    cfg = fcfg.copy()
    cfg.dp.enabled, cfg.dp.target_epsilon = True, 3.0
    with pytest.raises(ImportError, match=r"pip install -e '\.\[federated\]'"):
        FederatedTVAE(tiny_synth, cfg).fit(partition_sites(cohort, "site"), COLS, log=None)
