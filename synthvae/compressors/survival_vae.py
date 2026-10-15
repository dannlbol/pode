"""Survival-supervised expression VAE (the recommended front end).

A VAE over the expression matrix with a *third head* that predicts discrete-time
survival from the latent. The survival term pushes the latent to retain
prognostically relevant variation that a pure reconstruction objective would
happily discard -- the documented weakness of vanilla VAEs on survival data.

Objective (per batch)::

    L = w_recon * gaussian_nll(x, x_hat)          # reconstruction
      + beta(t)  * KL_freebits(q(z|x) || N(0,I))  # regularisation, annealed
      + w_surv   * disc_time_nll(surv_logits)     # survival supervision

``w_recon`` / ``w_surv`` are either fixed (``uncertainty_weighting=False``) or
learned via homoscedastic uncertainty (``True``). ``beta`` is annealed from 0 to
``beta_max`` over ``kl_warmup_epochs`` and free bits floors each latent dim's KL.

``transform`` returns the posterior mean (a deterministic latent), which is what
the downstream SDV TVAE models. ``inverse_transform`` decodes latents back to
expression space. The class satisfies the :class:`Compressor` contract, so the
pipeline treats it identically to :class:`PCACompressor`.

Requires torch. If torch is unavailable, importing the class still works but
``fit`` raises a clear error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .base import Compressor
from ..survival.discretize import TimeGrid
from ..survival import losses as survloss

try:
    import torch
    import torch.nn as nn
    from ..losses.vae import gaussian_nll, kl_standard_normal, UncertaintyWeighter

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


@dataclass
class SurvivalVAEConfig:
    n_latent: int = 32
    encoder_hidden: tuple = (512, 256)
    decoder_hidden: tuple = (256, 512)
    survival_hidden: tuple = (64,)
    n_time_bins: int = 20
    time_scheme: str = "quantile"
    dropout: float = 0.1
    # objective
    beta_max: float = 1.0
    free_bits: float = 0.02          # nats per latent dim; guards survival dims
    kl_warmup_epochs: int = 20
    w_recon: float = 1.0             # used when uncertainty_weighting=False
    w_surv: float = 1.0
    uncertainty_weighting: bool = True
    recon_reduction: str = "mean_over_batch"
    learn_obs_scale: bool = True     # learn Gaussian obs log-scale for recon
    # optimisation
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 20               # early stopping on val objective
    val_fraction: float = 0.15
    seed: int = 0
    device: str = "auto"
    verbose: bool = False


if _HAS_TORCH:

    def _mlp(in_dim, hidden, out_dim, dropout, final_activation=None):
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.LeakyReLU(0.01)]
            if dropout:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        if final_activation is not None:
            layers += [final_activation]
        return nn.Sequential(*layers)

    class _SurvivalVAENet(nn.Module):
        def __init__(self, n_genes: int, cfg: SurvivalVAEConfig):
            super().__init__()
            self.cfg = cfg
            self.encoder = _mlp(n_genes, cfg.encoder_hidden, 2 * cfg.n_latent, cfg.dropout)
            self.decoder = _mlp(cfg.n_latent, cfg.decoder_hidden, n_genes, cfg.dropout)
            self.survival = _mlp(
                cfg.n_latent, cfg.survival_hidden, cfg.n_time_bins, cfg.dropout
            )
            if cfg.learn_obs_scale:
                self.log_obs_scale = nn.Parameter(torch.zeros(1))
            else:
                self.register_buffer("log_obs_scale", torch.zeros(1))

        def encode(self, x):
            h = self.encoder(x)
            mu, logvar = h.chunk(2, dim=1)
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            if self.training:
                std = torch.exp(0.5 * logvar)
                return mu + std * torch.randn_like(std)
            return mu

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_hat = self.decoder(z)
            surv_logits = self.survival(z)
            return {"mu": mu, "logvar": logvar, "z": z,
                    "x_hat": x_hat, "surv_logits": surv_logits}


class SurvivalVAE(Compressor):
    def __init__(self, config: SurvivalVAEConfig | None = None, **kwargs):
        if not _HAS_TORCH:
            # Allow import/construction; fail loudly only on fit.
            self.cfg = config or SurvivalVAEConfig(**kwargs)
            self.n_latent = self.cfg.n_latent
            self._net = None
            return
        self.cfg = config or SurvivalVAEConfig(**kwargs)
        self.n_latent = self.cfg.n_latent
        self._net = None
        self._grid: TimeGrid | None = None
        self._x_mean = None
        self._x_std = None
        self.history_: list = []

    # -- internals -----------------------------------------------------------
    def _device(self):
        if self.cfg.device != "auto":
            return torch.device(self.cfg.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _standardize(self, X, fit=False):
        if fit:
            self._x_mean = X.mean(axis=0, keepdims=True)
            self._x_std = X.std(axis=0, keepdims=True) + 1e-8
        return (X - self._x_mean) / self._x_std

    # -- Compressor API ------------------------------------------------------
    def fit(self, X, time=None, event=None):
        if not _HAS_TORCH:
            raise RuntimeError(
                "SurvivalVAE.fit requires torch. Install torch, or use "
                "PCACompressor for a torch-free front end."
            )
        if time is None or event is None:
            raise ValueError("SurvivalVAE requires `time` and `event` for supervision.")
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        X = np.asarray(X, dtype=float)
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=float)

        self._grid = TimeGrid.from_times(time, event, cfg.n_time_bins, cfg.time_scheme)
        idx = self._grid.index(time)
        Xs = self._standardize(X, fit=True)

        device = self._device()
        net = _SurvivalVAENet(X.shape[1], cfg).to(device)
        self._net = net

        # train/val split
        n = X.shape[0]
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(n)
        n_val = max(1, int(cfg.val_fraction * n))
        val_ix, tr_ix = perm[:n_val], perm[n_val:]

        def batch_tensors(ix):
            return (
                torch.tensor(Xs[ix], dtype=torch.float32, device=device),
                torch.tensor(idx[ix], dtype=torch.long, device=device),
                torch.tensor(event[ix], dtype=torch.float32, device=device),
            )

        weighter = None
        params = list(net.parameters())
        if cfg.uncertainty_weighting:
            weighter = UncertaintyWeighter(["recon", "surv"]).to(device)
            params = params + list(weighter.parameters())
        opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_val = float("inf")
        best_state = None
        bad = 0

        for epoch in range(cfg.max_epochs):
            beta = cfg.beta_max * min(1.0, (epoch + 1) / max(1, cfg.kl_warmup_epochs))
            net.train()
            ep_perm = rng.permutation(tr_ix)
            for start in range(0, len(ep_perm), cfg.batch_size):
                b = ep_perm[start : start + cfg.batch_size]
                xb, ib, eb = batch_tensors(b)
                out = net(xb)
                recon = gaussian_nll(
                    xb, out["x_hat"], net.log_obs_scale, reduction=cfg.recon_reduction
                )
                kl = kl_standard_normal(
                    out["mu"], out["logvar"], free_bits=cfg.free_bits
                )
                surv = survloss.nll_torch(out["surv_logits"], ib, eb)
                if weighter is not None:
                    data_term, _ = weighter({"recon": recon, "surv": surv})
                else:
                    data_term = cfg.w_recon * recon + cfg.w_surv * surv
                loss = data_term + beta * kl
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()

            # validation objective (no KL warmup discount; use beta_max)
            net.eval()
            with torch.no_grad():
                xb, ib, eb = batch_tensors(val_ix)
                out = net(xb)
                recon = gaussian_nll(
                    xb, out["x_hat"], net.log_obs_scale, reduction=cfg.recon_reduction
                )
                kl = kl_standard_normal(out["mu"], out["logvar"], free_bits=cfg.free_bits)
                surv = survloss.nll_torch(out["surv_logits"], ib, eb)
                val_obj = float(
                    (cfg.w_recon * recon + cfg.w_surv * surv + cfg.beta_max * kl).item()
                )
            self.history_.append(
                {"epoch": epoch, "val_obj": val_obj,
                 "recon": float(recon), "surv": float(surv), "kl": float(kl)}
            )
            if cfg.verbose:
                print(f"[{epoch:03d}] val_obj={val_obj:.4f} recon={float(recon):.4f} "
                      f"surv={float(surv):.4f} kl={float(kl):.4f} beta={beta:.3f}")

            if val_obj < best_val - 1e-4:
                best_val = val_obj
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        return self

    def transform(self, X):
        if not _HAS_TORCH or self._net is None:
            raise RuntimeError("SurvivalVAE is not fitted (or torch missing).")
        self._net.eval()
        # ensure we create tensors on the same device as the model
        model_device = next(self._net.parameters()).device
        Xs = self._standardize(np.asarray(X, dtype=float), fit=False)
        with torch.no_grad():
            xb = torch.tensor(Xs, dtype=torch.float32, device=model_device)
            mu, _ = self._net.encode(xb)
        return mu.cpu().numpy()

    def inverse_transform(self, Z):
        if not _HAS_TORCH or self._net is None:
            raise RuntimeError("SurvivalVAE is not fitted (or torch missing).")
        self._net.eval()
        # create tensors on the same device where the model parameters live
        model_device = next(self._net.parameters()).device
        with torch.no_grad():
            zb = torch.tensor(np.asarray(Z, dtype=float), dtype=torch.float32, device=model_device)
            xs = self._net.decoder(zb).cpu().numpy()
        return xs * self._x_std + self._x_mean

    def survival_logits(self, Z):
        """Predicted discrete-time hazard logits for latents ``Z`` (n, K)."""
        self._net.eval()
        model_device = next(self._net.parameters()).device
        with torch.no_grad():
            zb = torch.tensor(np.asarray(Z, dtype=float), dtype=torch.float32, device=model_device)
            return self._net.survival(zb).cpu().numpy()

    def save(self, path):
        torch.save(
            {
                "cfg": self.cfg,
                "state_dict": self._net.state_dict(),
                "n_genes": self._net.decoder[-1].out_features,
                "grid": self._grid.to_dict(),
                "x_mean": self._x_mean,
                "x_std": self._x_std,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(config=blob["cfg"])
        obj._grid = TimeGrid.from_dict(blob["grid"])
        obj._x_mean = blob["x_mean"]
        obj._x_std = blob["x_std"]
        net = _SurvivalVAENet(blob["n_genes"], blob["cfg"])
        net.load_state_dict(blob["state_dict"])
        obj._net = net
        return obj
