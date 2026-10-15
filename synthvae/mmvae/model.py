"""JointMultimodalVAE -- the shared-latent multimodal generative model.

Ties together per-modality encoders (encoders.py), a fusion strategy
(experts.py), optional gating (experts.GatingNetwork), and three decoder heads
(heads.py) under one config. The ``forward`` pass (encode -> fuse -> sample ->
decode -> per-head NLL) is implemented. The training loop and the generation
routine are provided as documented stubs -- the SurvivalVAE in
``synthvae.compressors.survival_vae`` is the reference implementation for the
loop (KL warmup, free bits, uncertainty weighting, early stopping), so an agent
should mirror it here.

See docs/SPECIFICATION.md for the full contract and the option matrix. This file
requires torch; import-guarded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

try:
    import torch
    import torch.nn as nn

    from .encoders import TabularEncoder, ModalityEncoder
    from .heads import TabularHead, ExpressionHead, BinaryHead, SurvivalHead
    from . import experts as experts_mod
    from . import data as data_mod
    from .objectives import ObjectiveAssembler, beta_schedule

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


@dataclass
class MMVAEConfig:
    # dimensions (set from data at build time)
    tabular_dim: int = 0
    expression_dim: int = 0
    # methylation is an optional 4th modality: 0 disables it (existing 3-modality
    # behaviour is then byte-identical). build_from_frames sets it when a
    # methylation matrix is supplied.
    methylation_dim: int = 0
    modalities: dict = field(default_factory=dict)
    modality_dims: dict = field(default_factory=dict)
    active_modalities: list = field(default_factory=list)
    n_time_bins: int = 20
    latent_dim: int = 32
    # ``shared`` is the historical architecture. ``shared_private`` adds a
    # small modality-specific posterior to molecular decoder inputs while
    # leaving clinical/survival on the fused shared latent.
    latent_structure: str = "shared"
    private_latent_dims: dict = field(default_factory=dict)

    # architecture
    encoder_hidden: tuple = (512, 256)
    tabular_head_hidden: tuple = (128,)
    expression_head_hidden: tuple = (256, 512)
    methylation_head_hidden: tuple = (256, 512)
    survival_head_hidden: tuple = (64,)
    dropout: float = 0.1

    # fusion + gating  (the options you asked to explore)
    fusion: str = "poe"                 # poe | moe | mopoe
    gating: bool = True                 # enable data-dependent modality gating
    include_prior_expert: bool = True

    # likelihoods
    expression_likelihood: str = "gaussian"   # gaussian | nb | zinb
    methylation_likelihood: str = "gaussian"  # gaussian on M-values (only option)

    # tabular encoding
    tabular_embeddings: bool = False          # per-column embeddings for categoricals

    # objective / weighting
    weighting_scheme: str = "uncertainty"     # fixed | uncertainty | gradnorm
    fixed_weights: dict = field(default_factory=lambda: {
        "tabular": 1.0, "expression": 1.0, "methylation": 1.0, "survival": 1.0
    })
    beta_max: float = 1.0
    free_bits: float = 0.02
    kl_warmup_epochs: int = 20
    recon_reduction: str = "mean_per_feature"
    gradnorm_alpha: float = 1.5               # asymmetry for gradnorm targets
    private_kl_weight: float = 1.0

    # optimisation (mirror SurvivalVAEConfig)
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 300
    patience: int = 25
    val_fraction: float = 0.15
    seed: int = 0
    device: str = "auto"
    verbose: bool = False

    # generation
    latent_source: str = "posterior"          # posterior (GMM) | prior
    gmm_components: int = 10
    posterior_samples_per_patient: int = 4     # draws used to fit the ex-post GMM
    # Gaussian observation generation.  ``independent`` is the historical
    # scalar-noise path and remains the fallback for old checkpoints.
    # ``residual_factor`` uses the fitted, correlated residual model stored with
    # new checkpoints; ``mean`` emits decoder means only.
    observation_sampler: str = "auto"  # auto | mean | independent | residual_factor
    residual_rank: int = 16
    observation_noise_scale: dict = field(default_factory=dict)

    def resolved_modalities(self, provided=()):
        """Normalize legacy settings and registry entries into modality specs."""
        specs = {
            "expression": {
                "kind": "count" if self.expression_likelihood in ("nb", "zinb") else "continuous",
                "likelihood": self.expression_likelihood,
                "head_hidden": tuple(self.expression_head_hidden),
            }
        }
        if "methylation" in provided or self.methylation_dim:
            specs["methylation"] = {
                "kind": "methylation", "likelihood": "gaussian",
                "head_hidden": tuple(self.methylation_head_hidden),
            }
        for name, raw in (self.modalities or {}).items():
            spec = dict(raw or {})
            if name == "expression":
                specs[name].update(spec)
            else:
                if "kind" not in spec:
                    raise ValueError(f"modality {name!r} must declare a kind")
                default_likelihood = (
                    "bernoulli" if spec["kind"] == "binary"
                    else "nb" if spec["kind"] == "count"
                    else "gaussian"
                )
                spec.setdefault("likelihood", default_likelihood)
                spec.setdefault("head_hidden", tuple(self.expression_head_hidden))
                specs[name] = spec
        return specs

    def resolved_private_dims(self, modality_names=()):
        """Resolve private dimensions without changing shared-only behaviour."""
        structure = getattr(self, "latent_structure", "shared")
        if structure not in {"shared", "shared_private"}:
            raise ValueError("latent_structure must be 'shared' or 'shared_private'")
        raw = dict(getattr(self, "private_latent_dims", {}) or {})
        names = list(modality_names)
        unknown = sorted(set(raw) - {"default", *names})
        if unknown:
            raise ValueError(
                "private_latent_dims contains unknown modalities: " + ", ".join(unknown))
        for name, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"private latent dimension for {name!r} must be an integer")
            if int(value) < 0:
                raise ValueError(f"private latent dimension for {name!r} cannot be negative")
        if structure == "shared":
            return {name: 0 for name in names}
        default = int(raw.get("default", 0))
        return {name: int(raw.get(name, default)) for name in names}


if _HAS_TORCH:

    class JointMultimodalVAE(nn.Module):
        def __init__(self, cfg: MMVAEConfig, tabular_column_specs):
            super().__init__()
            self.cfg = cfg
            self.tabular_column_specs = tabular_column_specs

            self.modality_specs = cfg.resolved_modalities(cfg.active_modalities)
            self.modality_names = list(cfg.active_modalities or self.modality_specs)
            if "expression" not in self.modality_names:
                self.modality_names.insert(0, "expression")
            if getattr(cfg, "observation_sampler", "auto") == "auto":
                # Preserve the established expression-only path while making old
                # multi-head configs opt into the safer correlated sampler.
                cfg.observation_sampler = (
                    "residual_factor" if len(self.modality_names) > 1 else "independent")
            self.has_methylation = "methylation" in self.modality_names
            self.private_dims = cfg.resolved_private_dims(self.modality_names)
            n_encoded = 1 + len(self.modality_names)

            self.enc_tab = TabularEncoder(
                cfg.tabular_dim, cfg.latent_dim, cfg.encoder_hidden, cfg.dropout
            )
            dims = dict(cfg.modality_dims or {})
            dims.setdefault("expression", cfg.expression_dim)
            if cfg.methylation_dim:
                dims.setdefault("methylation", cfg.methylation_dim)
            self.enc_modalities = nn.ModuleDict({
                name: ModalityEncoder(dims[name], cfg.latent_dim, cfg.encoder_hidden, cfg.dropout)
                for name in self.modality_names
            })
            if any(self.private_dims.values()):
                # Construct this registry only for the opt-in path. Shared-only
                # checkpoints therefore retain their historical state_dict keys.
                self.enc_private_modalities = nn.ModuleDict({
                    name: ModalityEncoder(
                        dims[name], self.private_dims[name], cfg.encoder_hidden, cfg.dropout)
                    for name in self.modality_names if self.private_dims[name] > 0
                })
            self.gating = (
                experts_mod.GatingNetwork(cfg.latent_dim, n_modalities=n_encoded)
                if cfg.gating else None
            )

            self.head_tab = TabularHead(
                cfg.latent_dim, tabular_column_specs, cfg.tabular_head_hidden, cfg.dropout
            )
            heads = {}
            for name in self.modality_names:
                spec = self.modality_specs[name]
                decoder_dim = cfg.latent_dim + self.private_dims[name]
                if spec.get("kind") == "binary":
                    heads[name] = BinaryHead(decoder_dim, dims[name], tuple(spec["head_hidden"]),
                                             cfg.dropout, cfg.recon_reduction)
                else:
                    heads[name] = ExpressionHead(decoder_dim, dims[name], tuple(spec["head_hidden"]),
                                                 cfg.dropout, likelihood=spec.get("likelihood", "gaussian"),
                                                 reduction=cfg.recon_reduction)
            self.head_modalities = nn.ModuleDict(heads)
            self.head_surv = SurvivalHead(
                cfg.latent_dim, cfg.n_time_bins, cfg.survival_head_hidden, cfg.dropout
            )

            modality_names = ["tabular", *self.modality_names, "survival"]
            self.objective = ObjectiveAssembler(
                modality_names,
                scheme=cfg.weighting_scheme,
                fixed_weights=cfg.fixed_weights,
                free_bits=cfg.free_bits,
                private_kl_weight=getattr(cfg, "private_kl_weight", 1.0),
            )

        # Legacy aliases intentionally remain properties, not registered modules.
        @property
        def enc_expr(self): return self.enc_modalities["expression"]
        @property
        def head_expr(self): return self.head_modalities["expression"]
        @property
        def enc_meth(self): return self.enc_modalities["methylation"] if "methylation" in self.enc_modalities else None
        @property
        def head_meth(self): return self.head_modalities["methylation"] if "methylation" in self.head_modalities else None

        # -- inference -------------------------------------------------------
        def fuse(self, mus, logvars):
            # Gating rebalances multiple experts; with a single present modality
            # there is nothing to rebalance, so fall back to unweighted fusion.
            weights = None
            if self.gating is not None and len(mus) == self.gating.n_modalities:
                weights = self.gating(mus)
            if self.cfg.fusion == "poe":
                return experts_mod.product_of_experts(
                    mus, logvars, weights, include_prior=self.cfg.include_prior_expert
                )
            if self.cfg.fusion == "moe":
                return experts_mod.mixture_of_experts(mus, logvars, weights)
            if self.cfg.fusion == "mopoe":
                return experts_mod.mixture_of_products_of_experts(mus, logvars, weights)
            raise ValueError(f"unknown fusion {self.cfg.fusion!r}")

        def encode(self, x_tab, x_expr=None, x_meth=None, present=None, modalities=None):
            """Encode present modalities and fuse. ``present`` lets you drop a
            modality at inference (missing-modality generation). It defaults to all
            encoded modalities the model was built with -- ``methylation`` is
            included only when the methylation encoder exists."""
            modalities = dict(modalities or {})
            if x_expr is not None:
                modalities.setdefault("expression", x_expr)
            if x_meth is not None:
                modalities.setdefault("methylation", x_meth)
            if present is None:
                present = ["tabular", *self.modality_names]
            mus, logvars = [], []
            if "tabular" in present:
                mu, lv = self.enc_tab(x_tab)
                mus.append(mu); logvars.append(lv)
            for name in self.modality_names:
                if name not in present:
                    continue
                if name not in modalities:
                    raise ValueError(f"missing active modality tensor {name!r}")
                mu, lv = self.enc_modalities[name](modalities[name])
                mus.append(mu); logvars.append(lv)
            return self.fuse(mus, logvars)

        def reparameterize(self, mu, logvar):
            if self.training:
                std = torch.exp(0.5 * logvar)
                return mu + std * torch.randn_like(std)
            return mu

        def encode_private(self, modalities):
            """Return modality-private posterior parameters for enabled heads."""
            encoders = getattr(self, "enc_private_modalities", None)
            if encoders is None:
                return {}
            return {name: encoders[name](modalities[name]) for name in encoders}

        def decoder_latent(self, name, shared_z, private_z=None):
            dim = self.private_dims.get(name, 0)
            if dim == 0:
                return shared_z
            if private_z is None or name not in private_z:
                raise ValueError(f"missing private latent for modality {name!r}")
            return torch.cat([shared_z, private_z[name]], dim=1)

        def latent_layout(self):
            """Stable slices for the joint shared/private ex-post density."""
            offset = self.cfg.latent_dim
            private = {}
            for name in self.modality_names:
                dim = self.private_dims.get(name, 0)
                if dim:
                    private[name] = (offset, offset + dim)
                    offset += dim
            return {"shared": (0, self.cfg.latent_dim), "private": private,
                    "total_dim": offset}

        def split_latents(self, joint_z):
            layout = self.latent_layout()
            lo, hi = layout["shared"]
            private = {name: joint_z[:, sl[0]:sl[1]]
                       for name, sl in layout["private"].items()}
            return joint_z[:, lo:hi], private

        def _encode_decode(self, batch):
            """Shared body of forward: encode -> sample z -> per-head NLL.

            Returns ``(recon, mu, logvar, z, private)``. ``z`` retains grad so GradNorm can
            differentiate per-task losses w.r.t. the shared latent.
            """
            modal = {name: values["x"] for name, values in batch["modalities"].items()}
            mu, logvar = self.encode(batch["x_tab"], modalities=modal)
            z = self.reparameterize(mu, logvar)
            private_posteriors = self.encode_private(modal)
            private_z = {name: self.reparameterize(*posterior)
                         for name, posterior in private_posteriors.items()}
            recon = {
                "tabular": self.head_tab.nll(z, batch["tab_target"]),
                "survival": self.head_surv.nll(z, batch["surv_target"]),
            }
            for name, values in batch["modalities"].items():
                recon[name] = self.head_modalities[name].nll(
                    self.decoder_latent(name, z, private_z),
                    values["target"], library=values.get("library"))
            return recon, mu, logvar, z, private_posteriors

        def forward(self, batch):
            """batch: dict with x_tab, x_expr, expr_target, library, tab_target,
            surv_target. Returns (total_loss, logs) for the current beta (pass
            beta via ``batch['beta']`` or default to beta_max)."""
            recon, mu, logvar, _, private = self._encode_decode(batch)
            beta = batch.get("beta", self.cfg.beta_max)
            return self.objective(recon, mu, logvar, beta, private)

        # -- generation ------------------------------------------------------
        def fit_latent_density(self, Z, n_components=None):
            """Fit a GMM on encoded training latents for ex-post sampling.

            Avoids prior-hole artefacts (SPEC sec. 6b) -- the same trick the hybrid
            uses when the TVAE models the latents. Silently no-ops if sklearn is
            unavailable, in which case ``sample`` falls back to the prior.
            """
            try:
                from sklearn.mixture import GaussianMixture
            except Exception:
                self._latent_gmm = None
                return self
            Z = np.asarray(Z, dtype=float)
            if not np.isfinite(Z).all():
                # training produced non-finite latents; fall back to prior sampling
                self._latent_gmm = None
                return self
            k = n_components or self.cfg.gmm_components
            k = max(1, min(k, len(Z)))
            self._latent_gmm = GaussianMixture(
                n_components=k, covariance_type="full", random_state=self.cfg.seed
            ).fit(Z)
            return self

        def _draw_latents(self, n, source, device):
            if source == "posterior" and getattr(self, "_latent_gmm", None) is not None:
                z, _ = self._latent_gmm.sample(n)
                rng = np.random.default_rng(self.cfg.seed)
                z = z[rng.permutation(n)]  # GMM.sample orders by component
                return torch.tensor(z, dtype=torch.float32, device=device)
            return torch.randn(n, self.latent_layout()["total_dim"], device=device)

        def _noise_scale(self, name):
            scales = getattr(self.cfg, "observation_noise_scale", {}) or {}
            if isinstance(scales, (int, float)):
                return float(scales)
            return float(scales.get(name, scales.get("default", 1.0)))

        def _draw_residuals(self, n):
            """Draw paired low-rank residuals for every fitted Gaussian head.

            Factor scores are sampled jointly, so residual correlations are
            retained both within and between molecular modalities.  The small
            diagonal remainder is per-feature rather than the former one-scale-
            fits-all white noise.
            """
            state = getattr(self, "_residual_sampler", None)
            if not state:
                return {}
            rng = np.random.default_rng(self.cfg.seed)
            score_dim = int(state.get("score_dim", 0))
            if score_dim:
                cov = np.asarray(state["score_cov"], dtype=float)
                scores = rng.multivariate_normal(
                    np.zeros(score_dim, dtype=float), cov, size=n,
                    check_valid="ignore",
                )
            else:
                scores = np.zeros((n, 0), dtype=float)
            draws = {}
            for name, block in state.get("modalities", {}).items():
                lo, hi = block["score_slice"]
                components = np.asarray(block["components"], dtype=float)
                mean = np.asarray(block["mean"], dtype=float)
                diag = np.asarray(block["diag_scale"], dtype=float)
                residual = np.broadcast_to(mean, (n, len(mean))).copy()
                if hi > lo:
                    residual += scores[:, lo:hi] @ components
                if np.any(diag > 0):
                    residual += rng.normal(size=residual.shape) * diag
                draws[name] = residual * self._noise_scale(name)
            return draws

        @torch.no_grad()
        def sample(self, n: int, latent_source=None, sample_categoricals=True,
                   sample_observations=True, device=None):
            """Decode ``n`` synthetic patients into *model-space* arrays.

            Returns a dict ``{z, tab_continuous, tab_cat_codes, expr, surv_logits}``.
            The orchestrator (:class:`generate.JointVAESynthesizer`) maps these back
            to raw DataFrames and samples ``(time, event)`` from ``surv_logits``.
            """
            self.eval()
            device = device or next(self.parameters()).device
            source = latent_source or self.cfg.latent_source
            # All stochastic generative stages (latent, categorical and molecular
            # likelihood draws) are reproducible for the configured seed.
            torch.manual_seed(self.cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.cfg.seed)
            joint_z = self._draw_latents(n, source, device)
            z, private_z = self.split_latents(joint_z)
            sampler = getattr(self.cfg, "observation_sampler", "independent")
            if sampler not in {"mean", "independent", "residual_factor"}:
                raise ValueError(f"unknown observation_sampler {sampler!r}")
            residual_draws = (
                self._draw_residuals(n)
                if sample_observations and sampler == "residual_factor"
                else {}
            )

            tab = self.head_tab.forward(z)
            tab_cont = tab["continuous"].cpu().numpy() if tab["continuous"] is not None else None
            cat_codes = []
            for logits in tab["categorical"]:
                if sample_categoricals:
                    probs = torch.softmax(logits, dim=1)
                    codes = torch.multinomial(probs, 1).squeeze(1)
                else:
                    codes = logits.argmax(dim=1)
                cat_codes.append(codes.cpu().numpy())

            modalities = {}
            for name, head in self.head_modalities.items():
                decoder_z = self.decoder_latent(name, z, private_z)
                out = head.forward(decoder_z)
                spec = self.modality_specs[name]
                if spec.get("kind") == "binary":
                    probs = torch.sigmoid(out)
                    out = torch.bernoulli(probs) if sample_observations else probs
                elif spec.get("likelihood") in ("nb", "zinb"):
                    mu = torch.nn.functional.softplus(out).clamp(min=1e-8)
                    if sample_observations:
                        theta = torch.exp(head.log_theta).clamp(min=1e-4)
                        # Gamma-Poisson construction of NB(mean=mu, dispersion=theta).
                        rate = theta / mu
                        rate = rate.expand_as(mu)
                        gamma = torch.distributions.Gamma(theta.expand_as(mu), rate).sample()
                        out = torch.poisson(gamma)
                        if spec.get("likelihood") == "zinb":
                            dropout = torch.bernoulli(
                                torch.sigmoid(head.dropout_logits(decoder_z))).bool()
                            out = torch.where(dropout, torch.zeros_like(out), out)
                    else:
                        out = mu
                elif sample_observations and sampler == "residual_factor" and name in residual_draws:
                    out = out + torch.as_tensor(
                        residual_draws[name], dtype=out.dtype, device=out.device)
                elif sample_observations and sampler == "independent":
                    out = out + (
                        self._noise_scale(name)
                        * torch.exp(head.log_scale).clamp(max=10.0)
                        * torch.randn_like(out)
                    )
                modalities[name] = out.cpu().numpy()

            surv_logits = self.head_surv.forward(z).cpu().numpy()
            out = {
                "z": z.cpu().numpy(),
                "tab_continuous": tab_cont,
                "tab_cat_codes": cat_codes,
                "expr": modalities["expression"],
                "modalities": modalities,
                "surv_logits": surv_logits,
            }
            if "methylation" in modalities:
                out["meth"] = modalities["methylation"]
            return out

    def _val_objective(model, loader, device, cfg):
        """Mean per-batch objective at beta_max (validation / early-stopping)."""
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = data_mod.move_batch(batch, device)
                batch["beta"] = cfg.beta_max
                loss, _ = model(batch)
                total += float(loss) * len(batch["x_tab"])
                count += len(batch["x_tab"])
        return total / max(1, count)

    def fit(model: "JointMultimodalVAE", train_loader, cfg: MMVAEConfig,
            val_loader=None):
        """Train the joint model. Mirrors ``SurvivalVAE.fit`` (KL warmup, free
        bits, chosen weighting, grad clip, early stopping on the val objective at
        beta_max, best-state restore). Returns the best-state-restored model."""
        torch.manual_seed(cfg.seed)
        device = (torch.device(cfg.device) if cfg.device != "auto"
                  else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)
        model.history_ = []

        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
        # GradNorm updates its task weights with a separate optimizer.
        w_opt = None
        if model.objective.is_gradnorm:
            w_opt = torch.optim.Adam(model.objective.weighter.parameters(), lr=cfg.lr)

        best_val = float("inf")
        best_state = None
        bad = 0
        eval_loader = val_loader or train_loader

        for epoch in range(cfg.max_epochs):
            beta = beta_schedule(epoch, cfg.beta_max, cfg.kl_warmup_epochs)
            model.train()
            last_logs = {}
            for batch in train_loader:
                batch = data_mod.move_batch(batch, device)
                if model.objective.is_gradnorm:
                    weighter = model.objective.weighter
                    recon, mu, logvar, z, private = model._encode_decode(batch)
                    data_term, logs = weighter.weighted(recon)
                    kl, kl_logs = model.objective.kl_loss(mu, logvar, private)
                    loss = data_term + beta * kl
                    opt.zero_grad()
                    w_opt.zero_grad()
                    loss.backward(retain_graph=True)
                    penalty = weighter.gradnorm_penalty(recon, z, cfg.gradnorm_alpha)
                    gw = torch.autograd.grad(penalty, weighter.raw)[0]
                    weighter.raw.grad = gw
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    w_opt.step()
                    weighter.renormalize()
                    logs.update(kl_logs)
                    logs["beta"] = beta
                else:
                    batch["beta"] = beta
                    loss, logs = model(batch)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                last_logs = logs

            val_obj = _val_objective(model, eval_loader, device, cfg)
            record = {"epoch": epoch, "val_obj": val_obj, "beta": beta}
            record.update({k: v for k, v in last_logs.items()})
            model.history_.append(record)
            if cfg.verbose:
                print(f"[{epoch:03d}] val_obj={val_obj:.4f} beta={beta:.3f} "
                      + " ".join(f"{k}={v:.3f}" for k, v in last_logs.items()
                                 if isinstance(v, float)))

            if val_obj < best_val - 1e-4:
                best_val = val_obj
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model
