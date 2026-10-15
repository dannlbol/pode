"""Generation orchestrator for the joint multimodal VAE.

:class:`JointVAESynthesizer` is the high-level object the CLI and tests use -- the
joint-model analogue of :class:`synthvae.pipeline.hybrid.HybridPipeline`. It owns
the fitted transformers, the trained :class:`JointMultimodalVAE`, the survival
grid, and the ex-post latent density, and exposes ``fit`` / ``sample`` /
``encode_real`` / ``save`` / ``load``.

``sample`` returns a :class:`synthvae.pipeline.hybrid.SynthesisResult` so its
output feeds ``synthvae.evaluation.report.evaluate`` unchanged.

Requires torch; import-guarded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..survival.discretize import TimeGrid
from ..pipeline.hybrid import SynthesisResult

try:
    import torch

    from .model import MMVAEConfig, JointMultimodalVAE, fit as fit_model
    from . import data as data_mod

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def _interval_times(grid: TimeGrid) -> np.ndarray:
    """Representative time for each discrete interval (midpoints; sensible edges)."""
    cuts = np.asarray(grid.cuts, dtype=float)
    if cuts.size == 0:
        return np.array([1.0])
    lo = max(cuts[0] * 0.5, 1e-6)
    reps = [lo]
    for k in range(1, len(cuts)):
        reps.append(0.5 * (cuts[k - 1] + cuts[k]))
    # open last interval (cuts[-1], +inf): extrapolate one bin-width past the edge
    last_w = cuts[-1] - cuts[-2] if len(cuts) > 1 else cuts[-1]
    reps.append(cuts[-1] + 0.5 * max(last_w, 1e-6))
    return np.asarray(reps, dtype=float)


def sample_survival(surv_logits: np.ndarray, grid: TimeGrid, seed: int = 0):
    """Sample ``(time, event)`` from discrete-time hazard logits.

    Per-interval Bernoulli on ``sigmoid(logits)``; the first success is the event
    interval (event=1), otherwise censor at the horizon (event=0). Interval indices
    map to times via the grid's interval midpoints.
    """
    rng = np.random.default_rng(seed)
    hazards = 1.0 / (1.0 + np.exp(-np.asarray(surv_logits, dtype=float)))
    n, K = hazards.shape
    reps = _interval_times(grid)
    draws = rng.random((n, K)) < hazards
    times = np.empty(n)
    events = np.empty(n, dtype=int)
    for i in range(n):
        hit = np.argmax(draws[i]) if draws[i].any() else -1
        if hit >= 0:
            times[i] = reps[hit]
            events[i] = 1
        else:
            times[i] = reps[-1]      # censored at the horizon
            events[i] = 0
    return times, events


if _HAS_TORCH:

    class JointVAESynthesizer:
        """Fit the joint model and generate aligned synthetic cohorts."""

        def __init__(self, cfg: MMVAEConfig | None = None,
                     categorical=None, **kwargs):
            self.cfg = cfg or MMVAEConfig(**kwargs)
            self.categorical = categorical
            self.model = None
            self.transforms = None
            self._clinical_index = None

        # -- fit ------------------------------------------------------------
        def fit(self, clinical: pd.DataFrame, expression: pd.DataFrame,
                survival: pd.DataFrame, methylation: pd.DataFrame | None = None,
                omics: dict[str, pd.DataFrame] | None = None) -> "JointVAESynthesizer":
            dataset, transforms, self.cfg = data_mod.build_from_frames(
                clinical, expression, survival, self.cfg, self.categorical,
                methylation=methylation, omics=omics,
            )
            self.transforms = transforms
            # Kept only in memory so tuning can recalibrate generation-only
            # settings without retraining the neural network. It is deliberately
            # not serialized in model.pt.
            self._fit_dataset = dataset
            # Seed before constructing the model so weight initialisation (not just
            # the training loop, which fit() re-seeds) is reproducible for a given
            # cfg.seed. Without this the init draws from unseeded global RNG and
            # runs vary process-to-process.
            torch.manual_seed(self.cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.cfg.seed)
            self.model = JointMultimodalVAE(self.cfg, transforms.column_specs)
            train_loader, val_loader = data_mod.make_loaders(dataset, self.cfg)
            fit_model(self.model, train_loader, self.cfg, val_loader)
            # Fit the ex-post latent density on encoded training latents.
            z_train = self._encode_dataset(
                dataset, posterior_samples=self.cfg.posterior_samples_per_patient
            )
            self.model.fit_latent_density(z_train)
            self._fit_residual_sampler(dataset)
            return self

        def _fit_residual_sampler(self, dataset):
            """Fit a compact correlated observation model in transformed space."""
            if getattr(self.cfg, "observation_sampler", "independent") != "residual_factor":
                self.model._residual_sampler = None
                return
            try:
                from sklearn.covariance import LedoitWolf
                from sklearn.decomposition import PCA
            except Exception as exc:  # pragma: no cover - sklearn is a core dependency
                raise RuntimeError("residual_factor sampling requires scikit-learn") from exc

            self.model.eval()
            device = next(self.model.parameters()).device
            blocks = {}
            score_frames = []
            offset = 0
            with torch.no_grad():
                modal = {
                    name: values["x"].to(device)
                    for name, values in dataset.modalities.items()
                }
                mu, _ = self.model.encode(dataset.x_tab.to(device), modalities=modal)
                private = {
                    name: posterior[0]
                    for name, posterior in self.model.encode_private(modal).items()
                }
                for name in self.model.modality_names:
                    spec = self.model.modality_specs[name]
                    if (spec.get("kind") not in {"continuous", "methylation"}
                            or spec.get("likelihood", "gaussian") != "gaussian"):
                        continue
                    target = dataset.modalities[name]["target"].cpu().numpy().astype(float)
                    decoded = self.model.head_modalities[name](
                        self.model.decoder_latent(name, mu, private)
                    ).cpu().numpy().astype(float)
                    residual = target - decoded
                    max_rank = max(0, min(
                        int(getattr(self.cfg, "residual_rank", 16)),
                        residual.shape[0] - 1,
                        residual.shape[1],
                    ))
                    if max_rank:
                        pca = PCA(
                            n_components=max_rank,
                            svd_solver="randomized",
                            random_state=self.cfg.seed,
                        ).fit(residual)
                        scores = pca.transform(residual)
                        structured = pca.inverse_transform(scores)
                        components = pca.components_
                        mean = pca.mean_
                    else:
                        scores = np.zeros((len(residual), 0), dtype=float)
                        structured = np.broadcast_to(residual.mean(0), residual.shape)
                        components = np.zeros((0, residual.shape[1]), dtype=float)
                        mean = residual.mean(0)
                    remainder = residual - structured
                    diag_scale = np.sqrt(np.maximum(np.var(remainder, axis=0), 0.0))
                    blocks[name] = {
                        "components": components.astype(np.float32),
                        "mean": np.asarray(mean, dtype=np.float32),
                        "diag_scale": diag_scale.astype(np.float32),
                        "score_slice": (offset, offset + scores.shape[1]),
                    }
                    offset += scores.shape[1]
                    score_frames.append(scores)

            if not blocks:
                self.model._residual_sampler = None
                return
            all_scores = np.concatenate(score_frames, axis=1) if offset else np.zeros((len(dataset), 0))
            if offset:
                score_cov = LedoitWolf().fit(all_scores).covariance_
                # Guard numerical round-off before multivariate draws.
                score_cov = 0.5 * (score_cov + score_cov.T)
                score_cov.flat[::offset + 1] += 1e-8
            else:
                score_cov = np.zeros((0, 0), dtype=float)
            self.model._residual_sampler = {
                "version": 1,
                "score_dim": offset,
                "score_cov": score_cov.astype(np.float32),
                "modalities": blocks,
            }

        def _encode_dataset(self, dataset, posterior_samples: int = 1) -> np.ndarray:
            """Posterior draws for the ex-post latent density.

            Fitting the GMM only to posterior means makes generated cohorts
            systematically tighter than the training distribution.  Drawing from
            ``q(z|x)`` retains encoder uncertainty while leaving evaluation
            embeddings (`encode_real`) deterministic posterior means.
            """
            self.model.eval()
            device = next(self.model.parameters()).device
            with torch.no_grad():
                modal = {name: values["x"].to(device) for name, values in dataset.modalities.items()}
                mu, logvar = self.model.encode(dataset.x_tab.to(device), modalities=modal)
                private_posteriors = self.model.encode_private(modal)
                draws = max(1, int(posterior_samples))
                if draws == 1:
                    parts = [mu, *[private_posteriors[name][0]
                                  for name in self.model.modality_names
                                  if name in private_posteriors]]
                    z = torch.cat(parts, dim=1)
                else:
                    generator = torch.Generator(device=device).manual_seed(self.cfg.seed)
                    def posterior_draws(pmu, plogvar):
                        eps = torch.randn(
                            (draws, *pmu.shape), generator=generator, device=device)
                        value = (pmu.unsqueeze(0)
                                 + torch.exp(0.5 * plogvar).unsqueeze(0) * eps)
                        return value.reshape(-1, pmu.shape[1])
                    parts = [posterior_draws(mu, logvar)]
                    parts.extend(posterior_draws(*private_posteriors[name])
                                 for name in self.model.modality_names
                                 if name in private_posteriors)
                    z = torch.cat(parts, dim=1)
            return z.cpu().numpy()

        # -- sample ---------------------------------------------------------
        def sample(self, n: int) -> SynthesisResult:
            if self.model is None:
                raise RuntimeError("JointVAESynthesizer is not fitted.")
            out = self.model.sample(n)
            tab = self.transforms.tabular
            clinical = tab.inverse(out["tab_continuous"], out["tab_cat_codes"])
            frames = {}
            for name, values in out["modalities"].items():
                tf = self.transforms.modalities[name]
                frames[name] = pd.DataFrame(tf.inverse(values), columns=getattr(tf, "columns", None))
            expression = frames["expression"]
            methylation = frames.get("methylation")
            times, events = sample_survival(
                out["surv_logits"], self.transforms.grid, seed=self.cfg.seed
            )
            survival = pd.DataFrame({"time": times, "event": events})
            return SynthesisResult(
                clinical=clinical, expression=expression,
                survival=survival, latents=out["z"], methylation=methylation,
                modalities={k: v for k, v in frames.items() if k not in ("expression", "methylation")},
            )

        # -- helpers --------------------------------------------------------
        def encode_real(self, clinical: pd.DataFrame, expression: pd.DataFrame,
                        methylation: pd.DataFrame | None = None,
                        omics: dict[str, pd.DataFrame] | None = None) -> np.ndarray:
            """Posterior-mean shared latents for a real cohort (for evaluation)."""
            tab = self.transforms.tabular
            x_tab, _, _ = tab.transform(clinical)
            X = expression.to_numpy(dtype=float) if hasattr(expression, "to_numpy") \
                else np.asarray(expression, dtype=float)
            x_expr, _, _ = self.transforms.expression.transform(X)
            omics = dict(omics or {})
            if methylation is not None:
                omics.setdefault("methylation", methylation)
            modal = {"expression": torch.tensor(x_expr, device=next(self.model.parameters()).device)}
            for name, frame in omics.items():
                if name not in self.transforms.modalities:
                    raise ValueError(f"model has no modality {name!r}")
                Xm = frame.to_numpy(dtype=float) if hasattr(frame, "to_numpy") else np.asarray(frame, dtype=float)
                transformed = self.transforms.modalities[name].transform(Xm)
                modal[name] = torch.tensor(transformed[0], device=next(self.model.parameters()).device)
            self.model.eval()
            device = next(self.model.parameters()).device
            with torch.no_grad():
                mu, _ = self.model.encode(torch.tensor(x_tab, device=device), modalities=modal)
            return mu.cpu().numpy()

        # -- persistence ----------------------------------------------------
        def save(self, path):
            torch.save({
                "format_version": 2,
                "cfg": self.cfg,
                "categorical": self.categorical,
                "state_dict": self.model.state_dict(),
                "column_specs": self.transforms.column_specs,
                "transforms": self.transforms,
                "latent_gmm": getattr(self.model, "_latent_gmm", None),
                "residual_sampler": getattr(self.model, "_residual_sampler", None),
                "latent_layout": self.model.latent_layout(),
            }, path)

        @classmethod
        def load(cls, path):
            blob = torch.load(path, map_location="cpu", weights_only=False)
            cfg = blob["cfg"]
            if blob.get("residual_sampler") is None and getattr(
                    cfg, "observation_sampler", "independent") in {"auto", "residual_factor"}:
                # A checkpoint cannot use factor draws without fitted state. Old
                # checkpoints therefore retain their historical independent draw.
                cfg.observation_sampler = "independent"
            obj = cls(cfg=cfg, categorical=blob["categorical"])
            obj.transforms = blob["transforms"]
            obj.model = JointMultimodalVAE(cfg, blob["column_specs"])
            obj.model.load_state_dict(blob["state_dict"])
            saved_layout = blob.get("latent_layout")
            if saved_layout is not None and saved_layout != obj.model.latent_layout():
                raise ValueError("checkpoint latent layout does not match its configuration")
            obj.model._latent_gmm = blob.get("latent_gmm")
            obj.model._residual_sampler = blob.get("residual_sampler")
            return obj
