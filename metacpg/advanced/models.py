from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from .base import cox_loss_gradient, cox_loss_torch, harrell_c


class LinearCoxModel:
    def __init__(self, *, l1: float = 0.0, l2: float = 1e-3, max_iter: int = 500,
                 learning_rate: float = 0.2, n_omics: int | None = None,
                 l2_factors: np.ndarray | None = None,
                 graph_edges: list[tuple[int, int, float]] | None = None,
                 graph_strength: float = 0.0):
        self.l1, self.l2 = float(l1), float(l2)
        self.max_iter, self.learning_rate = int(max_iter), float(learning_rate)
        self.n_omics = n_omics
        self.l2_factors = l2_factors
        self.graph_edges = graph_edges or []
        self.graph_strength = float(graph_strength)

    def _objective_grad(self, beta, x, time, status, strata=None):
        value, grad = cox_loss_gradient(beta, x, time, status, strata)
        factors = self.l2_factors
        if factors is None:
            factors = np.ones(len(beta))
            if self.n_omics is not None:
                factors[self.n_omics:] = 0.0
        value += .5 * self.l2 * float(np.sum(factors * beta * beta))
        grad += self.l2 * factors * beta
        if self.graph_edges and self.graph_strength:
            for left, right, weight in self.graph_edges:
                delta = beta[left] - beta[right]
                value += .5 * self.graph_strength * weight * delta * delta
                grad[left] += self.graph_strength * weight * delta
                grad[right] -= self.graph_strength * weight * delta
        return value, grad

    def fit(self, x, time, status, strata=None, **context):
        x, time, status = np.asarray(x, float), np.asarray(time, float), np.asarray(status, float)
        beta = np.zeros(x.shape[1])
        penalized = np.zeros(x.shape[1], bool)
        penalized[: self.n_omics if self.n_omics is not None else x.shape[1]] = True
        previous = math.inf
        for _ in range(self.max_iter):
            value, grad = self._objective_grad(beta, x, time, status, strata)
            step = self.learning_rate
            while step > 1e-8:
                candidate = beta - step * grad
                if self.l1:
                    candidate[penalized] = np.sign(candidate[penalized]) * np.maximum(
                        np.abs(candidate[penalized]) - step * self.l1, 0
                    )
                new_value, _ = self._objective_grad(candidate, x, time, status, strata)
                new_value += self.l1 * float(np.abs(candidate[penalized]).sum())
                if new_value <= value + self.l1 * float(np.abs(beta[penalized]).sum()) + 1e-10:
                    break
                step *= .5
            delta = float(np.max(np.abs(candidate - beta)))
            beta = candidate
            if delta < 1e-7 or abs(previous - new_value) < 1e-9:
                break
            previous = new_value
        self.coef_ = beta
        n = self.n_omics if self.n_omics is not None else len(beta)
        self.importance_ = beta[:n].copy()
        self.n_iter_ = _ + 1
        return self

    def predict(self, x, **context):
        return np.asarray(x, float) @ self.coef_


class CoxBoostModel:
    """Component-wise Cox boosting with joint mandatory-covariate updates."""
    def __init__(self, *, steps=100, learning_rate=.05, n_omics=None):
        self.steps, self.learning_rate, self.n_omics = int(steps), float(learning_rate), n_omics

    def fit(self, x, time, status, strata=None, **context):
        x = np.asarray(x, float)
        beta = np.zeros(x.shape[1])
        n = self.n_omics if self.n_omics is not None else x.shape[1]
        scale = np.mean(x * x, axis=0) + 1e-8
        path = []
        for _ in range(self.steps):
            _, grad = cox_loss_gradient(beta, x, time, status, strata)
            if n < x.shape[1]:
                beta[n:] -= self.learning_rate * grad[n:] / scale[n:]
            chosen = int(np.argmax(np.abs(grad[:n]) / np.sqrt(scale[:n])))
            beta[chosen] -= self.learning_rate * grad[chosen] / scale[chosen]
            path.append(chosen)
        self.coef_, self.path_ = beta, path
        self.importance_ = beta[:n].copy()
        return self

    def predict(self, x, **context):
        return np.asarray(x, float) @ self.coef_


class StabilityBoostModel:
    def __init__(self, *, repetitions=20, threshold=.6, steps=60, learning_rate=.05,
                 seed=1, n_omics=None):
        self.repetitions, self.threshold = int(repetitions), float(threshold)
        self.steps, self.learning_rate, self.seed, self.n_omics = int(steps), float(learning_rate), int(seed), n_omics

    def fit(self, x, time, status, strata=None, **context):
        rng = np.random.default_rng(self.seed)
        n_features = self.n_omics if self.n_omics is not None else x.shape[1]
        counts = np.zeros(n_features)
        fits = 0
        for _ in range(self.repetitions):
            first = rng.choice(len(x), max(4, len(x) // 2), replace=False)
            complement = np.setdiff1d(np.arange(len(x)), first)
            for indices in (first, complement):
                if len(indices) < 4 or np.sum(status[indices]) == 0:
                    continue
                model = CoxBoostModel(steps=self.steps, learning_rate=self.learning_rate, n_omics=n_features)
                model.fit(x[indices], time[indices], status[indices],
                          strata=strata[indices] if strata is not None else None)
                counts += np.abs(model.coef_[:n_features]) > 1e-10
                fits += 1
        self.selection_frequency_ = counts / max(fits, 1)
        selected = self.selection_frequency_ >= self.threshold
        if not selected.any():
            selected[np.argmax(self.selection_frequency_)] = True
        penalty = np.ones(n_features)
        penalty[selected] = 0.05
        full_factors = np.r_[penalty, np.zeros(x.shape[1] - n_features)]
        self.final_model_ = LinearCoxModel(l1=1e-3, l2=1e-3, n_omics=n_features,
                                           l2_factors=full_factors).fit(x, time, status, strata=strata)
        self.coef_ = self.final_model_.coef_
        self.importance_ = self.selection_frequency_.copy()
        return self

    def predict(self, x, **context):
        return self.final_model_.predict(x)


class PLSCoxModel:
    def __init__(self, *, components=3, n_omics=None):
        self.components, self.n_omics = int(components), n_omics

    def fit(self, x, time, status, strata=None, **context):
        n = self.n_omics if self.n_omics is not None else x.shape[1]
        omics = np.asarray(x[:, :n], float)
        # Event-weighted log-time target is the documented PODE adaptation of PLS-Cox.
        target = -np.log(np.asarray(time, float)) * (.5 + np.asarray(status, float))
        components = max(1, min(self.components, omics.shape[0] - 1, omics.shape[1]))
        self.pls_ = PLSRegression(n_components=components, scale=False).fit(omics, target)
        scores = self.pls_.transform(omics)
        design = np.column_stack([scores, x[:, n:]])
        self.cox_ = LinearCoxModel(l2=1e-3, n_omics=components).fit(design, time, status, strata=strata)
        component_coef = self.cox_.coef_[:components]
        self.importance_ = np.asarray(self.pls_.x_weights_) @ component_coef
        return self

    def predict(self, x, **context):
        n = self.n_omics if self.n_omics is not None else x.shape[1]
        scores = self.pls_.transform(x[:, :n])
        return self.cox_.predict(np.column_stack([scores, x[:, n:]]))


class SksurvModel:
    def __init__(self, kind: str, *, n_omics: int, seed=1, **params):
        self.kind, self.n_omics, self.seed, self.params = kind, n_omics, int(seed), params

    def fit(self, x, time, status, **context):
        try:
            from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
            from sksurv.svm import FastSurvivalSVM
            from sksurv.util import Surv
        except ImportError as exc:
            raise RuntimeError(
                "this method requires scikit-survival; install pode[signatures-advanced]"
            ) from exc
        y = Surv.from_arrays(np.asarray(status, bool), np.asarray(time, float))
        if self.kind == "random_survival_forest":
            defaults = dict(n_estimators=100, min_samples_leaf=3, random_state=self.seed, n_jobs=1)
            defaults.update(self.params)
            self.model_ = RandomSurvivalForest(**defaults)
        elif self.kind == "gradient_boosted_survival":
            defaults = dict(n_estimators=100, learning_rate=.05, max_depth=2, random_state=self.seed)
            defaults.update(self.params)
            self.model_ = GradientBoostingSurvivalAnalysis(**defaults)
        else:
            defaults = dict(alpha=1.0, rank_ratio=1.0, random_state=self.seed, max_iter=100)
            defaults.update(self.params)
            self.model_ = FastSurvivalSVM(**defaults)
        self.model_.fit(x, y)
        importance = None
        try:
            importance = np.asarray(self.model_.feature_importances_)
        except (AttributeError, NotImplementedError):
            try:
                importance = np.ravel(self.model_.coef_)
            except AttributeError:
                pass
        if importance is None:
            baseline = harrell_c(time, status, self.model_.predict(x))
            rng = np.random.default_rng(self.seed)
            importance = np.zeros(x.shape[1])
            for column in range(self.n_omics):
                changed = x.copy()
                changed[:, column] = rng.permutation(changed[:, column])
                importance[column] = baseline - harrell_c(time, status, self.model_.predict(changed))
        self.importance_ = importance[:self.n_omics]
        return self

    def predict(self, x, **context):
        return np.asarray(self.model_.predict(x), float)


class DeepCoxModel:
    def __init__(self, architecture: str, *, n_omics: int, block_slices: dict[str, tuple[int, int]],
                 epochs=80, hidden=32, learning_rate=1e-3, seed=1, pathway_path=None,
                 cohort_aware=False, n_tasks=1, task_index=0, **params):
        self.architecture, self.n_omics = architecture, n_omics
        self.block_slices, self.epochs, self.hidden = block_slices, int(epochs), int(hidden)
        self.learning_rate, self.seed, self.pathway_path = float(learning_rate), int(seed), pathway_path
        self.cohort_aware = cohort_aware
        self.n_tasks, self.task_index = int(n_tasks), int(task_index)

    def fit(self, x, time, status, **context):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        x = np.asarray(x, np.float32)
        masks = np.asarray(context.get("masks", np.ones((len(x), len(self.block_slices)))), np.float32)
        cohort = context.get("cohort")
        strata = context.get("strata")
        strata_t = None
        if strata is not None:
            strata_t = torch.tensor(np.unique(strata, return_inverse=True)[1], dtype=torch.long)
        input_dim = x.shape[1]
        n_tasks = self.n_tasks
        architecture = self.architecture
        hidden = self.hidden

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(.1),
                                             nn.Linear(hidden, hidden), nn.ReLU())
                self.head = nn.Linear(hidden, n_tasks)
            def forward(self, value, block_mask=None):
                latent = self.encoder(value)
                output = self.head(latent)
                return (output.squeeze(1) if n_tasks == 1 else output), latent, None

        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
                self.decoder = nn.Linear(hidden, input_dim)
                self.head = nn.Linear(hidden, n_tasks)
            def forward(self, value, block_mask=None):
                latent = self.encoder(value)
                output = self.head(latent)
                return (output.squeeze(1) if n_tasks == 1 else output), latent, self.decoder(latent)

        class BlockNetwork(nn.Module):
            def __init__(self, mode):
                super().__init__()
                self.names = list(self_outer.block_slices)
                self.encoders = nn.ModuleList([
                    nn.Sequential(nn.Linear(self_outer.block_slices[n][1] - self_outer.block_slices[n][0], hidden), nn.ReLU())
                    for n in self.names
                ])
                self.missing = nn.Parameter(torch.zeros(len(self.names), hidden))
                self.mode = mode
                if mode == "surv_transformer":
                    layer = nn.TransformerEncoderLayer(hidden, nhead=1, dim_feedforward=hidden * 2,
                                                       dropout=.1, batch_first=True)
                    self.fusion = nn.TransformerEncoder(layer, num_layers=1)
                    fused = hidden
                else:
                    self.fusion = nn.Sequential(nn.Linear(hidden * len(self.names), hidden), nn.ReLU())
                    fused = hidden
                self.prototypes = nn.Parameter(torch.randn(4, hidden)) if mode == "prototype_contrastive_survival" else None
                self.head = nn.Linear(fused, n_tasks)
            def forward(self, value, block_mask=None):
                tokens = []
                for i, (name, encoder) in enumerate(zip(self.names, self.encoders)):
                    left, right = self_outer.block_slices[name]
                    token = encoder(value[:, left:right])
                    if block_mask is not None:
                        present = block_mask[:, i:i+1]
                        token = present * token + (1 - present) * self.missing[i]
                    tokens.append(token)
                if self.mode == "surv_transformer":
                    latent = self.fusion(torch.stack(tokens, dim=1)).mean(dim=1)
                else:
                    latent = self.fusion(torch.cat(tokens, dim=1))
                output = self.head(latent)
                return (output.squeeze(1) if n_tasks == 1 else output), latent, None

        self_outer = self
        if architecture == "autoencoder_cox":
            model = Autoencoder()
        elif architecture in {"surv_transformer", "prototype_contrastive_survival", "moli_survival", "maml_cox"}:
            model = BlockNetwork(architecture)
        else:
            model = MLP()
        if architecture == "cox_nnet" and self.pathway_path:
            self._apply_pathway_mask(model, input_dim)

        xt = torch.tensor(x)
        tt = torch.tensor(np.asarray(time, np.float32))
        st = torch.tensor(np.asarray(status, np.float32))
        mt = torch.tensor(masks)
        task_times = context.get("task_times")
        task_status = context.get("task_status")
        task_observed = context.get("task_observed")
        if task_times is not None:
            task_tt = torch.tensor(np.asarray(task_times, np.float32))
            task_st = torch.tensor(np.asarray(task_status, np.float32))
            task_mask = torch.tensor(np.asarray(task_observed, bool))
        else:
            task_tt, task_st = tt[:, None], st[:, None]
            task_mask = torch.ones_like(task_tt, dtype=torch.bool)

        def multitask_loss(risk, indices=None):
            if risk.ndim == 1:
                risk = risk[:, None]
            if indices is None:
                indices = torch.arange(len(risk))
            losses = []
            for task in range(risk.shape[1]):
                observed = task_mask[indices, task]
                selected = indices[observed]
                if len(selected) and task_st[selected, task].sum() > 0:
                    losses.append(cox_loss_torch(risk[selected, task], task_tt[selected, task], task_st[selected, task],
                                                 strata=strata_t[selected] if strata_t is not None else None))
            return torch.stack(losses).mean() if losses else risk.sum() * 0
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        model.train()
        if architecture == "maml_cox" and cohort is not None and len(set(cohort)) >= 2:
            cohort = np.asarray(cohort)
            # First-order MAML/Reptile update: adapt one copy per cohort, then move
            # the shared initialization toward the mean adapted parameters.
            for _ in range(self.epochs):
                adapted_states = []
                for level in sorted(set(cohort)):
                    indices = np.flatnonzero(cohort == level)
                    if len(indices) < 2 or st[indices].sum() == 0:
                        continue
                    adapted = copy.deepcopy(model)
                    inner = torch.optim.SGD(adapted.parameters(), lr=self.learning_rate * 5)
                    inner.zero_grad()
                    risk, _, _ = adapted(xt[indices], mt[indices])
                    # The adapted prediction is local to indices; evaluate all available endpoint heads.
                    local_losses = []
                    local_risk = risk[:, None] if risk.ndim == 1 else risk
                    for task in range(local_risk.shape[1]):
                        observed = task_mask[indices, task]
                        if observed.any() and task_st[indices[observed], task].sum() > 0:
                            local_losses.append(cox_loss_torch(
                                local_risk[observed, task],
                                task_tt[indices[observed], task],
                                task_st[indices[observed], task],
                                strata=strata_t[indices[observed]] if strata_t is not None else None,
                            ))
                    (torch.stack(local_losses).mean() if local_losses else local_risk.sum() * 0).backward()
                    inner.step()
                    adapted_states.append({k: v.detach().clone() for k, v in adapted.state_dict().items()})
                if adapted_states:
                    current = model.state_dict()
                    with torch.no_grad():
                        for key in current:
                            target = torch.stack([state[key] for state in adapted_states]).mean(dim=0)
                            current[key].add_(.2 * (target - current[key]))
                    model.load_state_dict(current)
        else:
            for _ in range(self.epochs):
                optimizer.zero_grad()
                risk, latent, reconstruction = model(xt, mt)
                loss = multitask_loss(risk)
                if reconstruction is not None:
                    loss = loss + .1 * torch.mean((reconstruction - xt) ** 2)
                if architecture == "prototype_contrastive_survival" and getattr(model, "prototypes", None) is not None:
                    distance = torch.cdist(latent, model.prototypes)
                    loss = loss + .01 * distance.min(dim=1).values.mean()
                if architecture == "moli_survival" and len(latent) > 2:
                    loss = loss + .001 / (latent.var(dim=0).mean() + 1e-6)
                loss.backward()
                optimizer.step()
        self.model_ = model.eval()
        # Mean absolute input gradient gives an original-feature attribution for every architecture.
        attr_x = xt.clone().requires_grad_(True)
        risk, _, _ = self.model_(attr_x, mt)
        target_risk = risk if risk.ndim == 1 else risk[:, self.task_index]
        target_risk.sum().backward()
        self.importance_ = attr_x.grad.detach().abs().mean(dim=0).numpy()[:self.n_omics]
        self.training_masks_ = masks
        return self

    def _apply_pathway_mask(self, model, input_dim):
        # Pathway masking is enforced when the membership file maps integer input indices to hidden nodes.
        mapping = pd.read_csv(self.pathway_path)
        required = {"feature_index", "pathway_index"}
        if not required.issubset(mapping):
            raise ValueError("pathway_path must contain feature_index and pathway_index columns")
        first = model.encoder[0]
        import torch
        mask = torch.zeros_like(first.weight)
        mask[:, self.n_omics:] = 1
        for row in mapping.itertuples(index=False):
            if 0 <= int(row.feature_index) < input_dim and 0 <= int(row.pathway_index) < mask.shape[0]:
                mask[int(row.pathway_index), int(row.feature_index)] = 1
        first.weight.register_hook(lambda grad: grad * mask)
        with torch.no_grad():
            first.weight.mul_(mask)

    def predict(self, x, **context):
        import torch
        masks = context.get("masks")
        if masks is None:
            masks = np.ones((len(x), len(self.block_slices)), np.float32)
        with torch.no_grad():
            risk, _, _ = self.model_(torch.tensor(np.asarray(x, np.float32)),
                                     torch.tensor(np.asarray(masks, np.float32)))
        if risk.ndim > 1:
            risk = risk[:, self.task_index]
        return risk.numpy()


class HierarchicalCoxModel:
    def __init__(self, *, n_omics, l1=1e-3, deviation_penalty=.1, **params):
        self.n_omics, self.l1, self.deviation_penalty = n_omics, float(l1), float(deviation_penalty)

    def fit(self, x, time, status, strata=None, **context):
        cohort = np.asarray(context.get("cohort"))
        levels = sorted(set(cohort))
        if len(levels) < 2:
            raise ValueError("hierarchical_cox requires at least two cohorts")
        common = x[:, :self.n_omics]
        deviations = [common * (cohort == level)[:, None] for level in levels]
        augmented = np.column_stack([common, *deviations, x[:, self.n_omics:]])
        factors = np.r_[np.ones(self.n_omics),
                        np.full(self.n_omics * len(levels), self.deviation_penalty),
                        np.zeros(x.shape[1] - self.n_omics)]
        self.model_ = LinearCoxModel(l1=self.l1, l2=.1, n_omics=self.n_omics,
                                     l2_factors=factors).fit(augmented, time, status, strata=strata)
        self.levels_, self.original_width_ = levels, x.shape[1]
        self.importance_ = self.model_.coef_[:self.n_omics]
        self.study_coefficients_ = {
            level: self.importance_ + self.model_.coef_[self.n_omics * (i + 1):self.n_omics * (i + 2)]
            for i, level in enumerate(levels)
        }
        return self

    def _augment(self, x, cohort):
        cohort = np.asarray(cohort)
        common = x[:, :self.n_omics]
        deviations = [common * (cohort == level)[:, None] for level in self.levels_]
        return np.column_stack([common, *deviations, x[:, self.n_omics:]])

    def predict(self, x, **context):
        cohort = context.get("cohort")
        if cohort is None:
            # Unseen/unspecified cohorts receive the learned population mean.
            cohort = np.repeat("__population__", len(x))
        return self.model_.predict(self._augment(x, cohort))


class FrailtyCoxModel:
    def __init__(self, *, n_omics, frailty_penalty=1.0, **params):
        self.n_omics, self.frailty_penalty = n_omics, float(frailty_penalty)

    def fit(self, x, time, status, strata=None, **context):
        cohort = np.asarray(context.get("cohort"))
        levels = sorted(set(cohort))
        if len(levels) < 2:
            raise ValueError("frailty_cox requires at least two cohorts")
        indicators = np.column_stack([(cohort == level).astype(float) for level in levels[1:]])
        augmented = np.column_stack([x, indicators])
        factors = np.r_[np.ones(self.n_omics), np.zeros(x.shape[1] - self.n_omics),
                        np.full(indicators.shape[1], self.frailty_penalty)]
        self.model_ = LinearCoxModel(l1=1e-3, l2=1.0, n_omics=self.n_omics,
                                     l2_factors=factors).fit(augmented, time, status, strata=strata)
        self.levels_, self.importance_ = levels, self.model_.coef_[:self.n_omics]
        return self

    def predict(self, x, **context):
        cohort = np.asarray(context.get("cohort", np.repeat(self.levels_[0], len(x))))
        indicators = np.column_stack([(cohort == level).astype(float) for level in self.levels_[1:]])
        return self.model_.predict(np.column_stack([x, indicators]))


def graph_edges(path: str | None, feature_names: list[str]) -> list[tuple[int, int, float]]:
    if not path:
        raise ValueError("net_cox requires methods.config.net_cox.graph_path")
    table = pd.read_csv(Path(path))
    if not {"source", "target"}.issubset(table):
        raise ValueError("graph_path must contain source and target columns")
    lookup = {name: i for i, name in enumerate(feature_names)}
    suffixes: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        suffixes.setdefault(name.split("::", 1)[-1], []).append(i)
    for suffix, indices in suffixes.items():
        if len(indices) == 1:
            lookup.setdefault(suffix, indices[0])
    edges = []
    for row in table.itertuples(index=False):
        if str(row.source) in lookup and str(row.target) in lookup:
            edges.append((lookup[str(row.source)], lookup[str(row.target)], float(getattr(row, "weight", 1.0))))
    if not edges:
        raise ValueError("graph_path has no edges matching configured omics features")
    return edges
