from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


def harrell_c(time: np.ndarray, status: np.ndarray, risk: np.ndarray) -> float:
    time, status, risk = map(np.asarray, (time, status, risk))
    concordant = comparable = 0.0
    for i in range(len(time)):
        for j in range(i + 1, len(time)):
            if time[i] == time[j]:
                continue
            early, late = (i, j) if time[i] < time[j] else (j, i)
            if status[early] != 1:
                continue
            comparable += 1
            concordant += 1.0 if risk[early] > risk[late] else 0.5 if risk[early] == risk[late] else 0.0
    return concordant / comparable if comparable else math.nan


def cox_loss_gradient(beta: np.ndarray, x: np.ndarray, time: np.ndarray,
                      status: np.ndarray, strata: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Mean negative Breslow partial log-likelihood and gradient.

    When ``strata`` is given, risk sets are restricted to samples sharing the same
    stratum label, so each stratum gets its own implicit baseline hazard.
    """
    eta = np.clip(x @ beta, -40, 40)
    loss = 0.0
    grad = np.zeros_like(beta)
    events = 0
    labels = np.zeros(len(time), dtype=int) if strata is None else np.asarray(strata)
    for group in np.unique(labels):
        in_group = labels == group
        for event_time in np.unique(time[in_group & (status == 1)]):
            event = in_group & (time == event_time) & (status == 1)
            risk = in_group & (time >= event_time)
            d = int(event.sum())
            if not d or not risk.any():
                continue
            risk_eta = eta[risk]
            shift = float(risk_eta.max())
            weights = np.exp(risk_eta - shift)
            denom = float(weights.sum())
            mean_x = (weights[:, None] * x[risk]).sum(axis=0) / denom
            loss += d * (math.log(denom) + shift) - float(eta[event].sum())
            grad += d * mean_x - x[event].sum(axis=0)
            events += d
    scale = max(events, 1)
    return loss / scale, grad / scale


def cox_loss_torch(risk, time, status, strata=None):
    import torch

    losses = []
    labels = torch.zeros_like(time, dtype=torch.long) if strata is None else strata
    for group in torch.unique(labels):
        in_group = labels == group
        for event_time in torch.unique(time[in_group & (status > 0.5)]):
            event = in_group & (time == event_time) & (status > 0.5)
            at_risk = in_group & (time >= event_time)
            d = event.sum()
            losses.append(d * torch.logsumexp(risk[at_risk], dim=0) - risk[event].sum())
    if not losses:
        return risk.sum() * 0
    return torch.stack(losses).sum() / torch.clamp(status.sum(), min=1.0)


class Estimator(Protocol):
    importance_: np.ndarray

    def fit(self, x: np.ndarray, time: np.ndarray, status: np.ndarray, **context: Any) -> "Estimator": ...
    def predict(self, x: np.ndarray, **context: Any) -> np.ndarray: ...


class CohortUnsupportedError(ValueError):
    """A method cannot honour ``phenotype.cohort_column`` and skipping was requested."""


@dataclass
class MethodSpec:
    name: str
    title: str
    maturity: str
    family: str
    factory: Any
    optional_dependency: str | None = None
    requires_cohort: bool = False
    requires_graph: bool = False
    requires_pathways: bool = False
    supports_missing_blocks: bool = True
    description: str = ""
    #: How the method accounts for samples drawn from several training cohorts.
    #: ``stratified`` - cohort enters the Cox partial likelihood as an extra stratum;
    #: ``explicit``   - the model itself represents cohort structure;
    #: ``none``       - the method cannot represent cohorts at all.
    cohort_support: str = "stratified"
    #: Whether cohort labels are appended to the Cox strata for this method. False for
    #: methods that place cohort on the linear predictor (where a stratified baseline
    #: would make those terms unidentifiable) and for methods with no Cox strata.
    cohort_stratifies: bool = True


@dataclass
class FittedMethod:
    name: str
    estimator: Estimator
    feature_names: list[str]
    importance: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def rank_importance(feature_names: list[str], importance: np.ndarray) -> list[dict[str, Any]]:
    importance = np.asarray(importance, float)
    order = np.argsort(-np.nan_to_num(np.abs(importance), nan=-np.inf), kind="stable")
    return [
        {"feature": feature_names[i], "rank": rank + 1, "importance": float(importance[i])}
        for rank, i in enumerate(order)
    ]
