"""Evaluation primitives for the federated generator benchmark.

Every number is computed at one **evaluation site** against that site's *real*
held-out patients (never against synthetic rows):

* :func:`utility_at_site` - train a prognostic model (augmentation's
  ``cox_ph`` / ``coxnet`` / ...) on a table (synthetic, or real for the
  references) and score it on the site's real test rows: Harrell/Uno C, AUC,
  Brier, IPA and calibration at the horizon;
* :func:`fidelity_at_site` - how the synthetic table compares with the site's
  real training rows (marginals, correlations, real-vs-synthetic
  discriminator, survival curve, missingness);
* :func:`privacy_at_site` - the full :mod:`pode.privacy` assessment with the
  site's training rows as members and its test rows as non-members.

:func:`federated_cox` is the note's "conventional federated predictive model
trained on real local data": a ridge Cox model stratified by site, fitted by
Newton-Raphson where every site sends only its Efron score vector and
information matrix (``site_cox_score``) at the broadcast coefficients. Each
site keeps its own baseline hazard (its stratum), so predictions at a site use
that site's baseline.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sites import Site


@dataclass
class Endpoint:
    time: str
    event: str
    horizon: float
    numeric: list[str]
    categorical: list[str]

    @property
    def features(self) -> list[str]:
        return [*self.numeric, *self.categorical]


def _clean(df: pd.DataFrame, ep: Endpoint) -> pd.DataFrame:
    t = pd.to_numeric(df[ep.time], errors="coerce")
    e = pd.to_numeric(df[ep.event], errors="coerce")
    ok = t.notna() & e.notna() & (t > 0) & e.isin([0, 1])
    return df[ok].reset_index(drop=True)


def fit_model(spec, table: pd.DataFrame, ep: Endpoint, seed: int):
    """A downstream prognostic model fitted on ``table`` (all rows treated as training rows)."""
    from ..augmentation.models import build_model

    table = _clean(table, ep)
    if len(table) < 10 or table[ep.event].sum() < 3:
        return None
    from ..imputation.base import resolve_columns

    feats = resolve_columns(spec.features, ep.features) if spec.features else ep.features
    num = [c for c in ep.numeric if c in feats and c in table.columns]
    cat = [c for c in ep.categorical if c in feats and c in table.columns]
    model = build_model(spec, num, cat, ep.horizon, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(table, table[ep.time].to_numpy(float), table[ep.event].to_numpy(float), None,
                  np.ones(len(table)), np.ones(len(table), dtype=bool))
    return model


def utility_at_site(model, test: pd.DataFrame, train_real: pd.DataFrame, ep: Endpoint) -> tuple[dict, np.ndarray | None]:
    """Held-out metrics of ``model`` on the site's real ``test`` rows (+ the risk vector for pairing)."""
    from ..augmentation.metrics import utility_metrics

    test, train_real = _clean(test, ep), _clean(train_real, ep)
    if model is None or len(test) < 5:
        return {"fit_failed": model is None}, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        risk = model.risk(test)
        surv = model.surv(test, [ep.horizon])
        surv_h = None if surv is None else np.clip(surv[:, 0], 1e-6, 1 - 1e-6)
        m = utility_metrics(test[ep.time].to_numpy(float), test[ep.event].to_numpy(float), risk, surv_h, None, [],
                            train_real[ep.time].to_numpy(float), train_real[ep.event].to_numpy(float),
                            ep.horizon, [0.2])
    return {k: v for k, v in m.items() if isinstance(v, (int, float, np.number))}, risk


def fidelity_at_site(real: pd.DataFrame, synth: pd.DataFrame, ep: Endpoint, seed: int) -> dict:
    from ..augmentation.fidelity import corr_gap, discriminator_auc, marginal_gaps, survival_gaps

    cols = [c for c in ep.features + [ep.time, ep.event] if c in synth.columns and c in real.columns]
    num = [c for c in ep.numeric + [ep.time] if c in cols]
    cat = [c for c in ep.categorical + [ep.event] if c in cols]
    if synth.empty:
        return {"synthetic_rows": 0}
    out = {"synthetic_rows": int(len(synth)), **marginal_gaps(real, synth, num, cat),
           "corr_gap": corr_gap(real, synth, num),
           "discriminator_auc": discriminator_auc(real[cols], synth[cols], num, cat, seed)}
    rs, ss = _clean(real, ep), _clean(synth, ep)
    if len(rs) > 5 and len(ss) > 5:
        out.update(survival_gaps(rs, ss, ep.time, ep.event, ep.horizon))
    miss = [abs(real[c].isna().mean() - synth[c].isna().mean()) for c in cols]
    out["missing_rate_gap"] = float(np.mean(miss)) if miss else np.nan
    return out


def privacy_at_site(train: pd.DataFrame, test: pd.DataFrame, synth: pd.DataFrame, columns: list[str], schema,
                    pcfg, seed: int) -> dict:
    """Members = the site's training rows, non-members = its test rows."""
    import copy

    from ..privacy.assess import assess_privacy

    cols = [c for c in columns if c in synth.columns]
    if synth.empty or len(cols) < 2:
        return {"verdict": "NOT_ASSESSED"}
    p = copy.deepcopy(pcfg)
    p.seed = seed
    try:
        rep = assess_privacy(train[cols], synth[cols], test[cols], schema, p, log=lambda m: None)
    except ValueError as exc:  # pragma: no cover - degenerate inputs
        return {"verdict": "NOT_ASSESSED", "error": str(exc)}
    keys = ("exact_copies", "dcr_p5_ratio", "near_copy_excess_p5", "membership_auc", "membership_auc_ci_high",
            "membership_auc_max", "attribute_member_gap_max", "linkage_excess_max", "rare_combo_leaks")
    out = {k: rep.summary.get(k) for k in keys}
    out["verdict"] = rep.verdict
    fails = rep.gate[rep.gate["status"] == "FAIL"]["check"].unique().tolist()
    out["failed_checks"] = ";".join(fails)
    return out


# ---- federated Cox reference ----------------------------------------------------------
def site_cox_score(data: pd.DataFrame, pre, ep: Endpoint, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Efron score and information of the site's rows (its own stratum) at ``beta``."""
    from metacpg.core import _batched_score_info_general, _CoxBackend, _cox_strata_orders

    d = _clean(data, ep)
    X = pre.transform(d)
    t, e = d[ep.time].to_numpy(float), d[ep.event].to_numpy(float)
    be = _CoxBackend("numpy")
    orders = _cox_strata_orders(t, e, None)
    so = [(be.idx(o), [(end, be.idx(ev), k) for end, ev, k in steps]) for o, steps in orders]
    obs = X[e == 1].sum(axis=0)[None, :]
    u, info = _batched_score_info_general(X[:, :1], X[:, 1:], beta[None, :], obs, X.shape[1] - 1, so, be)
    return u[0], info[0]


def federated_cox(sites: list[Site], pre, ep: Endpoint, ridge: float = 1.0, max_iter: int = 50,
                  tol: float = 1e-9) -> np.ndarray:
    """Site-stratified ridge Cox from per-site score/information only."""
    p = len(pre.names)
    beta = np.zeros(p)
    for _ in range(max_iter):
        parts = [s.run(site_cox_score, pre, ep, beta, label="cox_score_info") for s in sites]
        u = sum(x[0] for x in parts) - ridge * beta
        info = sum(x[1] for x in parts) + ridge * np.eye(p)
        step = np.linalg.solve(info, u)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


class LinearCox:
    """A fitted linear Cox score with a site-local Breslow baseline (the federated reference)."""

    def __init__(self, pre, beta: np.ndarray, train: pd.DataFrame, ep: Endpoint):
        from ..augmentation.models import BreslowCalibrator

        self.pre, self.beta = pre, beta
        d = _clean(train, ep)
        self.cal = BreslowCalibrator().fit(self.risk(d), d[ep.time].to_numpy(float), d[ep.event].to_numpy(float),
                                           np.ones(len(d)))

    def risk(self, df: pd.DataFrame) -> np.ndarray:
        return self.pre.transform(df) @ self.beta

    def surv(self, df: pd.DataFrame, times) -> np.ndarray:
        return self.cal.surv(self.risk(df), times)
