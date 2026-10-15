"""Downstream prognostic models refitted on real (+ synthetic) training rows.

Every model shares one interface::

    model.fit(train, time, event, strata, weights, real_mask)
    model.risk(test)              # higher = worse prognosis
    model.surv(test, times)       # P(T > t) per row and time, or None

Preprocessing (imputation medians, one-hot levels, standardisation) and the
MetaCPG score scaling are estimated on the *real* training rows only, so
synthetic rows can change what a model learns but never how real patients are
encoded. Absolute survival probabilities come from a weighted Breslow fit of
the model's score on all training rows (real + synthetic): augmentation that
distorts the case mix shows up as miscalibration, as it should.

* ``metacpg_fixed``     - MetaCPG's legacy score on a fixed CpG list: per-CpG
  (stratified) univariable Cox standardised coefficients as weights, mean of
  weighted M-values, centred/scaled by the real rows' median/IQR;
* ``coxnet``            - elastic-net Cox (scikit-survival), penalty chosen by
  inner K-fold C-index scored on real rows only;
* ``cox_ph``            - ridge-penalised Cox (lifelines), with strata and weights;
* ``logistic_landmark`` - L2 logistic regression for survival past the horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelSpec


# ---- preprocessing -------------------------------------------------------------
def numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    """Columns as a float array: one bulk conversion (per-column coercion is slow
    over thousands of CpGs), falling back to coercion for mixed types."""
    try:
        return df.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError):
        return df.apply(pd.to_numeric, errors="coerce").to_numpy(float)


def beta_to_m(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    X = np.clip(X, eps, 1 - eps)
    return np.log2(X / (1 - X))


class Preprocessor:
    """Median-impute, one-hot and (optionally) standardise, fitted on real rows.

    One-hot columns are standardised too, as glmnet does by default: penalised
    models are scale-sensitive, and leaving 0/1 dummies unscaled penalises
    rare levels far more than common ones.

    Options for omics matrices (model ``params``):

    * ``m_value_columns`` - columns holding methylation betas, converted to
      M-values (log2 b/(1-b), clipped) before anything else;
    * ``top_variance`` / ``top_variance_columns`` - keep only the ``k`` most
      variable of these columns (default: the M-value columns, else all
      numeric ones), ranked on the **real training rows** of each fit, so the
      unsupervised filter never sees held-out or synthetic patients. Other
      numeric columns (e.g. Age) are always kept.
    """

    def __init__(self, numeric: list[str], categorical: list[str], standardise: bool = True,
                 m_value_columns: list[str] | None = None, top_variance: int | None = None,
                 top_variance_columns: list[str] | None = None):
        self.numeric, self.categorical, self.standardise = list(numeric), categorical, standardise
        self.m_value_columns = set(m_value_columns or [])
        self.top_variance = top_variance
        self.top_variance_columns = top_variance_columns

    def _num(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        X = numeric_matrix(df[cols])
        mv = [i for i, c in enumerate(cols) if c in self.m_value_columns]
        if mv:
            X[:, mv] = beta_to_m(X[:, mv])
        return X

    def fit(self, real: pd.DataFrame) -> "Preprocessor":
        if self.top_variance:
            pool = set(self.top_variance_columns or self.m_value_columns or self.numeric)
            cand = [c for c in self.numeric if c in pool]
            if len(cand) > self.top_variance:
                var = np.nanvar(self._num(real, cand), axis=0)
                keep = set(np.asarray(cand)[np.argsort(var)[::-1][: self.top_variance]])
                self.numeric = [c for c in self.numeric if c not in pool or c in keep]
        self.median = np.nanmedian(self._num(real, self.numeric), axis=0) if self.numeric else np.array([])
        self.levels = {c: sorted(real[c].dropna().astype(str).unique())[1:] for c in self.categorical}
        raw = self._raw(real)
        self.mean = raw.mean(axis=0)
        sd = raw.std(axis=0, ddof=1) if len(raw) > 1 else np.ones(raw.shape[1])
        self.sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        return self

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        X = self._num(df, self.numeric)
        X = np.where(np.isnan(X), self.median, X)
        parts = [X]
        for c, lv in self.levels.items():
            s = df[c].astype(str)
            parts.append(np.column_stack([(s == level).to_numpy(float) for level in lv]) if lv
                         else np.empty((len(df), 0)))
        return np.column_stack(parts)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._raw(df)
        return (raw - self.mean) / self.sd if self.standardise else raw

    @property
    def names(self) -> list[str]:
        return [*self.numeric, *[f"{c}={lv}" for c, lvs in self.levels.items() for lv in lvs]]


# ---- score -> survival -----------------------------------------------------------
def _risk_sets(time: np.ndarray):
    order = np.argsort(time, kind="mergesort")
    t = time[order]
    uniq, first = np.unique(t, return_index=True)
    return order, t, uniq, first


class BreslowCalibrator:
    """Weighted 1-D Cox of a score (Breslow ties) + Breslow baseline hazard."""

    def fit(self, score, time, event, weights=None, iters: int = 50) -> "BreslowCalibrator":
        score, time, event = (np.asarray(a, float) for a in (score, time, event))
        w = np.ones_like(score) if weights is None else np.asarray(weights, float)
        ok = np.isfinite(score) & np.isfinite(time) & np.isfinite(event)
        s, t, e, w = score[ok], time[ok], event[ok], w[ok]
        self.center = float(np.average(s, weights=w))
        s = s - self.center
        order, ts, uniq, first = _risk_sets(t)
        s, e, w = s[order], e[order], w[order]
        b = 0.0
        for _ in range(iters):
            r = w * np.exp(np.clip(b * s, -50, 50))
            s0 = np.cumsum(r[::-1])[::-1][first]
            s1 = np.cumsum((r * s)[::-1])[::-1][first]
            s2 = np.cumsum((r * s * s)[::-1])[::-1][first]
            idx = np.searchsorted(uniq, ts)
            d = np.bincount(idx, weights=w * e, minlength=len(uniq))
            ds = np.bincount(idx, weights=w * e * s, minlength=len(uniq))
            grad = float(ds.sum() - np.sum(d * s1 / s0))
            hess = float(np.sum(d * (s2 / s0 - (s1 / s0) ** 2)))
            if hess <= 1e-12:
                break
            step = grad / hess
            b += float(np.clip(step, -5, 5))
            if abs(step) < 1e-9:
                break
        r = w * np.exp(np.clip(b * s, -50, 50))
        s0 = np.cumsum(r[::-1])[::-1][first]
        d = np.bincount(np.searchsorted(uniq, ts), weights=w * e, minlength=len(uniq))
        self.coef = b
        self.times = uniq
        self.cumhaz = np.cumsum(d / s0)
        return self

    def surv(self, score, times) -> np.ndarray:
        score = np.asarray(score, float) - self.center
        idx = np.searchsorted(self.times, np.atleast_1d(times), side="right") - 1
        h0 = np.where(idx >= 0, self.cumhaz[np.clip(idx, 0, None)], 0.0)
        return np.exp(-np.outer(np.exp(np.clip(self.coef * score, -50, 50)), h0))


# ---- models ----------------------------------------------------------------------
class PrognosticModel:
    supports_weights = False
    supports_strata = False

    def __init__(self, spec: ModelSpec, numeric: list[str], categorical: list[str], horizon: float, seed: int):
        self.spec, self.numeric, self.categorical = spec, numeric, categorical
        self.horizon, self.seed = horizon, seed
        self.params = dict(spec.params)

    def _preprocessor(self, standardise: bool = True) -> Preprocessor:
        from ..imputation.base import resolve_columns

        p = self.params
        return Preprocessor(self.numeric, self.categorical, standardise,
                            m_value_columns=resolve_columns(p.get("m_value_columns"), self.numeric),
                            top_variance=p.get("top_variance"),
                            top_variance_columns=resolve_columns(p.get("top_variance_columns"), self.numeric) or None)

    def fit(self, train, time, event, strata, weights, real_mask) -> "PrognosticModel":
        self.pre = self._preprocessor(self._standardise).fit(train[real_mask])
        X = self.pre.transform(train)
        self._fit(X, np.asarray(time, float), np.asarray(event, float), strata, weights, real_mask)
        self.cal = BreslowCalibrator().fit(self._risk(X), time, event, weights)
        return self

    def risk(self, df: pd.DataFrame) -> np.ndarray:
        return self._risk(self.pre.transform(df))

    def surv(self, df: pd.DataFrame, times) -> np.ndarray | None:
        return self.cal.surv(self.risk(df), times)

    _standardise = True

    def _fit(self, X, time, event, strata, weights, real_mask) -> None:  # pragma: no cover
        raise NotImplementedError

    def _risk(self, X) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def n_nonzero(self) -> int | None:
        return None

    def diagnostics(self) -> dict:
        return {}


class MetaCpGFixed(PrognosticModel):
    """MetaCPG legacy score with weights refitted on the (augmented) training rows."""

    supports_strata = True
    _standardise = False

    def _fit(self, X, time, event, strata, weights, real_mask) -> None:
        from metacpg.core import _endpoint_stats, _legacy_scale

        names = self.pre.names
        matrix = pd.DataFrame(X.T, index=names)
        pheno = pd.DataFrame({"time": time, "status": event})
        covs = []
        if strata is not None:
            pheno["stratum"] = strata
            covs = [{"column": "stratum", "type": "categorical", "mode": "strata"}]
        stats = _endpoint_stats(matrix, pheno, {"name": "aug", "time": "time", "status": "status"}, covs,
                                engine=self.params.get("engine", "batched"))
        w = stats.set_index("feature").loc[names, "standardized_coef"].to_numpy(float)
        self.weights = np.nan_to_num(w)
        raw = X[real_mask] @ self.weights / X.shape[1]
        _, self.center, self.scale = _legacy_scale(raw)
        self.scale = self.scale or 1.0

    def _risk(self, X) -> np.ndarray:
        return ((X @ self.weights / X.shape[1]) - self.center) / self.scale

    def n_nonzero(self) -> int:
        return int(np.sum(self.weights != 0))


class MetaCpGSearch(PrognosticModel):
    """MetaCPG's own feature selection on the (augmented) training rows.

    Every matrix feature gets a univariable (stratified) Cox fit; the top
    ``params.n`` by |z| (default 35) form the score, weighted by their
    standardised coefficients, centred/scaled by the real rows' median/IQR.
    Unlike ``metacpg_fixed`` the CpGs themselves are chosen inside each split,
    so synthetic patients can change *which* features are selected.
    Features are beta values (``params.scale: beta``, logit-transformed to
    M-values after clipping) or M-values already (``m_value``).
    """

    supports_strata = True
    _standardise = False

    def fit(self, train, time, event, strata, weights, real_mask) -> "PrognosticModel":
        from metacpg.core import _endpoint_stats, _legacy_scale, _rank

        n = int(self.params.get("n", 35))
        M = self._m_values(train[self.numeric])
        pheno = pd.DataFrame({"time": np.asarray(time, float), "status": np.asarray(event, float)})
        covs = []
        if strata is not None:
            pheno["stratum"] = strata
            covs = [{"column": "stratum", "type": "categorical", "mode": "strata"}]
        stats = _endpoint_stats(pd.DataFrame(M.T, index=self.numeric), pheno,
                                {"name": "aug", "time": "time", "status": "status"}, covs,
                                engine=self.params.get("engine", "batched"))
        top = _rank(stats).dropna(subset=["standardized_coef"]).head(n)
        self.selected_ = top["feature"].tolist()
        self.weights = top["standardized_coef"].to_numpy(float)
        self._idx = [self.numeric.index(f) for f in self.selected_]
        raw = M[np.asarray(real_mask)][:, self._idx] @ self.weights / len(self._idx)
        _, self.center, self.scale = _legacy_scale(raw)
        self.scale = self.scale or 1.0
        self.cal = BreslowCalibrator().fit(self._score(M), time, event, weights)
        return self

    def _m_values(self, df: pd.DataFrame) -> np.ndarray:
        X = numeric_matrix(df)
        X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
        if self.params.get("scale", "beta") == "beta":
            X = beta_to_m(X, float(self.params.get("beta_epsilon", 1e-6)))
        return X

    def _score(self, M: np.ndarray) -> np.ndarray:
        return (M[:, self._idx] @ self.weights / len(self._idx) - self.center) / self.scale

    def risk(self, df: pd.DataFrame) -> np.ndarray:
        return self._score(self._m_values(df[self.numeric]))

    def n_nonzero(self) -> int:
        return len(self.selected_)

    def diagnostics(self) -> dict:
        return {"selected_features": ";".join(self.selected_)}


class Coxnet(PrognosticModel):
    """Elastic-net Cox; the penalty is chosen by inner CV scored on real rows only."""

    def _fit(self, X, time, event, strata, weights, real_mask) -> None:
        from sklearn.model_selection import KFold
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.metrics import concordance_index_censored
        from sksurv.util import Surv

        l1 = float(self.params.get("l1_ratio", 0.5))
        folds = int(self.params.get("inner_folds", 3))
        Y = Surv.from_arrays(event.astype(bool), time)
        last_exc = None
        # Very small penalties can fail numerically in high dimensions: retry shorter paths.
        for min_ratio in (0.01, 0.05, 0.1, 0.2):
            try:
                base = CoxnetSurvivalAnalysis(l1_ratio=l1, alpha_min_ratio=min_ratio, n_alphas=25,
                                              max_iter=100000).fit(X, Y)
                alphas = base.alphas_
                scores = np.zeros(len(alphas))
                real_idx = np.flatnonzero(real_mask)
                synth_idx = np.flatnonzero(~real_mask)
                for tr_r, va_r in KFold(folds, shuffle=True, random_state=self.seed).split(real_idx):
                    tr = np.r_[real_idx[tr_r], synth_idx]
                    va = real_idx[va_r]
                    m = CoxnetSurvivalAnalysis(l1_ratio=l1, alphas=alphas, max_iter=100000).fit(X[tr], Y[tr])
                    for i, a in enumerate(alphas):
                        if Y[va]["event"].any():
                            scores[i] += concordance_index_censored(
                                Y[va]["event"], Y[va]["time"], m.predict(X[va], alpha=a))[0]
                break
            except (ArithmeticError, ValueError) as exc:
                last_exc = exc
                continue
        else:
            raise RuntimeError(f"coxnet failed to fit: {last_exc}")
        best = int(np.argmax(scores))
        self.model, self.alpha = base, float(alphas[best])
        self.coef_ = base.coef_[:, best]

    def _risk(self, X) -> np.ndarray:
        return self.model.predict(X, alpha=self.alpha)

    def n_nonzero(self) -> int:
        return int(np.sum(self.coef_ != 0))


class CoxPH(PrognosticModel):
    """Ridge-penalised Cox (lifelines); honours strata and case weights."""

    supports_weights = True
    supports_strata = True

    def _fit(self, X, time, event, strata, weights, real_mask) -> None:
        import warnings

        from lifelines import CoxPHFitter

        cols = [f"x{i}" for i in range(X.shape[1])]
        d = pd.DataFrame(X, columns=cols).assign(_t=time, _e=event, _w=weights)
        kw = {}
        if strata is not None and len(set(strata)) > 1:
            d["_s"] = strata
            kw["strata"] = ["_s"]
        self.fitter = CoxPHFitter(penalizer=float(self.params.get("penalizer", 0.1)),
                                  l1_ratio=float(self.params.get("l1_ratio", 0.0)))
        # Only the point estimates are used, so lifelines' warning about naive
        # variances under non-integer case weights does not apply.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitter.fit(d, duration_col="_t", event_col="_e", weights_col="_w", robust=False, **kw)
        self.beta = self.fitter.params_.reindex(cols).to_numpy(float)

    def _risk(self, X) -> np.ndarray:
        return X @ self.beta


class LogisticLandmark(PrognosticModel):
    """P(survival past the horizon); censored-before-horizon rows are excluded."""

    supports_weights = True

    def _fit(self, X, time, event, strata, weights, real_mask) -> None:
        from sklearn.linear_model import LogisticRegressionCV

        ok = (time >= self.horizon) | (event == 1)
        y = (time[ok] >= self.horizon).astype(int)
        self.clf = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=3, scoring="roc_auc", max_iter=5000,
                                        random_state=self.seed).fit(X[ok], y, sample_weight=weights[ok])

    def _risk(self, X) -> np.ndarray:
        return 1.0 - self.clf.predict_proba(X)[:, 1]

    def surv(self, df: pd.DataFrame, times) -> np.ndarray | None:
        times = np.atleast_1d(times)
        if not np.allclose(times, self.horizon):
            return None
        p = 1.0 - self.risk(df)
        return np.repeat(p[:, None], len(times), axis=1)

    def fit(self, train, time, event, strata, weights, real_mask) -> "PrognosticModel":
        self.pre = self._preprocessor().fit(train[real_mask])
        self._fit(self.pre.transform(train), np.asarray(time, float), np.asarray(event, float),
                  strata, np.asarray(weights, float), real_mask)
        return self


MODELS = {
    "metacpg_fixed": MetaCpGFixed,
    "metacpg_search": MetaCpGSearch,
    "coxnet": Coxnet,
    "cox_ph": CoxPH,
    "logistic_landmark": LogisticLandmark,
}


def build_model(spec: ModelSpec, numeric: list[str], categorical: list[str], horizon: float,
                seed: int) -> PrognosticModel:
    return MODELS[spec.kind](spec, numeric, categorical, horizon, seed)
