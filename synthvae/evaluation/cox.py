"""Cox proportional-hazards fitting for TSTR and hazard-ratio comparison.

Prefers ``lifelines`` when available (robust, handles ties/regularisation well).
Falls back to a compact NumPy Newton-Raphson Breslow implementation so the harness
runs with no extra dependencies. Both expose the same tiny API: ``fit(X, time,
event)`` then ``.coef_`` (log hazard ratios) and ``.predict_risk(X)`` (linear
predictor = risk score).
"""
from __future__ import annotations

import numpy as np
import warnings

try:
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError, ConvergenceWarning
    import pandas as pd

    _HAS_LIFELINES = True
except Exception:  # pragma: no cover
    _HAS_LIFELINES = False


class _NumpyCoxPH:
    """Breslow-tie Cox PH via Newton-Raphson with a small ridge for stability."""

    def __init__(self, ridge: float = 1e-3, max_iter: int = 50, tol: float = 1e-6):
        self.ridge = ridge
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self._x_mean = None
        self._x_std = None

    def fit(self, X, time, event):
        X = np.asarray(X, dtype=float)
        time = np.asarray(time, dtype=float)
        event = np.asarray(event).astype(int)
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        Xs = (X - self._x_mean) / self._x_std
        n, p = Xs.shape
        beta = np.zeros(p)

        # process risk sets by descending time
        order = np.argsort(-time)
        Xo, to, eo = Xs[order], time[order], event[order]

        for _ in range(self.max_iter):
            eta = Xo @ beta
            if not np.isfinite(eta).all():
                raise FloatingPointError("non-finite Cox linear predictor")
            # A global max shift alone is insufficient: early risk sets can omit
            # the globally highest-risk row, allowing every weight in that set to
            # underflow to zero.  Clipping the shifted scores keeps ratios stable
            # while retaining a strictly positive denominator for every risk set.
            w = np.exp(np.clip(eta - eta.max(), -50.0, 0.0))
            # cumulative sums over descending time = risk-set sums (time >= t)
            S0 = np.cumsum(w)
            S1 = np.cumsum(w[:, None] * Xo, axis=0)
            # S2 per row is outer product; accumulate incrementally
            grad = np.zeros(p)
            info = np.zeros((p, p))
            # We need, at each event, the risk-set sums for time >= t_event.
            # Because sorted descending, the risk set for row i is rows[0..i]
            # ONLY if there are no ties; Breslow ties: all rows with time >= t.
            # Map each row to the last index sharing its time (largest position).
            # Compute group boundaries.
            # last_idx[i] = max position j with to[j] == to[i]
            # Since descending & possibly tied, find for each unique time the max idx.
            uniq_t, inv = np.unique(-to, return_inverse=True)
            # position of last occurrence per group
            last_pos = np.zeros(len(uniq_t), dtype=int)
            for pos in range(n):
                last_pos[inv[pos]] = pos
            risk_end = last_pos[inv]  # for each row, index bounding its risk set

            # Precompute cumulative S2
            # S2_cum[i] = sum_{k<=i} w_k x_k x_k^T
            S2_cum = np.zeros((n, p, p))
            acc = np.zeros((p, p))
            for i in range(n):
                acc = acc + w[i] * np.outer(Xo[i], Xo[i])
                S2_cum[i] = acc

            for i in range(n):
                if eo[i] != 1:
                    continue
                r = risk_end[i]
                s0 = S0[r]
                s1 = S1[r]
                s2 = S2_cum[r]
                if not np.isfinite(s0) or s0 <= 0.0:
                    raise FloatingPointError("empty or non-finite Cox risk set")
                mean = s1 / s0
                grad += Xo[i] - mean
                info += s2 / s0 - np.outer(mean, mean)

            info += self.ridge * np.eye(p)
            if not np.isfinite(info).all() or not np.isfinite(grad).all():
                raise FloatingPointError("non-finite Cox information matrix")
            try:
                step = np.linalg.solve(info, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(info, grad, rcond=None)[0]
            if not np.isfinite(step).all():
                raise FloatingPointError("non-finite Cox Newton step")
            beta_new = beta + step
            if not np.isfinite(beta_new).all():
                raise FloatingPointError("non-finite Cox coefficients")
            if np.max(np.abs(step)) < self.tol:
                beta = beta_new
                break
            beta = beta_new

        # convert back to raw-feature scale so coefficients are hazard-ratio logs
        self.coef_ = beta / self._x_std
        return self

    def predict_risk(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_


class CoxModel:
    """Uniform Cox interface over lifelines (preferred) or the numpy fallback."""

    def __init__(self, prefer_lifelines: bool = True, penalizer: float = 0.01):
        self.prefer_lifelines = prefer_lifelines and _HAS_LIFELINES
        self.penalizer = penalizer
        self._impl = None
        self._using_lifelines = False
        self._feature_names = None
        self._keep_mask = None
        self._kept_feature_names = None

    def fit(self, X, time, event, feature_names=None):
        X = np.asarray(X, dtype=float)
        self._feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]
        self._keep_mask = np.std(X, axis=0) > 1e-12
        if not np.any(self._keep_mask):
            warnings.warn(
                "All Cox features are constant or nearly constant; returning zero coefficients.",
                RuntimeWarning,
            )
            self._impl = None
            self._using_lifelines = False
            self.coef_ = np.zeros(X.shape[1], dtype=float)
            return self

        Xk = X[:, self._keep_mask]
        names = [n for n, keep in zip(self._feature_names, self._keep_mask) if keep]
        self._kept_feature_names = names
        if self.prefer_lifelines:
            try:
                df = pd.DataFrame(Xk, columns=names)
                df["time"] = np.asarray(time, dtype=float)
                df["event"] = np.asarray(event).astype(int)
                cph = CoxPHFitter(penalizer=self.penalizer)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    cph.fit(df, duration_col="time", event_col="event")
                convergence_warnings = [
                    w for w in caught if issubclass(w.category, ConvergenceWarning)
                ]
                if convergence_warnings:
                    joined = "; ".join(str(w.message) for w in convergence_warnings)
                    raise ConvergenceError(joined)
                self._impl = cph
                self._using_lifelines = True
                self.coef_ = np.zeros(X.shape[1], dtype=float)
                self.coef_[self._keep_mask] = cph.params_.reindex(names).fillna(0.0).values
                return self
            except Exception:  # pragma: no cover - exercised in runtime sweeps
                warnings.warn(
                    "lifelines Cox did not converge; using numpy Cox fallback for this score.",
                    RuntimeWarning,
                )

        self.coef_ = np.zeros(X.shape[1], dtype=float)
        try:
            self._impl = _NumpyCoxPH().fit(Xk, time, event)
            if not np.isfinite(self._impl.coef_).all():
                raise FloatingPointError("non-finite numpy Cox coefficients")
            self.coef_[self._keep_mask] = self._impl.coef_
        except Exception:
            warnings.warn(
                "numpy Cox fallback was unstable; using zero coefficients for this score.",
                RuntimeWarning,
            )
            self._impl = None
        self._using_lifelines = False
        return self

    def predict_risk(self, X):
        X = np.asarray(X, dtype=float)
        if self._keep_mask is not None:
            X = X[:, self._keep_mask]
        if self._using_lifelines and self._impl is not None:
            df = pd.DataFrame(X, columns=self._kept_feature_names)
            # partial hazard is proportional to exp(risk); use log for a risk score
            return np.log(self._impl.predict_partial_hazard(df).values)
        if self._impl is not None:
            return self._impl.predict_risk(X)
        return np.zeros(X.shape[0], dtype=float)

    def hazard_ratios(self):
        return np.exp(self.coef_)
