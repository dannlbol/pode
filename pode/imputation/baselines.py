"""Classical imputation baselines sharing the :class:`Imputer` interface.

* ``mean_mode``  - column mean / mode (single imputation).
* ``knn``        - scikit-learn ``KNNImputer`` on standardised + one-hot data.
* ``mice``       - chained equations (Bayesian ridge / logistic regression with
                   bootstrap parameter draws), one chain per draw.
* ``missforest`` - ``IterativeImputer(ExtraTreesRegressor)`` with ordinal-coded
                   categoricals (single imputation; categories rounded/clipped).

The kNN / MICE / missForest engines are inherently transductive: they are run
on the training rows stacked with the incomplete rows. Hidden cells are never
visible to them, so there is no leakage of the benchmark truth.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..schema.schema import DatasetSchema
from .base import Imputer, category_key, parallel_map


class TabularCodec:
    """Numeric encoding of a mixed-type table (NaN preserved) and its inverse."""

    def __init__(self, schema: DatasetSchema, df: pd.DataFrame, onehot: bool = True, standardise: bool = True):
        self.onehot = onehot
        self.columns = [c for c in schema.columns if c in df.columns]
        self.numeric = {c: schema.types[c].is_numeric for c in self.columns}
        self.stats: dict[str, tuple[float, float]] = {}
        self.categories: dict[str, list] = {}
        for c in self.columns:
            if self.numeric[c]:
                x = pd.to_numeric(df[c], errors="coerce")
                mu, sd = float(x.mean()), float(x.std())
                mu = mu if np.isfinite(mu) else 0.0
                sd = sd if np.isfinite(sd) and sd > 0 else 1.0
                self.stats[c] = (mu, sd) if standardise else (0.0, 1.0)
            else:
                vals = df[c].dropna()
                keys = category_key(vals.to_numpy())
                seen: dict[str, object] = {}
                for k, v in zip(keys, vals.to_numpy()):
                    seen.setdefault(k, v)
                self.categories[c] = list(seen.values())
        self._keys = {c: list(category_key(v)) for c, v in self.categories.items()}

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        blocks = []
        for c in self.columns:
            s = df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
            if self.numeric[c]:
                mu, sd = self.stats[c]
                blocks.append(((pd.to_numeric(s, errors="coerce").to_numpy(float) - mu) / sd)[:, None])
                continue
            lookup = {k: i for i, k in enumerate(self._keys[c])}
            codes = np.array([lookup.get(k, np.nan) if k is not None else np.nan for k in category_key(s.to_numpy())], float)
            if self.onehot:
                block = np.zeros((len(s), len(self._keys[c])))
                ok = ~np.isnan(codes)
                block[np.flatnonzero(ok), codes[ok].astype(int)] = 1.0
                block[~ok] = np.nan
                blocks.append(block)
            else:
                blocks.append(codes[:, None])
        return np.hstack(blocks) if blocks else np.zeros((len(df), 0))

    def decode(self, mat: np.ndarray, index=None) -> pd.DataFrame:
        out, st = {}, 0
        for c in self.columns:
            if self.numeric[c]:
                mu, sd = self.stats[c]
                out[c] = mat[:, st] * sd + mu
                st += 1
                continue
            cats = self.categories[c]
            if self.onehot:
                block = mat[:, st : st + len(cats)]
                idx = np.argmax(np.nan_to_num(block, nan=-np.inf), axis=1)
                st += len(cats)
            else:
                idx = np.clip(np.round(mat[:, st]), 0, len(cats) - 1).astype(int)
                st += 1
            out[c] = pd.Series([cats[i] for i in idx], dtype=object).infer_objects().to_numpy()
        return pd.DataFrame(out, index=index)


FIT_SCOPES = ("stacked", "reference")


class _StackedImputer(Imputer):
    """Base for engines that run on (training rows + incomplete rows).

    ``fit_scope='stacked'`` (default) lets the incomplete rows' observed values
    inform the imputation models, as in standard MICE practice.
    ``fit_scope='reference'`` learns the models from the training (reference)
    rows only and applies them to the incomplete rows, so the rows being
    imputed never shape the model, which matches deployment on new patients.
    """

    onehot = True
    fit_scope = "stacked"

    def _set_scope(self, fit_scope: str) -> None:
        if fit_scope not in FIT_SCOPES:
            raise ValueError(f"fit_scope must be one of {FIT_SCOPES}, got '{fit_scope}'.")
        self.fit_scope = fit_scope

    def _fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "_StackedImputer":
        self.schema = schema
        self.columns = [c for c in schema.columns if c in df.columns]
        self.reference = df[self.columns].reset_index(drop=True)
        self.codec = TabularCodec(schema, self.reference, onehot=self.onehot)
        return self

    def _stack(self, df: pd.DataFrame) -> tuple[np.ndarray, int, int]:
        """Return ``(matrix, n_learn, start)``.

        Rows ``[:n_learn]`` may train models under ``fit_scope='reference'``;
        rows ``[start:]`` are the ones being imputed. When ``df`` is the
        fitting data itself (transductive use), it is not stacked twice, which
        would duplicate every observation and overstate certainty.
        """
        ref = self.codec.encode(self.reference)
        if len(df) == len(self.reference) and df.reset_index(drop=True).equals(self.reference):
            return ref, len(ref), 0
        return np.vstack([ref, self.codec.encode(df)]), len(ref), len(ref)

    def _drop_all_nan_columns(self, mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = ~np.all(np.isnan(mat), axis=0)
        return mat[:, keep], keep

    def _run(self, mat: np.ndarray, seed: int, n_ref: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _fit_transform(self, imputer, mat: np.ndarray, n_ref: int) -> np.ndarray:
        """Stacked: fit on all rows. Reference: fit on the first ``n_ref`` rows."""
        if self.fit_scope == "reference":
            return imputer.fit(mat[:n_ref]).transform(mat)
        return imputer.fit_transform(mat)

    def _impute(self, df: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
        mat, n_ref, start = self._stack(df)
        sub, keep = self._drop_all_nan_columns(mat)
        n_runs = m if self.multiple else 1
        draws = []
        for i in range(n_runs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filled = self._run(sub.copy(), seed + i, n_ref)
            full = np.zeros_like(mat)
            full[:, keep] = filled
            draws.append(self.codec.decode(full[start:], index=df.index))
        return draws if self.multiple else [draws[0].copy() for _ in range(m)]


class MeanModeImputer(Imputer):
    name = "mean_mode"
    multiple = False

    def _fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "MeanModeImputer":
        self.schema = schema
        self.columns = [c for c in schema.columns if c in df.columns]
        self.fill = {}
        for c in self.columns:
            if schema.types[c].is_numeric:
                self.fill[c] = pd.to_numeric(df[c], errors="coerce").mean()
            else:
                mode = df[c].mode()
                self.fill[c] = mode.iloc[0] if not mode.empty else np.nan
        return self

    def _impute(self, df: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
        filled = df.copy()
        for c in self.columns:
            filled[c] = filled[c].astype(object).where(filled[c].notna(), self.fill[c])
        return [filled.infer_objects() for _ in range(m)]


class KNNEngine(_StackedImputer):
    name = "knn"
    multiple = False

    def __init__(self, n_neighbors: int = 5, fit_scope: str = "stacked"):
        super().__init__()
        self.n_neighbors = n_neighbors
        self._set_scope(fit_scope)

    def _run(self, mat: np.ndarray, seed: int, n_ref: int) -> np.ndarray:
        from sklearn.impute import KNNImputer

        imp = KNNImputer(n_neighbors=self.n_neighbors, keep_empty_features=True)
        return self._fit_transform(imp, mat, n_ref)


class MICEEngine(_StackedImputer):
    """Chained equations with per-type conditional models.

    Continuous columns: Bayesian ridge, imputed by sampling its predictive
    distribution. Categorical/binary columns: multinomial logistic regression,
    imputed by sampling predicted probabilities. Each conditional model is fit
    on a bootstrap of the observed rows so parameter uncertainty propagates
    (approximately proper multiple imputation, cf. ``mice``'s ``norm.boot`` /
    ``logreg.boot``). One independent chain per draw.
    """

    name = "mice"
    multiple = True

    def __init__(self, max_iter: int = 5, n_jobs: int = 4, fit_scope: str = "stacked"):
        super().__init__()
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self._set_scope(fit_scope)

    def _slices(self) -> dict[str, slice]:
        out, st = {}, 0
        for c in self.codec.columns:
            width = 1 if self.codec.numeric[c] else len(self.codec.categories[c])
            out[c] = slice(st, st + width)
            st += width
        return out

    def _impute(self, df: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
        from joblib import delayed

        mat, n_ref, start = self._stack(df)
        slices = self._slices()
        # Chains are independent, so draws run in parallel processes.
        chains = parallel_map(
            [delayed(self._chain)(mat, slices, seed + i, n_ref) for i in range(m)],
            1 if m == 1 else self.n_jobs,
        )
        return [self.codec.decode(E[start:], index=df.index) for E in chains]

    def _chain(self, mat: np.ndarray, slices: dict[str, slice], seed: int, n_ref: int) -> np.ndarray:
        from sklearn.linear_model import BayesianRidge, LogisticRegression

        rng = np.random.default_rng(seed)
        E = mat.copy()
        miss = {c: np.isnan(E[:, sl.start]) for c, sl in slices.items()}
        # Rows whose observed values may train the conditional models.
        learn = np.ones(E.shape[0], dtype=bool)
        if self.fit_scope == "reference":
            learn[n_ref:] = False
        fit_obs = {c: ~miss[c] & learn for c in slices}
        todo = [c for c in slices if miss[c].any() and fit_obs[c].sum() >= 2]
        # Initialise: standardised numerics at 0 (the mean), categoricals by
        # sampling observed levels.
        for c, sl in slices.items():
            rows = np.flatnonzero(miss[c])
            if rows.size == 0:
                continue
            if self.codec.numeric[c]:
                E[rows, sl.start] = 0.0
                continue
            observed = E[fit_obs[c], sl]
            k = sl.stop - sl.start
            if observed.size:
                p = observed.sum(0) / observed.sum()
                picks = rng.choice(k, size=rows.size, p=p)
            else:
                picks = np.zeros(rows.size, dtype=int)
            E[rows, sl] = 0.0
            E[rows, sl.start + picks] = 1.0

        all_idx = np.arange(E.shape[1])
        for _ in range(self.max_iter):
            for c in todo:
                sl = slices[c]
                X = E[:, np.setdiff1d(all_idx, np.arange(sl.start, sl.stop))]
                obs_rows = np.flatnonzero(fit_obs[c])
                mis_rows = np.flatnonzero(miss[c])
                boot = rng.choice(obs_rows, size=obs_rows.size, replace=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if self.codec.numeric[c]:
                        model = BayesianRidge().fit(X[boot], E[boot, sl.start])
                        mu, sd = model.predict(X[mis_rows], return_std=True)
                        E[mis_rows, sl.start] = mu + sd * rng.standard_normal(mis_rows.size)
                        continue
                    y = np.argmax(E[boot, sl], axis=1)
                    E[mis_rows, sl] = 0.0
                    classes = np.unique(y)
                    if classes.size == 1:
                        E[mis_rows, sl.start + classes[0]] = 1.0
                        continue
                    model = LogisticRegression(max_iter=300).fit(X[boot], y)
                    proba = model.predict_proba(X[mis_rows])
                    cum = np.cumsum(proba, axis=1)
                    pick = (rng.random((mis_rows.size, 1)) > cum).sum(axis=1)
                    pick = np.minimum(pick, classes.size - 1)
                    E[mis_rows, sl.start + model.classes_[pick]] = 1.0
        return E


class MissForestEngine(_StackedImputer):
    name = "missforest"
    multiple = False
    onehot = False

    def __init__(self, n_estimators: int = 30, max_iter: int = 5, n_jobs: int = 1, fit_scope: str = "stacked"):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self._set_scope(fit_scope)

    def _run(self, mat: np.ndarray, seed: int, n_ref: int) -> np.ndarray:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        imp = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=self.n_estimators, max_features="sqrt",  # mtry = sqrt(p), as in missForest
                random_state=seed, n_jobs=self.n_jobs,
            ),
            max_iter=self.max_iter,
            random_state=seed,
            keep_empty_features=True,
        )
        return self._fit_transform(imp, mat, n_ref)
