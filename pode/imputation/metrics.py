"""Imputation quality metrics.

Everything is computed on the cells a benchmark hid (``mask``), using all M
draws per cell, so point accuracy, probabilistic calibration and uncertainty
are assessed together:

* continuous  - MAE / RMSE of the draw mean (also SD-normalised), bias, CRPS,
                50% / 90% interval coverage and width, Wasserstein-1;
* discrete    - accuracy / balanced accuracy of the modal draw, Brier and log
                score of the draw frequencies, prevalence TVD;
* dependence  - Spearman and Cramér's V structure of completed vs true tables;
* downstream  - Cox models pooled across draws with Rubin's rules and compared
                with the fit on the unmasked data, plus MNAR delta-adjustment.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ..evaluation.metrics import total_variation_distance
from ..schema.schema import DatasetSchema
from .base import category_key

LAPLACE = 0.5


# ---- cell-level --------------------------------------------------------------
def crps_ensemble(draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS per cell from an ensemble ``draws`` (M, n) and truth ``y`` (n,).

    Energy form: E|X - y| - 0.5 E|X - X'|. Zero for a point mass at the truth.
    """
    draws = np.asarray(draws, float)
    term1 = np.mean(np.abs(draws - y[None, :]), axis=0)
    term2 = np.mean(np.abs(draws[:, None, :] - draws[None, :, :]), axis=(0, 1))
    return term1 - 0.5 * term2


def interval_coverage(draws: np.ndarray, y: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Central ``level`` interval from the draws: (covered (n,), width (n,))."""
    lo = np.quantile(draws, (1 - level) / 2, axis=0)
    hi = np.quantile(draws, 1 - (1 - level) / 2, axis=0)
    return (y >= lo) & (y <= hi), hi - lo


def _continuous_metrics(y: np.ndarray, X: np.ndarray, sd: float) -> dict:
    sd = sd if np.isfinite(sd) and sd > 0 else 1.0
    point = X.mean(axis=0)
    err = point - y
    crps = crps_ensemble(X, y)
    cov50, _ = interval_coverage(X, y, 0.5)
    cov90, w90 = interval_coverage(X, y, 0.9)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "nmae": float(np.mean(np.abs(err)) / sd),
        "nrmse": float(np.sqrt(np.mean(err**2)) / sd),
        "bias_norm": float(np.mean(err) / sd),
        "crps": float(np.mean(crps)),
        "crps_norm": float(np.mean(crps) / sd),
        "coverage50": float(np.mean(cov50)),
        "coverage90": float(np.mean(cov90)),
        "width90_norm": float(np.mean(w90) / sd),
        "w1_norm": float(stats.wasserstein_distance(y, X.ravel()) / sd),
    }


def _discrete_metrics(y: np.ndarray, X: np.ndarray, levels: list[str]) -> dict:
    from sklearn.metrics import balanced_accuracy_score

    levels = sorted((set(levels) | set(y.tolist()) | set(X.ravel().tolist())) - {None})
    k = len(levels)
    pos = {lv: i for i, lv in enumerate(levels)}
    counts = np.zeros((len(y), k))
    for row in X:
        counts[np.arange(len(y)), [pos[v] for v in row]] += 1
    m = X.shape[0]
    probs = (counts + LAPLACE) / (m + LAPLACE * k)
    truth_idx = np.array([pos[v] for v in y])
    onehot = np.eye(k)[truth_idx]
    modal = np.array(levels, dtype=object)[np.argmax(counts, axis=1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bal = balanced_accuracy_score(y, modal)
    return {
        "accuracy": float(np.mean(modal == y)),
        "balanced_accuracy": float(bal),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "log_score": float(-np.mean(np.log(probs[np.arange(len(y)), truth_idx]))),
        "tvd": total_variation_distance(pd.Series(y), pd.Series(X.ravel())),
    }


def cell_metrics(
    truth: pd.DataFrame,
    draws: list[pd.DataFrame],
    mask: np.ndarray,
    schema: DatasetSchema,
) -> list[dict]:
    """Per-column metrics over masked cells. ``mask`` aligns with ``truth``."""
    rows = []
    for j, col in enumerate(truth.columns):
        hidden = mask[:, j]
        if not hidden.any() or col not in schema.types:
            continue
        base = {"column": col, "type": schema.types[col].value, "n_masked": int(hidden.sum())}
        if schema.types[col].is_numeric:
            y = pd.to_numeric(truth[col], errors="coerce").to_numpy(float)[hidden]
            X = np.vstack([pd.to_numeric(d[col], errors="coerce").to_numpy(float)[hidden] for d in draws])
            sd = float(pd.to_numeric(truth[col], errors="coerce").std())
            base.update(_continuous_metrics(y, X, sd))
        else:
            y = category_key(truth[col].to_numpy()[hidden])
            X = np.vstack([category_key(d[col].to_numpy()[hidden]) for d in draws])
            levels = [v for v in category_key(truth[col].dropna().unique())]
            base.update(_discrete_metrics(y, X, levels))
        rows.append(base)
    return rows


# ---- dependence ---------------------------------------------------------------
def _codes(df: pd.DataFrame, cols: list[str], schema: DatasetSchema, ref: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in cols:
        if schema.types[c].is_numeric:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            order = {k: i for i, k in enumerate(category_key(ref[c].value_counts().index.to_numpy()))}
            out[c] = pd.Series([order.get(k, np.nan) for k in category_key(df[c].to_numpy())], index=df.index)
    return pd.DataFrame(out)


def _cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def dependence_metrics(
    truth: pd.DataFrame, draws: list[pd.DataFrame], schema: DatasetSchema, max_categorical: int = 15
) -> dict:
    """Structure preservation: mean |ΔSpearman| and mean |ΔCramér's V| over pairs."""
    cols = [c for c in truth.columns if c in schema.types and truth[c].nunique(dropna=True) > 1]
    if len(cols) < 2:
        return {"spearman_mad": np.nan, "cramers_v_mad": np.nan}
    t_codes = _codes(truth, cols, schema, truth)
    r_true = t_codes.corr(method="spearman").to_numpy()
    iu = np.triu_indices(len(cols), 1)
    sp = []
    for d in draws:
        r = _codes(d, cols, schema, truth).corr(method="spearman").to_numpy()
        sp.append(np.nanmean(np.abs(r[iu] - r_true[iu])))

    cats = [c for c in cols if schema.types[c].is_discrete][:max_categorical]
    cv = []
    if len(cats) >= 2:
        pairs = [(a, b) for i, a in enumerate(cats) for b in cats[i + 1 :]]
        tk = {c: category_key(truth[c].to_numpy()) for c in cats}
        v_true = np.array([_cramers_v(tk[a], tk[b]) for a, b in pairs])
        for d in draws:
            dk = {c: category_key(d[c].to_numpy()) for c in cats}
            v = np.array([_cramers_v(dk[a], dk[b]) for a, b in pairs])
            cv.append(np.nanmean(np.abs(v - v_true)))
    return {
        "spearman_mad": float(np.mean(sp)),
        "cramers_v_mad": float(np.mean(cv)) if cv else np.nan,
    }


# ---- downstream Cox + Rubin ----------------------------------------------------
def _cox_frame(df: pd.DataFrame, covariates: list[str], schema: DatasetSchema, ref: pd.DataFrame, time_col: str, event_col: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            time_col: pd.to_numeric(df[time_col], errors="coerce"),
            event_col: pd.to_numeric(df[event_col], errors="coerce"),
        },
        index=df.index,
    )
    for c in covariates:
        if schema.types[c].is_numeric:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            levels = list(category_key(ref[c].value_counts().index.to_numpy()))
            out[c] = pd.Categorical(category_key(df[c].to_numpy()), categories=levels)
    return out.dropna()


def fit_cox(
    df: pd.DataFrame,
    covariates: list[str],
    schema: DatasetSchema,
    time_col: str,
    event_col: str,
    ref: pd.DataFrame,
    mode: str = "univariate",
    penalizer: float = 0.0,
) -> pd.DataFrame:
    """Cox fits returning one row per term: covariate, term, coef, se, n."""
    from lifelines import CoxPHFitter

    def term(c: str) -> str:
        return f"C(`{c}`)" if schema.types[c].is_discrete else f"`{c}`"

    groups = [[c] for c in covariates] if mode == "univariate" else [list(covariates)]
    frame = _cox_frame(df, covariates, schema, ref, time_col, event_col)
    rows = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for group in groups:
            sub = frame[[time_col, event_col, *group]].dropna()
            if len(sub) < 15 or any(sub[c].nunique() < 2 for c in group):
                continue
            try:
                cph = CoxPHFitter(penalizer=penalizer)
                cph.fit(sub, time_col, event_col, formula=" + ".join(term(c) for c in group))
            except Exception:
                continue
            for name, r in cph.summary.iterrows():
                owner = next((c for c in group if c in str(name)), group[0])
                rows.append({"covariate": owner, "term": str(name), "coef": float(r["coef"]), "se": float(r["se(coef)"]), "n": len(sub)})
    return pd.DataFrame(rows, columns=["covariate", "term", "coef", "se", "n"])


def rubin_pool(q: np.ndarray, se: np.ndarray, n: int, k: int = 1, alpha: float = 0.05) -> dict:
    """Pool M estimates with Rubin's rules (Barnard-Rubin small-sample df)."""
    q, u = np.asarray(q, float), np.asarray(se, float) ** 2
    m = len(q)
    qbar, ubar = float(q.mean()), float(u.mean())
    b = float(q.var(ddof=1)) if m > 1 else 0.0
    t = ubar + (1 + 1 / m) * b
    lam = (1 + 1 / m) * b / t if t > 0 else 0.0
    nu_com = max(n - k, 1)
    nu_obs = (nu_com + 1) / (nu_com + 3) * nu_com * (1 - lam)
    if m > 1 and lam > 0:
        nu = 1 / (1 / ((m - 1) / lam**2) + 1 / nu_obs)
    else:
        nu = nu_obs
    r = (1 + 1 / m) * b / ubar if ubar > 0 else 0.0
    fmi = (r + 2 / (nu + 3)) / (r + 1)
    se_t = float(np.sqrt(t))
    tcrit = float(stats.t.ppf(1 - alpha / 2, nu))
    p = float(2 * stats.t.sf(abs(qbar / se_t), nu)) if se_t > 0 else np.nan
    return {
        "coef": qbar, "se": se_t, "within_var": ubar, "between_var": b, "df": float(nu),
        "fmi": float(fmi), "ci_lo": qbar - tcrit * se_t, "ci_hi": qbar + tcrit * se_t, "p": p,
    }


def pooled_cox(
    draws: list[pd.DataFrame],
    covariates: list[str],
    schema: DatasetSchema,
    time_col: str,
    event_col: str,
    ref: pd.DataFrame,
    mode: str = "univariate",
    penalizer: float = 0.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Fit Cox on each completed dataset and pool every term with Rubin's rules."""
    fits = [fit_cox(d, covariates, schema, time_col, event_col, ref, mode, penalizer).assign(draw=i) for i, d in enumerate(draws)]
    allfits = pd.concat(fits, ignore_index=True) if fits else pd.DataFrame()
    rows = []
    if allfits.empty:
        return pd.DataFrame()
    for (cov, term), g in allfits.groupby(["covariate", "term"], sort=False):
        k = 1 if mode == "univariate" else len(covariates)
        pooled = rubin_pool(g["coef"].to_numpy(), g["se"].to_numpy(), int(g["n"].max()), k, alpha)
        rows.append({"covariate": cov, "term": term, "n_draws_fitted": len(g), **pooled})
    return pd.DataFrame(rows)


def compare_to_truth(pooled: pd.DataFrame, truth_fit: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Join pooled estimates with the full-data fit and score agreement."""
    if pooled.empty or truth_fit.empty:
        return pd.DataFrame()
    z = stats.norm.ppf(1 - alpha / 2)
    tf = truth_fit.rename(columns={"coef": "coef_true", "se": "se_true"})[["covariate", "term", "coef_true", "se_true"]]
    out = pooled.merge(tf, on=["covariate", "term"], how="inner")
    out["p_true"] = 2 * stats.norm.sf(np.abs(out["coef_true"] / out["se_true"]))
    out["hr"] = np.exp(out["coef"])
    out["hr_true"] = np.exp(out["coef_true"])
    out["bias"] = out["coef"] - out["coef_true"]
    out["abs_hr_error"] = np.abs(out["hr"] - out["hr_true"])
    out["covers_truth"] = (out["ci_lo"] <= out["coef_true"]) & (out["coef_true"] <= out["ci_hi"])
    out["ci_width_ratio"] = (out["ci_hi"] - out["ci_lo"]) / (2 * z * out["se_true"])
    out["significance_agrees"] = (out["p"] < alpha) == (out["p_true"] < alpha)
    return out


# ---- MNAR delta adjustment -----------------------------------------------------
def apply_delta(
    draws: list[pd.DataFrame],
    mask: np.ndarray,
    columns: list[str],
    schema: DatasetSchema,
    truth: pd.DataFrame,
    delta: float,
    rng: np.random.Generator,
) -> list[pd.DataFrame]:
    """Shift imputed (masked) cells in an MNAR direction.

    Continuous columns move by ``delta`` x SD(column); binary columns have the
    per-cell probability of their second level (from draw frequencies) shifted
    by ``delta`` on the log-odds scale and are re-sampled. Other discrete
    columns are left unchanged.
    """
    if delta == 0:
        return draws
    out = [d.copy() for d in draws]
    col_pos = {c: j for j, c in enumerate(truth.columns)}
    for c in columns:
        hidden = mask[:, col_pos[c]]
        if not hidden.any():
            continue
        if schema.types[c].is_numeric:
            sd = float(pd.to_numeric(truth[c], errors="coerce").std())
            for d in out:
                vals = pd.to_numeric(d[c], errors="coerce").to_numpy(float)
                vals[hidden] = vals[hidden] + delta * sd
                d[c] = vals
            continue
        levels = sorted(set(category_key(truth[c].dropna().unique())))
        if len(levels) != 2:
            continue
        raw_levels = {k: v for k, v in zip(category_key(truth[c].dropna().to_numpy()), truth[c].dropna().to_numpy())}
        keys = np.vstack([category_key(d[c].to_numpy()[hidden]) for d in draws])
        p = np.clip((keys == levels[1]).mean(axis=0), 0.02, 0.98)
        p_new = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + delta)))
        for d in out:
            pick = rng.random(p_new.size) < p_new
            col = d[c].astype(object).to_numpy()
            col[hidden] = np.where(pick, raw_levels[levels[1]], raw_levels[levels[0]])
            d[c] = pd.Series(col, index=d.index).infer_objects()
    return out
