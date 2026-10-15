"""Fidelity of a synthetic pool against the real rows its generator was fitted on.

Reference points: TVD (categorical) and KS (continuous) are 0 for identical
marginals and 1 for disjoint ones; < 0.1 is close. The discriminator AUC is
how well a classifier tells synthetic from real rows: 0.5 = indistinguishable,
> 0.8 = easily told apart. ``km_diff_h`` is the absolute difference in
Kaplan-Meier survival at the horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import km


def tvd(a: pd.Series, b: pd.Series) -> float:
    pa = a.astype(str).value_counts(normalize=True)
    pb = b.astype(str).value_counts(normalize=True)
    levels = pa.index.union(pb.index)
    return float(0.5 * (pa.reindex(levels, fill_value=0) - pb.reindex(levels, fill_value=0)).abs().sum())


def marginal_gaps(real: pd.DataFrame, synth: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    ks = [stats.ks_2samp(real[c].dropna(), synth[c].dropna()).statistic
          for c in numeric if real[c].notna().sum() > 2 and synth[c].notna().sum() > 2]
    tv = [tvd(real[c], synth[c]) for c in categorical]
    return {"ks_mean": float(np.mean(ks)) if ks else np.nan, "ks_max": float(np.max(ks)) if ks else np.nan,
            "tvd_mean": float(np.mean(tv)) if tv else np.nan, "tvd_max": float(np.max(tv)) if tv else np.nan}


def corr_gap(real: pd.DataFrame, synth: pd.DataFrame, numeric: list[str]) -> float:
    cols = [c for c in numeric if real[c].std() > 0 and synth[c].std() > 0]
    if len(cols) < 2:
        return np.nan
    iu = np.triu_indices(len(cols), 1)
    r = real[cols].corr().to_numpy()[iu]
    s = synth[cols].corr().to_numpy()[iu]
    ok = np.isfinite(r) & np.isfinite(s)
    return float(np.abs(r[ok] - s[ok]).mean()) if ok.any() else np.nan


def discriminator_auc(real: pd.DataFrame, synth: pd.DataFrame, numeric: list[str], categorical: list[str],
                      seed: int) -> float:
    """Cross-validated AUC of a gradient-boosted real-vs-synthetic classifier."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    n = min(len(real), len(synth))
    if n < 10:
        return np.nan
    rng = np.random.default_rng(seed)
    # Balance the classes so 0.5 is chance whatever the pool size.
    s = synth.iloc[rng.choice(len(synth), n, replace=False)]
    r = real.iloc[rng.choice(len(real), n, replace=False)]
    joint = pd.concat([r, s], ignore_index=True)
    parts = [joint[numeric].apply(pd.to_numeric, errors="coerce")] if numeric else []
    if categorical:
        parts.append(pd.get_dummies(joint[categorical].astype(str), dtype=float))
    X = pd.concat(parts, axis=1)
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=seed)
    folds = min(5, n // 2)
    return float(cross_val_score(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                                 scoring="roc_auc").mean())


def survival_gaps(real: pd.DataFrame, synth: pd.DataFrame, time: str, event: str, horizon: float) -> dict:
    from lifelines.statistics import logrank_test

    rt, re = real[time].to_numpy(float), real[event].to_numpy(float)
    st, se = synth[time].to_numpy(float), synth[event].to_numpy(float)
    out = {
        "km_diff_h": float(abs(km(rt, re, horizon)[0] - km(st, se, horizon)[0])),
        "event_rate_diff": float(abs(re.mean() - se.mean())),
    }
    try:
        out["logrank_p"] = float(logrank_test(rt, st, re, se).p_value)
    except Exception:  # pragma: no cover - degenerate inputs
        out["logrank_p"] = np.nan
    # Event-time coupling: do events happen at realistic times?
    ev_r, ev_s = rt[re == 1], st[se == 1]
    out["event_time_ks"] = (float(stats.ks_2samp(ev_r, ev_s).statistic)
                            if len(ev_r) > 2 and len(ev_s) > 2 else np.nan)
    return out


def pool_fidelity(
    train: pd.DataFrame,
    pool: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    time: str | None,
    event: str | None,
    horizon: float,
    target_rows: np.ndarray,
    fixed_columns: list[str],
    seed: int,
) -> dict:
    """Fidelity of a TVAE pool vs the split's real training rows.

    For targeted pools ``target_rows`` (positions in ``train``) restricts the
    comparison to the real rows of the target stratum, and ``fixed_columns``
    (the conditioned-on columns) are left out of the target-stratum TVD.
    """
    numeric = [c for c in numeric if c in pool.columns]
    categorical = [c for c in categorical if c in pool.columns]
    if pool.empty:
        return {"pool_rows": 0}
    out = {"pool_rows": int(len(pool)), **marginal_gaps(train, pool, numeric, categorical),
           "corr_gap": corr_gap(train, pool, numeric),
           "discriminator_auc": discriminator_auc(train, pool, numeric, categorical, seed)}
    if time and event and time in pool.columns and event in pool.columns:
        out.update(survival_gaps(train, pool, time, event, horizon))
    if len(target_rows) and len(target_rows) < len(train):
        real_t = train.iloc[target_rows]
        cats = [c for c in categorical if c not in fixed_columns]
        gaps = marginal_gaps(real_t, pool, [c for c in numeric if c not in fixed_columns], cats)
        out["target_tvd_mean"] = gaps["tvd_mean"]
        out["target_ks_mean"] = gaps["ks_mean"]
        if time and event and time not in fixed_columns and len(real_t) >= 5:
            out["target_km_diff_h"] = survival_gaps(real_t, pool, time, event, horizon)["km_diff_h"]
    return out


def matrix_fidelity(real: pd.DataFrame, synth: pd.DataFrame, top: int, matrix_kind: str, seed: int) -> dict:
    """Decoded-matrix fidelity on the ``top`` highest-variance features (real rows).

    Methylation betas are compared as M-values. ``matrix_mean_gap`` is the mean
    |Δ mean| in real-SD units; ``matrix_sd_ratio`` the median synthetic/real SD
    (decoders shrink variance: well below 1 means over-smooth samples);
    ``matrix_corr_gap`` the mean |Δ correlation| among the top 50 features;
    ``matrix_discriminator_auc`` a real-vs-synthetic classifier on the top features.
    """
    R, S = real.to_numpy(float), synth.to_numpy(float)
    if matrix_kind == "methylation":
        R, S = (np.log2(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6))) for x in (R, S))
    sd = R.std(axis=0, ddof=1)
    idx = np.argsort(sd)[::-1][: min(top, R.shape[1])]
    R, S, sd = R[:, idx], S[:, idx], sd[idx]
    out = {"matrix_mean_gap": float(np.mean(np.abs(S.mean(0) - R.mean(0)) / np.where(sd > 0, sd, 1))),
           "matrix_sd_ratio": float(np.median(S.std(0, ddof=1) / np.where(sd > 0, sd, 1)))}
    k = min(50, R.shape[1])
    if k >= 3 and len(S) > 3:
        iu = np.triu_indices(k, 1)
        out["matrix_corr_gap"] = float(np.nanmean(np.abs(np.corrcoef(R[:, :k], rowvar=False)[iu]
                                                          - np.corrcoef(S[:, :k], rowvar=False)[iu])))
    cols = [f"f{i}" for i in range(R.shape[1])]
    out["matrix_discriminator_auc"] = discriminator_auc(pd.DataFrame(R, columns=cols), pd.DataFrame(S, columns=cols),
                                                        cols, [], seed)
    return out
