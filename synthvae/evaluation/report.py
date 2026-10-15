"""Assemble the four evaluation signals into one report.

1. **TSTR concordance** -- fit Cox on synthetic (clinical + latents), score real
   held-out. Compare against TRTR (train real / test real) as the ceiling.
2. **KM distance** -- integrated absolute difference real vs synthetic, overall
   and in late follow-up (where VAEs tend to drift).
3. **Hazard-ratio preservation** -- correlation / max abs log-HR difference
   between a Cox fit on real and one fit on synthetic.
4. **Latent fidelity** -- how close synthetic latent marginals/covariance are to
   real (a quick joint-distribution sanity check).

Everything is NumPy + optional lifelines; nothing here needs torch or SDV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cindex import concordance_index
from .km import km_distance
from .cox import CoxModel


def _feature_frame(clinical: pd.DataFrame, latents: np.ndarray) -> pd.DataFrame:
    """Numeric feature frame from clinical (numeric or one-hot) + latents."""
    parts: list[pd.DataFrame] = []
    for c in clinical.columns:
        col = clinical[c]
        if col.dtype == object or str(col.dtype).startswith("category"):
            dummies = pd.get_dummies(col, prefix=c, drop_first=True)
            parts.append(dummies.astype(float))
        else:
            parts.append(pd.DataFrame({c: col.to_numpy(dtype=float)}))
    lat_df = pd.DataFrame(
        np.asarray(latents, dtype=float),
        columns=[f"expr_latent_{i}" for i in range(latents.shape[1])],
    )
    parts.append(lat_df)
    if not parts:
        return pd.DataFrame(index=np.arange(len(latents)))
    return pd.concat(parts, axis=1)


def _align_feature_frames(reference: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    aligned = other.reindex(columns=reference.columns, fill_value=0.0)
    return aligned.astype(float)


def _cox_mask(train_frame: pd.DataFrame, variance_floor: float = 1e-12) -> np.ndarray:
    if train_frame.shape[1] == 0:
        return np.zeros(0, dtype=bool)
    std = train_frame.to_numpy(dtype=float).std(axis=0)
    return std > variance_floor


def evaluate(real, synth, train_real=None, prefer_lifelines: bool = True) -> dict:
    """Compute the report.

    Parameters
    ----------
    real : dict with keys clinical (DataFrame), latents (ndarray),
           time (ndarray), event (ndarray)
    synth : same schema, for synthetic data
    train_real : same schema as ``real`` but for the real training split. When
        provided, Cox-based metrics use a fixed feature set derived from this
        split so all sweep candidates are scored on the same columns.
    """
    train_frame = _feature_frame(
        train_real["clinical"], train_real["latents"]
    ) if train_real is not None else _feature_frame(real["clinical"], real["latents"])
    real_frame = _align_feature_frames(train_frame, _feature_frame(real["clinical"], real["latents"]))
    synth_frame = _align_feature_frames(train_frame, _feature_frame(synth["clinical"], synth["latents"]))
    keep_mask = _cox_mask(train_frame)
    names = list(train_frame.columns[keep_mask])
    Xr = real_frame.to_numpy(dtype=float)[:, keep_mask] if keep_mask.size else np.zeros((len(real_frame), 0))
    Xs = synth_frame.to_numpy(dtype=float)[:, keep_mask] if keep_mask.size else np.zeros((len(synth_frame), 0))

    # --- TSTR / TRTR concordance ---
    cox_syn = CoxModel(prefer_lifelines=prefer_lifelines).fit(
        Xs, synth["time"], synth["event"], feature_names=names
    )
    risk_on_real = cox_syn.predict_risk(Xr)
    tstr_c = concordance_index(real["time"], real["event"], risk_on_real)

    cox_real = CoxModel(prefer_lifelines=prefer_lifelines).fit(
        Xr, real["time"], real["event"], feature_names=names
    )
    risk_real = cox_real.predict_risk(Xr)
    trtr_c = concordance_index(real["time"], real["event"], risk_real)

    # --- hazard-ratio preservation ---
    log_hr_real = cox_real.coef_
    log_hr_syn = cox_syn.coef_
    hr_corr = float(np.corrcoef(log_hr_real, log_hr_syn)[0, 1]) if len(names) > 1 else float("nan")
    hr_max_abs_diff = float(np.max(np.abs(log_hr_real - log_hr_syn)))

    # --- KM distance ---
    km = km_distance(real["time"], real["event"], synth["time"], synth["event"])

    # --- latent fidelity (marginal + covariance) ---
    Lr = np.asarray(real["latents"], dtype=float)
    Ls = np.asarray(synth["latents"], dtype=float)
    mean_gap = float(np.mean(np.abs(Lr.mean(0) - Ls.mean(0))))
    std_gap = float(np.mean(np.abs(Lr.std(0) - Ls.std(0))))
    cov_gap = float(np.linalg.norm(np.cov(Lr.T) - np.cov(Ls.T)) /
                    (np.linalg.norm(np.cov(Lr.T)) + 1e-8))

    return {
        "tstr_concordance": tstr_c,
        "trtr_concordance": trtr_c,
        "concordance_gap": (trtr_c - tstr_c) if not np.isnan(tstr_c) else float("nan"),
        "hr_logcorr": hr_corr,
        "hr_max_abs_logdiff": hr_max_abs_diff,
        "cox_features_used": int(keep_mask.sum()),
        "cox_features_dropped": int(len(keep_mask) - keep_mask.sum()),
        "km_iad": km["km_iad"],
        "km_iad_late": km["km_iad_late"],
        "latent_mean_gap": mean_gap,
        "latent_std_gap": std_gap,
        "latent_cov_relgap": cov_gap,
    }


def format_report(metrics: dict) -> str:
    order = [
        ("TSTR concordance (train synth / test real)", "tstr_concordance"),
        ("TRTR concordance (ceiling)", "trtr_concordance"),
        ("Concordance gap (lower better)", "concordance_gap"),
        ("Hazard-ratio log-correlation (higher better)", "hr_logcorr"),
        ("Hazard-ratio max |log diff| (lower better)", "hr_max_abs_logdiff"),
        ("KM integrated abs diff (lower better)", "km_iad"),
        ("KM integrated abs diff, late follow-up", "km_iad_late"),
        ("Latent mean gap", "latent_mean_gap"),
        ("Latent std gap", "latent_std_gap"),
        ("Latent covariance rel. gap", "latent_cov_relgap"),
    ]
    width = max(len(lbl) for lbl, _ in order)
    lines = ["Evaluation report", "=" * (width + 12)]
    for lbl, key in order:
        v = metrics.get(key, float("nan"))
        lines.append(f"{lbl:<{width}} : {v: .4f}")
    return "\n".join(lines)
