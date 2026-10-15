"""The individual privacy checks. Each returns ``(summary_dict, table_or_None)``.

Every distance-based check compares the synthetic data against a *reference*:
how close genuinely new patients (the holdout, never seen by the generator)
are to the training patients. Without a holdout the reference falls back to
each training patient's leave-one-out nearest real neighbour, and the checks
that need non-members (membership, attribute member gap, linkage excess) are
not assessed.

Reading the numbers:

* ``exact_copies``: training patients reproduced as a whole record. Compared
  with how many holdout patients the synthetic data "copies" by coincidence,
  which happens in coarse, mostly categorical tables.
* ``dcr_p5_ratio``: 5th percentile of synthetic distance-to-closest-record
  (DCR) / the reference's 5th percentile. ~1 = synthetic rows are no closer to
  training patients than new patients are; well below 1 = near-copies.
* ``near_copy_excess_pX``: fraction of synthetic rows inside the reference's
  X% closest band, divided by X% (the rate new patients give). 1 = expected;
  2 = twice as many near-copies as chance.
* ``membership_auc``: distance attack - real patients scored by closeness to
  the nearest synthetic row; AUC for telling training patients from holdout
  patients. 0.5 = no signal, 0.6 = worrying, 1 = membership fully revealed.
* ``attribute member gap``: an attacker who knows a patient's quasi-identifiers
  looks up the closest synthetic record(s) and reads off a sensitive value.
  Accuracy on training patients minus accuracy on unseen patients is the
  disclosure caused by being in the training data; accuracy that is equally
  high for unseen patients is population-level inference (the data's utility).
* ``linkage_excess``: fraction of patients singled out (unique quasi-identifier
  combination, matched by exactly one synthetic row) - training minus holdout,
  in equal-sized cohorts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .config import PrivacyConfig
from .distance import MISSING, Binner, Encoded, GowerSpace, row_keys


def qname(q: float) -> str:
    return f"p{q * 100:g}"


@dataclass
class Context:
    """Everything the checks share: the tables, their encodings and nearest-neighbour results."""

    train: pd.DataFrame
    holdout: pd.DataFrame | None  # None when absent or too small to use
    synth: pd.DataFrame  # assessed (possibly subsampled) synthetic rows
    synth_full: pd.DataFrame
    synth_rows: np.ndarray  # positions of ``synth`` rows in ``synth_full``
    numeric: list[str]
    categorical: list[str]
    sensitive: list[str]
    tiers: dict[str, list[str]]
    strata: list[str]
    cfg: PrivacyConfig
    train_ids: np.ndarray | None = None
    log: Callable[[str], None] = print
    # Filled by ``prepare``.
    space: GowerSpace = None
    e_train: Encoded = None
    e_hold: Encoded = None
    e_synth: Encoded = None
    e_synth_full: Encoded = None
    st_d: np.ndarray = None  # synthetic -> training, k=2
    st_i: np.ndarray = None
    ref_d: np.ndarray = None  # holdout -> training (or training LOO), k=2
    ref_kind: str = "holdout"
    loo_d: np.ndarray = None  # training LOO nearest real, k=1
    ts_d: np.ndarray = None  # training -> synthetic, k=1
    ts_i: np.ndarray = None
    hs_d: np.ndarray = None  # holdout -> synthetic, k=1
    thresholds: dict[float, float] = field(default_factory=dict)
    binner: Binner = None
    rng: np.random.Generator = None

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical

    @property
    def has_holdout(self) -> bool:
        return self.holdout is not None

    @property
    def near_thr(self) -> float:
        return self.thresholds[max(self.thresholds)]

    def prepare(self) -> "Context":
        cfg, n_jobs = self.cfg, self.cfg.n_jobs
        self.rng = np.random.default_rng(cfg.seed)
        self.space = GowerSpace(self.train, self.numeric, self.categorical, cfg.weights)
        self.e_train = self.space.encode(self.train)
        self.e_synth = self.space.encode(self.synth)
        self.e_synth_full = self.space.encode(self.synth_full)
        self.st_d, self.st_i = self.space.nearest(self.e_synth, self.e_train, k=2, n_jobs=n_jobs)
        self.loo_d, _ = self.space.nearest(self.e_train, self.e_train, k=2, exclude_self=True, n_jobs=n_jobs)
        self.ts_d, self.ts_i = self.space.nearest(self.e_train, self.e_synth, k=1, n_jobs=n_jobs)
        if self.has_holdout:
            self.e_hold = self.space.encode(self.holdout)
            self.ref_d, _ = self.space.nearest(self.e_hold, self.e_train, k=2, n_jobs=n_jobs)
            self.hs_d, _ = self.space.nearest(self.e_hold, self.e_synth, k=1, n_jobs=n_jobs)
            self.ref_kind = "holdout"
        else:
            self.ref_d, self.ref_kind = self.loo_d, "leave-one-out"
        ref0 = _finite(self.ref_d[:, 0])
        self.thresholds = {q: float(np.quantile(ref0, q)) if len(ref0) else np.nan
                           for q in sorted(self.cfg.near_copy_quantiles)}
        keyed = sorted({c for cols in self.tiers.values() for c in cols} | set(self.strata))
        self.binner = Binner(self.train, keyed, set(self.numeric), cfg.numeric_bins)
        return self

    def strata_small_mask(self, df: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Rows of ``df`` (default: training) in a stratum level with <= rare_k training patients."""
        df = self.train if df is None else df
        small = np.zeros(len(df), dtype=bool)
        reason = np.full(len(df), "", dtype=object)
        for s in self.strata:
            counts = pd.Series(self.binner.labels(self.train, s)).value_counts()
            lab = self.binner.labels(df, s)
            hit = np.array([counts.get(v, 0) <= self.cfg.rare_k for v in lab])
            for i in np.flatnonzero(hit & ~small):
                reason[i] = f"small stratum {s}={self.binner.describe(s, lab[i])}"
            small |= hit
        return small, reason


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _nndr(d: np.ndarray) -> np.ndarray:
    d1, d2 = d[:, 0], d[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(d2 > 0, d1 / d2, 1.0)
    return np.where(np.isfinite(d2), r, np.nan)


def excess_pvalue(count: int, n: int, rate: float, n_ref: int) -> float:
    """One-sided binomial p-value for >= ``count`` of ``n`` events at the reference ``rate``.

    A zero reference rate is floored at half an event in the reference sample,
    so a single synthetic event is not automatically "infinitely" significant.
    """
    from scipy.stats import binom

    if n == 0 or count == 0:
        return 1.0
    floor = 0.5 / max(n_ref, 1)
    return float(binom.sf(count - 1, n, min(max(rate, floor), 1.0)))


def auc_stats(members: np.ndarray, nonmembers: np.ndarray, n_boot: int, rng) -> dict:
    """AUC (members scoring higher = membership signal), bootstrap 95% CI, advantage, TPR at 10% FPR."""
    from sklearn.metrics import roc_auc_score, roc_curve

    members, nonmembers = members[np.isfinite(members)], nonmembers[np.isfinite(nonmembers)]
    y = np.r_[np.ones(len(members)), np.zeros(len(nonmembers))]
    s = np.r_[members, nonmembers]
    auc = float(roc_auc_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)
    boots = []
    for _ in range(n_boot):
        m = rng.choice(members, len(members))
        n = rng.choice(nonmembers, len(nonmembers))
        boots.append(roc_auc_score(y, np.r_[m, n]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return {"auc": auc, "ci_low": float(lo), "ci_high": float(hi),
            "advantage": float(np.max(tpr - fpr)), "tpr_at_fpr10": float(tpr[fpr <= 0.1].max()),
            "n_members": len(members), "n_nonmembers": len(nonmembers)}


# ---------------------------------------------------------------------------
# 1. Metadata / identifier leakage
# ---------------------------------------------------------------------------

def check_identifiers(real: pd.DataFrame, synth: pd.DataFrame, schema=None, id_column: str | None = None):
    """Identifier, date and free-text columns in the synthetic table, and real ID values inside it."""
    from pandas.api import types as ptypes

    from ..schema.column_types import ColumnRole
    from ..schema.detect import detect_identifiers

    rows = []
    real_ids = set(detect_identifiers(real)) | ({id_column} if id_column and id_column in real else set())
    if schema is not None:
        real_ids |= {c for c in schema.columns_of_role(ColumnRole.IDENTIFIER) if c in real}
    for c in sorted((real_ids | set(detect_identifiers(synth))) & set(synth.columns)):
        rows.append({"column": c, "issue": "identifier column", "severity": "FAIL",
                     "detail": "direct/row identifier present in the synthetic table"})
    flagged = {r["column"] for r in rows}
    for c in synth.columns:
        if c in flagged:
            continue
        s = synth[c].dropna()
        if s.empty:
            continue
        if ptypes.is_datetime64_any_dtype(s):
            rows.append({"column": c, "issue": "date column", "severity": "REVIEW",
                         "detail": "exact dates; shift to relative time or coarsen"})
            continue
        if ptypes.is_numeric_dtype(s):
            continue
        text = s.astype(str)
        sample = text.sample(min(len(text), 200), random_state=0)
        if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.9:
            continue
        with _quiet():
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.8:
            rows.append({"column": c, "issue": "date column", "severity": "REVIEW",
                         "detail": "values parse as dates; shift to relative time or coarsen"})
        elif text.str.len().mean() > 30 or (text.str.contains(" ").mean() > 0.5 and text.nunique() / len(text) > 0.5):
            rows.append({"column": c, "issue": "free text", "severity": "REVIEW",
                         "detail": "long or mostly-unique text; remove before training"})
    # Real identifier values appearing anywhere in the synthetic table.
    values = set()
    for c in real_ids:
        values |= {str(v) for v in real[c].dropna().unique()}
    if values:
        for c in synth.columns:
            if ptypes.is_numeric_dtype(synth[c]):
                continue
            n = int(synth[c].dropna().astype(str).isin(values).sum())
            if n:
                rows.append({"column": c, "issue": "real identifier value", "severity": "FAIL",
                             "detail": f"{n} synthetic cells equal a real identifier value"})
    table = pd.DataFrame(rows, columns=["column", "issue", "severity", "detail"])
    return {"identifier_issues_fail": int((table["severity"] == "FAIL").sum()),
            "identifier_issues_review": int((table["severity"] == "REVIEW").sum())}, table


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


# ---------------------------------------------------------------------------
# 2. Exact memorisation
# ---------------------------------------------------------------------------

def check_exact_copies(ctx: Context):
    dec = ctx.cfg.exact_decimals
    tk = row_keys(ctx.train, ctx.numeric, ctx.categorical, dec)
    sk = row_keys(ctx.synth_full, ctx.numeric, ctx.categorical, dec)
    sset = set(sk)
    copied = np.isin(tk, list(sset))
    small, _ = ctx.strata_small_mask()
    out = {
        "exact_copies": int(copied.sum()),
        "exact_copy_frac_train": float(copied.mean()),
        "exact_copy_synthetic_rows": int(np.isin(sk, list(set(tk))).sum()),
        "exact_copies_small_strata": int((copied & small).sum()),
    }
    if ctx.has_holdout:
        hk = row_keys(ctx.holdout, ctx.numeric, ctx.categorical, dec)
        out["exact_copy_frac_holdout"] = float(np.isin(hk, list(sset)).mean())
    return out, None


# ---------------------------------------------------------------------------
# 3. Near-copying (distance to closest record)
# ---------------------------------------------------------------------------

def check_dcr(ctx: Context):
    dcr, ref = ctx.st_d[:, 0], _finite(ctx.ref_d[:, 0])
    out = {"reference": ctx.ref_kind}
    for name, q in (("p1", 1), ("p5", 5), ("median", 50)):
        out[f"dcr_synthetic_{name}"] = float(np.percentile(dcr, q))
        out[f"dcr_reference_{name}"] = float(np.percentile(ref, q)) if len(ref) else np.nan
    p5r = out["dcr_reference_p5"]
    out["dcr_p5_ratio"] = float(out["dcr_synthetic_p5"] / p5r) if p5r and p5r > 0 else np.nan
    for q, thr in ctx.thresholds.items():
        count, band = int((dcr <= thr).sum()), int((ref <= thr).sum())
        obs, exp = count / len(dcr), band / max(len(ref), 1)
        out[f"near_copy_frac_{qname(q)}"] = obs
        out[f"near_copy_expected_{qname(q)}"] = exp
        out[f"near_copy_excess_{qname(q)}"] = obs / exp if exp > 0 else np.nan
        # Reference rows inside the band: below ~5 the band itself is too noisy to gate on.
        out[f"near_copy_band_rows_{qname(q)}"] = band
        out[f"near_copy_pvalue_{qname(q)}"] = excess_pvalue(count, len(dcr), exp, len(ref))
    ns, nr = _nndr(ctx.st_d), _nndr(ctx.ref_d)
    out["nndr_mean_synthetic"] = float(np.nanmean(ns))
    out["nndr_mean_reference"] = float(np.nanmean(nr)) if np.isfinite(nr).any() else np.nan
    iso = (dcr <= ctx.near_thr) & (ns < 0.5)
    iso_s, iso_r = float(iso.mean()), float(((ctx.ref_d[:, 0] <= ctx.near_thr) & (nr < 0.5)).mean())
    out["isolated_near_copy_pvalue"] = excess_pvalue(int(iso.sum()), len(dcr), iso_r, len(ctx.ref_d))
    out["isolated_near_copy_frac_synthetic"] = iso_s
    out["isolated_near_copy_frac_reference"] = iso_r
    out["isolated_near_copy_excess"] = iso_s / iso_r if iso_r > 0 else (np.inf if iso_s > 0 else np.nan)
    return out, None


# ---------------------------------------------------------------------------
# 4. Stratified near-copying and membership
# ---------------------------------------------------------------------------

def check_strata(ctx: Context):
    if not ctx.strata:
        return {"strata_flagged": np.nan}, None
    cfg, thr = ctx.cfg, ctx.near_thr
    expected = float((_finite(ctx.ref_d[:, 0]) <= thr).mean())
    ref_rows = ctx.holdout if ctx.has_holdout else ctx.train
    rows = []
    for s in ctx.strata:
        lt, ls, lr = (ctx.binner.labels(df, s) for df in (ctx.train, ctx.synth, ref_rows))
        lh = lr if ctx.has_holdout else None
        for level in sorted(set(lt) | set(ls)):
            mt, ms, mr = lt == level, ls == level, lr == level
            n_s = int(ms.sum())
            d_s, d_r = ctx.st_d[ms, 0], _finite(ctx.ref_d[mr, 0])
            near = int((d_s <= thr).sum())
            frac = near / n_s if n_s else np.nan
            row = {"stratum": s, "level": ctx.binner.describe(s, level), "n_train": int(mt.sum()),
                   "n_synthetic": n_s, "n_reference": int(mr.sum()), "near_copies": near,
                   "near_copy_frac": frac, "near_copy_excess": frac / expected if n_s and expected > 0 else np.nan,
                   "dcr_p5_ratio": np.nan, "membership_auc": np.nan}
            if n_s >= cfg.min_group and len(d_r) >= cfg.min_group and np.percentile(d_r, 5) > 0:
                row["dcr_p5_ratio"] = float(np.percentile(d_s, 5) / np.percentile(d_r, 5))
            if lh is not None and mt.sum() >= cfg.min_group and (lh == level).sum() >= cfg.min_group:
                row["membership_auc"] = auc_stats(-ctx.ts_d[mt, 0], -ctx.hs_d[lh == level, 0], 0, ctx.rng)["auc"]
            row["small_stratum"] = row["n_train"] <= cfg.rare_k
            reasons = []
            row["near_copy_pvalue"] = excess_pvalue(near, n_s, expected, len(ctx.ref_d))
            if row["near_copy_excess"] > cfg.gates.max_near_copy_excess and row["near_copy_pvalue"] < cfg.gates.alpha:
                reasons.append("near-copy excess")
            if row["small_stratum"] and near:
                reasons.append("near-copies of a small stratum")
            if row["small_stratum"] and n_s and row["n_train"] <= 2:
                reasons.append("synthetic rows generated from <=2 real patients")
            if np.isfinite(row["membership_auc"]) and row["membership_auc"] > cfg.gates.max_membership_auc:
                reasons.append("membership AUC above gate")
            row["flag"] = "; ".join(reasons)
            rows.append(row)
    table = pd.DataFrame(rows)
    return {"strata_levels": len(table), "strata_flagged": int((table["flag"] != "").sum()),
            "small_strata_with_synthetic": int((table["small_stratum"] & (table["n_synthetic"] > 0)).sum())}, table


# ---------------------------------------------------------------------------
# 5. Membership inference
# ---------------------------------------------------------------------------

def check_membership(ctx: Context):
    if not ctx.has_holdout:
        return {"membership_auc": np.nan}, None
    cfg = ctx.cfg
    rows = [{"attack": "all_columns", "columns": len(ctx.columns),
             **auc_stats(-ctx.ts_d[:, 0], -ctx.hs_d[:, 0], cfg.n_bootstrap, ctx.rng)}]
    for tier, cols in ctx.tiers.items():
        num = [c for c in cols if c in ctx.numeric]
        cat = [c for c in cols if c in ctx.categorical]
        space = GowerSpace(ctx.train, num, cat, cfg.weights)
        es = space.encode(ctx.synth)
        dt, _ = space.nearest(space.encode(ctx.train), es, n_jobs=cfg.n_jobs)
        dh, _ = space.nearest(space.encode(ctx.holdout), es, n_jobs=cfg.n_jobs)
        rows.append({"attack": f"qi:{tier}", "columns": len(cols),
                     **auc_stats(-dt[:, 0], -dh[:, 0], cfg.n_bootstrap, ctx.rng)})
    table = pd.DataFrame(rows)
    main = table.iloc[0]
    worst = table.loc[table["auc"].idxmax()]
    return {"membership_auc": float(main["auc"]), "membership_auc_ci_low": float(main["ci_low"]),
            "membership_auc_ci_high": float(main["ci_high"]), "membership_advantage": float(main["advantage"]),
            "membership_tpr_at_fpr10": float(main["tpr_at_fpr10"]),
            "membership_auc_max": float(worst["auc"]), "membership_auc_max_ci_high": float(worst["ci_high"]),
            "membership_worst_attack": str(worst["attack"])}, table


# ---------------------------------------------------------------------------
# 6. Attribute inference
# ---------------------------------------------------------------------------

def _predict(neighbours: np.ndarray, values: np.ndarray, numeric: bool) -> np.ndarray:
    if numeric:
        with _quiet():
            return np.nanmedian(np.where(neighbours >= 0, values[np.maximum(neighbours, 0)], np.nan), axis=1)
    out = np.full(len(neighbours), MISSING, dtype=object)
    for i, row in enumerate(neighbours):
        v = [values[j] for j in row if j >= 0 and values[j] != MISSING]
        if v:
            levels, counts = np.unique(v, return_counts=True)
            out[i] = levels[np.argmax(counts)]
    return out


def _target(df: pd.DataFrame, col: str, numeric: bool) -> np.ndarray:
    if numeric:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    from .distance import _cat_strings

    return _cat_strings(df[col].to_numpy())


def _correct(pred, truth, numeric: bool, tol: float) -> np.ndarray:
    """1/0 per row where the truth is known (NaN where it is not)."""
    if numeric:
        ok = np.isfinite(truth)
        hit = np.abs(pred - truth) <= tol
        return np.where(ok, np.where(np.isfinite(pred), hit, False), np.nan).astype(float)
    ok = truth != MISSING
    return np.where(ok, (pred == truth).astype(float), np.nan)


def _boot_diff(a: np.ndarray, b: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    if not n_boot or not len(a) or not len(b):
        return np.nan, np.nan
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def check_attribute_inference(ctx: Context):
    cfg, rows = ctx.cfg, []
    for tier, cols in ctx.tiers.items():
        num = [c for c in cols if c in ctx.numeric]
        cat = [c for c in cols if c in ctx.categorical]
        targets = [t for t in ctx.sensitive if t not in cols]
        if not cols or not targets:
            continue
        space = GowerSpace(ctx.train, num, cat, cfg.weights)
        es = space.encode(ctx.synth)
        _, ti = space.nearest(space.encode(ctx.train), es, k=5, n_jobs=cfg.n_jobs)
        hi = space.nearest(space.encode(ctx.holdout), es, k=5, n_jobs=cfg.n_jobs)[1] if ctx.has_holdout else None
        for t in targets:
            is_num = t in ctx.numeric
            sv = _target(ctx.synth, t, is_num)
            truth_t = _target(ctx.train, t, is_num)
            tol = 0.0
            if is_num:
                rng_ = np.nanmax(truth_t) - np.nanmin(truth_t) if np.isfinite(truth_t).any() else 0.0
                tol = cfg.attribute_tolerance * (rng_ if rng_ > 0 else 1.0)
                base_pred = np.nanmedian(sv) if np.isfinite(sv).any() else np.nan
            else:
                known = sv[sv != MISSING]
                base_pred = pd.Series(known).mode().iloc[0] if len(known) else MISSING
            base = _correct(np.full(len(truth_t), base_pred, dtype=float if is_num else object), truth_t, is_num, tol)
            for k in (1, 5):
                cm = _correct(_predict(ti[:, :k], sv, is_num), truth_t, is_num, tol)
                cm = cm[np.isfinite(cm)]
                row = {"tier": tier, "target": t, "attacker": f"{k}-nn", "type": "numeric" if is_num else "categorical",
                       "n_members": len(cm), "acc_members": float(cm.mean()) if len(cm) else np.nan,
                       "acc_baseline": float(np.nanmean(base)) if np.isfinite(base).any() else np.nan,
                       "acc_nonmembers": np.nan, "member_gap": np.nan, "gap_ci_low": np.nan, "gap_ci_high": np.nan}
                if hi is not None:
                    truth_h = _target(ctx.holdout, t, is_num)
                    cn = _correct(_predict(hi[:, :k], sv, is_num), truth_h, is_num, tol)
                    cn = cn[np.isfinite(cn)]
                    if len(cn) >= cfg.min_group and len(cm):
                        row.update(acc_nonmembers=float(cn.mean()), member_gap=float(cm.mean() - cn.mean()))
                        row["gap_ci_low"], row["gap_ci_high"] = _boot_diff(cm, cn, cfg.n_bootstrap, ctx.rng)
                row["gain_over_baseline"] = row["acc_members"] - row["acc_baseline"]
                rows.append(row)
    if not rows:
        return {"attribute_member_gap_max": np.nan}, None
    table = pd.DataFrame(rows)
    gaps = table["member_gap"]
    if gaps.notna().any():
        w = table.loc[gaps.idxmax()]
        return {"attribute_member_gap_max": float(w["member_gap"]), "attribute_gap_ci_low": float(w["gap_ci_low"]),
                "attribute_worst": f"{w['tier']}/{w['target']}/{w['attacker']}",
                "attribute_gain_over_baseline_max": float(table["gain_over_baseline"].max())}, table
    return {"attribute_member_gap_max": np.nan,
            "attribute_gain_over_baseline_max": float(table["gain_over_baseline"].max())}, table


# ---------------------------------------------------------------------------
# 7. Singling-out / linkage
# ---------------------------------------------------------------------------

def _match_distance(ctx: Context, keys: np.ndarray, enc: Encoded, synth_counts: pd.Series,
                    first: dict) -> np.ndarray:
    """Distance from each patient to the single synthetic row sharing their key (NaN if not exactly one)."""
    rows = np.array([first.get(k, -1) if synth_counts.get(k, 0) == 1 else -1 for k in keys])
    out = np.full(len(keys), np.nan)
    has = rows >= 0
    if has.any():
        out[has] = ctx.space.paired(enc, np.flatnonzero(has), ctx.e_synth_full, rows[has])
    return out


def _link_rates(keys: np.ndarray, match_d: np.ndarray, thr: float) -> tuple[float, float]:
    """(singled-out rate, disclosive-link rate) of a cohort: unique key in the cohort AND one synthetic match."""
    counts = pd.Series(keys).value_counts()
    unique = counts.reindex(keys).to_numpy() == 1
    single = unique & np.isfinite(match_d)
    return float(single.mean()), float((single & (match_d <= thr)).mean())


def _two_prop_p(a: float, b: float, n: int) -> float:
    """One-sided p-value that rate ``a`` exceeds rate ``b`` (both over ``n`` patients)."""
    from scipy.stats import norm

    p = (a + b) / 2
    se = np.sqrt(max(2 * p * (1 - p) / max(n, 1), 1e-12))
    return float(norm.sf((a - b) / se))


def check_linkage(ctx: Context, repeats: int = 20):
    """Singling-out: a key unique among the patients and matched by exactly one synthetic row.

    A *disclosive* link also requires that synthetic row to be a near-copy of
    the patient's whole record, i.e. it really is "their" record. Rates for
    training and holdout patients are compared in equal-sized cohorts, because
    uniqueness depends on cohort size.
    """
    cfg, rows, thr = ctx.cfg, [], ctx.near_thr
    for tier, cols in ctx.tiers.items():
        b = Binner(ctx.train, cols, set(ctx.numeric), cfg.numeric_bins)
        tk, sk = b.keys(ctx.train), b.keys(ctx.synth_full)
        tc, sc = pd.Series(tk).value_counts(), pd.Series(sk).value_counts()
        first: dict = {}
        for i, k in enumerate(sk):
            first.setdefault(k, i)
        td = _match_distance(ctx, tk, ctx.e_train, sc, first)
        singled = (tc.reindex(tk).to_numpy() == 1) & np.isfinite(td)
        row = {"tier": tier, "columns": len(cols), "train_unique_frac": float((tc.reindex(tk).to_numpy() == 1).mean()),
               "singled_out_members": int(singled.sum()),
               "disclosive_links_members": int((singled & (td <= thr)).sum()),
               "sensitive_reproduced_frac": np.nan}
        targets = [t for t in ctx.sensitive if t not in cols]
        if singled.any() and targets:
            hits = np.ones(int(singled.sum()), dtype=bool)
            ti = np.flatnonzero(singled)
            si = np.array([first[tk[i]] for i in ti])
            for t in targets:
                is_num = t in ctx.numeric
                truth = _target(ctx.train, t, is_num)[ti]
                guess = _target(ctx.synth_full, t, is_num)[si]
                tol = 0.0
                if is_num:
                    v = _target(ctx.train, t, True)
                    tol = cfg.attribute_tolerance * ((np.nanmax(v) - np.nanmin(v)) or 1.0)
                hits &= np.nan_to_num(_correct(guess, truth, is_num, tol), nan=0.0).astype(bool)
            row["sensitive_reproduced_frac"] = float(hits.mean())
        if ctx.has_holdout:
            hk = b.keys(ctx.holdout)
            hd = _match_distance(ctx, hk, ctx.e_hold, sc, first)
            m = min(len(tk), len(hk))
            rm, rh = [], []
            for _ in range(repeats):
                i = ctx.rng.choice(len(tk), m, replace=False)
                j = ctx.rng.choice(len(hk), m, replace=False)
                rm.append(_link_rates(tk[i], td[i], thr))
                rh.append(_link_rates(hk[j], hd[j], thr))
            (sm, dm), (sh, dh) = np.mean(rm, axis=0), np.mean(rh, axis=0)
            # Disclosive links are rare under chance, so they are also tested as counts over the full
            # cohorts. The holdout is smaller, so its keys are unique more often: conservative for members.
            full_h = _link_rates(hk, hd, thr)[1]
            row.update(cohort_size=m, link_rate_members=sm, link_rate_holdout=sh, linkage_excess=sm - sh,
                       linkage_pvalue=_two_prop_p(sm, sh, m), disclosive_rate_members=dm,
                       disclosive_rate_holdout=dh, disclosive_excess=dm - dh,
                       disclosive_rate_holdout_full=full_h,
                       disclosive_pvalue=excess_pvalue(row["disclosive_links_members"], len(tk), full_h, len(hk)))
        rows.append(row)
    if not rows:
        return {"linkage_excess_max": np.nan}, None
    table = pd.DataFrame(rows)
    out = {"singled_out_members_max": int(table["singled_out_members"].max()),
           "disclosive_links_members_max": int(table["disclosive_links_members"].max())}
    if "linkage_excess" in table:
        w = table.loc[table["linkage_excess"].idxmax()]
        out.update(linkage_excess_max=float(w["linkage_excess"]), linkage_pvalue=float(w["linkage_pvalue"]),
                   linkage_worst_tier=str(w["tier"]))
        d = table.loc[table["disclosive_pvalue"].idxmin()]
        out.update(disclosive_links=int(d["disclosive_links_members"]),
                   disclosive_rate_members=float(d["disclosive_links_members"] / len(ctx.train)),
                   disclosive_rate_holdout=float(d["disclosive_rate_holdout_full"]),
                   disclosive_pvalue=float(d["disclosive_pvalue"]), disclosive_worst_tier=str(d["tier"]))
    else:
        out["linkage_excess_max"] = np.nan
    return out, table


# ---------------------------------------------------------------------------
# 8. Rare-combination disclosure
# ---------------------------------------------------------------------------

def check_rare_combinations(ctx: Context):
    """Synthetic rows reproducing a quasi-identifier combination of <= rare_k training patients,
    and those that are also near-copies; the holdout gives the chance rate of the same event."""
    cfg, rows, thr = ctx.cfg, [], ctx.near_thr
    for tier, cols in ctx.tiers.items():
        b = Binner(ctx.train, cols, set(ctx.numeric), cfg.numeric_bins)
        tc = pd.Series(b.keys(ctx.train)).value_counts()
        rare = (tc.reindex(b.keys(ctx.synth)).fillna(0).to_numpy() >= 1) & \
               (tc.reindex(b.keys(ctx.synth)).fillna(0).to_numpy() <= cfg.rare_k)
        leaks = rare & (ctx.st_d[:, 0] <= thr)
        sc = pd.Series(b.keys(ctx.synth_full)).value_counts()
        tcs = tc.reindex(sc.index).fillna(0)
        row = {"tier": tier, "rare_combo_rows": int(rare.sum()), "rare_combo_leaks": int(leaks.sum()),
               "leak_rate_synthetic": float(leaks.mean()), "leak_rate_holdout": np.nan, "leak_pvalue": np.nan,
               "rare_cells_reproduced": int(((sc <= cfg.rare_k) & (tcs >= 1) & (tcs <= cfg.rare_k)).sum())}
        if ctx.has_holdout:
            hc = tc.reindex(b.keys(ctx.holdout)).fillna(0).to_numpy()
            rate_h = float(((hc >= 1) & (hc <= cfg.rare_k) & (ctx.ref_d[:, 0] <= thr)).mean())
            row.update(leak_rate_holdout=rate_h,
                       leak_pvalue=excess_pvalue(int(leaks.sum()), len(leaks), rate_h, len(ctx.holdout)))
        rows.append(row)
    if not rows:
        return {"rare_combo_leaks": np.nan}, None
    table = pd.DataFrame(rows)
    w = table.loc[table["rare_combo_leaks"].idxmax()]
    return {"rare_combo_leaks": int(w["rare_combo_leaks"]), "rare_combo_leak_rate": float(w["leak_rate_synthetic"]),
            "rare_combo_leak_rate_holdout": float(w["leak_rate_holdout"]), "rare_combo_pvalue": float(w["leak_pvalue"]),
            "rare_combo_worst_tier": str(w["tier"])}, table


# ---------------------------------------------------------------------------
# 9. Outlier replication (and the closest pairs, for case review)
# ---------------------------------------------------------------------------

def _label(ctx: Context, train_rows: np.ndarray) -> np.ndarray:
    if ctx.train_ids is None:
        return np.full(len(train_rows), "", dtype=object)
    return np.array([str(ctx.train_ids[i]) for i in train_rows], dtype=object)


def _outliers(d_real: np.ndarray, d_syn: np.ndarray, small: np.ndarray, thr: float):
    """Outlier mask (top 5% distance to the nearest real patient, or a small stratum), ratio, replicated mask."""
    finite = _finite(d_real)
    q95 = float(np.quantile(finite, 0.95)) if len(finite) else np.inf
    isolated = np.isfinite(d_real) & (d_real >= q95) & (d_real > 0)
    outlier = isolated | small
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d_real > 0, d_syn / d_real, np.nan)
    return isolated, outlier, ratio, outlier & ((d_syn <= thr) | (ratio < 0.5))


def check_outliers(ctx: Context):
    """Unusual training patients (isolated or in a small stratum) with a synthetic row much closer than any
    real patient. Holdout outliers, measured the same way, give the rate expected by chance."""
    cfg, thr = ctx.cfg, ctx.near_thr
    d_loo, d_syn = ctx.loo_d[:, 0], ctx.ts_d[:, 0]
    small, reason = ctx.strata_small_mask()
    isolated, outlier, ratio, replicated = _outliers(d_loo, d_syn, small, thr)
    out = {"outliers": int(outlier.sum()), "outliers_replicated": int(replicated.sum()),
           "outlier_replication_rate": float(replicated.sum() / max(outlier.sum(), 1)),
           "outlier_replication_rate_holdout": np.nan, "outlier_pvalue": np.nan}
    if ctx.has_holdout:
        h_small, _ = ctx.strata_small_mask(ctx.holdout)
        _, h_out, _, h_rep = _outliers(ctx.ref_d[:, 0], ctx.hs_d[:, 0], h_small, thr)
        rate_h = float(h_rep.sum() / max(h_out.sum(), 1))
        out.update(outlier_replication_rate_holdout=rate_h,
                   outlier_pvalue=excess_pvalue(int(replicated.sum()), int(outlier.sum()), rate_h, int(h_out.sum())))

    idx = np.flatnonzero(outlier)
    order = idx[np.argsort(np.nan_to_num(ratio[idx], nan=np.inf), kind="stable")]
    keep = np.r_[order[replicated[order]], order[~replicated[order]]][:max(cfg.review_top_n, int(replicated.sum()))]
    why = np.where(isolated, "isolated (top 5% distance to nearest real patient)", "")
    why = np.array([("; ".join(x for x in (a, b) if x)) for a, b in zip(why, reason)], dtype=object)
    outliers = pd.DataFrame({
        "train_row": keep, "id": _label(ctx, keep), "nearest_synthetic_row": ctx.synth_rows[ctx.ts_i[keep, 0]],
        "d_nearest_synthetic": d_syn[keep], "d_nearest_real": d_loo[keep], "ratio": ratio[keep],
        "replicated": replicated[keep], "reason": why[keep]})

    top = np.argsort(ctx.st_d[:, 0], kind="stable")[:cfg.review_top_n]
    ti = ctx.st_i[top, 0]
    closest = pd.DataFrame({
        "synthetic_row": ctx.synth_rows[top], "train_row": ti, "id": _label(ctx, ti),
        "distance": ctx.st_d[top, 0], "nndr": _nndr(ctx.st_d)[top],
        "columns_differing": _columns_differing(ctx.e_synth, top, ctx.e_train, ti),
        "columns_total": len(ctx.columns)})
    return out, (outliers, closest)


def _columns_differing(a: Encoded, ai: np.ndarray, b: Encoded, bi: np.ndarray) -> np.ndarray:
    na, nb = a.num[ai], b.num[bi]
    same = (np.abs(na - nb) < 1e-12) | (np.isnan(na) & np.isnan(nb))
    return (~same).sum(axis=1) + (a.cat[ai] != b.cat[bi]).sum(axis=1)


# ---------------------------------------------------------------------------
# 10. Missingness patterns
# ---------------------------------------------------------------------------

def check_missingness(ctx: Context):
    def patterns(df):
        m = df[ctx.columns].isna().to_numpy()
        return np.array(["".join("1" if x else "0" for x in r) for r in m], dtype=object), m.any()

    tp, t_any = patterns(ctx.train)
    sp, s_any = patterns(ctx.synth)
    if not (t_any or s_any):
        return {"missingness_present": False}, None
    tc = pd.Series(tp).value_counts()
    rare = np.array([1 <= tc.get(p, 0) <= ctx.cfg.rare_k for p in sp])
    out = {"missingness_present": True, "rare_missingness_rows": int(rare.sum()),
           "rare_missingness_rate": float(rare.mean()), "rare_missingness_rate_holdout": np.nan}
    if ctx.has_holdout:
        hp, _ = patterns(ctx.holdout)
        rate_h = float(np.mean([1 <= tc.get(p, 0) <= ctx.cfg.rare_k for p in hp]))
        out["rare_missingness_rate_holdout"] = rate_h
        out["rare_missingness_pvalue"] = excess_pvalue(int(rare.sum()), len(rare), rate_h, len(hp))
    return out, None
