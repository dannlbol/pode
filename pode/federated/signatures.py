"""Federated MetaCPG: prognostic CpG signatures from per-site statistics only.

MetaCPG (``metacpg.core``, ``methodology: legacy_exact``) ranks every CpG by a
univariable (optionally stratified / covariate-adjusted) Cox model, scores
patients by the mean of standardised-coefficient-weighted M-values of the top
CpGs, and picks the number of CpGs by leave-one-out (LOO) cross-validation.
Every ingredient decomposes over sites:

* **Cox fits.** For a pooled design whose strata nest within sites (e.g.
  ``cohort x age`` strata), the Efron score vector and information
  matrix are sums over strata, hence over sites. The coordinator runs MetaCPG's
  own batched Newton-Raphson, but each iteration's score/information is the sum
  of the sites' contributions at the broadcast coefficients
  (``site_score_info``, which calls ``metacpg.core._batched_score_info_general``
  on the site's rows). The result is the pooled fit, up to floating-point
  summation order. Strata that span sites, or an unstratified pooled risk set,
  do not decompose; they are refused with an explanation.
* **Meta-analysis** (``group_column`` designs): each site fits its own Cox
  models (``site_endpoint_stats``) and the coordinator pools them with
  MetaCPG's ``_random_effects``. This takes one round.
* **Standardised weights** need the pooled SD of each CpG: sites send
  (n, mean, M2) and the coordinator merges them (Chan et al.).
* **Score scaling** needs the median and IQR of training raw scores: an exact
  count-based protocol (:mod:`.quantiles`).
* **LOO.** Holding out patient *i* changes only their own site's contribution;
  every other site recomputes at the broadcast coefficients. ``exact_loo``
  therefore reproduces the centralised search and extraction.

Two evaluation modes:

* ``pooled``: held-out patients' LOO scores, times and statuses go to the
  coordinator, exactly as MetaCPG computes its metrics. This is per-patient
  disclosure, and the ledger says so.
* ``within_site``: sites keep their LOO scores and send only concordant/
  comparable pair counts and a per-site AUC. The C-index is then the
  site-stratified C.

``loso`` (leave one site out) is the deployment-realistic check. The signature
is built from k-1 sites and scored at the held-out site, which reports only
aggregates.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .ledger import DisclosureLedger
from .quantiles import federated_order_stats, median_iqr_from, median_iqr_ranks
from .sites import Site, stateful


# ---- site data -----------------------------------------------------------------------
@dataclass
class SiteOmics:
    """A site's methylation matrix (features x samples, M-values) and aligned phenotype."""

    matrix: pd.DataFrame
    pheno: pd.DataFrame

    def __len__(self) -> int:
        return int(self.matrix.shape[1])


def omics_sites(data, site_column: str, ledger: DisclosureLedger | None = None,
                include: list[str] | None = None) -> list[Site]:
    """Partition a validated MetaCPG ``InputData`` into sites by a phenotype column (simulation)."""
    if site_column not in data.phenotype.columns:
        raise ValueError(f"site column '{site_column}' is not in the phenotype table.")
    labels = data.phenotype[site_column].astype(str)
    levels = sorted(labels[data.phenotype[site_column].notna()].unique())
    if include:
        levels = [lv for lv in levels if lv in set(map(str, include))]
    sites = []
    for lv in levels:
        cols = data.phenotype.index[(labels == lv).to_numpy()]
        sites.append(Site(lv, SiteOmics(data.matrix.loc[:, cols], data.phenotype.loc[cols]), ledger))
    if len(sites) < 2:
        raise ValueError(f"Federated MetaCPG needs at least two sites; '{site_column}' gives {len(sites)}.")
    return sites


def resolve_design(data, site_column: str | None) -> tuple[str, str]:
    """('meta', group column) for group_column designs, else ('pooled', site column)."""
    groups = [ep.get("_group") for ep in data.endpoints]
    if any(g is not None for g in groups):
        column = groups[0][0]
        if site_column and site_column != column:
            raise ValueError(f"This config meta-analyses by group_column '{column}'; the sites must be its levels "
                             f"(got --site-column {site_column}).")
        return "meta", column
    if len(data.endpoints) != 1:
        raise ValueError("Federated MetaCPG supports one endpoint (or a group_column meta-analysis).")
    if not site_column:
        raise ValueError("A pooled design needs --site-column (the phenotype column naming each patient's site).")
    if not any(c.get("mode", "adjust") == "strata" for c in data.covariates):
        raise ValueError(
            "A pooled design without strata has one risk set spanning every site, which does not decompose "
            "over sites. Stratify by the site column (covariates: [{column: SITE, type: categorical, "
            "mode: strata}]) or meta-analyse with endpoints[0].group_column.")
    return "pooled", site_column


# ---- site operations -----------------------------------------------------------------
def _design_with_levels(pheno: pd.DataFrame, covariates: list[dict], levels: dict | None):
    """``metacpg.core._design`` with fixed (federation-wide) categorical levels."""
    columns, strata_cols = [], []
    for cov in covariates:
        series = pheno[cov["column"]]
        if cov.get("mode", "adjust") == "strata":
            strata_cols.append(series.astype(object))
        elif cov["type"] == "numeric":
            columns.append(pd.to_numeric(series).to_numpy(float)[:, None])
        else:
            cats = (levels or {}).get(cov["column"])
            cat = pd.Categorical(series, categories=cats) if cats is not None else series.astype("category")
            dummy = pd.get_dummies(cat, drop_first=True, dtype=float)
            if dummy.shape[1]:
                columns.append(dummy.to_numpy())
    z = np.column_stack(columns) if columns else np.empty((len(pheno), 0))
    strata = None
    if strata_cols:
        frame = pd.concat(strata_cols, axis=1).fillna("__MISSING__").astype(str)
        strata = frame.agg("|".join, axis=1).to_numpy()
    return z, strata


def site_levels(omics: SiteOmics, covariates: list[dict]) -> dict[str, list]:
    """Categorical adjustment levels present at the site (category names only)."""
    out = {}
    for cov in covariates:
        if cov.get("mode", "adjust") == "adjust" and cov["type"] != "numeric":
            out[cov["column"]] = sorted(omics.pheno[cov["column"]].dropna().astype("category").cat.categories)
    return out


@stateful
def site_prepare(omics: SiteOmics, local: dict, endpoint: dict, covariates: list[dict], levels: dict | None,
                 key: str = "sig") -> dict:
    """Select the endpoint-eligible rows and build the site's design; returns its count and strata labels."""
    from metacpg.core import InputData, _valid_endpoint_rows

    data = InputData(omics.matrix, omics.pheno, [endpoint], covariates)
    rows = _valid_endpoint_rows(data, endpoint)
    pheno = omics.pheno.iloc[rows]
    z, strata = _design_with_levels(pheno, covariates, levels)
    local[key] = {
        "rows": rows,
        "matrix": omics.matrix.iloc[:, rows],
        "X": np.ascontiguousarray(omics.matrix.to_numpy(float)[:, rows].T),
        "time": pd.to_numeric(pheno[endpoint["time"]], errors="coerce").to_numpy(float),
        "status": pd.to_numeric(pheno[endpoint["status"]], errors="coerce").to_numpy(float),
        "z": z, "strata": strata, "pheno": pheno, "endpoint": endpoint, "covariates": covariates,
        "ids": list(omics.matrix.columns[rows]),
    }
    local.pop("cox_cache", None)
    return {"n": int(len(rows)), "n_cov": int(z.shape[1]),
            "strata": None if strata is None else sorted(set(strata))}


def _keep(n: int, exclude: int | None) -> np.ndarray:
    idx = np.arange(n)
    return idx if exclude is None else np.delete(idx, exclude)


@stateful
def site_score_info(omics, local: dict, beta: np.ndarray, exclude: int | None, backend_kind: str,
                    key: str = "sig") -> tuple[np.ndarray, np.ndarray]:
    """Efron score U [F, p] and information I [F, p, p] of the site's rows at ``beta``."""
    from metacpg.core import _batched_score_info_general, _CoxBackend, _cox_strata_orders

    sig = local[key]
    cache = local.setdefault("cox_cache", {})
    ck = (key, exclude, backend_kind)
    if ck not in cache:
        cache.clear()
        keep = _keep(len(sig["rows"]), exclude)
        x, z = sig["X"][keep], sig["z"][keep]
        t, s = sig["time"][keep], sig["status"][keep]
        strata = None if sig["strata"] is None else sig["strata"][keep]
        backend = _CoxBackend(backend_kind)
        orders = _cox_strata_orders(t, s, strata)
        so = [(backend.idx(o), [(end, backend.idx(ev), d) for end, ev, d in steps]) for o, steps in orders]
        n_cov = z.shape[1]
        obs = np.zeros((x.shape[1], n_cov + 1))
        events = np.flatnonzero(s == 1)
        obs[:, 0] = x[events].sum(axis=0)
        if n_cov:
            obs[:, 1:] = z[events].sum(axis=0)[None, :]
        cache[ck] = (backend, backend.to(x), backend.to(z if n_cov else np.zeros((len(keep), 0))),
                     backend.to(obs), so, n_cov)
    backend, x, z, obs, so, n_cov = cache[ck]
    u, info = _batched_score_info_general(x, z, backend.to(beta), obs, n_cov, so, backend)
    return backend.get(u), backend.get(info)


@stateful
def site_moments(omics, local: dict, exclude: int | None, key: str = "sig") -> tuple[int, np.ndarray, np.ndarray]:
    """(n, mean, M2) per feature over the site's eligible rows."""
    sig = local[key]
    x = sig["X"][_keep(len(sig["rows"]), exclude)]
    mean = x.mean(axis=0)
    return int(len(x)), mean, ((x - mean) ** 2).sum(axis=0)


@stateful
def site_endpoint_stats(omics, local: dict, exclude: int | None, engine: str, key: str = "sig") -> pd.DataFrame:
    """MetaCPG's per-feature Cox statistics on the site's own rows (meta-analysis designs)."""
    from metacpg.core import _endpoint_stats, _make_backend

    sig = local[key]
    keep = _keep(len(sig["rows"]), exclude)
    return _endpoint_stats(sig["matrix"].iloc[:, keep], sig["pheno"].iloc[keep], sig["endpoint"],
                           sig["covariates"], None, None, engine, _make_backend(engine))


@stateful
def site_raw_scores(omics, local: dict, features: list[list[str]], weights: list[np.ndarray],
                    exclude: int | None, store: str, key: str = "sig") -> int:
    """Raw scores (mean weighted M-value) of the site's training rows for each target, kept at the site."""
    sig = local[key]
    m = sig["matrix"].iloc[:, _keep(len(sig["rows"]), exclude)]
    local[store] = np.stack([np.mean(m.loc[f].to_numpy(float) * w[:, None], axis=0) if m.shape[1] else
                             np.zeros(0) for f, w in zip(features, weights)]) if features else np.zeros((0, 0))
    return int(m.shape[1])


@stateful
def site_heldout_scores(omics, local: dict, index: int, features: list[list[str]], weights: list[np.ndarray],
                        centers: np.ndarray, scales: np.ndarray, keep_local: bool, counts: list[int],
                        key: str = "sig") -> dict | None:
    """LOO score(s) of held-out patient ``index``. Returned (pooled metrics) or kept at the site."""
    sig = local[key]
    col = sig["matrix"].iloc[:, index]
    raw = np.array([float(np.mean(col.loc[f].to_numpy(float) * w)) for f, w in zip(features, weights)])
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(scales != 0, (raw - centers) / scales, np.nan)
    rec = {"sample_id": sig["ids"][index], "score": score, "center": centers, "scale": scales,
           "time": float(sig["time"][index]), "status": float(sig["status"][index]),
           "z": sig["z"][index], "stratum": None if sig["strata"] is None else sig["strata"][index],
           "local_index": int(index), "counts": list(counts)}
    if keep_local:
        local.setdefault("loo_scores", []).append(rec)
        return None
    return rec


@stateful
def site_within_metrics(omics, local: dict, counts: list[int], horizon: float, source: str = "loo_scores",
                        records: list | None = None) -> pd.DataFrame:
    """Per count: concordant/comparable pairs and binary AUC of the site's own held-out scores."""
    from metacpg.core import _binary_auc_at_time

    recs = records if records is not None else local.get(source, [])
    if not recs:
        return pd.DataFrame(columns=["feature_count", "concordant", "comparable", "auc", "n", "events"])
    t = np.array([r["time"] for r in recs])
    s = np.array([r["status"] for r in recs])
    scores = np.stack([r["score"] for r in recs])  # [n, counts the LOO ran]
    rows = []
    for n in counts:
        j = recs[0]["counts"].index(n)
        conc, comp = _pair_counts(t, s, scores[:, j])
        rows.append({"feature_count": n, "concordant": conc, "comparable": comp,
                     "auc": _binary_auc_at_time(t, s, scores[:, j], horizon), "n": len(t), "events": int(s.sum())})
    return pd.DataFrame(rows)


@stateful
def site_score_all(omics, local: dict, features: list[str], weights: np.ndarray, center: float, scale: float,
                   counts_only: bool = False, horizon: float | None = None, key: str = "sig"):
    """Score every phenotype-matched patient at the site with a frozen model.

    ``counts_only``: return only aggregate validation metrics (C pair counts, AUC,
    the score's Cox HR) on the eligible rows; else the per-patient risk scores
    (the site's own output).
    """
    from metacpg.core import _binary_auc_at_time, _cox_fit

    raw = np.mean(omics.matrix.loc[features].to_numpy(float) * np.asarray(weights)[:, None], axis=0)
    scaled = (raw - center) / scale if scale else np.full_like(raw, np.nan)
    if not counts_only:
        return pd.DataFrame({"sample_id": omics.matrix.columns, "risk_score": scaled, "raw_score": raw})
    sig = local[key]
    sc = scaled[sig["rows"]]
    t, s = sig["time"], sig["status"]
    conc, comp = _pair_counts(t, s, sc)
    try:
        beta, se = _cox_fit(t, s, sc, sig["strata"])
        hr, p = float(np.exp(beta[0])), float(2 * norm.sf(abs(beta[0] / se[0])))
    except Exception:  # pragma: no cover - degenerate site
        hr, p = np.nan, np.nan
    return {"n": int(len(t)), "events": int(s.sum()), "concordant": conc, "comparable": comp,
            "c_index": conc / comp if comp else np.nan,
            "auc": _binary_auc_at_time(t, s, sc, horizon) if horizon else np.nan, "hr_per_unit_score": hr, "hr_p": p}


def _pair_counts(t: np.ndarray, s: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Harrell's concordant (ties 0.5) and comparable pair counts (higher score = earlier event)."""
    ok = np.isfinite(t) & np.isfinite(s) & np.isfinite(score)
    t, s, score = t[ok], s[ok], score[ok]
    early = (t[:, None] < t[None, :]) & (s[:, None] == 1)
    conc = (score[:, None] > score[None, :]) + 0.5 * (score[:, None] == score[None, :])
    return float((conc * early).sum()), float(early.sum())


# ---- coordinator ---------------------------------------------------------------------------
class _SD:
    """Duck-typed stand-in for the pooled matrix in ``metacpg.core._random_effects`` (only ``.std`` is used)."""

    def __init__(self, sd: pd.Series):
        self.sd = sd

    def std(self, axis=1, ddof=1):
        return self.sd


def merge_moments(parts: list[tuple[int, np.ndarray, np.ndarray]]) -> tuple[int, np.ndarray]:
    """Chan's parallel merge of per-site (n, mean, M2) -> (N, sample SD)."""
    parts = [p for p in parts if p[0] > 0]
    n = sum(p[0] for p in parts)
    mean = sum(p[0] * p[1] for p in parts) / n
    m2 = sum(p[2] + p[0] * (p[1] - mean) ** 2 for p in parts)
    with np.errstate(invalid="ignore", divide="ignore"):
        return n, np.sqrt(m2 / (n - 1))


@dataclass
class FederatedSignature:
    """Coordinator state for one federated MetaCPG analysis."""

    sites: list[Site]
    design: str
    features: list[str]
    endpoint_name: str
    covariates: list[dict]
    engine: str = "batched"
    backend_kind: str = "numpy"
    log: Callable[[str], None] | None = None

    # ---- statistics ----------------------------------------------------------------------
    def prepare(self, endpoints_by_site: dict[str, dict], covariates: list[dict]) -> dict[str, int]:
        levels = None
        if self.design == "pooled":
            per = [s.run(site_levels, covariates, label="covariate_levels") for s in self.sites]
            levels = {}
            for d in per:
                for c, lv in d.items():
                    levels[c] = sorted(set(levels.get(c, [])) | set(lv))
        info = {s.name: s.run(site_prepare, endpoints_by_site[s.name], covariates, levels, label="site_prepare")
                for s in self.sites}
        if self.design == "pooled":
            seen: dict[str, str] = {}
            for name, d in info.items():
                for label in d["strata"] or []:
                    if label in seen and seen[label] != name:
                        raise ValueError(
                            f"Stratum '{label}' occurs at sites '{seen[label]}' and '{name}': its risk set spans "
                            "sites and does not decompose. Make the site column one of the strata.")
                    seen[label] = name
        self.n = {k: v["n"] for k, v in info.items()}
        self.n_cov = max(v["n_cov"] for v in info.values())
        return self.n

    def stats(self, exclude: dict[str, int] | None = None, sites: list[Site] | None = None) -> pd.DataFrame:
        """Per-feature Cox statistics as MetaCPG would compute them on the pooled rows."""
        sites = sites or self.sites
        exclude = exclude or {}
        moments = [s.run(site_moments, exclude.get(s.name), label="feature_moments") for s in sites]
        _, sd = merge_moments(moments)
        if self.design == "meta":
            from metacpg.core import _random_effects

            stats = [self._site_stats(s, exclude.get(s.name)) for s in sites]
            if len(stats) == 1:
                return stats[0]
            return _random_effects(stats, _SD(pd.Series(sd, index=self.features)))
        beta, se = self._newton(sites, exclude)
        with np.errstate(divide="ignore", invalid="ignore"):
            standardized = beta / sd
            p = 2 * norm.sf(np.abs(beta / se))
            out = pd.DataFrame({"feature": self.features, "coef": beta, "se": se, "ordinary_z": beta / se,
                                "standardized_coef": standardized, "standardized_z": standardized / se,
                                "p_value": p})
        out["endpoint"] = self.endpoint_name
        return out

    def _site_stats(self, site: Site, exclude: int | None) -> pd.DataFrame:
        cache = getattr(self, "_stats_cache", {})
        self._stats_cache = cache
        key = (site.name, exclude)
        if exclude is None and key in cache:
            return cache[key]
        st = site.run(site_endpoint_stats, exclude, self.engine, label="site_cox_stats")
        if exclude is None:
            cache[key] = st
        return st

    def _newton(self, sites: list[Site], exclude: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
        """``metacpg.core._batched_cox_fit_general`` with the score/information summed over sites."""
        from metacpg.core import _BATCH_MAX_ITER, _BATCH_TOL, _active_features

        beta = np.zeros((len(self.features), 1 + self.n_cov))

        def total(b):
            parts = [s.run(site_score_info, b, exclude.get(s.name), self.backend_kind, label="cox_score_info")
                     for s in sites]
            return sum(x[0] for x in parts), sum(x[1] for x in parts)

        active = np.ones(len(self.features), dtype=bool)
        for _ in range(_BATCH_MAX_ITER):
            u, info = total(beta)
            active = _active_features(info[:, 0, 0])
            step = np.zeros_like(beta)
            if active.any():
                step[active] = np.einsum("aij,aj->ai", np.linalg.pinv(info[active]), u[active])
            beta = beta + step
            moved = np.abs(step[active, 0]) if active.any() else np.zeros(0)
            if not moved.size or float(moved.max()) < _BATCH_TOL:
                break
        _, info = total(beta)
        active = _active_features(info[:, 0, 0])
        b, se = np.zeros(len(self.features)), np.zeros(len(self.features))
        if active.any():
            pinv = np.linalg.pinv(info[active])
            b[active] = beta[active, 0]
            se[active] = np.sqrt(np.clip(pinv[:, 0, 0], 0, np.inf))
        return b, se

    # ---- scoring ------------------------------------------------------------------------------
    def scaling(self, ranking: pd.DataFrame, counts: list[int], exclude: dict[str, int] | None = None,
                sites: list[Site] | None = None) -> tuple[list, list, np.ndarray, np.ndarray]:
        """Top-n features/weights per count and the federated median/IQR of training raw scores."""
        sites = sites or self.sites
        exclude = exclude or {}
        top = ranking.dropna(subset=["standardized_coef"])
        feats = [top.head(n)["feature"].tolist() for n in counts]
        weights = [top.head(n)["standardized_coef"].to_numpy(float) for n in counts]
        ns = [s.run(site_raw_scores, feats, weights, exclude.get(s.name), "raw", label="raw_scores_kept_local")
              for s in sites]
        ranks, recipe = median_iqr_ranks(int(sum(ns)))
        values = federated_order_stats(sites, "raw", np.repeat(ranks[None, :], len(counts), axis=0))
        centers, scales = zip(*(median_iqr_from(values[i], recipe) for i in range(len(counts))))
        return feats, weights, np.array(centers), np.array(scales)


# ---- drivers ---------------------------------------------------------------------------------
def _counts(cfg: dict) -> list[int]:
    s = cfg.get("search", {})
    return list(range(int(s.get("min_features", 5)), int(s.get("max_features", 60)) + 1, int(s.get("step", 1))))


def build_federation(cfg: dict, site_column: str | None, ledger: DisclosureLedger, engine: str | None = None,
                     log=None):
    """Validate a MetaCPG config, partition its samples into sites, prepare every site."""
    from metacpg.core import _resolve_backend_kind, _resolve_engine, validate_inputs

    data = validate_inputs(cfg)
    design, column = resolve_design(data, site_column)
    sites = omics_sites(data, column, ledger)
    eng = engine or _resolve_engine(cfg)
    batched = "batched" if eng == "exact" else eng
    fed = FederatedSignature(sites, design, list(data.matrix.index), data.endpoints[0]["name"],
                             data.covariates, engine=eng, backend_kind=_resolve_backend_kind(batched), log=log)
    if design == "meta":
        by_site = {ep["_group"][1]: ep for ep in data.endpoints}
    else:
        by_site = {s.name: data.endpoints[0] for s in sites}
    fed.prepare(by_site, data.covariates)
    fed.endpoints = data.endpoints
    fed.endpoint_index = {s.name: (i if design == "meta" else 0) for i, s in enumerate(sites)}
    fed.horizon = min(float(ep.get("prediction_time", 1095)) for ep in data.endpoints)
    return fed, data


def federated_loo(fed: FederatedSignature, counts: list[int], metrics: str = "pooled",
                  phase: str = "search") -> dict:
    """Exact federated LOO over every eligible patient; returns MetaCPG-shaped records."""
    records, t0 = [], _time.perf_counter()
    keep_local = metrics == "within_site"
    total = sum(fed.n.values())
    done = 0
    for s in fed.sites:
        for i in range(fed.n[s.name]):
            exclude = {s.name: i}
            ranking = _rank(fed.stats(exclude))
            feats, weights, centers, scales = fed.scaling(ranking, counts, exclude)
            rec = s.run(site_heldout_scores, i, feats, weights, centers, scales, keep_local, counts,
                        per_patient=not keep_local, label="loo_heldout_score")
            if rec is not None:
                for j, n in enumerate(counts):
                    row = {"sample_id": rec["sample_id"], "endpoint": fed.endpoints[fed.endpoint_index[s.name]]["name"],
                           "endpoint_index": fed.endpoint_index[s.name], "heldout_index": done,
                           "feature_count": n, "risk_score": rec["score"][j], "fold_center": centers[j],
                           "fold_scale": scales[j], "time": rec["time"], "status": rec["status"],
                           "site": s.name, "_z": rec["z"], "_stratum": rec["stratum"]}
                    records.append((row, [{"feature_count": n, "feature": f, "endpoint": row["endpoint"]}
                                          for f in feats[j]]))
            else:
                for j, n in enumerate(counts):
                    records.append(({"site": s.name, "heldout_index": done, "feature_count": n,
                                     "endpoint_index": fed.endpoint_index[s.name]},
                                    [{"feature_count": n, "feature": f} for f in feats[j]]))
            done += 1
            if fed.log and (done == 1 or done % 25 == 0 or done == total):
                fed.log(f"  {phase} LOO {done}/{total} ({_time.perf_counter() - t0:.0f}s)")
    return _assemble(records, fed, total)


def _rank(stats: pd.DataFrame) -> pd.DataFrame:
    from metacpg.core import _rank as rank

    return rank(stats)


def _assemble(records, fed: FederatedSignature, denominator: int) -> dict:
    """``metacpg.core._assemble_loo`` with the denominator from site counts."""
    scores = pd.DataFrame([r for r, _ in records])
    scores = scores.sort_values(["feature_count", "endpoint_index", "heldout_index"]).reset_index(drop=True)
    sel = [x for _, s in records for x in s]
    freq = (pd.DataFrame(sel).groupby(["feature_count", "feature"], sort=True).size()
            .rename("fold_count").reset_index())
    totals = scores.groupby("feature_count").size().rename("total_folds")
    freq = freq.join(totals, on="feature_count")
    freq["stability_denominator"] = denominator
    freq["inclusion_percent"] = 100 * freq["fold_count"] / freq["stability_denominator"]
    freq = freq.sort_values(["feature_count", "inclusion_percent", "feature"], ascending=[True, False, True])
    return {"scores": scores, "inclusion_frequency": freq}


def loo_metrics(fed: FederatedSignature, loo: dict, counts: list[int], metrics: str) -> pd.DataFrame:
    """Search metrics per feature count, as MetaCPG computes them (pooled) or site-stratified."""
    from metacpg.core import _legacy_performance, _legacy_performance_adjusted, _neglog10

    rows = []
    if metrics == "within_site":
        per = pd.concat([s.run(site_within_metrics, counts, fed.horizon, label="loo_metrics_aggregates")
                         .assign(site=s.name) for s in fed.sites], ignore_index=True)
        for n, g in per.groupby("feature_count"):
            rows.append({"feature_count": int(n), "c_index": g["concordant"].sum() / g["comparable"].sum(),
                         "auc": float(np.average(g["auc"].fillna(0.5), weights=g["n"])),
                         "neglog10_p": np.nan, "metric_mode": "within_site"})
        return pd.DataFrame(rows)
    scores = loo["scores"]
    for n, g in scores.groupby("feature_count", sort=True):
        t = pd.to_numeric(g["time"], errors="coerce").to_numpy(float)
        st = pd.to_numeric(g["status"], errors="coerce").to_numpy(float)
        sc = g["risk_score"].to_numpy(float)
        auc, c, p = _legacy_performance(t, st, sc, fed.horizon)
        row = {"feature_count": int(n), "auc": auc, "c_index": c, "neglog10_p": _neglog10(p)}
        if fed.covariates:
            z = np.stack(g["_z"].to_numpy()) if len(g) else np.empty((0, 0))
            strata = g["_stratum"].to_numpy() if g["_stratum"].notna().any() else None
            a2, c2, p2 = _legacy_performance_adjusted(t, st, sc, z, strata, fed.horizon)
            row.update({"auc_adjusted": a2, "c_index_adjusted": c2, "neglog10_p_adjusted": _neglog10(p2)})
        rows.append(row)
    return pd.DataFrame(rows)


def federated_metacpg(cfg: dict, site_column: str | None, ledger: DisclosureLedger | None = None,
                      metrics: str = "pooled", feature_count: int | None = None, extraction_only: bool = False,
                      engine: str | None = None, log=None) -> dict:
    """Federated equivalent of ``metacpg.core.fit_metacpg`` (exact LOO). Returns MetaCPG's result dict."""
    from metacpg.core import _validate_weight_refit, select_feature_count

    t0 = _time.perf_counter()
    ledger = ledger or DisclosureLedger()
    fed, data = build_federation(cfg, site_column, ledger, engine, log)
    counts = _counts(cfg)
    configured = cfg.get("selection", {}).get("feature_count", cfg.get("search", {}).get("feature_count_override"))
    mode = cfg.get("selection", {}).get("mode", "manual")
    result: dict[str, Any] = {}
    if extraction_only:
        selected = feature_count if feature_count is not None else configured
        if selected is None:
            raise ValueError("Extraction-only needs a feature count (--feature-count or selection.feature_count).")
        selected = int(selected)
        loo = federated_loo(fed, [selected], metrics, "extraction")
    else:
        full = fed.stats()
        result["ranking"] = _rank(full)
        result["endpoint_stats"] = full
        loo = federated_loo(fed, counts, metrics, "search")
        result["scores"] = _public_scores(loo["scores"])
        result["inclusion_frequency"] = loo["inclusion_frequency"]
        result["metrics"] = loo_metrics(fed, loo, counts, metrics)
        selected = select_feature_count(result["metrics"], feature_count if feature_count is not None else configured,
                                        mode)
    # Extraction: the LOO ranking per fold is count-independent, so the fresh extraction LOO
    # MetaCPG runs at the selected count equals the search LOO restricted to that count.
    freq = loo["inclusion_frequency"]
    freq = freq[freq["feature_count"] == selected]
    chosen = freq.head(selected)["feature"].tolist()
    weight_refit = _validate_weight_refit(cfg)
    final = _final_stats(fed, weight_refit)
    chosen_stats = final.set_index("feature").loc[chosen].reset_index()
    weights = chosen_stats["standardized_coef"].to_numpy(float)
    feats_l, w_l, centers, scales = fed.scaling(final.set_index("feature").loc[chosen].reset_index(), [selected])
    center, scale = float(centers[0]), float(scales[0])
    risk = pd.concat([s.run(site_score_all, chosen, weights, center, scale, per_patient=True,
                            label="final_risk_scores_site_output") for s in fed.sites], ignore_index=True)
    selected_features = chosen_stats.copy()
    selected_features.insert(1, "rank", np.arange(1, len(selected_features) + 1))
    ext_scores = loo["scores"][loo["scores"]["feature_count"] == selected].reset_index(drop=True)
    perf = {"feature_count": selected}
    if metrics == "pooled":
        m = loo_metrics(fed, {"scores": ext_scores}, [selected], metrics).iloc[0].to_dict()
    else:
        m = loo_metrics(fed, loo, [selected], metrics).iloc[0].to_dict()
    perf.update({k: v for k, v in m.items() if k not in ("feature_count", "metric_mode")})
    result.update({"selected_features": selected_features, "risk_scores": risk,
                   "scaling": {"center": center, "scale": scale}, "performance": perf,
                   "weight_refit": weight_refit, "extraction_scores": _public_scores(ext_scores),
                   "extraction_inclusion_frequency": freq, "selected_count": selected,
                   "site_counts": dict(fed.n), "design": fed.design, "metric_mode": metrics,
                   "elapsed_seconds": _time.perf_counter() - t0})
    return result


def _public_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return scores.drop(columns=[c for c in ("_z", "_stratum") if c in scores.columns])


def _final_stats(fed: FederatedSignature, weight_refit: str) -> pd.DataFrame:
    """Full-data statistics for the weight refit (``covariate_aware`` or ``legacy_unadjusted``)."""
    if weight_refit == "covariate_aware" or not fed.covariates:
        return fed.stats()
    if fed.design == "pooled":
        raise ValueError("final_model.weight_refit: legacy_unadjusted drops the strata, leaving one pooled risk set "
                         "that does not decompose over sites; use covariate_aware for a federated pooled design.")
    # Meta designs: refit each site without covariates.
    for s in fed.sites:
        sig = s.local["sig"]
        s.local["sig_unadj"] = {**sig, "covariates": []}
    fed2 = FederatedSignature(fed.sites, fed.design, fed.features, fed.endpoint_name, [], fed.engine,
                              fed.backend_kind, fed.log)
    fed2.n = fed.n
    moments = [s.run(site_moments, None, label="feature_moments") for s in fed.sites]
    _, sd = merge_moments(moments)
    from metacpg.core import _random_effects

    stats = [s.run(site_endpoint_stats, None, fed.engine, "sig_unadj", label="site_cox_stats") for s in fed.sites]
    return _random_effects(stats, _SD(pd.Series(sd, index=fed.features)))


# ---- leave one site out -------------------------------------------------------------------------
def federated_loso(cfg: dict, site_column: str | None, ledger: DisclosureLedger | None = None,
                   counts: list[int] | None = None, engine: str | None = None, log=None) -> dict:
    """Build the signature on k-1 sites, score the held-out site; aggregates only from the held-out site."""
    ledger = ledger or DisclosureLedger()
    fed, data = build_federation(cfg, site_column, ledger, engine, log)
    counts = counts or _counts(cfg)
    rows, tops = [], {}
    for h in fed.sites:
        train = [s for s in fed.sites if s is not h]
        ranking = _rank(fed.stats(sites=train))
        feats, weights, centers, scales = fed.scaling(ranking, counts, sites=train)
        tops[h.name] = feats
        for j, n in enumerate(counts):
            m = h.run(site_score_all, feats[j], weights[j], float(centers[j]), float(scales[j]), True, fed.horizon,
                      label="loso_validation_aggregates")
            rows.append({"heldout_site": h.name, "feature_count": n, "trained_on": ";".join(s.name for s in train),
                         **m})
        if log:
            log(f"  LOSO: held out {h.name} ({fed.n[h.name]} patients)")
    per = pd.DataFrame(rows)
    pooled = (per.groupby("feature_count")
              .apply(lambda g: pd.Series({"c_index_stratified": g["concordant"].sum() / g["comparable"].sum(),
                                          "auc_mean": float(np.average(g["auc"].fillna(0.5), weights=g["n"])),
                                          "c_index_worst_site": float(g["c_index"].min())}), include_groups=False)
              .reset_index())
    stability = []
    names = list(tops)
    for j, n in enumerate(counts):
        sets = [set(tops[k][j]) for k in names]
        jac = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
        stability.append({"feature_count": n, "mean_jaccard": float(np.mean(jac)) if jac else np.nan})
    pooled = pooled.merge(pd.DataFrame(stability), on="feature_count")
    return {"per_site": per, "curve": pooled, "top_features": tops, "fed": fed}


def site_local_signature_transfer(fed: FederatedSignature, counts: list[int]) -> pd.DataFrame:
    """Each site alone: its own top-n signature (own ranking, own scaling) scored at every other site."""
    rows = []
    for src in fed.sites:
        ranking = _rank(fed.stats(sites=[src]))
        feats, weights, centers, scales = fed.scaling(ranking, counts, sites=[src])
        for dst in fed.sites:
            if dst is src:
                continue
            for j, n in enumerate(counts):
                m = dst.run(site_score_all, feats[j], weights[j], float(centers[j]), float(scales[j]), True,
                            fed.horizon, label="local_signature_transfer_aggregates")
                rows.append({"source_site": src.name, "heldout_site": dst.name, "feature_count": n, **m})
    return pd.DataFrame(rows)


def loso_final(fed: FederatedSignature, n: int, curve: pd.DataFrame) -> dict:
    """The deployable model after a LOSO-chosen count: top-``n`` of the all-site ranking."""
    t0 = _time.perf_counter()
    full = fed.stats()
    ranking = _rank(full)
    feats, weights, centers, scales = fed.scaling(ranking, [n])
    top = ranking.dropna(subset=["standardized_coef"]).head(n).reset_index(drop=True)
    top = top.drop(columns=["rank", "abs_z"], errors="ignore")
    top.insert(1, "rank", np.arange(1, len(top) + 1))
    risk = pd.concat([s.run(site_score_all, feats[0], weights[0], float(centers[0]), float(scales[0]),
                            per_patient=True, label="final_risk_scores_site_output") for s in fed.sites],
                     ignore_index=True)
    row = curve[curve["feature_count"] == n].iloc[0]
    return {"ranking": ranking, "endpoint_stats": full, "selected_features": top, "risk_scores": risk,
            "scaling": {"center": float(centers[0]), "scale": float(scales[0])},
            "performance": {"feature_count": n, "c_index_loso_stratified": float(row["c_index_stratified"]),
                            "auc_loso_mean": float(row["auc_mean"]),
                            "c_index_loso_worst_site": float(row["c_index_worst_site"])},
            "weight_refit": "covariate_aware", "selected_count": n, "site_counts": dict(fed.n),
            "design": fed.design, "metric_mode": "loso", "elapsed_seconds": _time.perf_counter() - t0}
