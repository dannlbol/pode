"""``pode federated benchmark``: is a federated TVAE useful, and is it safe?

Design (per repeat):

1. Every site's patients are split into **train** and **test** (stratified on
   the event). Every generator sees site *train* rows only.
2. Each condition (see :data:`~pode.federated.config.CONDITIONS`) produces
   synthetic data:
   * ``local`` - each site's own TVAE (the synthetic rows stay at that site);
   * ``exchange`` - local TVAEs whose releases passed the privacy gate, pooled;
   * ``fedavg`` / ``fedprox`` / ``fed_personalised`` / ``fed_dp_eps<E>`` -
     federated TVAEs (personalised: each site samples its own fine-tuned copy);
   * ``central_*`` - generators fitted on pooled train rows (the "if pooling
     were permitted" baselines, incl. PODE's SDV TVAE and a Gaussian copula).
3. At every evaluation site, using **only that site's real test patients**:
   * utility: train a prognostic model on the condition's synthetic rows
     (TSTR) and score it; references are the same model trained on the site's
     own real rows (``real_local``), on pooled real rows (``real_pooled``) and
     a federated site-stratified Cox on real rows (``real_federated_cox``);
   * fidelity of the synthetic rows against the site's real training rows;
   * privacy: the :mod:`pode.privacy` suite, with the site's training rows as
     members and its test rows as non-members.
4. A leave-one-site-out arm (``sites.loso``) holds a site out of everything
   and evaluates on all of its patients: does the generator transport?

Paired differences in C-index (condition - reference, same test patients)
are summarised over sites and repeats with the corrected repeated-split CI.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..config import SynthesizerConfig
from ..imputation.base import limit_cpu, resolve_columns
from .config import FederatedConfig, dp_epsilon
from .evaluate import (Endpoint, LinearCox, federated_cox, fidelity_at_site, fit_model, privacy_at_site,
                       utility_at_site)
from .ledger import DisclosureLedger
from .sites import Site

REFERENCES = ("real_local", "real_pooled", "real_federated_cox")
GLOBAL = ("fedavg", "fedprox", "central_fed_tvae", "central_sdv_tvae", "central_copula")


def resolve_endpoint(data: pd.DataFrame, schema, cfg: FederatedConfig, columns: list[str]) -> Endpoint:
    ev = cfg.evaluation
    time_col, event_col = ev.time, ev.event
    if time_col is None or event_col is None:
        pair = schema.primary_survival_pair()
        if pair is None:
            raise ValueError("No survival endpoint: set evaluation.time and evaluation.event.")
        time_col, event_col = time_col or pair.time_col, event_col or pair.event_col
    feats = resolve_columns(ev.features, columns) if ev.features else [c for c in columns
                                                                        if c not in (time_col, event_col)]
    feats = [c for c in feats if c not in (time_col, event_col)]
    numeric = [c for c in feats if schema.types[c].is_numeric]
    categorical = [c for c in feats if c not in numeric]
    horizon = ev.horizon or float(pd.to_numeric(data[time_col], errors="coerce").median())
    return Endpoint(time_col, event_col, horizon, numeric, categorical)


def split_sites(data: pd.DataFrame, site_col: str, ep: Endpoint, frac: float, seed: int) -> dict:
    """Per site: (train positions, test positions) into that site's rows, stratified on the event."""
    from sklearn.model_selection import train_test_split

    out = {}
    for name, idx in data.groupby(data[site_col].astype(str)).indices.items():
        ev = data.iloc[idx][ep.event]
        strat = ev.astype(str) if ev.notna().all() and ev.value_counts().min() >= 2 else None
        tr, te = train_test_split(np.arange(len(idx)), test_size=frac, random_state=seed, stratify=strat)
        out[name] = (np.sort(tr), np.sort(te))
    return out


class Benchmark:
    def __init__(self, data: pd.DataFrame, schema, columns: list[str], synth: SynthesizerConfig,
                 cfg: FederatedConfig, pcfg, censor_threshold=None, log: Callable[[str], None] = print):
        self.data, self.schema, self.columns = data.reset_index(drop=True), schema, columns
        self.synth, self.cfg, self.pcfg = synth, cfg, pcfg
        self.censor = censor_threshold
        self.log = log
        self.site_col = cfg.sites.column
        self.ep = resolve_endpoint(self.data, schema, cfg, columns)
        from ..augmentation.config import ModelSpec

        self.models = [ModelSpec(**m) for m in cfg.evaluation.models]
        names = sorted(self.data[self.site_col].astype(str).unique())
        include = [str(x) for x in cfg.sites.include] if cfg.sites.include else names
        self.site_names = [s for s in names if s in include]
        self.frames = {s: self.data[self.data[self.site_col].astype(str) == s].drop(columns=[self.site_col])
                       .reset_index(drop=True) for s in self.site_names}

    # ---- generators -------------------------------------------------------------------
    def _fed_cfg(self, cond: str) -> FederatedConfig:
        c = self.cfg.copy()
        g = c.generator
        g.algorithm = "fedprox" if cond == "fedprox" else "fedavg"
        if cond == "fedprox" and g.prox_mu <= 0:
            g.prox_mu = 0.01
        g.personalise_epochs = (max(g.personalise_epochs, g.local_epochs * 4) if cond == "fed_personalised" else 0)
        g.site_in_model = False
        eps = dp_epsilon(cond)
        c.dp.enabled = eps is not None
        if eps is not None:
            c.dp.target_epsilon, c.dp.noise_multiplier = eps, None
        return c

    def _synthetic(self, cond: str, trains: dict[str, pd.DataFrame], tests: dict[str, pd.DataFrame], seed: int,
                   splits: dict | None) -> dict:
        """Synthetic table per evaluation site, plus generator metadata."""
        from .exchange import local_exchange
        from .generators import fit_generator
        from .tvae import FederatedTVAE

        from ..imputation.base import subset_schema

        cols = self.columns
        n_total = sum(len(t) for t in trains.values())
        n_syn = self.cfg.evaluation.n_synthetic
        t0 = time.perf_counter()
        meta: dict = {"privacy": {}, "bytes": 0}
        if cond in ("local", "exchange"):
            cache = getattr(self, "_exchange_cache", None)
            if cache is None or cache[0] != (seed, tuple(trains)):
                ledger = DisclosureLedger()
                sites = [Site(s, pd.concat([trains[s], tests[s]], ignore_index=True), ledger) for s in trains]
                spl = {s: (np.arange(len(trains[s])), np.arange(len(trains[s]), len(trains[s]) + len(tests[s])))
                       for s in trains}
                fc = self.cfg.copy()
                fc.seed = seed
                fc.dp.enabled = False
                fc.exchange.num_samples = n_syn
                res = local_exchange(sites, cols, self.schema, self.synth, fc, copy.deepcopy(self.pcfg), self.censor,
                                     spl, log=None, return_candidates=True)
                self._exchange_cache = ((seed, tuple(trains)), res, time.perf_counter() - t0)
            _, res, secs = self._exchange_cache
            meta["seconds"] = secs
            meta["release"] = res.sites.to_dict(orient="records")
            if cond == "local":
                return {"tables": {s: o["candidate"] for s, o in res.outcomes.items()}, **meta}
            pooled = res.released.drop(columns=["_site"]) if len(res.released) else pd.DataFrame(columns=cols)
            meta["bytes"] = int(pooled.memory_usage(deep=True).sum())
            return {"tables": {s: pooled for s in trains}, **meta}
        if cond.startswith("central_"):
            pooled = pd.concat(list(trains.values()), ignore_index=True)
            kind = {"central_fed_tvae": "fed_tvae", "central_sdv_tvae": "sdv_tvae", "central_copula": "copula"}[cond]
            gen = fit_generator(kind, pooled, cols, self.schema, self.synth, self.cfg, seed, self.censor, "pooled")
            table = gen.sample(n_syn or n_total, seed=seed)
            return {"tables": {s: table for s in trains}, **meta, "seconds": time.perf_counter() - t0}
        # federated
        fc = self._fed_cfg(cond)
        fc.seed = seed
        ledger = DisclosureLedger()
        sites = [Site(s, trains[s], ledger) for s in trains]
        model = FederatedTVAE(self.synth, fc, subset_schema(self.schema, cols), self.censor)
        model.fit(sites, cols, log=None)
        meta.update({"seconds": time.perf_counter() - t0, "bytes": ledger.total_bytes,
                     "privacy": model.record.privacy, "rounds": fc.generator.rounds})
        if cond == "fed_personalised":
            tables = {s.name: model.sample_personalised(s, n_syn or len(trains[s.name]), seed=seed) for s in sites}
        else:
            table = model.sample(n_syn or n_total, seed=seed)
            tables = {s: table for s in trains}
        return {"tables": tables, **meta}

    # ---- references ---------------------------------------------------------------------
    def _references(self, trains: dict[str, pd.DataFrame], seed: int) -> dict:
        from ..augmentation.models import Preprocessor

        pooled = pd.concat(list(trains.values()), ignore_index=True)
        out = {"real_local": {s: trains[s] for s in trains}, "real_pooled": {s: pooled for s in trains}}
        pre = Preprocessor(self.ep.numeric, self.ep.categorical).fit(pooled)
        sites = [Site(s, trains[s], DisclosureLedger()) for s in trains]
        try:
            beta = federated_cox(sites, pre, self.ep, ridge=1.0)
            out["real_federated_cox"] = {s: LinearCox(pre, beta, trains[s], self.ep) for s in trains}
        except np.linalg.LinAlgError:  # pragma: no cover
            pass
        return out

    # ---- one evaluation ----------------------------------------------------------------
    def _evaluate(self, cond: str, tables: dict, trains: dict, tests: dict, seed: int, arm: str,
                  privacy: bool) -> tuple[list[dict], dict]:
        rows, risks = [], {}
        for s, test in tests.items():
            table = tables.get(s)
            if table is None:
                continue
            fitted = None
            for spec in self.models:
                if isinstance(table, LinearCox):
                    fitted, label = table, "federated_cox"
                else:
                    fitted, label = fit_model(spec, table, self.ep, seed), spec.key
                m, risk = utility_at_site(fitted, test, trains.get(s, pd.concat(list(trains.values()))), self.ep)
                risks[(s, label)] = risk
                row = {"arm": arm, "condition": cond, "site": s, "model": label, "n_train_table": (
                    None if isinstance(table, LinearCox) else len(table)), **m}
                rows.append(row)
                if isinstance(table, LinearCox):
                    break
            if isinstance(table, pd.DataFrame) and cond not in REFERENCES:
                real_train = trains.get(s, pd.concat(list(trains.values())))
                fid = fidelity_at_site(real_train, table, self.ep, seed)
                priv = privacy_at_site(trains[s], test, table, self.columns, self.schema, self.pcfg, seed) \
                    if privacy and s in trains else {}
                for r in rows:
                    if r["site"] == s and r["condition"] == cond:
                        r.update({f"fid_{k}": v for k, v in fid.items()})
                        r.update({f"priv_{k}": v for k, v in priv.items()})
        return rows, risks

    # ---- driver ------------------------------------------------------------------------------
    def run(self, out_dir: Path) -> dict:
        cfg = self.cfg
        out_dir.mkdir(parents=True, exist_ok=True)
        ck_dir = out_dir / "checkpoints"
        ck_dir.mkdir(exist_ok=True)
        key = self._key()
        all_rows, pairs, gens = [], [], []
        conditions = list(cfg.evaluation.conditions)
        for r in range(cfg.evaluation.repeats):
            seed = cfg.seed + r
            splits = split_sites(self.data, self.site_col, self.ep, cfg.evaluation.holdout_frac, seed)
            trains = {s: self.frames[s].iloc[splits[s][0]].reset_index(drop=True) for s in self.site_names}
            tests = {s: self.frames[s].iloc[splits[s][1]].reset_index(drop=True) for s in self.site_names}
            self._exchange_cache = None
            ref_risks = {}
            for cond in [*REFERENCES, *conditions]:
                path = ck_dir / f"r{r}_{cond}.pkl"
                saved = self._load(path, key)
                if saved is None:
                    self.log(f"repeat {r}: {cond}")
                    t0 = time.perf_counter()
                    with limit_cpu(cfg.n_jobs):
                        if cond in REFERENCES:
                            tables, meta = self._references(trains, seed).get(cond, {}), {}
                        else:
                            res = self._synthetic(cond, trains, tests, seed, splits)
                            tables, meta = res["tables"], {k: v for k, v in res.items() if k != "tables"}
                        rows, risks = self._evaluate(cond, tables, trains, tests, seed, "within_site",
                                                     privacy=cond not in REFERENCES)
                    meta["seconds_total"] = time.perf_counter() - t0
                    saved = {"rows": rows, "risks": risks, "meta": meta}
                    self._save(path, key, saved)
                for row in saved["rows"]:
                    all_rows.append({**row, "repeat": r})
                gens.append({"repeat": r, "condition": cond, **_flat_meta(saved["meta"])})
                if cond in REFERENCES:
                    ref_risks[cond] = saved["risks"]
                else:
                    pairs.extend(self._pairs(cond, saved["risks"], ref_risks, tests, r))
                    if cond == "central_fed_tvae":
                        ref_risks["central_fed_tvae"] = saved["risks"]
                    if cond == "local":
                        ref_risks["local"] = saved["risks"]
            # pairs against the generator references computed after both exist
            for cond in conditions:
                path = ck_dir / f"r{r}_{cond}.pkl"
                saved = self._load(path, key)
                for ref in ("central_fed_tvae", "local"):
                    if ref in ref_risks and cond != ref:
                        pairs.extend(self._pair(cond, saved["risks"], ref, ref_risks[ref], tests, r))
        loso_rows = self._loso() if cfg.sites.loso else []
        tables = {"results": pd.DataFrame(all_rows), "pairs": pd.DataFrame(_dedup(pairs)),
                  "generators": pd.DataFrame(gens), "loso": pd.DataFrame(loso_rows)}
        return tables

    def _pairs(self, cond, risks, ref_risks, tests, r) -> list[dict]:
        out = []
        for ref in REFERENCES:
            if ref in ref_risks:
                out.extend(self._pair(cond, risks, ref, ref_risks[ref], tests, r))
        return out

    def _pair(self, cond, risks, ref, rrisks, tests, r) -> list[dict]:
        from ..augmentation.metrics import harrell_c, paired_bootstrap

        from .evaluate import _clean

        out = []
        for (s, label), risk in risks.items():
            ref_key = (s, "federated_cox") if ref == "real_federated_cox" else (s, label)
            rr = rrisks.get(ref_key)
            if risk is None or rr is None:
                continue
            test = _clean(tests[s], self.ep)
            t, e = test[self.ep.time].to_numpy(float), test[self.ep.event].to_numpy(float)
            boot = paired_bootstrap(t, e, risk, rr, self.cfg.evaluation.n_bootstrap, self.cfg.seed + r)
            out.append({"repeat": r, "site": s, "model": label, "condition": cond, "reference": ref,
                        "delta_c": harrell_c(t, e, risk) - harrell_c(t, e, rr), "boot_low": boot["ci_low"],
                        "boot_high": boot["ci_high"], "n_test": len(t)})
        return out

    def _loso(self) -> list[dict]:
        """Leave each ``sites.loso`` site out of everything; evaluate on all of its patients."""
        rows = []
        conds = [c for c in self.cfg.evaluation.conditions if c not in ("local", "fed_personalised")]
        for h in [str(x) for x in self.cfg.sites.loso if str(x) in self.site_names]:
            trains = {s: self.frames[s] for s in self.site_names if s != h}
            tests = {h: self.frames[h]}
            seed = self.cfg.seed
            refs = self._references(trains, seed)
            for cond in ["real_pooled", "real_federated_cox", *conds]:
                self.log(f"LOSO {h}: {cond}")
                with limit_cpu(self.cfg.n_jobs):
                    if cond in REFERENCES:
                        tabs = refs.get(cond, {})
                        tables = {h: next(iter(tabs.values()))} if tabs else {}
                        if cond == "real_federated_cox" and tabs:
                            fc = tables[h]  # baseline hazard from the pooled training sites
                            tables[h] = LinearCox(fc.pre, fc.beta, pd.concat(list(trains.values())), self.ep)
                    else:
                        self._exchange_cache = None
                        res = self._synthetic(cond, trains, {s: self.frames[s].iloc[:0] for s in trains}, seed,
                                              None) if cond != "exchange" else self._loso_exchange(trains, seed)
                        tables = {h: next(iter(res["tables"].values()))}
                    r_rows, _ = self._evaluate(cond, tables, {h: self.frames[h]}, tests, seed, "loso",
                                               privacy=False)
                rows.extend({**x, "heldout_site": h} for x in r_rows)
        return rows

    def _loso_exchange(self, trains, seed) -> dict:
        """Exchange among the training sites (each gated against its own holdout share)."""
        from .exchange import local_exchange

        sites = [Site(s, trains[s], DisclosureLedger()) for s in trains]
        fc = self.cfg.copy()
        fc.seed = seed
        fc.dp.enabled = False
        res = local_exchange(sites, self.columns, self.schema, self.synth, fc, copy.deepcopy(self.pcfg),
                             self.censor, log=None)
        pooled = res.released.drop(columns=["_site"]) if len(res.released) else pd.DataFrame(columns=self.columns)
        return {"tables": {"pooled": pooled}}

    # ---- checkpoints ----------------------------------------------------------------------------
    def _key(self) -> str:
        h = hashlib.sha256()
        h.update(pd.util.hash_pandas_object(self.data.round(6).astype(str), index=False).values.tobytes())
        h.update(json.dumps(self.cfg.to_dict(), sort_keys=True, default=str).encode())
        h.update(json.dumps(vars(self.synth), sort_keys=True, default=str).encode())
        h.update(json.dumps(self.pcfg.to_dict(), sort_keys=True, default=str).encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _load(path: Path, key: str):
        if path.is_file():
            with open(path, "rb") as f:
                saved = pickle.load(f)
            if saved.get("key") == key:
                return saved["value"]
        return None

    @staticmethod
    def _save(path: Path, key: str, value) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"key": key, "value": value}, f)
        tmp.replace(path)


def _flat_meta(meta: dict) -> dict:
    out = {k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool)) or v is None}
    priv = meta.get("privacy") or {}
    eps = [p.get("epsilon_total") for p in priv.values() if isinstance(p, dict) and p.get("dp")]
    if eps:
        out["epsilon_max"] = float(max(eps))
        out["epsilon_min"] = float(min(eps))
    rel = meta.get("release")
    if rel:
        out["sites_released"] = int(sum(bool(x["released"]) for x in rel))
    return out


def _dedup(pairs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for p in pairs:
        k = (p["repeat"], p["site"], p["model"], p["condition"], p["reference"])
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def run_benchmark(data, schema, columns, synth, cfg, pcfg, out_dir, censor_threshold=None, log=print) -> dict:
    """Run, summarise and write the benchmark (see module docstring)."""
    from .report import summarise, write_report

    out = Path(out_dir)
    t0 = time.perf_counter()
    bench = Benchmark(data, schema, columns, synth, cfg, pcfg, censor_threshold, log)
    tables = bench.run(out)
    tables.update(summarise(tables, cfg))
    for name, t in tables.items():
        if isinstance(t, pd.DataFrame):
            t.to_csv(out / f"{name}.csv", index=False)
    cfg.to_yaml(out / "federated_config_used.yaml")
    meta = {"sites": {s: int(len(bench.frames[s])) for s in bench.site_names}, "endpoint": vars(bench.ep),
            "columns": columns, "elapsed_seconds": time.perf_counter() - t0}
    (out / "benchmark_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    write_report(tables, cfg, meta, out)
    return tables
