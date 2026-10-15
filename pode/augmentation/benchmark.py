"""Augmentation benchmark: split first, synthesise second, score on real patients.

For every repeated-K-fold split of the real development rows (and once more
for the external cohort, when configured):

1. fit a TVAE on the split's real training rows only (cached per split);
2. build one pool per strategy (TVAE natural/targeted/subgroup, duplicate,
   jitter, weight) and check TVAE pools for fidelity and privacy, using the
   split's held-out patients as the never-seen reference;
3. for each model, fit the real-only arm, the real-only learning curve and
   every (strategy, ratio, draw) augmented arm;
4. score each fit on the split's real held-out patients (or the external
   cohort). Synthetic rows never enter any test set.

Results are then summarised into paired gains, real-data equivalents,
dose-response, compute cost and a gated scorecard (see :mod:`.efficiency`,
:mod:`.scorecard`).
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import delayed

from ..config import PodeConfig
from ..imputation.base import limit_cpu, n_workers, parallel_map, resolve_columns
from ..schema.column_types import ColumnType
from ..schema.schema import DatasetSchema
from . import efficiency as E
from .config import AugmentationConfig, ModelSpec
from .fidelity import matrix_fidelity, pool_fidelity
from .generator import HybridSplitGenerator, SplitGenerator, generator_columns, matrix_columns, with_survival_rules
from .metrics import _ibs_grid, paired_bootstrap, subgroup_metrics, utility_metrics
from .models import MODELS, build_model
from .privacy import privacy_metrics
from .scorecard import build_scorecard
from .selection import load_reference, selection_stability
from .splits import make_splits, split_external
from .strategies import SOURCE, build_pool, draw, target_config


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---- column roles ----------------------------------------------------------------
class Columns:
    """Endpoint, features, strata and generator columns resolved against the data."""

    def __init__(self, df: pd.DataFrame, schema: DatasetSchema, cfg: AugmentationConfig):
        pair = schema.primary_survival_pair()
        self.time = cfg.endpoint.time or (pair.time_col if pair else None)
        self.event = cfg.endpoint.event or (pair.event_col if pair else None)
        if not self.time or not self.event:
            raise ValueError("No survival endpoint: set endpoint.time/event or mark a survival pair in the schema.")
        self.outcomes = sorted({self.time, self.event, *schema.columns_of_type(
            ColumnType.SURVIVAL_TIME, ColumnType.SURVIVAL_EVENT)})
        self.strata = list(cfg.strata)
        # Hybrid generators decode the matrix columns, so they count as generated.
        self.matrix = matrix_columns(df, cfg)
        self.generated = generator_columns(df, schema, cfg) + self.matrix
        missing = [c for c in [*self.strata, *cfg.subgroups, *cfg.splits.stratify] if c not in df.columns]
        if missing:
            raise ValueError(f"Configured columns not in the data: {missing}")
        blocked = set(self.outcomes) | set(self.strata) | {cfg.external.column, cfg.id_column}
        pool = [c for c in self.generated if c not in blocked]
        self.features = resolve_columns(cfg.features, pool) if cfg.features else pool
        self.schema = schema

    def split_types(self, cols: list[str]) -> tuple[list[str], list[str]]:
        num = [c for c in cols if c in self.schema.types and self.schema.types[c].is_numeric]
        return num, [c for c in cols if c not in num]

    def model_features(self, spec: ModelSpec) -> tuple[list[str], list[str]]:
        cols = resolve_columns(spec.features, self.features) if spec.features else self.features
        num, cat = self.split_types(cols)
        if spec.kind in ("metacpg_fixed", "metacpg_search"):
            cat = []  # the MetaCPG score is a weighted mean of numeric (M-value) features
        return num, cat


def strata_labels(df: pd.DataFrame, cfg: AugmentationConfig) -> np.ndarray | None:
    if not cfg.strata:
        return None
    lab = df[cfg.strata].astype(str).agg("|".join, axis=1).to_numpy(dtype=object)
    synthetic = df[SOURCE].astype(str).str.startswith("synthetic").to_numpy() if SOURCE in df else \
        np.zeros(len(df), bool)
    missing = df[cfg.strata].isna().any(axis=1).to_numpy()
    own = synthetic & (missing | (cfg.generator.synthetic_strata == "own"))
    lab[own] = cfg.generator.synthetic_label
    return lab


# ---- one model fit + score (runs in workers) ---------------------------------------
def _build_train(task: dict, ctx: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """The task's training set, built where it is used (wide matrices are never
    materialised for every task at once)."""
    b = task["build"]
    train = ctx["train"]
    if b["kind"] == "augment":
        return draw(ctx["pools"][b["pool"]], train, b["n"], np.random.default_rng(b["seed"]))
    real = train.assign(**{SOURCE: "real"})
    if b["kind"] == "subsample":
        real = _subsample_real(real, b["frac"], ctx["cols"].event, np.random.default_rng(b["seed"]))
    return real, np.ones(len(real))


def _fit_and_score(task: dict, ctx: dict) -> dict:
    t0 = time.perf_counter()
    train, w = _build_train(task, ctx)
    cols, cfg = ctx["cols"], ctx["cfg"]
    spec: ModelSpec = task["spec"]
    num, cat = cols.model_features(spec)
    model = build_model(spec, num, cat, cfg.endpoint.horizon, task["seed"])
    real = (train[SOURCE] == "real").to_numpy()
    strata = strata_labels(train, cfg) if model.supports_strata else None
    # Rows added, or for the weight strategy the effective count the weights add.
    n_added = int((~real).sum()) + int(round(w[real].sum() - real.sum()))
    if not model.supports_weights:
        w = np.ones(len(train))
    model.fit(train, train[cols.time].to_numpy(float), train[cols.event].to_numpy(float), strata, w, real)
    fit_s = time.perf_counter() - t0

    test = ctx["test"]
    tt, te = test[cols.time].to_numpy(float), test[cols.event].to_numpy(float)
    risk = model.risk(test)
    grid = ctx["grid"]
    s_h = model.surv(test, [cfg.endpoint.horizon])
    s_grid = model.surv(test, grid) if len(grid) >= 2 else None
    s_h = s_h[:, 0] if s_h is not None else None
    rt, re_ = train.loc[real, cols.time].to_numpy(float), train.loc[real, cols.event].to_numpy(float)
    metrics = utility_metrics(tt, te, risk, s_h, s_grid, grid, rt, re_, cfg.endpoint.horizon,
                              cfg.net_benefit_thresholds)
    subs = subgroup_metrics(test, tt, te, risk, s_h, cfg.subgroups, cfg.endpoint.horizon,
                            cfg.min_subgroup_n, cfg.min_subgroup_events)
    meta = {k: task[k] for k in ("split", "evaluation", "model", "arm", "strategy", "ratio", "draw", "fraction")}
    row = {**meta, "n_real": int(real.sum()), "n_added": n_added,
           "fit_seconds": fit_s, "score_seconds": time.perf_counter() - t0 - fit_s,
           "n_nonzero": model.n_nonzero(), **model.diagnostics(), **metrics}
    return {"row": row, "subgroups": [{**meta, **s} for s in subs],
            "risk": risk if task["evaluation"] == "external" else None}


def _safe_chunk(chunk: list[dict], ctx: dict) -> list[dict]:
    outs = []
    for task in chunk:
        try:
            outs.append(_fit_and_score(task, ctx))
        except Exception as exc:  # recorded, never fatal
            outs.append({"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=3),
                         "meta": {k: task[k] for k in ("split", "model", "arm", "strategy", "ratio", "draw")}})
    return outs


def _run_tasks(tasks: list[dict], ctx: dict, n_jobs: int, threaded: bool) -> list[dict]:
    """Run tasks in chunks of one (model, strategy) each.

    Worker processes receive ``ctx`` (training rows, pools, test rows) once per
    chunk. Wide tables (omics matrices) use threads instead, so nothing is
    copied; BLAS is pinned to one thread per worker to keep within ``n_jobs``.
    """
    chunks: dict[tuple, list[dict]] = {}
    for t in tasks:
        chunks.setdefault((t["model"], t["strategy"]), []).append(t)
    calls = [delayed(_safe_chunk)(c, ctx) for c in chunks.values()]
    if threaded and n_workers(n_jobs) > 1:
        from joblib import Parallel
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            results = Parallel(n_jobs=n_workers(n_jobs), backend="threading")(calls)
    else:
        results = parallel_map(calls, n_jobs)
    return [o for chunk in results for o in chunk]


def _subsample_real(train: pd.DataFrame, frac: float, event: str, rng: np.random.Generator) -> pd.DataFrame:
    """Event-stratified subsample of real rows (learning curve)."""
    parts = []
    for _, g in train.groupby(train[event].astype(str)):
        k = max(1, int(round(frac * len(g))))
        parts.append(g.iloc[rng.choice(len(g), k, replace=False)])
    return pd.concat(parts).reset_index(drop=True)


# ---- driver ----------------------------------------------------------------------
def run_benchmark(
    df: pd.DataFrame,
    schema: DatasetSchema,
    pode_config: PodeConfig,
    cfg: AugmentationConfig,
    out_dir: str | Path,
    generator_model: str | Path | None = None,
    log: Callable[[str], None] = _log,
) -> dict:
    """Run the benchmark and write all tables/plots/reports into ``out_dir``.

    CPU use is capped at ``cfg.n_jobs`` cores: worker processes, BLAS/OpenMP
    thread pools and torch's CPU threads all respect the limit.
    ``generator_model`` (a pre-trained SDV ``model.pkl``) is used **only** for the
    external cohort; internal splits always fit their own generator, because a
    model trained on all development rows has seen every internal test patient.
    """
    with limit_cpu(cfg.n_jobs):
        return _run(df, schema, pode_config, cfg, Path(out_dir), generator_model, log)


def plan(df: pd.DataFrame, schema: DatasetSchema, cfg: AugmentationConfig, generator_model=None) -> str:
    """What a benchmark would do, resolved against the data, without fitting anything."""
    cols = Columns(df, schema, cfg)
    keep = df[[cols.time, cols.event, *cfg.strata]].notna().all(axis=1)
    dev, ext = split_external(df[keep].reset_index(drop=True), cfg)
    splits = make_splits(dev, cfg, cols.event)
    n_train = int(np.mean([len(s.train) for s in splits]))

    def short(xs, k=8):
        return ", ".join(xs[:k]) + (f", … (+{len(xs) - k})" if len(xs) > k else "") if xs else "-"

    lines = [
        f"Endpoint: {cols.time} / {cols.event}, horizon {cfg.endpoint.horizon:g}",
        f"Rows: {len(df)} ({int((~keep).sum())} dropped for missing endpoint/strata); development {len(dev)} "
        f"({int(dev[cols.event].sum())} events); external {len(ext)}"
        + (f" ({cfg.external.column} in {cfg.external.values})" if cfg.external.column else ""),
        f"Splits: {cfg.splits.folds}-fold x {cfg.splits.repeats} repeats = {len(splits)} "
        f"(~{n_train} train rows each), stratified on event + {cfg.splits.stratify or 'nothing else'}",
        f"Generator: {cfg.generator.backend}; TVAE columns ({len(generator_columns(df, schema, cfg))}): "
        f"{short(generator_columns(df, schema, cfg))}",
    ]
    if cols.matrix:
        lines.append(f"  matrix columns decoded by the compressor: {len(cols.matrix)} ({short(cols.matrix, 4)})")
    lines += [f"Strata: {cfg.strata or 'none'} (synthetic rows: {cfg.generator.synthetic_strata})",
              f"Subgroup audit: {cfg.subgroups or 'none'}",
              f"Strategies ({len(cfg.strategies)}): " + ", ".join(f"{s.key} [{s.kind}]" for s in cfg.strategies),
              f"Ratios: {cfg.ratios}, draws: {cfg.draws}"]
    per_split = 0
    for spec in cfg.models:
        num, cat = cols.model_features(spec)
        n_strat = sum(1 for s in cfg.strategies if s.kind != "weight" or MODELS[spec.kind].supports_weights)
        fits = 1 + (len(cfg.learning_curve.fractions) * cfg.learning_curve.draws if cfg.learning_curve.enabled
                    else 0) + n_strat * len(cfg.ratios) * cfg.draws
        per_split += fits
        tv = spec.params.get("top_variance")
        lines.append(f"Model {spec.key} [{spec.kind}]: {len(num)} numeric + {len(cat)} categorical features "
                     f"({short(num + cat, 6)})" + (f"; top {tv} by variance per fit" if tv else "")
                     + f"; {fits} fits per split")
    lines.append(f"Workload: {len(splits) + (1 if len(ext) else 0)} evaluations x {per_split} model fits"
                 + (f"; external generator: {generator_model}" if generator_model else ""))
    return "\n".join(lines)


def _evaluations(dev, ext, cfg, cols):
    for s in make_splits(dev, cfg, cols.event):
        yield {"key": s.key, "evaluation": "internal", "train": dev.iloc[s.train].reset_index(drop=True),
               "test": dev.iloc[s.test].reset_index(drop=True), "seed": cfg.seed + 1000 * s.repeat + 10 * s.fold}
    if not ext.empty:
        yield {"key": "external", "evaluation": "external", "train": dev, "test": ext, "seed": cfg.seed + 999_999}


def _run(df, schema, pode_config, cfg, out_dir: Path, generator_model, log) -> dict:
    # Import what the end of the run needs now: in an editable install a long run
    # otherwise depends on the working tree still holding these files hours later.
    from . import plots as _plots, report as _report  # noqa: F401

    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    cols = Columns(df, schema, cfg)
    if generator_model and cfg.generator.backend == "hybrid":
        raise ValueError("--generator-model is a plain TVAE model.pkl; it cannot stand in for the hybrid backend.")
    need = [cols.time, cols.event, *cfg.strata]
    keep = df[need].notna().all(axis=1)
    if (~keep).any():
        log(f"Dropping {int((~keep).sum())} rows with missing endpoint/strata values.")
    df = df[keep].reset_index(drop=True)
    dev, ext = split_external(df, cfg)
    log(f"Development rows: {len(dev)} ({int(dev[cols.event].sum())} events); external rows: {len(ext)}.")
    ungenerated = [c for c in cols.features if c not in cols.generated]
    if ungenerated:
        log(f"warning: model features not synthesised (median-filled in synthetic rows): {ungenerated}")
    constraints = with_survival_rules(schema, cfg.generator.constraints)
    strategies = cfg.strategies
    tvae_keys = {s.key for s in strategies if s.uses_tvae}
    max_ratio = max(cfg.ratios)
    gpu_peak = _reset_gpu_peak()

    results, subgroups, fidelity, privacy, timings, failures, pools_diag = [], [], [], [], [], [], []
    membership = []
    ext_risk: dict[tuple, list[np.ndarray]] = {}

    # Checkpoint every finished split, so a crash or interruption never loses
    # completed work: a re-run with the same config and data resumes from them.
    ckpt_dir = out_dir / "checkpoints"
    ckpt_key = _checkpoint_key(cfg, dev, ext, generator_model)
    lists = {"results": results, "subgroups": subgroups, "fidelity": fidelity, "privacy": privacy,
             "timings": timings, "failures": failures, "pools_diag": pools_diag, "membership": membership}

    for ev in _evaluations(dev, ext, cfg, cols):
        key, train, test, seed = ev["key"], ev["train"], ev["test"], ev["seed"]
        part = _load_checkpoint(ckpt_dir / f"{key}.pkl", ckpt_key)
        if part is not None:
            for name, items in part["lists"].items():
                lists[name].extend(items)
            ext_risk.update(part["ext_risk"])
            log(f"[{key}] resumed from checkpoint")
            continue
        start = {name: len(items) for name, items in lists.items()}
        ext_before = set(ext_risk)
        log(f"[{key}] train {len(train)} / test {len(test)}")
        if cfg.id_column:
            membership += [{"split": key, "role": role, cfg.id_column: i}
                           for role, part in (("train", train), ("test", test)) for i in part[cfg.id_column]]
        gen = None
        if tvae_keys:
            if ev["evaluation"] == "external" and generator_model:
                gen = SplitGenerator.from_model(generator_model, schema, cfg)
                gen_src = str(generator_model)
            elif cfg.generator.backend == "hybrid":
                gen = HybridSplitGenerator.fit(train, schema, pode_config, cfg, seed, out_dir / "generators" / key,
                                               endpoint=(cols.time, cols.event))
                gen_src = "cache" if gen.cached else "fitted"
            else:
                gen = SplitGenerator.fit(train, schema, pode_config, cfg, seed,
                                         out_dir / "generators" / key)
                gen_src = "cache" if gen.cached else "fitted"
            train_view, test_view = gen.view(train), gen.view(test)
            # Cached models report the fit time recorded when they were trained.
            timings.append({"split": key, "stage": "generator_fit", "strategy": "-", "tvae": True,
                            "seconds": gen.fit_seconds, "source": gen_src})
            log(f"  generator {gen_src} ({timings[-1]['seconds']:.1f}s)")

        n_pool = int(math.ceil(cfg.generator.pool_factor * max_ratio * len(train)))
        pools = {}
        for i, spec in enumerate(strategies):
            t0 = time.perf_counter()
            try:
                pool = build_pool(spec, train, schema, gen, n_pool, seed + 101 * (i + 1), constraints,
                                  cols.outcomes)
            except Exception as exc:
                failures.append({"split": key, "stage": "pool", "strategy": spec.key,
                                 "error": f"{type(exc).__name__}: {exc}"})
                log(f"  ! pool {spec.key} failed: {exc}")
                continue
            timings.append({"split": key, "stage": "pool", "strategy": spec.key, "tvae": spec.uses_tvae,
                            "seconds": time.perf_counter() - t0})
            pools_diag.append({"split": key, "evaluation": ev["evaluation"],
                               **{k: v for k, v in pool.diagnostics.items() if not isinstance(v, (dict, list))}})
            if pool.rows.empty and not pool.is_weight:
                # Otherwise its "augmented" arms would silently equal real-only.
                failures.append({"split": key, "stage": "pool", "strategy": spec.key,
                                 "error": "no usable rows (the generator never produced the target, or every "
                                          "row broke a constraint); strategy skipped for this split"})
                log(f"  ! pool {spec.key}: no usable rows, skipped")
                continue
            pools[spec.key] = pool
            if spec.uses_tvae and not pool.rows.empty:
                fixed = _conditioned_columns(spec)
                num, cat = _split_by(gen.schema, gen.columns)
                if cfg.fidelity:
                    fid = pool_fidelity(train_view, pool.rows, num, cat, cols.time, cols.event,
                                        cfg.endpoint.horizon, pool.target_index, fixed, seed)
                    if cols.matrix:
                        fid.update(matrix_fidelity(train[cols.matrix], pool.rows[cols.matrix],
                                                   cfg.generator.hybrid.fidelity_top_features,
                                                   cfg.generator.hybrid.matrix_kind, seed))
                    fidelity.append({"split": key, "evaluation": ev["evaluation"], "strategy": spec.key, **fid})
                # Privacy needs never-seen patients from the same population as the
                # generator's training rows; an external cohort differs systematically,
                # so it would make synthetic rows look spuriously close. Internal only.
                if cfg.privacy.enabled and ev["evaluation"] == "internal":
                    pcols = (resolve_columns(cfg.privacy.columns, gen.columns) if cfg.privacy.columns
                             else [c for c in gen.columns if c not in cols.outcomes])
                    pn, pc = _split_by(gen.schema, pcols)
                    privacy.append({"split": key, "evaluation": ev["evaluation"], "strategy": spec.key,
                                    **privacy_metrics(train_view, test_view, pool.rows, pn, pc, cfg.privacy.rare_k,
                                                      cfg.privacy.max_synthetic, seed)})

        tasks = _tasks(ev, train, pools, cfg, cols)
        ctx = {"cols": cols, "cfg": cfg, "test": test, "train": train, "pools": pools,
               "grid": _ibs_grid(test[cols.time].to_numpy(float), cfg.endpoint.horizon)}
        log(f"  {len(tasks)} model fits")
        wide = len(train.columns) > 1000 or any(m.kind == "metacpg_search" for m in cfg.models)
        outs = _run_tasks(tasks, ctx, cfg.n_jobs, threaded=wide)
        for out in outs:
            if "error" in out:
                failures.append({**out["meta"], "stage": "model", "error": out["error"]})
                log(f"  ! {out['meta']}: {out['error']}")
                continue
            results.append(out["row"])
            subgroups.extend(out["subgroups"])
            if out["risk"] is not None:
                r = out["row"]
                ext_risk.setdefault((r["model"], r["arm"], r["strategy"], r["ratio"]), []).append(out["risk"])
        _save_checkpoint(ckpt_dir / f"{key}.pkl", ckpt_key,
                         {name: items[start[name]:] for name, items in lists.items()},
                         {k: v for k, v in ext_risk.items() if k not in ext_before})

    tables = _summarise(results, subgroups, fidelity, privacy, timings, failures, pools_diag,
                        ext_risk, ext, cols, cfg, tvae_keys)
    tables["splits"] = pd.DataFrame(membership)
    report = _write_outputs(tables, out_dir, cfg, cols, dev, ext, schema, gpu_peak,
                            time.perf_counter() - t_start, generator_model)
    return {"tables": tables, "report": report}


def _checkpoint_key(cfg: AugmentationConfig, dev: pd.DataFrame, ext: pd.DataFrame, generator_model) -> str:
    import hashlib

    # Settings that only affect execution or end-of-run summaries never invalidate a split.
    relevant = {k: v for k, v in cfg.to_dict().items() if k not in ("n_jobs", "plots", "gates", "bootstrap")}
    h = hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode())
    for part in (dev, ext):
        h.update(pd.util.hash_pandas_object(part, index=False).values.tobytes())
    h.update(str(generator_model).encode())
    return h.hexdigest()[:16]


def _load_checkpoint(path: Path, key: str) -> dict | None:
    import pickle

    if not path.is_file():
        return None
    try:
        part = pickle.loads(path.read_bytes())
    except Exception:  # a checkpoint torn by a crash mid-write is simply redone
        return None
    return part if part.get("key") == key else None


def _save_checkpoint(path: Path, key: str, lists: dict, ext_risk: dict) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps({"key": key, "lists": lists, "ext_risk": ext_risk}))
    tmp.replace(path)  # atomic: a crash never leaves a half-written checkpoint


def _split_by(schema: DatasetSchema, cols: list[str]) -> tuple[list[str], list[str]]:
    """Numeric / categorical split of ``cols`` by ``schema`` types."""
    num = [c for c in cols if c in schema.types and schema.types[c].is_numeric]
    return num, [c for c in cols if c not in num]


def _conditioned_columns(spec) -> list[str]:
    cols = []
    for g in target_config(spec.target).get("groups", []):
        cols += [*(g.get("equals") or {}), *(g.get("ranges") or {})]
    cols += list(spec.params.get("columns", []))
    return sorted(set(cols))


def _tasks(ev, train, pools, cfg: AugmentationConfig, cols: Columns) -> list[dict]:
    base = {"split": ev["key"], "evaluation": ev["evaluation"]}
    tasks = []
    for mi, spec in enumerate(cfg.models):
        common = {**base, "spec": spec, "model": spec.key}
        tasks.append({**common, "arm": "real_only", "strategy": "real_only", "ratio": 0.0, "draw": 0,
                      "fraction": 1.0, "build": {"kind": "real"}, "seed": ev["seed"]})
        if ev["evaluation"] == "internal" and cfg.learning_curve.enabled:
            for fi, frac in enumerate(cfg.learning_curve.fractions):
                for d in range(cfg.learning_curve.draws):
                    tasks.append({**common, "arm": "learning_curve", "strategy": "real_only", "ratio": 0.0,
                                  "draw": d, "fraction": float(frac), "seed": ev["seed"] + d,
                                  "build": {"kind": "subsample", "frac": float(frac), "seed": [ev["seed"], 7, fi, d]}})
        for si, (skey, pool) in enumerate(pools.items()):
            if pool.is_weight and not MODELS[spec.kind].supports_weights:
                continue  # case weights only reach models that accept them
            for ri, ratio in enumerate(cfg.ratios):
                n = int(round(ratio * len(train)))
                for d in range(cfg.draws):
                    tasks.append({**common, "arm": "augmented", "strategy": skey, "ratio": float(ratio),
                                  "draw": d, "fraction": 1.0, "seed": ev["seed"] + d + 31 * mi,
                                  "build": {"kind": "augment", "pool": skey, "n": n,
                                            "seed": [ev["seed"], si, ri, d, 13]}})
    return tasks


def _reset_gpu_peak():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return lambda: torch.cuda.max_memory_allocated() / 2**20
    except Exception:  # pragma: no cover
        pass
    return lambda: 0.0


# ---- summaries -------------------------------------------------------------------
def _external_paired(ext_risk, ext, cols, cfg) -> pd.DataFrame:
    if ext.empty or not ext_risk:
        return pd.DataFrame()
    t, e = ext[cols.time].to_numpy(float), ext[cols.event].to_numpy(float)

    def ensemble(rs):  # mean standardised risk over draws
        return np.mean([(r - r.mean()) / (r.std() or 1.0) for r in rs], axis=0)

    rows = []
    for (model, arm, strategy, ratio), rs in ext_risk.items():
        if arm != "augmented" or (model, "real_only", "real_only", 0.0) not in ext_risk:
            continue
        base = ensemble(ext_risk[(model, "real_only", "real_only", 0.0)])
        res = paired_bootstrap(t, e, ensemble(rs), base, cfg.bootstrap, cfg.seed, strata=e)
        rows.append({"model": model, "strategy": strategy, "ratio": ratio, "metric": "c_harrell",
                     "delta": res["mean"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                     "p_better": res["p_better"], "n_boot": res["n_boot"]})
    return pd.DataFrame(rows)


def _summarise(results, subgroups, fidelity, privacy, timings, failures, pools_diag, ext_risk, ext, cols, cfg,
               tvae_keys) -> dict[str, pd.DataFrame]:
    tables = {"results_by_fold": pd.DataFrame(results), "subgroups": pd.DataFrame(subgroups),
              "fidelity": pd.DataFrame(fidelity), "privacy": pd.DataFrame(privacy),
              "timings": pd.DataFrame(timings), "failures": pd.DataFrame(failures),
              "pools": pd.DataFrame(pools_diag), "external_paired": _external_paired(ext_risk, ext, cols, cfg)}
    return summarise_tables(tables, cfg, tvae_keys)


def summarise_tables(tables: dict[str, pd.DataFrame], cfg: AugmentationConfig,
                     tvae_keys: set[str]) -> dict[str, pd.DataFrame]:
    """Derived tables (paired gains, efficiency, dose-response, cost, scorecard) from the raw ones.

    Used at the end of a run and by ``augment report``, so a re-render picks up
    changes to the summaries without refitting anything.
    """
    res = tables["results_by_fold"]
    priv = tables.get("privacy", pd.DataFrame())
    if not priv.empty and "evaluation" in priv:
        tables["privacy"] = priv = priv[priv["evaluation"] == "internal"].reset_index(drop=True)
    if res.empty:
        return tables
    pm = cfg.gates.primary_metric
    paired = E.paired_differences(res)
    ext_res = res[res["evaluation"] == "external"]
    ext_paired = tables.get("external_paired", pd.DataFrame())
    tables.update({
        "paired_differences": paired,
        "external_results": (ext_res.groupby(["model", "arm", "strategy", "ratio"]).mean(numeric_only=True)
                             .reset_index() if not ext_res.empty else pd.DataFrame()),
        "learning_curve": E.learning_curve(res, pm),
        "learning_curve_slopes": pd.concat([E.learning_curve_slopes(res, m) for m in (pm, "ibs") if m in res],
                                           ignore_index=True),
        "efficiency": E.real_data_equivalent(paired, res, (pm, "ibs")) if not paired.empty else pd.DataFrame(),
        "dose_response": E.dose_response(paired, pm) if not paired.empty else pd.DataFrame(),
        "compute_cost": (E.compute_cost(tables["timings"], res) if not tables.get("timings", pd.DataFrame()).empty
                         else pd.DataFrame()),
        **dict(zip(("selection_stability", "selection_frequency"),
                   selection_stability(res, load_reference(cfg.reference_signature)))),
        "scorecard": build_scorecard(paired, tables.get("subgroups", pd.DataFrame()),
                                     tables.get("fidelity", pd.DataFrame()), priv, ext_paired, cfg, tvae_keys),
    })
    return tables


def _write_outputs(tables, out_dir: Path, cfg, cols, dev, ext, schema, gpu_peak, elapsed, generator_model) -> dict:
    from .report import run_manifest, write_markdown_report

    for name, t in tables.items():
        if isinstance(t, pd.DataFrame) and not t.empty:
            t.to_csv(out_dir / f"{name}.csv", index=False)
    report = {
        "data": {"development_rows": len(dev), "development_events": int(dev[cols.event].sum()),
                 "external_rows": len(ext), "external_events": int(ext[cols.event].sum()) if len(ext) else 0},
        "columns": {"time": cols.time, "event": cols.event, "features": cols.features, "strata": cols.strata,
                    "generated": cols.generated},
        "config": cfg.to_dict(),
        "generator_model_external": str(generator_model) if generator_model else None,
        "gpu_peak_mib": gpu_peak(),
        "elapsed_seconds": elapsed,
        "failures": tables["failures"].to_dict("records") if not tables["failures"].empty else [],
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(run_manifest(cfg), f, indent=2, default=str)
    write_markdown_report(tables, report, out_dir / "report.md")
    if cfg.plots:
        try:
            from .plots import write_plots

            write_plots(tables, out_dir / "plots", cfg)
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            _log(f"warning: plots failed: {exc}")
    return report
