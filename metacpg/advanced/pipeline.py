from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import pickle
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.model_selection import KFold, StratifiedKFold

from .base import CohortUnsupportedError, harrell_c, rank_importance
from .data import (
    AdvancedData,
    ClinicalPreprocessor,
    Preprocessor,
    load_advanced_data,
    read_advanced_config,
    strata_labels,
)
from .models import LinearCoxModel
from .registry import REGISTRY, build_estimator, cohort_report, method_catalog


ProgressCallback = Callable[[str], None]


def _announce(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _cohort_labels(data: AdvancedData, rows: np.ndarray) -> np.ndarray | None:
    if data.cohort is None:
        return None
    return data.cohort.iloc[rows].astype(str).to_numpy()


def _combine_strata(covariate_strata: np.ndarray | None,
                    cohort: np.ndarray | None) -> np.ndarray | None:
    """Cross covariate strata with cohort labels.

    Crossing cohort into the Cox strata gives every cohort its own unspecified baseline
    hazard while the coefficients stay shared, which is what ``phenotype.cohort_column``
    means: the training samples come from several sources whose underlying survival rates
    may differ, not from separate train/test populations.
    """
    if cohort is None:
        return covariate_strata
    cohort = np.asarray(cohort, dtype=str)
    if covariate_strata is None:
        return cohort
    return np.array([f"{a}|{b}" for a, b in zip(np.asarray(covariate_strata, dtype=str), cohort)])


@dataclass
class Prepared:
    x: np.ndarray
    masks: np.ndarray
    feature_names: list[str]
    design_names: list[str]
    n_omics: int
    preprocessor: Preprocessor
    clinical: ClinicalPreprocessor
    strata: np.ndarray | None = None
    cohort_strata: np.ndarray | None = None

    def method_strata(self, stratifies: bool) -> np.ndarray | None:
        """Strata for the method's own fit; cohort is only crossed in when it applies."""
        return self.cohort_strata if stratifies and self.cohort_strata is not None else self.strata

    def refit_strata(self) -> np.ndarray | None:
        """Strata for the reduced-signature and clinical-only ridge Cox refits.

        These are always plain Cox fits, so they stratify on cohort whenever one is
        configured regardless of how the parent method handles cohorts.
        """
        return self.cohort_strata if self.cohort_strata is not None else self.strata


def _prepare(data: AdvancedData, fit_rows: np.ndarray, rows: np.ndarray,
             preprocessor: Preprocessor | None = None,
             clinical: ClinicalPreprocessor | None = None) -> Prepared:
    preprocessor = preprocessor or Preprocessor.fit(data, fit_rows)
    clinical = clinical or ClinicalPreprocessor.fit(data, fit_rows)
    omics, masks = preprocessor.transform(data, rows)
    clinical_values = clinical.transform(data, rows)
    technical_names = [f"technical::block_present::{name}" for name in preprocessor.block_names]
    x = np.column_stack([omics, masks, clinical_values])
    names = preprocessor.feature_names + technical_names + clinical.feature_names
    covariate_strata = strata_labels(data, rows)
    return Prepared(x, masks, preprocessor.feature_names, names, len(preprocessor.feature_names),
                    preprocessor, clinical, covariate_strata,
                    _combine_strata(covariate_strata, _cohort_labels(data, rows)))


def _endpoint_rows(data: AdvancedData, endpoint: dict[str, Any]) -> np.ndarray:
    time_v = pd.to_numeric(data.phenotype[endpoint["time"]], errors="coerce").to_numpy(float)
    status = pd.to_numeric(data.phenotype[endpoint["status"]], errors="coerce").to_numpy(float)
    return np.flatnonzero(np.isfinite(time_v) & np.isfinite(status))


def _outcome(data: AdvancedData, endpoint: dict[str, Any], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_v = pd.to_numeric(data.phenotype.iloc[rows][endpoint["time"]], errors="coerce").to_numpy(float)
    status = pd.to_numeric(data.phenotype.iloc[rows][endpoint["status"]], errors="coerce").to_numpy(float)
    return time_v, status


def _stratified_split(rows: np.ndarray, status: np.ndarray, cohorts: np.ndarray | None,
                      n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split ``rows`` into ``n_splits`` folds, balancing cohort and event status.

    Balancing on ``cohort x event`` keeps every cohort represented in every training fold,
    so cohort-specific baselines and random effects stay estimable and no fold accidentally
    holds out an entire cohort. Degrades to event-only stratification and then to plain
    K-fold when the smallest cell is too small to spread across the folds.
    """
    event = status.astype(int).astype(str)
    layers = [np.char.add(np.char.add(np.asarray(cohorts, dtype=str), "|"), event)] if cohorts is not None else []
    layers.append(event)
    for labels in layers:
        _, counts = np.unique(labels, return_counts=True)
        if counts.min() >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(splitter.split(rows, labels))
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(rows))


def _leave_one_cohort_out(rows: np.ndarray, cohorts: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
    levels = sorted(set(cohorts))
    if len(levels) < 2:
        raise ValueError(
            "evaluation.cohort_validation: leave_one_cohort_out requires at least two cohorts "
            f"in phenotype.cohort_column; found {len(levels)}"
        )
    result = []
    for level in levels:
        test_mask = cohorts == level
        train, test = rows[~test_mask], rows[test_mask]
        if len(train) >= 4 and len(test):
            result.append((train, test, f"cohort:{level}"))
    if not result:
        raise ValueError(
            "evaluation.cohort_validation: leave_one_cohort_out could not form any usable fold "
            "(every held-out cohort leaves fewer than four training samples)"
        )
    return result


def _folds(data: AdvancedData, endpoint: dict[str, Any], rows: np.ndarray, folds: int, seed: int,
           mode: str = "stratified_kfold") -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Build the outer cross-validation folds.

    ``phenotype.cohort_column`` describes heterogeneity *within* the training data and does
    not, by itself, define a split. Under the default ``stratified_kfold`` mode the folds are
    balanced on cohort and event status; ``leave_one_cohort_out`` must be requested explicitly
    via ``evaluation.cohort_validation`` when cross-cohort external validation is wanted.
    """
    _, status = _outcome(data, endpoint, rows)
    cohorts = _cohort_labels(data, rows)
    if mode == "leave_one_cohort_out":
        if cohorts is None:
            raise ValueError(
                "evaluation.cohort_validation: leave_one_cohort_out requires phenotype.cohort_column"
            )
        return _leave_one_cohort_out(rows, cohorts)
    n_splits = min(max(2, int(folds)), len(rows))
    indices = _stratified_split(rows, status, cohorts, n_splits, seed)
    return [(rows[train], rows[test], f"fold:{i}") for i, (train, test) in enumerate(indices)]


def _binary_auc(time_v: np.ndarray, status: np.ndarray, risk: np.ndarray, horizon: float) -> float:
    cases = (time_v <= horizon) & (status == 1)
    controls = time_v > horizon
    if not cases.any() or not controls.any():
        return math.nan
    ranks = rankdata(np.r_[risk[cases], risk[controls]])
    n1, n0 = int(cases.sum()), int(controls.sum())
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _breslow_survival(train_time, train_status, train_risk, test_risk, times):
    event_times = np.sort(np.unique(train_time[train_status == 1]))
    increments = []
    exp_risk = np.exp(np.clip(train_risk, -40, 40))
    for event_time in event_times:
        d = np.sum((train_time == event_time) & (train_status == 1))
        denom = exp_risk[train_time >= event_time].sum()
        increments.append(d / denom if denom else 0.0)
    cumulative = np.cumsum(increments)
    base = np.array([cumulative[np.searchsorted(event_times, t, side="right") - 1]
                     if np.searchsorted(event_times, t, side="right") else 0.0 for t in times])
    return np.exp(-np.exp(np.clip(test_risk, -40, 40))[:, None] * base[None, :])


def _within_cohort_c_index(time_v: np.ndarray, status: np.ndarray, risk: np.ndarray,
                           cohort: np.ndarray | None) -> float:
    """Event-weighted mean of the per-cohort concordance.

    Pooled Harrell's C is contaminated when cohorts have different baseline survival rates,
    because samples from a low-risk cohort outlive those from a high-risk one for reasons the
    signature does not model. Comparing only within cohorts removes that component.
    """
    if cohort is None:
        return math.nan
    cohort = np.asarray(cohort, dtype=str)
    values, weights = [], []
    for level in np.unique(cohort):
        mask = cohort == level
        events = float(np.sum(status[mask] == 1))
        value = harrell_c(time_v[mask], status[mask], risk[mask])
        if events and np.isfinite(value):
            values.append(value)
            weights.append(events)
    if not values:
        return math.nan
    return float(np.average(values, weights=weights))


def _metrics(train_time, train_status, train_risk, test_time, test_status, test_risk,
             horizon: float, test_cohort: np.ndarray | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "c_index": harrell_c(test_time, test_status, test_risk),
        "c_index_within_cohort": _within_cohort_c_index(test_time, test_status, test_risk, test_cohort),
        "time_dependent_auc": _binary_auc(test_time, test_status, test_risk, horizon),
        "ipcw_c_index": math.nan,
        "integrated_brier_score": math.nan,
        "metric_warning": None,
    }
    try:
        from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
        from sksurv.util import Surv
        y_train = Surv.from_arrays(train_status.astype(bool), train_time)
        y_test = Surv.from_arrays(test_status.astype(bool), test_time)
        upper = min(float(np.max(train_time)), float(np.max(test_time)))
        tau = np.nextafter(upper, 0)
        result["ipcw_c_index"] = float(concordance_index_ipcw(y_train, y_test, test_risk, tau=tau)[0])
        if float(np.min(test_time)) < horizon < upper:
            result["time_dependent_auc"] = float(cumulative_dynamic_auc(y_train, y_test, test_risk, [horizon])[0][0])
        lower = max(float(np.min(test_time)), float(np.min(train_time[train_status == 1])))
        grid = np.linspace(lower, tau, 20)
        grid = grid[(grid > np.min(test_time)) & (grid < np.max(test_time))]
        if len(grid) >= 2:
            survival = _breslow_survival(train_time, train_status, train_risk, test_risk, grid)
            result["integrated_brier_score"] = float(integrated_brier_score(y_train, y_test, survival, grid))
    except Exception as exc:
        result["metric_warning"] = str(exc)
    return result


def _context(prepared: Prepared, data: AdvancedData, rows: np.ndarray, seed: int,
             cohort_handling: str = "warn") -> dict[str, Any]:
    cohort = _cohort_labels(data, rows)
    return {
        "n_omics": prepared.n_omics, "feature_names": prepared.feature_names,
        "design_names": prepared.design_names, "block_slices": prepared.preprocessor.block_slices,
        # ``strata`` stays covariate-only: the scikit-survival guard in build_estimator must
        # keep rejecting covariates[].mode: strata without tripping on cohort strata, which
        # are applied locally in _fit_method instead.
        "seed": seed, "cohort": cohort, "strata": prepared.strata,
        "has_cohort": cohort is not None, "cohort_handling": cohort_handling,
    }


def _fit_method(name: str, cfg: dict[str, Any], data: AdvancedData, endpoint: dict[str, Any],
                train_rows: np.ndarray, test_rows: np.ndarray, seed: int,
                cohort_handling: str = "warn"):
    train = _prepare(data, train_rows, train_rows)
    test = _prepare(data, train_rows, test_rows, train.preprocessor, train.clinical)
    train_time, train_status = _outcome(data, endpoint, train_rows)
    test_time, test_status = _outcome(data, endpoint, test_rows)
    context = _context(train, data, train_rows, seed, cohort_handling)
    endpoint_index = next(i for i, value in enumerate(data.endpoints) if value["name"] == endpoint["name"])
    context.update({"n_tasks": len(data.endpoints), "task_index": endpoint_index})
    estimator = build_estimator(name, context, cfg)
    task_times = np.column_stack([
        pd.to_numeric(data.phenotype.iloc[train_rows][value["time"]], errors="coerce").to_numpy(float)
        for value in data.endpoints
    ])
    task_status = np.column_stack([
        pd.to_numeric(data.phenotype.iloc[train_rows][value["status"]], errors="coerce").to_numpy(float)
        for value in data.endpoints
    ])
    task_observed = np.isfinite(task_times) & np.isfinite(task_status)
    estimator.fit(train.x, train_time, train_status, masks=train.masks, cohort=context["cohort"],
                  strata=train.method_strata(REGISTRY[name].cohort_stratifies),
                  task_times=np.nan_to_num(task_times, nan=1.0),
                  task_status=np.nan_to_num(task_status, nan=0.0), task_observed=task_observed)
    train_risk = estimator.predict(train.x, masks=train.masks, cohort=context["cohort"])
    test_cohort = _cohort_labels(data, test_rows)
    test_risk = estimator.predict(test.x, masks=test.masks, cohort=test_cohort)
    direction = 1.0 if not np.isfinite(harrell_c(train_time, train_status, train_risk)) or harrell_c(train_time, train_status, train_risk) >= .5 else -1.0
    return estimator, train, test, train_time, train_status, test_time, test_status, train_risk * direction, test_risk * direction, direction


def _candidate_counts(config: dict[str, Any], n_features: int) -> list[int]:
    selection = config.get("selection", {})
    configured = selection.get("feature_counts", [5, 10, 20])
    counts = sorted({max(1, min(int(value), n_features)) for value in configured})
    return counts or [min(10, n_features)]


def _reduced_design(prepared: Prepared, selected: list[int]) -> np.ndarray:
    technical_start = prepared.n_omics
    return np.column_stack([prepared.x[:, selected], prepared.x[:, technical_start:]])


def _parameter_candidates(method_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    base = copy.deepcopy(method_cfg)
    space = base.pop("search_space", {}) or {}
    if not space:
        return [base]
    keys = sorted(space)
    values = [list(space[key]) for key in keys]
    if any(not value for value in values):
        raise ValueError("method search_space values must be non-empty lists")
    candidates = []
    for combination in itertools.product(*values):
        candidate = copy.deepcopy(base)
        for key, value in zip(keys, combination):
            target = candidate
            parts = key.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        candidates.append(candidate)
    return candidates


def _nested_select(data: AdvancedData, endpoint: dict[str, Any], method: str,
                   method_cfg: dict[str, Any], train_rows: np.ndarray,
                   candidates: list[int], folds: int, seed: int,
                   cohort_handling: str = "warn") -> tuple[dict[str, Any], int, dict[str, float]]:
    if len(train_rows) < 6:
        return _parameter_candidates(method_cfg)[0], candidates[0], {f"params=0,k={candidates[0]}": math.nan}
    _, outer_status = _outcome(data, endpoint, train_rows)
    n_splits = min(max(2, folds), len(train_rows))
    split = _stratified_split(train_rows, outer_status, _cohort_labels(data, train_rows), n_splits, seed)
    splits = [(train_rows[train], train_rows[test]) for train, test in split]
    configurations = _parameter_candidates(method_cfg)
    collected: dict[tuple[int, int], list[float]] = {
        (index, count): [] for index in range(len(configurations)) for count in candidates
    }
    for config_index, candidate_cfg in enumerate(configurations):
        for fold_index, (inner_train, inner_test) in enumerate(splits):
            fitted = _fit_method(method, candidate_cfg, data, endpoint, inner_train, inner_test,
                                 seed + config_index * 100 + fold_index, cohort_handling)
            estimator, train, test, train_time, train_status, test_time, test_status, *_ = fitted
            ranking = np.argsort(-np.nan_to_num(np.abs(estimator.importance_), nan=-np.inf), kind="stable")
            for count in candidates:
                selected = ranking[:count].tolist()
                train_design = _reduced_design(train, selected)
                test_design = _reduced_design(test, selected)
                model = LinearCoxModel(l2=.01, n_omics=count).fit(train_design, train_time, train_status,
                                                                   strata=train.refit_strata())
                collected[(config_index, count)].append(
                    harrell_c(test_time, test_status, model.predict(test_design))
                )
    scores = {}
    for (config_index, count), values in collected.items():
        finite = [value for value in values if np.isfinite(value)]
        scores[f"params={config_index},k={count}"] = float(np.mean(finite)) if finite else math.nan
    finite = [(key, value) for key, value in scores.items() if np.isfinite(value)]
    if not finite:
        return configurations[0], candidates[0], scores
    best_value = max(value for _, value in finite)
    best_keys = [key for key, value in finite if value == best_value]
    parsed = sorted((int(key.split(",k=")[0].split("=")[1]), int(key.split(",k=")[1])) for key in best_keys)
    config_index, count = min(parsed, key=lambda value: (value[1], value[0]))
    return configurations[config_index], count, scores


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _fingerprint(cfg: dict[str, Any]) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    digest = hashlib.sha256(json.dumps(clean, sort_keys=True).encode())
    paths = [cfg["phenotype"]["path"]] + [block["path"] for block in cfg["omics"]]
    for method_cfg in (cfg.get("methods", {}).get("config", {}) or {}).values():
        paths.extend(method_cfg[key] for key in ("graph_path", "pathway_path", "similarity_path")
                     if method_cfg.get(key))
    for value in paths:
        stat = Path(value).stat()
        digest.update(f"{value}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _write_endpoint_model(path: Path, method: str, endpoint: dict[str, Any], prepared: Prepared,
                          selected: list[int], reduced, direction: float, method_config: dict[str, Any],
                          native_estimator: Any, cohort: dict[str, Any] | None = None) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    signature_names = [prepared.feature_names[i] for i in selected]
    design_names = signature_names + prepared.design_names[prepared.n_omics:]
    native_artifact = None
    native_state = None
    try:
        with (path / "native_model.pkl").open("wb") as handle:
            pickle.dump(native_estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
        native_artifact = "native_model.pkl"
    except Exception:
        (path / "native_model.pkl").unlink(missing_ok=True)
        try:
            import torch
            if hasattr(native_estimator, "model_") and hasattr(native_estimator.model_, "state_dict"):
                torch.save({"architecture": getattr(native_estimator, "architecture", method),
                            "state_dict": native_estimator.model_.state_dict()}, path / "native_state.pt")
                native_state = "native_state.pt"
        except Exception:
            pass
    manifest = {
        "artifact_version": 1, "method": method, "endpoint": endpoint,
        "method_config": _safe_json(method_config),
        "preprocessor": prepared.preprocessor.to_dict(), "clinical": prepared.clinical.to_dict(),
        "selected_indices": selected, "selected_features": signature_names,
        "design_names": design_names, "coefficients": reduced.coef_.tolist(),
        "native_direction": direction, "schema_version": 2,
        "native_model": native_artifact,
        "native_state": native_state,
        "cohort": cohort or {"cohort_handling": "none", "cohort_warning": None},
        "adaptation_notice": "Canonical PODE adaptation; not an exact upstream research implementation.",
    }
    (path / "manifest.json").write_text(json.dumps(_safe_json(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def _method_config(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    return copy.deepcopy((cfg.get("methods", {}).get("config", {}) or {}).get(name, {}))


def run_advanced_pipeline(config: dict[str, Any] | str | Path, method: str | None = None,
                          *, write_results: bool = True,
                          task_filter: tuple[str, int] | None = None,
                          finalize: bool = True,
                          progress: ProgressCallback | None = None) -> dict[str, Any]:
    cfg = read_advanced_config(config) if not isinstance(config, dict) else copy.deepcopy(config)
    _announce(progress, "Loading schema-v2 signature data")
    data = load_advanced_data(cfg)
    enabled = cfg["methods"]["enabled"]
    method = method or (enabled[0] if len(enabled) == 1 else None)
    if method is None:
        raise ValueError("multiple methods are enabled; use benchmark or specify --method")
    if method not in enabled:
        raise ValueError(f"method {method!r} is not listed in methods.enabled")
    if method not in REGISTRY:
        raise ValueError(f"unknown signature method: {method}")
    method_cfg = _method_config(cfg, method)
    seed = int(cfg.get("execution", {}).get("seed", 1))
    outer_folds = int(cfg.get("evaluation", {}).get("outer_folds", 5))
    inner_folds = int(cfg.get("evaluation", {}).get("inner_folds", 3))
    cohort_validation = str(cfg.get("evaluation", {}).get("cohort_validation", "stratified_kfold"))
    cohort_handling = str(cfg.get("phenotype", {}).get("cohort_handling", "warn"))
    cohort_status = cohort_report(method, data.cohort is not None, cohort_validation)
    if cohort_status["cohort_warning"]:
        _announce(progress, f"{method}: WARNING - {cohort_status['cohort_warning']}")
    started = time.monotonic()
    fold_predictions: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    endpoint_manifests: dict[str, Any] = {}

    output = Path(cfg["output_dir"]) / "python" / method
    task_cache = Path(cfg["_advanced_task_cache"]) if cfg.get("_advanced_task_cache") else None
    _announce(progress, f"{method}: evaluating {len(data.endpoints)} endpoint(s) with {outer_folds} outer fold(s)")
    for endpoint_index, endpoint in enumerate(data.endpoints):
        if task_filter is not None and endpoint["name"] != task_filter[0]:
            continue
        rows = _endpoint_rows(data, endpoint)
        if len(rows) < 5:
            raise ValueError(f"endpoint {endpoint['name']} has fewer than five complete outcomes")
        _announce(progress, f"{method}: endpoint {endpoint['name']} ({endpoint_index + 1}/{len(data.endpoints)})")
        fold_counts: list[int] = []
        fold_configs: list[dict[str, Any]] = []
        folds = list(_folds(data, endpoint, rows, outer_folds, seed + endpoint_index,
                            cohort_validation))
        for fold_index, (train_rows, test_rows, fold_name) in enumerate(folds):
            if task_filter is not None and fold_index != task_filter[1]:
                continue
            cache_path = task_cache / method / endpoint["name"] / f"fold-{fold_index}.json" if task_cache else None
            action = "reusing cached results" if cache_path and cache_path.is_file() else "fitting"
            _announce(progress, f"{method}: {endpoint['name']} fold {fold_index + 1}/{len(folds)} ({action})")
            if cache_path and cache_path.is_file():
                cached = json.loads(cache_path.read_text())
                fold_counts.append(int(cached["feature_count"]))
                fold_configs.append(cached.get("selected_parameters", _parameter_candidates(method_cfg)[0]))
                fold_metrics.extend(cached["fold_metrics"])
                fold_predictions.extend(cached["fold_predictions"])
                continue
            metric_start, prediction_start = len(fold_metrics), len(fold_predictions)
            available_features = sum(len(block.values.columns) for block in data.blocks.values())
            candidates = _candidate_counts(cfg, available_features)
            chosen_cfg, chosen, inner_scores = _nested_select(
                data, endpoint, method, method_cfg, train_rows, candidates,
                inner_folds, seed + endpoint_index * 1000 + fold_index, cohort_handling,
            )
            fold_configs.append(chosen_cfg)
            result = _fit_method(method, chosen_cfg, data, endpoint, train_rows, test_rows,
                                 seed + endpoint_index * 1000 + fold_index, cohort_handling)
            estimator, train, test, train_time, train_status, test_time, test_status, train_risk, test_risk, direction = result
            importance = np.asarray(estimator.importance_, float)
            rank = np.argsort(-np.nan_to_num(np.abs(importance), nan=-np.inf), kind="stable")
            fold_counts.append(chosen)
            selected = rank[:chosen].tolist()
            reduced_train = _reduced_design(train, selected)
            reduced_test = _reduced_design(test, selected)
            reduced = LinearCoxModel(l2=.01, n_omics=chosen).fit(reduced_train, train_time, train_status,
                                                                  strata=train.refit_strata())
            reduced_train_risk = reduced.predict(reduced_train)
            reduced_test_risk = reduced.predict(reduced_test)
            horizon = float(endpoint.get("prediction_time", np.median(train_time)))
            test_cohort = _cohort_labels(data, test_rows)
            native_metrics = _metrics(train_time, train_status, train_risk,
                                      test_time, test_status, test_risk, horizon, test_cohort)
            reduced_metrics = _metrics(train_time, train_status, reduced_train_risk,
                                       test_time, test_status, reduced_test_risk, horizon, test_cohort)
            clinical_width = train.x.shape[1] - train.n_omics - train.masks.shape[1]
            if clinical_width:
                clinical_train = train.x[:, -clinical_width:]
                clinical_test = test.x[:, -clinical_width:]
                clinical_model = LinearCoxModel(l2=.01, n_omics=0).fit(clinical_train, train_time, train_status,
                                                                        strata=train.refit_strata())
                clinical_train_risk = clinical_model.predict(clinical_train)
                clinical_test_risk = clinical_model.predict(clinical_test)
                clinical_metrics = _metrics(train_time, train_status, clinical_train_risk,
                                            test_time, test_status, clinical_test_risk, horizon, test_cohort)
            else:
                clinical_metrics = {"c_index": math.nan}
            for model_type, values in (("native", native_metrics), ("reduced_signature", reduced_metrics)):
                fold_metrics.append({
                    "method": method, "endpoint": endpoint["name"], "fold": fold_name,
                    "fold_index": fold_index, "model_type": model_type, "feature_count": chosen,
                    "cohort_handling": cohort_status["cohort_handling"],
                    "clinical_c_index": clinical_metrics.get("c_index"),
                    "incremental_c_index": values.get("c_index", math.nan) - clinical_metrics.get("c_index", math.nan),
                    "inner_scores": json.dumps(_safe_json(inner_scores), sort_keys=True), **values,
                    "selected_parameters": json.dumps(_safe_json(chosen_cfg), sort_keys=True),
                })
            for local, global_row in enumerate(test_rows):
                for model_type, risk in (("native", test_risk[local]), ("reduced_signature", reduced_test_risk[local])):
                    fold_predictions.append({
                        "sample_id": data.sample_ids[global_row], "endpoint": endpoint["name"],
                        "fold": fold_name, "model_type": model_type, "risk_score": float(risk),
                        "time": float(test_time[local]), "status": int(test_status[local]),
                    })

            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "method": method, "endpoint": endpoint["name"], "fold_index": fold_index,
                    "feature_count": chosen,
                    "selected_parameters": chosen_cfg,
                    "fold_metrics": _safe_json(fold_metrics[metric_start:]),
                    "fold_predictions": _safe_json(fold_predictions[prediction_start:]),
                }
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                os.replace(temporary, cache_path)

        # Full-data refit. The modal unbiased fold choice is used; ties choose the smaller signature.
        if not finalize:
            continue
        if not fold_counts:
            raise ValueError(f"no completed folds are available for endpoint {endpoint['name']}")
        _announce(progress, f"{method}: refitting final model for {endpoint['name']}")
        chosen = min(set(fold_counts), key=lambda value: (-fold_counts.count(value), value))
        serialized_configs = [json.dumps(_safe_json(value), sort_keys=True) for value in fold_configs]
        selected_serialized = min(set(serialized_configs),
                                  key=lambda value: (-serialized_configs.count(value), value))
        final_method_cfg = json.loads(selected_serialized)
        prepared = _prepare(data, rows, rows)
        time_v, status = _outcome(data, endpoint, rows)
        context = _context(prepared, data, rows, seed + endpoint_index, cohort_handling)
        context.update({"n_tasks": len(data.endpoints), "task_index": endpoint_index})
        estimator = build_estimator(method, context, final_method_cfg)
        task_times = np.column_stack([
            pd.to_numeric(data.phenotype.iloc[rows][value["time"]], errors="coerce").to_numpy(float)
            for value in data.endpoints
        ])
        task_status = np.column_stack([
            pd.to_numeric(data.phenotype.iloc[rows][value["status"]], errors="coerce").to_numpy(float)
            for value in data.endpoints
        ])
        task_observed = np.isfinite(task_times) & np.isfinite(task_status)
        estimator.fit(prepared.x, time_v, status, masks=prepared.masks, cohort=context["cohort"],
                      strata=prepared.method_strata(REGISTRY[method].cohort_stratifies),
                      task_times=np.nan_to_num(task_times, nan=1.0),
                      task_status=np.nan_to_num(task_status, nan=0.0), task_observed=task_observed)
        native_risk = estimator.predict(prepared.x, masks=prepared.masks, cohort=context["cohort"])
        direction = 1.0 if not np.isfinite(harrell_c(time_v, status, native_risk)) or harrell_c(time_v, status, native_risk) >= .5 else -1.0
        native_risk *= direction
        ranking_records = rank_importance(prepared.feature_names, estimator.importance_)
        selected_indices = [prepared.feature_names.index(record["feature"]) for record in ranking_records[:chosen]]
        reduced_design = _reduced_design(prepared, selected_indices)
        reduced = LinearCoxModel(l2=.01, n_omics=chosen).fit(reduced_design, time_v, status,
                                                              strata=prepared.refit_strata())
        reduced_risk = reduced.predict(reduced_design)
        for record in ranking_records:
            rankings.append({"method": method, "endpoint": endpoint["name"], **record})
        for rank_value, feature_index in enumerate(selected_indices, 1):
            selected_rows.append({
                "method": method, "endpoint": endpoint["name"], "feature": prepared.feature_names[feature_index],
                "rank": rank_value, "importance": float(estimator.importance_[feature_index]),
            })
        design_names = [prepared.feature_names[i] for i in selected_indices] + prepared.design_names[prepared.n_omics:]
        for name, coefficient in zip(design_names, reduced.coef_):
            coefficient_rows.append({
                "method": method, "endpoint": endpoint["name"], "term": name,
                "coefficient": float(coefficient), "signature_feature": name in prepared.feature_names,
            })
        for local, global_row in enumerate(rows):
            score_rows.extend([
                {"sample_id": data.sample_ids[global_row], "endpoint": endpoint["name"],
                 "model_type": "native", "risk_score": float(native_risk[local])},
                {"sample_id": data.sample_ids[global_row], "endpoint": endpoint["name"],
                 "model_type": "reduced_signature", "risk_score": float(reduced_risk[local])},
            ])
        if hasattr(estimator, "selection_frequency_"):
            for feature, frequency in zip(prepared.feature_names, estimator.selection_frequency_):
                stability_rows.append({"method": method, "endpoint": endpoint["name"],
                                       "feature": feature, "selection_frequency": float(frequency)})
        if hasattr(estimator, "study_coefficients_"):
            for cohort, coefficients in estimator.study_coefficients_.items():
                for feature, coefficient in zip(prepared.feature_names, coefficients):
                    coefficient_rows.append({"method": method, "endpoint": endpoint["name"],
                                             "term": feature, "cohort": cohort,
                                             "coefficient": float(coefficient), "signature_feature": True})
        if write_results:
            endpoint_manifests[endpoint["name"]] = _write_endpoint_model(
                output / "model" / endpoint["name"], method, endpoint, prepared,
                selected_indices, reduced, direction, final_method_cfg, estimator,
                cohort_status,
            )
            _announce(progress, f"{method}: wrote model for {endpoint['name']}")

    frames = {
        "fold_predictions": pd.DataFrame(fold_predictions),
        "fold_metrics": pd.DataFrame(fold_metrics),
        "feature_rankings": pd.DataFrame(rankings),
        "selected_features": pd.DataFrame(selected_rows),
        "signature_coefficients": pd.DataFrame(coefficient_rows),
        "risk_scores": pd.DataFrame(score_rows),
        "stability_frequencies": pd.DataFrame(stability_rows),
    }
    elapsed = time.monotonic() - started
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "scikit-survival"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    summary = {
        "analysis_id": cfg["analysis_id"], "method": method,
        "maturity": REGISTRY[method].maturity, "schema_version": 2,
        "engine": "python", "fingerprint": _fingerprint(cfg), "seed": seed,
        "elapsed_seconds": elapsed, "python": sys.version, "platform": platform.platform(),
        "dependencies": versions,
        "adaptation_notice": "Canonical PODE adaptation; not exact upstream numerical parity.",
        "endpoints": list(endpoint_manifests),
        **cohort_status,
    }
    if write_results:
        _announce(progress, f"{method}: writing results")
        output.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            if not frame.empty:
                frame.to_csv(output / f"{name}.csv", index=False)
        (output / "result.json").write_text(json.dumps(_safe_json(summary), indent=2) + "\n", encoding="utf-8")
        root_manifest = {"artifact_version": 1, "method": method,
                         "endpoints": {name: f"{name}/manifest.json" for name in endpoint_manifests}}
        (output / "model" / "manifest.json").write_text(json.dumps(root_manifest, indent=2) + "\n", encoding="utf-8")
        summary["output_dir"] = str(output)
        _announce(progress, f"{method}: complete ({elapsed:.1f}s)")
    summary.update(frames)
    return summary


def _preset_config(cfg: dict[str, Any], preset: str) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    if preset not in {"quick", "balanced", "exhaustive"}:
        raise ValueError("preset must be quick, balanced, or exhaustive")
    if preset == "quick":
        cfg.setdefault("evaluation", {})["outer_folds"] = min(3, int(cfg.get("evaluation", {}).get("outer_folds", 3)))
        cfg["evaluation"]["inner_folds"] = 2
        cfg.setdefault("selection", {})["feature_counts"] = cfg.get("selection", {}).get("feature_counts", [5, 10])[:2]
        for name in cfg["methods"]["enabled"]:
            method_cfg = cfg["methods"].setdefault("config", {}).setdefault(name, {})
            if method_cfg.get("search_space"):
                method_cfg["search_space"] = {
                    key: list(values)[:2] for key, values in method_cfg["search_space"].items()
                }
            if REGISTRY.get(name) and REGISTRY[name].family == "deep":
                method_cfg["epochs"] = min(int(method_cfg.get("epochs", 20)), 20)
            if name == "stability_coxboost":
                method_cfg["repetitions"] = min(int(method_cfg.get("repetitions", 8)), 8)
            if name in {"random_survival_forest", "gradient_boosted_survival"}:
                method_cfg.setdefault("params", {})["n_estimators"] = 30
    elif preset == "balanced":
        for name in cfg["methods"]["enabled"]:
            method_cfg = cfg["methods"].setdefault("config", {}).setdefault(name, {})
            if method_cfg.get("search_space"):
                method_cfg["search_space"] = {
                    key: list(values)[:3] for key, values in method_cfg["search_space"].items()
                }
            if REGISTRY.get(name) and REGISTRY[name].family == "deep":
                method_cfg["epochs"] = min(int(method_cfg.get("epochs", 80)), 80)
            if name == "stability_coxboost":
                method_cfg["repetitions"] = min(int(method_cfg.get("repetitions", 20)), 20)
            if name in {"random_survival_forest", "gradient_boosted_survival"}:
                method_cfg.setdefault("params", {})["n_estimators"] = min(
                    int(method_cfg.get("params", {}).get("n_estimators", 100)), 100
                )
    return cfg


# Every fold-level metric that is comparable across methods, mapped to its direction.
# Ranking, the tidy metrics table and the report all read this so they cannot disagree.
_BENCHMARK_METRICS: dict[str, bool] = {
    "c_index": True,
    "ipcw_c_index": True,
    "c_index_within_cohort": True,
    "time_dependent_auc": True,
    "integrated_brier_score": False,
    "clinical_c_index": True,
    "incremental_c_index": True,
}

# evaluation.primary_metric names the metric in user terms; these are its column names.
_PRIMARY_METRIC_ALIASES: dict[str, str] = {
    "harrell_c": "c_index",
    "ipcw_c": "ipcw_c_index",
    "within_cohort_c": "c_index_within_cohort",
    "auc": "time_dependent_auc",
    "brier": "integrated_brier_score",
    **{name: name for name in _BENCHMARK_METRICS},
}

_PAIRED_METRICS = ("c_index", "ipcw_c_index")
_SUMMARY_KEYS = ("mean", "sd", "se", "ci_low", "ci_high", "median", "min", "max")


def _summarise(values: Any) -> dict[str, Any]:
    """Dispersion summary over the finite values only; never raises on empty input."""
    array = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").to_numpy(float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, **{key: math.nan for key in _SUMMARY_KEYS}}
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if array.size > 1 else math.nan
    se = sd / math.sqrt(array.size) if np.isfinite(sd) else math.nan
    half = 1.96 * se if np.isfinite(se) else math.nan
    return {"n": int(array.size), "mean": mean, "sd": sd, "se": se,
            "ci_low": mean - half if np.isfinite(half) else math.nan,
            "ci_high": mean + half if np.isfinite(half) else math.nan,
            "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max())}


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame[name] if name in frame.columns else pd.Series(dtype="float64")


def _primary_metric(cfg: dict[str, Any]) -> str:
    requested = str(cfg.get("evaluation", {}).get("primary_metric", "harrell_c"))
    return _PRIMARY_METRIC_ALIASES.get(requested, "c_index")


def _leaderboard_metrics(metrics: pd.DataFrame) -> dict[str, Any]:
    """Per-method leaderboard columns, pooled over endpoints, from the reduced signature."""
    reduced = metrics[metrics["model_type"] == "reduced_signature"]
    native = metrics[metrics["model_type"] == "native"]
    endpoints = sorted(reduced["endpoint"].astype(str).unique().tolist()) if "endpoint" in reduced else []
    row: dict[str, Any] = {"n_folds": int(len(reduced)), "n_endpoints": len(endpoints),
                           "endpoints": ",".join(endpoints)}
    for metric in _BENCHMARK_METRICS:
        stats = _summarise(_column(reduced, metric))
        row[f"mean_{metric}"] = stats["mean"]
        row[f"sd_{metric}"] = stats["sd"]
        row[f"ci_low_{metric}"] = stats["ci_low"]
        row[f"ci_high_{metric}"] = stats["ci_high"]
    for metric in _PAIRED_METRICS:
        row[f"native_mean_{metric}"] = _summarise(_column(native, metric))["mean"]
    counts = pd.to_numeric(_column(reduced, "feature_count"), errors="coerce").dropna()
    row["median_feature_count"] = float(counts.median()) if len(counts) else math.nan
    row["feature_count_range"] = f"{int(counts.min())}-{int(counts.max())}" if len(counts) else None
    row["n_metric_warnings"] = int(_column(reduced, "metric_warning").notna().sum())
    chosen = _column(reduced, "selected_parameters").dropna()
    row["selected_parameters_modal"] = str(chosen.mode().iloc[0]) if len(chosen) else None
    return row


def _rank_leaderboard(rows: list[dict[str, Any]], metric: str) -> list[str]:
    """Assign 1..n over completed rows on the primary metric; return the ranked names."""
    column = f"mean_{metric}"
    higher_is_better = _BENCHMARK_METRICS.get(metric, True)
    scored = [row for row in rows
              if row.get("status") == "completed" and np.isfinite(row.get(column, math.nan))]
    scored.sort(key=lambda row: (-row[column] if higher_is_better else row[column], row["method"]))
    for position, row in enumerate(scored, 1):
        row["rank"] = position
    for row in rows:
        row.setdefault("rank", math.nan)
    return [row["method"] for row in scored]


def _benchmark_metrics_frame(results: dict[str, Any]) -> pd.DataFrame:
    """Tidy method x endpoint x model_type x metric table -- the per-endpoint detail."""
    rows = []
    for method, result in results.items():
        metrics = result.get("fold_metrics")
        if metrics is None or metrics.empty:
            continue
        for (endpoint, model_type), block in metrics.groupby(["endpoint", "model_type"], sort=True):
            for metric, higher_is_better in _BENCHMARK_METRICS.items():
                stats = _summarise(_column(block, metric))
                if not stats["n"]:
                    continue
                rows.append({"method": method, "endpoint": str(endpoint), "model_type": str(model_type),
                             "metric": metric, "higher_is_better": higher_is_better,
                             "n_folds": stats["n"],
                             **{key: stats[key] for key in _SUMMARY_KEYS}})
    return pd.DataFrame(rows)


def _pooled_fold_metrics(results: dict[str, Any]) -> pd.DataFrame:
    """Every method's per-fold rows in one table. inner_scores is dropped -- it is bulky
    and is already kept in full in each per-method directory."""
    frames = [result["fold_metrics"] for result in results.values()
              if result.get("fold_metrics") is not None and not result["fold_metrics"].empty]
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, ignore_index=True, sort=False)
    return pooled.drop(columns=[name for name in ("inner_scores",) if name in pooled.columns])


def _paired_test(diff: np.ndarray) -> tuple[float, float]:
    if diff.size < 3 or not np.any(diff):
        return math.nan, math.nan
    try:
        result = wilcoxon(diff)
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return math.nan, math.nan


def _method_comparison_frame(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Paired per-fold differences between methods. Valid because _folds is seeded from
    the endpoint alone, so every method in a benchmark sees identical splits."""
    if fold_metrics.empty or "model_type" not in fold_metrics.columns:
        return pd.DataFrame()
    reduced = fold_metrics[fold_metrics["model_type"] == "reduced_signature"]
    rows = []
    for endpoint, block in reduced.groupby("endpoint", sort=True):
        for metric in _PAIRED_METRICS:
            if metric not in block.columns:
                continue
            table = block.pivot_table(index="fold_index", columns="method", values=metric, aggfunc="mean")
            for first, second in itertools.permutations(sorted(table.columns), 2):
                paired = table[[first, second]].dropna()
                if paired.empty:
                    continue
                diff = (paired[first] - paired[second]).to_numpy(float)
                statistic, p_value = _paired_test(diff)
                rows.append({
                    "endpoint": str(endpoint), "method_a": first, "method_b": second, "metric": metric,
                    "n_pairs": int(diff.size), "mean_diff": float(diff.mean()),
                    "sd_diff": float(diff.std(ddof=1)) if diff.size > 1 else math.nan,
                    "wins_a": int((diff > 0).sum()), "wins_b": int((diff < 0).sum()),
                    "ties": int((diff == 0).sum()),
                    "wilcoxon_stat": statistic, "wilcoxon_p": p_value,
                })
    return pd.DataFrame(rows)


def _truncate_names(names: list[str], limit: int = 20) -> str:
    if len(names) <= limit:
        return ",".join(names)
    return ",".join(names[:limit]) + ",..."


def _signature_overlap_frame(results: dict[str, Any]) -> pd.DataFrame:
    """Jaccard agreement between each method pair's selected features, per endpoint."""
    features: dict[tuple[str, str], set[str]] = {}
    for method, result in results.items():
        frame = result.get("selected_features")
        if frame is None or frame.empty:
            continue
        for endpoint, block in frame.groupby("endpoint", sort=True):
            features[(str(endpoint), method)] = set(block["feature"].astype(str))
    rows = []
    for endpoint in sorted({endpoint for endpoint, _ in features}):
        methods = sorted(method for key, method in features if key == endpoint)
        for first, second in itertools.permutations(methods, 2):
            left, right = features[(endpoint, first)], features[(endpoint, second)]
            shared = sorted(left & right)
            union = len(left | right)
            rows.append({"endpoint": endpoint, "method_a": first, "method_b": second,
                         "size_a": len(left), "size_b": len(right), "n_shared": len(shared),
                         "jaccard": len(shared) / union if union else math.nan,
                         "shared_features": _truncate_names(shared)})
    return pd.DataFrame(rows)


def _format_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Pipe table without requiring tabulate, which is not a declared dependency."""
    if frame.empty:
        return "_no rows_\n"
    columns = [str(name) for name in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for record in frame.to_dict(orient="records"):
        lines.append("| " + " | ".join(_format_cell(record[name]) for name in frame.columns) + " |")
    return "\n".join(lines) + "\n"


def _interval(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"mean_{metric}", math.nan)
    if not np.isfinite(mean):
        return "-"
    low, high = row.get(f"ci_low_{metric}", math.nan), row.get(f"ci_high_{metric}", math.nan)
    if not (np.isfinite(low) and np.isfinite(high)):
        return f"{mean:.3f}"
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def _benchmark_report(cfg: dict[str, Any], preset: str, rows: list[dict[str, Any]],
                      frames: dict[str, pd.DataFrame], metric: str, fingerprint: str) -> str:
    fold_scheme = str(cfg.get("evaluation", {}).get("cohort_validation", "stratified_kfold"))
    completed = [row for row in rows if row.get("status") == "completed"]
    completed.sort(key=lambda row: row.get("rank") if np.isfinite(row.get("rank", math.nan)) else math.inf)
    parts = [
        f"# Benchmark report: {cfg['analysis_id']}",
        "",
        f"- Preset: `{preset}`",
        f"- Primary metric: `{metric}`",
        f"- Outer validation: `{fold_scheme}`",
        f"- Methods: {len(completed)} completed of {len(rows)} configured",
        f"- Fingerprint: `{fingerprint}`",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Ranking (reduced signature, pooled over endpoints)",
        "",
    ]
    ranked = pd.DataFrame([{
        "rank": row.get("rank"), "method": row["method"],
        "c_index [95% CI]": _interval(row, "c_index"),
        "ipcw_c": _format_cell(row.get("mean_ipcw_c_index")),
        "within_cohort_c": _format_cell(row.get("mean_c_index_within_cohort")),
        "auc": _format_cell(row.get("mean_time_dependent_auc")),
        "brier": _format_cell(row.get("mean_integrated_brier_score")),
        "incremental_c": _format_cell(row.get("mean_incremental_c_index")),
        "native_c": _format_cell(row.get("native_mean_c_index")),
        "features": _format_cell(row.get("median_feature_count")),
        "folds": row.get("n_folds"), "seconds": _format_cell(row.get("elapsed_seconds")),
    } for row in completed])
    parts.append(_markdown_table(ranked))

    tidy = frames.get("benchmark_metrics", pd.DataFrame())
    endpoints = sorted(tidy["endpoint"].unique().tolist()) if not tidy.empty else []
    if len(endpoints) > 1:
        parts += ["", "## Per endpoint (reduced signature)", ""]
        for endpoint in endpoints:
            block = tidy[(tidy["endpoint"] == endpoint) & (tidy["model_type"] == "reduced_signature")
                         & (tidy["metric"] == metric)].sort_values("mean", ascending=not _BENCHMARK_METRICS[metric])
            parts += [f"### {endpoint}", "",
                      _markdown_table(block[["method", "n_folds", "mean", "sd", "ci_low", "ci_high"]]), ""]

    other = [row for row in rows if row.get("status") != "completed"]
    if other:
        parts += ["", "## Not completed", "",
                  _markdown_table(pd.DataFrame([{"method": row["method"], "status": row["status"],
                                                 "reason": row.get("reason")} for row in other]))]

    comparison = frames.get("method_comparison", pd.DataFrame())
    if not comparison.empty:
        top = comparison[comparison["metric"] == metric].copy() if metric in _PAIRED_METRICS else comparison.copy()
        # The CSV keeps both orderings so it can be joined on (method_a, method_b);
        # the report only needs one row per unordered pair.
        top = top[top["method_a"] < top["method_b"]]
        if not top.empty:
            top = top.reindex(top["mean_diff"].abs().sort_values(ascending=False).index).head(15)
            parts += ["", f"## Largest paired differences ({metric})", "",
                      _markdown_table(top[["endpoint", "method_a", "method_b", "n_pairs",
                                           "mean_diff", "wins_a", "wins_b", "wilcoxon_p"]]),
                      "",
                      f"With {int(top['n_pairs'].max())} paired folds the smallest attainable two-sided "
                      "Wilcoxon p is bounded from below; read `n_pairs` before reading `wilcoxon_p`."]

    notes = [f"- `{row['method']}`: {row['cohort_warning']}" for row in completed if row.get("cohort_warning")]
    notes += [f"- `{row['method']}`: {row['n_metric_warnings']} fold(s) fell back to unweighted metrics"
              for row in completed if row.get("n_metric_warnings")]
    if notes:
        parts += ["", "## Warnings", ""] + notes
    return "\n".join(parts) + "\n"


def benchmark_methods(config: dict[str, Any] | str | Path, preset: str = "quick",
                      *, progress: ProgressCallback | None = None) -> dict[str, Any]:
    cfg = read_advanced_config(config) if not isinstance(config, dict) else copy.deepcopy(config)
    cfg = _preset_config(cfg, preset)
    catalog = {row["name"]: row for row in method_catalog()}
    methods = list(cfg["methods"]["enabled"])
    if preset == "quick":
        methods = [name for name in methods if name in REGISTRY and REGISTRY[name].maturity == "stable"]
    _announce(progress, f"Benchmark preset '{preset}': {len(methods)} method(s) selected")
    rows = []
    results = {}
    total_methods = len(cfg["methods"]["enabled"])
    for method_index, name in enumerate(cfg["methods"]["enabled"], 1):
        if name not in methods:
            _announce(progress, f"Benchmark {method_index}/{total_methods}: skipping {name} (not in preset)")
            rows.append({"method": name, "status": "skipped", "reason": "not included in quick preset"})
            continue
        if name not in catalog:
            _announce(progress, f"Benchmark {method_index}/{total_methods}: skipping unknown method {name}")
            rows.append({"method": name, "status": "failed", "reason": "unknown method"})
            continue
        if not catalog[name]["available"]:
            _announce(progress, f"Benchmark {method_index}/{total_methods}: {name} unavailable")
            rows.append({"method": name, "status": "unavailable", "reason": catalog[name]["unavailable_reason"]})
            continue
        _announce(progress, f"Benchmark {method_index}/{total_methods}: starting {name}")
        method_started = time.monotonic()
        try:
            result = run_advanced_pipeline(cfg, name, progress=progress)
            summary = _leaderboard_metrics(result["fold_metrics"])
            rows.append({
                "method": name, "status": "completed", "reason": None,
                "maturity": REGISTRY[name].maturity, "preset": preset,
                "mean_c_index": summary["mean_c_index"],
                "mean_ipcw_c_index": summary["mean_ipcw_c_index"],
                "mean_c_index_within_cohort": summary["mean_c_index_within_cohort"],
                "cohort_handling": result["cohort_handling"],
                "cohort_warning": result["cohort_warning"],
                "elapsed_seconds": result["elapsed_seconds"],
                **summary,
            })
            results[name] = result
            _announce(progress, f"Benchmark {method_index}/{total_methods}: {name} complete ({time.monotonic() - method_started:.1f}s)")
        except CohortUnsupportedError as exc:
            _announce(progress, f"Benchmark {method_index}/{total_methods}: skipping {name} ({exc})")
            rows.append({"method": name, "status": "skipped", "reason": str(exc),
                         "maturity": REGISTRY[name].maturity, "preset": preset,
                         "cohort_handling": "ignored", "cohort_warning": str(exc)})
        except Exception as exc:
            _announce(progress, f"Benchmark {method_index}/{total_methods}: {name} failed: {exc}")
            rows.append({"method": name, "status": "failed", "reason": str(exc),
                         "maturity": REGISTRY[name].maturity, "preset": preset})
    metric = _primary_metric(cfg)
    ranking = _rank_leaderboard(rows, metric)
    pooled = _pooled_fold_metrics(results)
    leaderboard = pd.DataFrame(rows)
    frames = {
        "leaderboard": leaderboard,
        "benchmark_metrics": _benchmark_metrics_frame(results),
        "fold_metrics": pooled,
        "method_comparison": _method_comparison_frame(pooled),
        "signature_overlap": _signature_overlap_frame(results),
    }
    out = Path(cfg["output_dir"]) / "python" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for name, frame in frames.items():
        if frame.empty and name != "leaderboard":
            continue
        frame.to_csv(out / f"{name}.csv", index=False)
        artifacts.append(f"{name}.csv")
        _announce(progress, f"Benchmark: wrote {name}.csv ({len(frame)} row(s))")
    fingerprint = _fingerprint(cfg)
    report = _benchmark_report(cfg, preset, rows, frames, metric, fingerprint)
    (out / "benchmark_report.md").write_text(report, encoding="utf-8")
    artifacts.append("benchmark_report.md")
    _announce(progress, "Benchmark: wrote benchmark_report.md")
    manifest = {"analysis_id": cfg["analysis_id"], "schema_version": 2, "preset": preset,
                "primary_metric": metric, "ranking": ranking,
                "endpoints": [endpoint["name"] for endpoint in cfg["endpoints"]],
                "fold_scheme": str(cfg.get("evaluation", {}).get("cohort_validation", "stratified_kfold")),
                "artifacts": artifacts + ["result.json"],
                "methods": _safe_json(rows), "fingerprint": fingerprint}
    (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _announce(progress, f"Benchmark complete; leaderboard written to {out} "
                        f"with {len(artifacts)} companion artifact(s)")
    return {**frames, "results": results, "output_dir": str(out),
            "primary_metric": metric, "ranking": ranking, "artifacts": artifacts + ["result.json"]}


def predict_signature(model_dir: str | Path, config: dict[str, Any] | str | Path,
                      output_dir: str | Path | None = None,
                      *, progress: ProgressCallback | None = None) -> pd.DataFrame:
    cfg = read_advanced_config(config) if not isinstance(config, dict) else config
    _announce(progress, "Loading schema-v2 data for prediction")
    data = load_advanced_data(cfg)
    model_dir = Path(model_dir)
    root = json.loads((model_dir / "manifest.json").read_text())
    rows = []
    for endpoint_index, (endpoint_name, relative) in enumerate(root["endpoints"].items(), 1):
        _announce(progress, f"Predicting endpoint {endpoint_name} ({endpoint_index}/{len(root['endpoints'])})")
        manifest = json.loads((model_dir / relative).read_text())
        preprocessor = Preprocessor.from_dict(manifest["preprocessor"])
        clinical = ClinicalPreprocessor.from_dict(manifest["clinical"])
        missing_blocks = set(preprocessor.block_names) - set(data.blocks)
        if missing_blocks:
            raise ValueError(f"prediction config is missing omics blocks: {', '.join(sorted(missing_blocks))}")
        all_rows = np.arange(len(data.sample_ids))
        prepared = _prepare(data, all_rows, all_rows, preprocessor, clinical)
        selected = [prepared.feature_names.index(name) for name in manifest["selected_features"]]
        design = _reduced_design(prepared, selected)
        coefficients = np.asarray(manifest["coefficients"], float)
        if design.shape[1] != len(coefficients):
            raise ValueError(f"model design mismatch for endpoint {endpoint_name}")
        risk = design @ coefficients
        for sample_id, value in zip(data.sample_ids, risk):
            rows.append({"sample_id": sample_id, "endpoint": endpoint_name,
                         "model_type": "reduced_signature", "risk_score": float(value)})
        if manifest.get("native_model"):
            with (model_dir / Path(relative).parent / manifest["native_model"]).open("rb") as handle:
                native = pickle.load(handle)
            cohort = data.cohort.astype(str).to_numpy() if data.cohort is not None else None
            native_risk = native.predict(prepared.x, masks=prepared.masks, cohort=cohort)
            native_risk = np.asarray(native_risk, float) * float(manifest.get("native_direction", 1.0))
            for sample_id, value in zip(data.sample_ids, native_risk):
                rows.append({"sample_id": sample_id, "endpoint": endpoint_name,
                             "model_type": "native", "risk_score": float(value)})
    frame = pd.DataFrame(rows)
    destination = Path(output_dir) if output_dir else Path(cfg["output_dir"]) / "python" / "predictions"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "risk_scores.csv", index=False)
    _announce(progress, f"Prediction complete; results written to {destination}")
    return frame
