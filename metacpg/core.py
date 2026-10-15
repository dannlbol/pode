from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.stats import norm, rankdata, spearmanr


SCHEMA_VERSION = 1
_MEBIBYTE = 1024 ** 2
_DEFAULT_HEARTBEAT_SECONDS = 300.0
_WEIGHT_REFIT_MODES = ("covariate_aware", "legacy_unadjusted")


def _weight_refit_mode(cfg: dict[str, Any]) -> str:
    final_model = cfg.get("final_model") or {}
    return str(final_model.get("weight_refit", "covariate_aware"))


def _weight_refit_explicit(cfg: dict[str, Any]) -> bool:
    final_model = cfg.get("final_model")
    return bool(cfg.get("_weight_refit_explicit", False)) or (
        isinstance(final_model, dict) and "weight_refit" in final_model)


def _validate_weight_refit(cfg: dict[str, Any]) -> str:
    final_model = cfg.get("final_model")
    if final_model is not None and not isinstance(final_model, dict):
        raise ValueError("final_model must be an object")
    mode = _weight_refit_mode(cfg)
    if mode not in _WEIGHT_REFIT_MODES:
        raise ValueError(f"final_model.weight_refit must be one of {_WEIGHT_REFIT_MODES}")
    return mode


def parse_memory_budget(value: str | int | float) -> int:
    """Parse a positive byte value or a binary B/K/M/G/T size string."""
    if isinstance(value, bool):
        raise ValueError("memory budget must be a positive size")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError("memory budget must be positive")
        return int(value)
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(?:i?B)?\s*", str(value), re.IGNORECASE)
    if match is None:
        raise ValueError("memory budget must use B, K, M, G, or T units (for example 8G)")
    magnitude = float(match.group(1))
    if magnitude <= 0:
        raise ValueError("memory budget must be positive")
    unit = match.group(2).upper()
    multiplier = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}[unit]
    return int(magnitude * multiplier)


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "unavailable"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def _process_memory(pid: int, proc_root: Path = Path("/proc")) -> tuple[int | None, str]:
    """Return PSS when available, otherwise RSS, for a Linux process."""
    proc = proc_root / str(pid)
    try:
        for line in (proc / "smaps_rollup").read_text(encoding="utf-8").splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024, "PSS"
    except (FileNotFoundError, PermissionError, OSError):
        pass
    try:
        for line in (proc / "status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024, "RSS"
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return None, "unavailable"


def heartbeat_seconds_from_config(cfg: dict[str, Any]) -> float:
    value = cfg.get("execution", {}).get("heartbeat_seconds", _DEFAULT_HEARTBEAT_SECONDS)
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution.heartbeat_seconds must be a non-negative number") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("execution.heartbeat_seconds must be a non-negative number")
    return seconds


def progress_enabled_from_config(cfg: dict[str, Any]) -> bool:
    value = cfg.get("execution", {}).get("progress", True)
    if not isinstance(value, bool):
        raise ValueError("execution.progress must be true or false")
    return value


def read_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    final_model = cfg.get("final_model")
    cfg["_weight_refit_explicit"] = isinstance(final_model, dict) and "weight_refit" in final_model
    _validate_weight_refit(cfg)
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    for section, key in (("matrix", "path"), ("phenotype", "path"), ("annotation", "path")):
        if section in cfg and cfg[section] and cfg[section].get(key):
            p = Path(cfg[section][key]).expanduser()
            if not p.is_absolute():
                p = path.parent / p
            cfg[section][key] = str(p.resolve())
    out = Path(cfg.get("output_dir", "results")).expanduser()
    if not out.is_absolute():
        out = path.parent / out
    cfg["output_dir"] = str(out.resolve())
    return cfg


def beta_to_m(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.nanmin(values) <= 0 or np.nanmax(values) >= 1:
        raise ValueError("beta values must be strictly between 0 and 1")
    return np.log2(values / (1.0 - values))


@dataclass
class InputData:
    matrix: pd.DataFrame
    phenotype: pd.DataFrame
    endpoints: list[dict[str, Any]]
    covariates: list[dict[str, Any]]


class FitCounter:
    """Thread-safe count of attempted per-feature Cox fits."""
    def __init__(self) -> None:
        self.value = 0
        self._lock = threading.Lock()

    def add(self, amount: int) -> None:
        with self._lock:
            self.value += amount


class ProgressReporter:
    """Report coarse workflow stages and completed Cox feature fits on stderr.

    The reporter deliberately owns no statistical state: ``FitCounter`` keeps
    recording attempted fits for provenance, while this class records completed
    work (including cache hits) solely for user-facing progress.
    """

    def __init__(self, stream: Any | None = None,
                 heartbeat_seconds: float = _DEFAULT_HEARTBEAT_SECONDS,
                 show_progress: bool = True) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.show_progress = bool(show_progress)
        self._lock = threading.Lock()
        self._phase: str | None = None
        self._completed = 0
        self._total = 0
        self._last_percent = -1
        self._bar: Any | None = None
        self._memory_observer: Callable[[], str | None] | None = None
        self._closed = False
        self._stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self.resume_heartbeats()

    def _write(self, message: str) -> None:
        print(message, file=self.stream, flush=True)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self.heartbeat()

    def pause_heartbeats(self) -> None:
        """Stop the reporter thread before forking worker processes."""
        with self._lock:
            thread = self._heartbeat_thread
            self._heartbeat_thread = None
            self._stop.set()
        if thread is not None:
            thread.join(timeout=1.0)

    def resume_heartbeats(self) -> None:
        """Start the reporter thread when no process fork is in progress."""
        with self._lock:
            if self._closed or self.heartbeat_seconds <= 0 or self._heartbeat_thread is not None:
                return
            self._stop = threading.Event()
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop,
                                                       name="metacpg-progress", daemon=True)
            self._heartbeat_thread.start()

    def stage(self, name: str) -> None:
        """Announce a non-Cox workflow stage such as validation or writing."""
        with self._lock:
            if self.interactive and self._bar is not None:
                self._bar.write(name, file=self.stream)
            else:
                self._write(name)

    def start_phase(self, phase: str, total: int) -> None:
        """Start a Cox-fit progress phase with a known number of work units."""
        with self._lock:
            self._finish_phase_locked()
            self._phase = phase
            self._completed = 0
            self._total = max(0, int(total))
            self._last_percent = -1
            if self.interactive and self.show_progress:
                from tqdm import tqdm
                self._bar = tqdm(total=self._total, desc=f"{phase}: Cox fits", unit="fit",
                                 file=self.stream, dynamic_ncols=True)
            else:
                self._write(f"{phase}: 0/{self._total} Cox feature fits (0.0%)")

    def advance(self, amount: int) -> None:
        """Advance the active Cox-fit phase; safe to call from worker threads."""
        if amount <= 0:
            return
        with self._lock:
            if self._phase is None:
                return
            self._completed = min(self._total, self._completed + int(amount))
            if self._bar is not None:
                self._bar.update(int(amount))
                return
            if self.show_progress:
                self._report_milestone_locked()

    def heartbeat(self) -> None:
        """Emit a plain-text status line for long non-interactive phases."""
        with self._lock:
            if self._phase is not None and not self.interactive and self.show_progress:
                self._write(self._status_locked())
            if self._memory_observer is not None:
                message = self._memory_observer()
                if message:
                    if self.interactive and self._bar is not None:
                        self._bar.write(message, file=self.stream)
                    else:
                        self._write(message)

    def set_memory_observer(self, observer: Callable[[], str | None] | None) -> None:
        """Register a parent-side memory sampler used by heartbeat updates."""
        with self._lock:
            self._memory_observer = observer

    def finish_phase(self) -> None:
        with self._lock:
            self._finish_phase_locked()

    def _finish_phase_locked(self) -> None:
        if self._phase is None:
            return
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        elif self._completed < self._total and self.show_progress:
            self._write(self._status_locked())
        self._phase = None

    def _report_milestone_locked(self) -> None:
        if not self._total:
            return
        percent = int(100 * self._completed / self._total)
        milestone = (percent // 5) * 5
        if milestone > self._last_percent:
            self._last_percent = milestone
            self._write(self._status_locked())

    def _status_locked(self) -> str:
        percent = 100 * self._completed / self._total if self._total else 100.0
        return f"{self._phase}: {self._completed}/{self._total} Cox feature fits ({percent:.1f}%)"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._finish_phase_locked()
            self._closed = True
        self.pause_heartbeats()


def _read_table(path: str, sep: str | None = None, **kwargs: Any) -> pd.DataFrame:
    if sep is None:
        sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep, **kwargs)


def _read_matrix(path: str, orientation: str = "features_by_samples",
                 feature_id_column: str | int = "first", separator: str | None = None) -> pd.DataFrame:
    """Read a methylation matrix into a numeric features-by-samples DataFrame.

    Shared by :func:`validate_inputs` and :func:`apply_model` so training and new-cohort
    matrices are parsed identically (same feature-id column, orientation, and coercion).
    """
    feature_col = 0 if feature_id_column == "first" else feature_id_column
    raw = _read_table(path, separator)
    feature_name = raw.columns[feature_col] if isinstance(feature_col, int) else feature_col
    matrix = raw.set_index(feature_name)
    if orientation == "samples_by_features":
        matrix = matrix.T
    return matrix.apply(pd.to_numeric, errors="coerce")


def validate_inputs(config: dict[str, Any] | str | Path) -> InputData:
    cfg = read_config(config) if not isinstance(config, dict) else config
    errors: list[str] = []
    for key in ("analysis_id", "matrix", "phenotype", "endpoints"):
        if key not in cfg:
            errors.append(f"missing required key: {key}")
    if errors:
        raise ValueError("; ".join(errors))
    if cfg.get("methodology", "legacy_exact") != "legacy_exact":
        errors.append("methodology must be legacy_exact")
    if cfg.get("search", {}).get("cross_validation", "loo") != "loo":
        errors.append("legacy_exact requires search.cross_validation: loo")
    if cfg.get("selection", {}).get("mode", "manual") not in {"manual", "max_auc"}:
        errors.append("selection.mode must be manual or max_auc")
    try:
        _validate_weight_refit(cfg)
    except ValueError as exc:
        errors.append(str(exc))
    endpoints = cfg["endpoints"]
    if not isinstance(endpoints, list) or not 1 <= len(endpoints) <= 2:
        errors.append("endpoints must contain one or two entries")
    mcfg, pcfg = cfg["matrix"], cfg["phenotype"]
    for p in (mcfg.get("path"), pcfg.get("path")):
        if not p or not Path(p).is_file():
            errors.append(f"input file does not exist: {p}")
    if errors:
        raise ValueError("; ".join(errors))

    matrix = _read_matrix(mcfg["path"], mcfg.get("orientation", "features_by_samples"),
                          mcfg.get("feature_id_column", "first"), mcfg.get("separator"))
    if matrix.index.duplicated().any():
        errors.append("matrix feature identifiers are not unique")
    if matrix.columns.duplicated().any():
        errors.append("matrix sample identifiers are not unique")

    pheno = _read_table(pcfg["path"], pcfg.get("separator"))
    sample_col = pcfg["sample_id_column"]
    if sample_col not in pheno:
        errors.append(f"phenotype sample_id_column is missing: {sample_col}")
    elif pheno[sample_col].astype(str).duplicated().any():
        errors.append("phenotype sample identifiers are not unique")
    required = [x[k] for x in endpoints for k in ("time", "status")]
    covariates = cfg.get("covariates", []) or []
    required += [x["column"] for x in covariates]
    for col in required:
        if col not in pheno:
            errors.append(f"phenotype column is missing: {col}")
    if errors:
        raise ValueError("; ".join(errors))

    pheno[sample_col] = pheno[sample_col].astype(str)
    pheno = pheno.set_index(sample_col, drop=False)
    matrix.columns = matrix.columns.astype(str)
    unmatched_matrix = sorted(set(matrix.columns) - set(pheno.index))
    unmatched_pheno = sorted(set(pheno.index) - set(matrix.columns))
    allow_unmatched = cfg.get("validation", {}).get("allow_unmatched_samples", False)
    if (unmatched_matrix or unmatched_pheno) and not allow_unmatched:
        raise ValueError(
            f"sample IDs do not match: {len(unmatched_matrix)} matrix-only, "
            f"{len(unmatched_pheno)} phenotype-only"
        )
    common = [x for x in matrix.columns if x in pheno.index]
    if not common:
        raise ValueError("matrix and phenotype have no sample IDs in common")
    matrix = matrix.loc[:, common]
    pheno = pheno.loc[common].copy()
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("matrix contains missing or non-finite values")
    if mcfg.get("scale", "beta") == "beta":
        matrix.iloc[:, :] = beta_to_m(matrix.to_numpy())
    elif mcfg.get("scale") != "m_value":
        raise ValueError("matrix.scale must be beta or m_value")
    endpoints = _expand_group_endpoints(endpoints, pheno)
    for endpoint in endpoints:
        t = pd.to_numeric(pheno[endpoint["time"]], errors="coerce")
        s = pd.to_numeric(pheno[endpoint["status"]], errors="coerce")
        bad = t.notna() & (t <= 0)
        if bad.any():
            raise ValueError(f"endpoint {endpoint['name']} contains non-positive times")
        if not set(s.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"endpoint {endpoint['name']} status must be 0/1")
    for cov in covariates:
        if cov.get("type") not in {"categorical", "numeric"}:
            raise ValueError("covariate.type must be categorical or numeric")
        if cov.get("mode", "adjust") not in {"strata", "adjust"}:
            raise ValueError("covariate.mode must be strata or adjust")
        if cov.get("type") == "numeric":
            pd.to_numeric(pheno[cov["column"]], errors="raise")
    return InputData(matrix, pheno, endpoints, covariates)


def _expand_group_endpoints(endpoints: list[dict[str, Any]], pheno: pd.DataFrame) -> list[dict[str, Any]]:
    """Expand an endpoint with ``group_column`` into per-cohort endpoints for a stacked-table meta.

    A single (time, status) endpoint plus a cohort column with two or more levels becomes one
    internal endpoint per cohort, each carrying a ``_group`` (column, value) filter applied by
    :func:`_valid_endpoint_rows`. The rest of the pipeline then runs the same per-cohort Cox +
    random-effects pooling (``_random_effects`` handles any number of cohorts) on the batched/GPU
    engine.
    """
    if not any(ep.get("group_column") for ep in endpoints):
        return endpoints
    if len(endpoints) != 1:
        raise ValueError("group_column requires exactly one endpoint entry (it expands into per-cohort endpoints)")
    endpoint = endpoints[0]
    column = endpoint["group_column"]
    if column not in pheno:
        raise ValueError(f"endpoint group_column is missing from the phenotype: {column}")
    levels = sorted(pheno[column].dropna().astype(str).unique().tolist())
    if len(levels) < 2:
        raise ValueError(
            f"group_column '{column}' must have at least two non-null cohort levels for meta-analysis; "
            f"found {len(levels)} ({', '.join(levels[:5]) or 'none'})."
        )
    expanded = []
    for level in levels:
        child = {key: value for key, value in endpoint.items() if key != "group_column"}
        child["name"] = f"{endpoint['name']}:{level}"
        child["_group"] = (column, level)
        expanded.append(child)
    return expanded


def _design(pheno: pd.DataFrame, covariates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray | None]:
    columns: list[np.ndarray] = []
    strata_cols: list[pd.Series] = []
    for cov in covariates:
        series = pheno[cov["column"]]
        if cov.get("mode", "adjust") == "strata":
            strata_cols.append(series.astype(object))
        elif cov["type"] == "numeric":
            columns.append(pd.to_numeric(series).to_numpy(float)[:, None])
        else:
            dummy = pd.get_dummies(series.astype("category"), drop_first=True, dtype=float)
            if dummy.shape[1]:
                columns.append(dummy.to_numpy())
    z = np.column_stack(columns) if columns else np.empty((len(pheno), 0))
    strata = None
    if strata_cols:
        # pandas 3 may preserve missing categorical values as float NaN even
        # after ``astype(str)``.  These rows are excluded by
        # ``_valid_endpoint_rows``, but the full phenotype design is built
        # first for adjusted performance metrics.  Give them a harmless label
        # so joining multiple stratum columns remains type-safe.
        strata_frame = pd.concat(strata_cols, axis=1).fillna("__MISSING__").astype(str)
        strata = strata_frame.agg("|".join, axis=1).to_numpy()
    return z, strata


def _cox_fit(time_v: np.ndarray, status: np.ndarray, x: np.ndarray,
             strata: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    labels = np.zeros(len(time_v), dtype=int) if strata is None else np.asarray(strata)
    groups = np.unique(labels)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        ll = 0.0
        grad = np.zeros_like(beta)
        eta = np.clip(x @ beta, -40, 40)
        risk = np.exp(eta)
        for group in groups:
            idx = np.flatnonzero(labels == group)
            order = idx[np.argsort(-time_v[idx], kind="stable")]
            cum_risk = np.cumsum(risk[order])
            cum_xrisk = np.cumsum(risk[order, None] * x[order], axis=0)
            ordered_time = time_v[order]
            for event_time in np.unique(ordered_time[status[order] == 1]):
                at_time = ordered_time == event_time
                event_mask = at_time & (status[order] == 1)
                d = int(event_mask.sum())
                end = int(np.flatnonzero(at_time)[-1])
                s0, s1 = cum_risk[end], cum_xrisk[end]
                e0 = risk[order][event_mask].sum()
                e1 = (risk[order][event_mask, None] * x[order][event_mask]).sum(axis=0)
                ll += eta[order][event_mask].sum()
                grad += x[order][event_mask].sum(axis=0)
                for tie in range(d):
                    frac = tie / d; denom = s0 - frac * e0
                    ll -= np.log(denom); grad -= (s1 - frac * e1) / denom
        return -ll, -grad

    result = minimize(lambda b: objective(b), np.zeros(x.shape[1]), jac=True, method="BFGS",
                      options={"gtol": 1e-10, "maxiter": 1000})
    beta = result.x
    # Observed information with Efron ties, matching survival::coxph's default.
    info = np.zeros((x.shape[1], x.shape[1]))
    eta = np.clip(x @ beta, -40, 40)
    risk = np.exp(eta)
    for group in groups:
        idx = np.flatnonzero(labels == group)
        order = idx[np.argsort(-time_v[idx], kind="stable")]
        sr = np.cumsum(risk[order])
        sx = np.cumsum(risk[order, None] * x[order], axis=0)
        sxx = np.cumsum(risk[order, None, None] * np.einsum("ni,nj->nij", x[order], x[order]), axis=0)
        ordered_time = time_v[order]
        for event_time in np.unique(ordered_time[status[order] == 1]):
            at_time = ordered_time == event_time
            event_mask = at_time & (status[order] == 1)
            d = int(event_mask.sum()); end = int(np.flatnonzero(at_time)[-1])
            e0 = risk[order][event_mask].sum()
            e1 = (risk[order][event_mask, None] * x[order][event_mask]).sum(axis=0)
            e2 = (risk[order][event_mask, None, None] *
                  np.einsum("ni,nj->nij", x[order][event_mask], x[order][event_mask])).sum(axis=0)
            for tie in range(d):
                frac = tie / d; denom = sr[end] - frac * e0
                mean = (sx[end] - frac * e1) / denom
                info += (sxx[end] - frac * e2) / denom - np.outer(mean, mean)
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv(info)), 0, np.inf))
    return beta, se


# --- Batched univariate Efron Cox solver (numpy / torch backends) --------------
#
# Every CpG is a single-covariate Cox model that shares the same survival outcome
# (time, status) and therefore the same risk sets and tie structure. This lets all
# features be fit simultaneously with a batched Newton-Raphson on the *exact* Efron
# partial likelihood implemented term-for-term against ``_cox_fit`` above, so the
# betas/SEs/z-scores match the per-feature scipy solver to numerical tolerance. The
# same kernel runs on numpy (CPU) or torch (CPU/CUDA); both use float64 so the CPU
# and GPU results are identical, which the verify-parity command checks.

_BATCH_MAX_ITER = 50
_BATCH_TOL = 1e-11


class _CoxBackend:
    """Minimal array adapter so one kernel runs identically on numpy and torch."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        if kind == "numpy":
            self.exp = np.exp
            self.sqrt = np.sqrt
            self.clamp = lambda a: np.clip(a, -40.0, 40.0)
            self.cumsum0 = lambda a: np.cumsum(a, axis=0)
            self.sum0 = lambda a: a.sum(axis=0)
            self.to = lambda a: np.asarray(a, dtype=np.float64)
            self.idx = lambda a: np.asarray(a, dtype=np.int64)
            self.get = lambda a: np.asarray(a, dtype=np.float64)
            self.zeros = lambda shape: np.zeros(shape, dtype=np.float64)
            self.ones = lambda shape: np.ones(shape, dtype=np.float64)
            self.cat_last = lambda arrays: np.concatenate(arrays, axis=-1)
        elif kind in ("torch_cpu", "torch_cuda"):
            import torch

            device = torch.device("cuda" if kind == "torch_cuda" else "cpu")
            self.exp = torch.exp
            self.sqrt = torch.sqrt
            self.clamp = lambda a: torch.clamp(a, -40.0, 40.0)
            self.cumsum0 = lambda a: torch.cumsum(a, dim=0)
            self.sum0 = lambda a: a.sum(dim=0)
            self.to = lambda a: torch.as_tensor(np.ascontiguousarray(np.asarray(a, dtype=np.float64)),
                                                dtype=torch.float64, device=device)
            self.idx = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64),
                                                  dtype=torch.long, device=device)
            self.get = lambda a: a.detach().to("cpu", dtype=torch.float64).numpy()
            self.zeros = lambda shape: torch.zeros(shape, dtype=torch.float64, device=device)
            self.ones = lambda shape: torch.ones(shape, dtype=torch.float64, device=device)
            self.cat_last = lambda arrays: torch.cat(arrays, dim=-1)
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unknown Cox backend: {kind}")


def _resolve_backend_kind(engine: str) -> str:
    """Map an engine name to a concrete Cox backend, honouring CUDA availability."""
    if engine == "batched":
        return "numpy"
    if engine in ("gpu", "auto"):
        try:
            import torch

            if torch.cuda.is_available():
                return "torch_cuda"
        except Exception:
            pass
        if engine == "gpu":
            raise RuntimeError(
                "engine 'gpu' requires a CUDA-capable torch build; none is available. "
                "Use engine 'auto' to fall back to the CPU batched solver."
            )
        return "numpy"
    raise ValueError(f"engine {engine!r} does not use the batched solver")


_VALID_ENGINES = ("exact", "batched", "gpu", "auto")


def _resolve_engine(cfg: dict[str, Any]) -> str:
    """Resolve ``execution.engine`` to a concrete engine, mapping ``auto`` by availability."""
    engine = str(cfg.get("execution", {}).get("engine", "exact") or "exact").lower()
    if engine not in _VALID_ENGINES:
        raise ValueError(f"execution.engine must be one of {_VALID_ENGINES}")
    if engine == "auto":
        try:
            import torch

            return "gpu" if torch.cuda.is_available() else "batched"
        except Exception:
            return "batched"
    return engine


def _make_backend(engine: str) -> _CoxBackend | None:
    """Build the array backend for an engine (None for the exact per-feature solver)."""
    if engine == "exact":
        return None
    return _CoxBackend(_resolve_backend_kind(engine))


def _cox_skeleton(time_ordered: np.ndarray, status_ordered: np.ndarray) -> list[tuple[int, np.ndarray, int]]:
    """Precompute per-event-time risk-set boundaries and tie groups in descending-time order."""
    steps: list[tuple[int, np.ndarray, int]] = []
    event_times = np.unique(time_ordered[status_ordered == 1])
    for event_time in event_times:
        at_time = time_ordered == event_time
        end = int(np.flatnonzero(at_time)[-1])
        event_idx = np.flatnonzero(at_time & (status_ordered == 1))
        steps.append((end, event_idx, int(len(event_idx))))
    return steps


def _batched_score_info(x_ordered: Any, beta: Any, obs: Any,
                        steps: list[tuple[int, Any, int]], backend: _CoxBackend) -> tuple[Any, Any]:
    """Vectorized Efron score U(beta) and observed information I(beta) for every feature."""
    eta = backend.clamp(x_ordered * beta)
    risk = backend.exp(eta)
    rx = risk * x_ordered
    rxx = rx * x_ordered
    cum0 = backend.cumsum0(risk)
    cum1 = backend.cumsum0(rx)
    cum2 = backend.cumsum0(rxx)
    u = obs * 1.0
    info = obs * 0.0
    for end, event_idx, d in steps:
        s0, s1, s2 = cum0[end], cum1[end], cum2[end]
        e0 = backend.sum0(risk[event_idx])
        e1 = backend.sum0(rx[event_idx])
        e2 = backend.sum0(rxx[event_idx])
        for tie in range(d):
            frac = tie / d
            denom = s0 - frac * e0
            num1 = s1 - frac * e1
            mean = num1 / denom
            u = u - num1 / denom
            info = info + (s2 - frac * e2) / denom - mean * mean
    return u, info


def _active_features(info_np: np.ndarray) -> np.ndarray:
    """Features whose Fisher information is comfortably positive (identifiable model).

    Constant or near-constant covariates give an information near machine zero, where the
    Newton step u/info is numerically meaningless and blows up. Such features carry no
    survival signal, so they are frozen at beta=0 (they sort to the bottom of the ranking,
    exactly as a genuinely non-informative feature would). The floor is relative to the
    largest information in the batch so it is invariant to the covariate scale.
    """
    positive = np.isfinite(info_np) & (info_np > 0)
    scale = float(info_np[positive].max()) if positive.any() else 1.0
    return np.isfinite(info_np) & (info_np > 1e-10 * scale)


def _batched_cox_fit(x_ordered_np: np.ndarray, status_ordered: np.ndarray,
                     steps: list[tuple[int, np.ndarray, int]],
                     backend: _CoxBackend) -> tuple[np.ndarray, np.ndarray]:
    """Fit one univariate Cox model per column of ``x_ordered_np`` and return (beta, se).

    The device backend computes only the (identical, float64) score and information; the
    per-feature scalar Newton update is done in numpy. Keeping the update host-side makes
    the CPU and CUDA paths bit-for-bit identical and lets non-identifiable features be
    frozen cleanly instead of diverging.
    """
    x_ordered = backend.to(x_ordered_np)
    event_rows = backend.idx(np.flatnonzero(status_ordered == 1))
    obs = backend.sum0(x_ordered[event_rows])
    device_steps = [(end, backend.idx(event_idx), d) for end, event_idx, d in steps]
    beta_np = np.zeros(x_ordered_np.shape[1], dtype=np.float64)
    active = np.ones_like(beta_np, dtype=bool)
    for _ in range(_BATCH_MAX_ITER):
        u, info = _batched_score_info(x_ordered, backend.to(beta_np), obs, device_steps, backend)
        u_np = backend.get(u)
        info_np = backend.get(info)
        active = _active_features(info_np)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(active & np.isfinite(u_np), u_np / info_np, 0.0)
        beta_np = beta_np + step
        moved = np.abs(step[active])
        if not moved.size or float(moved.max()) < _BATCH_TOL:
            break
    _, info = _batched_score_info(x_ordered, backend.to(beta_np), obs, device_steps, backend)
    info_np = backend.get(info)
    active = _active_features(info_np)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = 1.0 / np.sqrt(info_np)
    # Non-identified features settle at beta=0 with se=0 so they sort to the bottom and the
    # two engines agree on them deterministically (rather than emitting solver-dependent junk).
    beta_np = np.where(active, beta_np, 0.0)
    se = np.where(active, se, 0.0)
    return beta_np, se


def _endpoint_stats_batched(matrix: pd.DataFrame, time_v: np.ndarray, status: np.ndarray,
                            valid: np.ndarray, endpoint: dict[str, Any], backend: _CoxBackend,
                            counter: FitCounter | None = None,
                            progress: ProgressReporter | None = None) -> pd.DataFrame:
    """Batched equivalent of ``_endpoint_stats`` for the univariate, no-strata case."""
    features = list(matrix.index)
    if counter:
        counter.add(len(features))
    values = matrix.to_numpy(float)[:, valid]  # [features, valid_samples]
    tv = time_v[valid]
    sv = status[valid]
    order = np.argsort(-tv, kind="stable")
    x_ordered = values[:, order].T  # [samples, features]
    steps = _cox_skeleton(tv[order], sv[order])
    beta, se = _batched_cox_fit(x_ordered, sv[order], steps, backend)
    # Mirror the exact per-feature arithmetic (division semantics included) so degenerate
    # features produce the same NaN/0 pattern as ``_endpoint_stats``.
    sd = np.std(values, axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = beta / sd
        ordinary_z = beta / se
        standardized_z = standardized / se
        p = 2 * norm.sf(np.abs(beta / se))
    out = pd.DataFrame({"feature": features, "coef": beta, "se": se, "ordinary_z": ordinary_z,
                        "standardized_coef": standardized, "standardized_z": standardized_z,
                        "p_value": p})
    out["endpoint"] = endpoint["name"]
    if progress is not None:
        progress.advance(len(features))
    return out


def _cox_strata_orders(time_v: np.ndarray, status: np.ndarray, strata: np.ndarray | None
                       ) -> list[tuple[np.ndarray, list[tuple[int, np.ndarray, int]]]]:
    """Per-stratum descending-time ordering and Efron skeleton (positions local to each stratum)."""
    labels = np.zeros(len(time_v), dtype=int) if strata is None else np.asarray(strata)
    orders = []
    for group in np.unique(labels):
        idx = np.flatnonzero(labels == group)
        order = idx[np.argsort(-time_v[idx], kind="stable")]
        orders.append((order, _cox_skeleton(time_v[order], status[order])))
    return orders


def _batched_score_info_general(x: Any, z: Any, beta: Any, obs: Any, n_cov: int,
                                strata_orders: list[tuple[Any, list[tuple[int, Any, int]]]],
                                backend: _CoxBackend) -> tuple[Any, Any]:
    """Batched Efron score U[C,p] and information I[C,p,p] for a stratified multivariate design.

    Per CpG the design is ``[x_cpg | z]`` (``z`` shared across CpGs). Contributions are summed
    over strata, mirroring the group loop in ``_cox_fit`` but vectorized over all CpGs.
    """
    eta = x * beta[:, 0]
    if n_cov:
        eta = eta + (z @ beta[:, 1:].T)
    risk = backend.exp(backend.clamp(eta))
    u = obs * 1.0
    info = obs[:, :, None] * obs[:, None, :] * 0.0
    for order_idx, steps in strata_orders:
        xg = x[order_idx]
        rg = risk[order_idx]
        design = xg[:, :, None]
        if n_cov:
            zg = z[order_idx]
            design = backend.cat_last([design, zg[:, None, :] * backend.ones((xg.shape[0], xg.shape[1], n_cov))])
        rd = rg[:, :, None] * design
        rdd = rd[:, :, :, None] * design[:, :, None, :]
        cum0 = backend.cumsum0(rg)
        cum1 = backend.cumsum0(rd)
        cum2 = backend.cumsum0(rdd)
        for end, event_idx, d in steps:
            s0, s1, s2 = cum0[end], cum1[end], cum2[end]
            e0 = backend.sum0(rg[event_idx])
            e1 = backend.sum0(rd[event_idx])
            e2 = backend.sum0(rdd[event_idx])
            for tie in range(d):
                frac = tie / d
                denom = s0 - frac * e0
                num1 = s1 - frac * e1
                mean = num1 / denom[:, None]
                u = u - num1 / denom[:, None]
                info = info + (s2 - frac * e2) / denom[:, None, None] - mean[:, :, None] * mean[:, None, :]
    return u, info


def _batched_cox_fit_general(x_np: np.ndarray, z_np: np.ndarray, time_v: np.ndarray,
                             status: np.ndarray, strata: np.ndarray | None,
                             backend: _CoxBackend) -> tuple[np.ndarray, np.ndarray]:
    """Fit a stratified multivariate Cox per CpG; return the CpG coefficient's (beta, se).

    The device backend computes the (float64) score/information; the per-CpG ``p x p`` Newton
    solve and standard errors are done host-side with ``pinv`` (matching ``_cox_fit``), so the
    CPU and CUDA paths stay identical and singular/degenerate CpGs are handled cleanly.
    """
    n, num_features = x_np.shape
    n_cov = z_np.shape[1]
    p = n_cov + 1
    orders_np = _cox_strata_orders(time_v, status, strata)
    strata_orders = [(backend.idx(order), [(end, backend.idx(ev), d) for end, ev, d in steps])
                     for order, steps in orders_np]
    x = backend.to(x_np)
    z = backend.to(z_np) if n_cov else backend.to(np.zeros((n, 0)))
    events = np.flatnonzero(status == 1)
    obs_np = np.zeros((num_features, p), dtype=np.float64)
    obs_np[:, 0] = x_np[events].sum(axis=0)
    if n_cov:
        obs_np[:, 1:] = z_np[events].sum(axis=0)[None, :]
    obs = backend.to(obs_np)
    beta_np = np.zeros((num_features, p), dtype=np.float64)
    active = np.ones(num_features, dtype=bool)
    for _ in range(_BATCH_MAX_ITER):
        u, info = _batched_score_info_general(x, z, backend.to(beta_np), obs, n_cov, strata_orders, backend)
        info_np = backend.get(info)
        u_np = backend.get(u)
        active = _active_features(info_np[:, 0, 0])
        step = np.zeros((num_features, p), dtype=np.float64)
        if active.any():
            pinv = np.linalg.pinv(info_np[active])
            step[active] = np.einsum("aij,aj->ai", pinv, u_np[active])
        beta_np = beta_np + step
        moved = np.abs(step[active, 0]) if active.any() else np.zeros(0)
        if not moved.size or float(moved.max()) < _BATCH_TOL:
            break
    _, info = _batched_score_info_general(x, z, backend.to(beta_np), obs, n_cov, strata_orders, backend)
    info_np = backend.get(info)
    active = _active_features(info_np[:, 0, 0])
    beta = np.zeros(num_features, dtype=np.float64)
    se = np.zeros(num_features, dtype=np.float64)
    if active.any():
        pinv = np.linalg.pinv(info_np[active])
        beta[active] = beta_np[active, 0]
        se[active] = np.sqrt(np.clip(pinv[:, 0, 0], 0, np.inf))
    return beta, se


def _endpoint_stats_batched_general(matrix: pd.DataFrame, time_v: np.ndarray, status: np.ndarray,
                                    zcov: np.ndarray, strata: np.ndarray | None, valid: np.ndarray,
                                    endpoint: dict[str, Any], backend: _CoxBackend,
                                    counter: FitCounter | None = None,
                                    progress: ProgressReporter | None = None) -> pd.DataFrame:
    """Batched ``_endpoint_stats`` for stratified and/or covariate-adjusted models."""
    features = list(matrix.index)
    if counter:
        counter.add(len(features))
    values = matrix.to_numpy(float)[:, valid]  # [features, valid_samples]
    x_np = values.T  # [samples, features]
    z_np = zcov[valid] if zcov.shape[1] else np.zeros((int(valid.sum()), 0))
    strata_v = None if strata is None else np.asarray(strata)[valid]
    beta, se = _batched_cox_fit_general(x_np, z_np, time_v[valid], status[valid], strata_v, backend)
    sd = np.std(values, axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = beta / sd
        ordinary_z = beta / se
        standardized_z = standardized / se
        p = 2 * norm.sf(np.abs(beta / se))
    out = pd.DataFrame({"feature": features, "coef": beta, "se": se, "ordinary_z": ordinary_z,
                        "standardized_coef": standardized, "standardized_z": standardized_z,
                        "p_value": p})
    out["endpoint"] = endpoint["name"]
    if progress is not None:
        progress.advance(len(features))
    return out


def _endpoint_stats(matrix: pd.DataFrame, pheno: pd.DataFrame, endpoint: dict[str, Any],
                    covariates: list[dict[str, Any]], counter: FitCounter | None = None,
                    progress: ProgressReporter | None = None,
                    engine: str = "exact", backend: _CoxBackend | None = None) -> pd.DataFrame:
    time_v = pd.to_numeric(pheno[endpoint["time"]], errors="coerce").to_numpy(float)
    status = pd.to_numeric(pheno[endpoint["status"]], errors="coerce").to_numpy(float)
    zcov, strata = _design(pheno, covariates)
    valid = np.isfinite(time_v) & np.isfinite(status)
    if zcov.size:
        valid &= np.isfinite(zcov).all(axis=1)
    # The batched kernels need the same valid-sample mask for every feature, so they require a
    # finite matrix. The whole-matrix scan runs last so the exact engine's per-fold cost is zero.
    if engine != "exact" and np.isfinite(matrix.to_numpy(float)).all():
        backend = backend or _CoxBackend(_resolve_backend_kind(engine))
        if strata is None and zcov.shape[1] == 0:
            return _endpoint_stats_batched(matrix, time_v, status, valid, endpoint,
                                           backend, counter, progress)
        return _endpoint_stats_batched_general(matrix, time_v, status, zcov, strata, valid,
                                               endpoint, backend, counter, progress)
    rows = []
    if counter:
        counter.add(len(matrix))
    completed_since_update = 0
    for feature, values in matrix.iterrows():
        xv = values.to_numpy(float)
        mask = valid & np.isfinite(xv)
        try:
            design = np.column_stack([xv[mask], zcov[mask]])
            beta, se = _cox_fit(time_v[mask], status[mask], design, None if strata is None else strata[mask])
            sd = np.std(xv[mask], ddof=1)
            standardized = beta[0] / sd
            ordinary_z = beta[0] / se[0]
            standardized_z = standardized / se[0]
            p = 2 * norm.sf(abs(beta[0] / se[0]))
            rows.append((feature, beta[0], se[0], ordinary_z, standardized, standardized_z, p))
        except Exception:
            rows.append((feature, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        completed_since_update += 1
        if progress is not None and completed_since_update >= 128:
            progress.advance(completed_since_update)
            completed_since_update = 0
    if progress is not None and completed_since_update:
        progress.advance(completed_since_update)
    out = pd.DataFrame(rows, columns=["feature", "coef", "se", "ordinary_z", "standardized_coef", "standardized_z", "p_value"])
    out["endpoint"] = endpoint["name"]
    return out


def _random_effects(stats: list[pd.DataFrame], matrix: pd.DataFrame) -> pd.DataFrame:
    merged = stats[0][["feature"]].copy()
    bs, ses = [], []
    for i, frame in enumerate(stats):
        aligned = frame.set_index("feature").loc[merged["feature"]]
        bs.append(aligned["coef"].to_numpy(float))
        ses.append(aligned["se"].to_numpy(float))
    b, se = np.column_stack(bs), np.column_stack(ses)
    w = 1 / np.square(se)
    fixed = np.sum(w * b, axis=1) / np.sum(w, axis=1)
    q = np.sum(w * np.square(b - fixed[:, None]), axis=1)
    c = np.sum(w, axis=1) - np.sum(np.square(w), axis=1) / np.sum(w, axis=1)
    # Empirical-Bayes tau² iteration, initialized by the moment estimate.
    tau2 = np.maximum(0, (q - (b.shape[1] - 1)) / c)
    tau2 = np.where(tau2 == 0, np.finfo(float).eps, tau2)
    for _ in range(100):
        wi = 1 / (np.square(se) + tau2[:, None])
        mu = np.sum(wi * b, axis=1) / np.sum(wi, axis=1)
        updated = tau2 * np.sum(wi * np.square(b - mu[:, None]), axis=1) / (b.shape[1] - 1)
        updated = np.maximum(updated, 0)
        if np.nanmax(np.abs(updated - tau2) / np.maximum(tau2, 1e-12)) < 1e-8:
            tau2 = updated
            break
        tau2 = updated
    wr = 1 / (np.square(se) + tau2[:, None])
    coef = np.sum(wr * b, axis=1) / np.sum(wr, axis=1)
    out_se = np.sqrt(1 / np.sum(wr, axis=1))
    merged["coef"] = coef
    merged["se"] = out_se
    merged["ordinary_z"] = coef / out_se
    pooled_sd = matrix.std(axis=1, ddof=1)
    merged["standardized_coef"] = coef / merged["feature"].map(pooled_sd)
    merged["standardized_z"] = merged["standardized_coef"] / out_se
    merged["p_value"] = 2 * norm.sf(np.abs(merged["ordinary_z"]))
    merged["tau2"] = tau2
    merged["endpoint"] = "meta"
    return merged


def _rank(stats: pd.DataFrame) -> pd.DataFrame:
    out = stats.copy()
    out["abs_z"] = out["ordinary_z"].abs()
    # Stable sorting preserves input feature order for exact legacy tie behavior.
    out = out.sort_values("abs_z", ascending=False, na_position="last", kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _legacy_scale(raw: np.ndarray) -> tuple[np.ndarray, float, float]:
    center = float(np.median(raw))
    scale = float(np.percentile(raw, 75) - np.percentile(raw, 25))
    scaled = (raw - center) / scale if scale else np.full_like(raw, np.nan, dtype=float)
    return scaled, center, scale


def _legacy_fold_score(ranking: pd.DataFrame, n: int, training_matrix: pd.DataFrame,
                       heldout: pd.Series) -> tuple[float, list[str], float, float]:
    top = ranking.dropna(subset=["standardized_coef"]).head(n)
    features = top["feature"].tolist()
    weights = top["standardized_coef"].to_numpy(float)
    train_raw = np.mean(training_matrix.loc[features].to_numpy(float) * weights[:, None], axis=0)
    _, center, scale = _legacy_scale(train_raw)
    heldout_raw = float(np.mean(heldout.loc[features].to_numpy(float) * weights))
    score = (heldout_raw - center) / scale if scale else math.nan
    return score, features, center, scale


def _legacy_full_scores(matrix: pd.DataFrame, stats: pd.DataFrame, features: list[str],
                        scaling_matrix: pd.DataFrame | None = None
                        ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Score ``matrix`` using centre/IQR estimated from fitted training rows only.

    ``matrix`` can include phenotype-matched samples without complete endpoint or
    covariate data; retaining them in ``risk_scores.csv`` is useful, but they were
    not model-training observations and must not influence the frozen model scale.
    Callers therefore pass the endpoint-eligible training matrix as
    ``scaling_matrix``.  The default preserves the helper's historical behaviour
    for direct callers that score their training matrix.
    """
    chosen = stats.set_index("feature").loc[features].reset_index()
    weights = chosen["standardized_coef"].to_numpy(float)
    raw = np.mean(matrix.loc[features].to_numpy(float) * weights[:, None], axis=0)
    scaling_matrix = matrix if scaling_matrix is None else scaling_matrix
    scaling_raw = np.mean(scaling_matrix.loc[features].to_numpy(float) * weights[:, None], axis=0)
    _, center, scale = _legacy_scale(scaling_raw)
    scaled = (raw - center) / scale if scale else np.full_like(raw, np.nan, dtype=float)
    scores = pd.DataFrame({"sample_id": matrix.columns, "risk_score": scaled, "raw_score": raw})
    return scores, chosen, {"center": center, "scale": scale}


def _valid_endpoint_rows(data: InputData, endpoint: dict[str, Any]) -> np.ndarray:
    p = data.phenotype
    valid = pd.to_numeric(p[endpoint["time"]], errors="coerce").notna().to_numpy(copy=True)
    valid &= pd.to_numeric(p[endpoint["status"]], errors="coerce").notna().to_numpy()
    for cov in data.covariates:
        valid &= p[cov["column"]].notna().to_numpy()
    group = endpoint.get("_group")
    if group is not None:  # stacked-table meta-analysis: restrict to this cohort's samples
        column, level = group
        valid &= (p[column].astype(str) == level).to_numpy()
    return np.flatnonzero(valid)


def _loo_fit_total(data: InputData, counts: Iterable[int]) -> int:
    """Return the feature-wise Cox fits performed by the requested LOO tasks.

    The per-fold ranking is independent of the feature count, so each held-out fold
    scans every feature exactly once regardless of how many candidate counts are
    evaluated; the count grid no longer multiplies the fit total.
    """
    list(counts)  # accepted for signature stability; the fit total is count-independent
    return (len(data.matrix) * len(data.endpoints) *
            sum(len(_valid_endpoint_rows(data, endpoint)) for endpoint in data.endpoints))


def _fold_ranking(data: InputData, endpoint_index: int, heldout: int, counter: FitCounter,
                  engine: str = "exact", backend: _CoxBackend | None = None,
                  progress: ProgressReporter | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the count-independent per-fold feature ranking and normalization matrix.

    This is the expensive part of a leave-one-out fold: it scans every feature once
    (via ``_endpoint_stats``) with the held-out sample removed from its own endpoint.
    Because the ranking never depends on the candidate feature count, computing it once
    per fold and reusing it across the whole count grid is exactly equivalent to the
    legacy per-count recomputation but avoids the redundant scans.
    """
    endpoint_sets: list[tuple[np.ndarray, pd.DataFrame]] = []
    fold_stats = []
    for other_index, other_endpoint in enumerate(data.endpoints):
        rows = _valid_endpoint_rows(data, other_endpoint)
        if other_index == endpoint_index:
            rows = rows[rows != heldout]
        endpoint_sets.append((rows, data.matrix.iloc[:, rows]))
        fold_stats.append(_endpoint_stats(data.matrix.iloc[:, rows], data.phenotype.iloc[rows],
                                          other_endpoint, data.covariates, counter, progress,
                                          engine, backend))
    if len(fold_stats) == 1:
        combined = fold_stats[0]
        normalization_matrix = endpoint_sets[0][1]
    else:
        combined_matrix = pd.concat([x[1] for x in endpoint_sets], axis=1)
        combined = _random_effects(fold_stats, combined_matrix)
        if data.endpoints[endpoint_index].get("_group") is not None:
            # group_column meta (two or more cohorts): normalize the held-out score against every
            # cohort's training samples (the held-out sample is already removed from its own cohort).
            normalization_matrix = combined_matrix
        elif endpoint_index == 0:
            secondary_rows = _valid_endpoint_rows(data, data.endpoints[1])
            normalization_matrix = pd.concat([endpoint_sets[0][1], data.matrix.iloc[:, secondary_rows]], axis=1)
        else:
            # Preserve the legacy second-endpoint normalization matrix, including its duplicated cohort.
            secondary_rows = _valid_endpoint_rows(data, data.endpoints[1])
            normalization_matrix = pd.concat([data.matrix.iloc[:, secondary_rows], endpoint_sets[1][1]], axis=1)
    return _rank(combined), normalization_matrix


def _fold_score_row(data: InputData, endpoint_index: int, heldout: int, n: int,
                    ranking: pd.DataFrame, normalization_matrix: pd.DataFrame
                    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the LOO score row and selected-feature list for one feature count ``n``."""
    endpoint = data.endpoints[endpoint_index]
    score, features, center, scale = _legacy_fold_score(
        ranking, n, normalization_matrix, data.matrix.iloc[:, heldout]
    )
    row = {"sample_id": data.matrix.columns[heldout], "endpoint": endpoint["name"],
           "endpoint_index": endpoint_index, "heldout_index": heldout, "feature_count": n,
           "risk_score": score, "fold_center": center, "fold_scale": scale,
           "time": data.phenotype.iloc[heldout][endpoint["time"]],
           "status": data.phenotype.iloc[heldout][endpoint["status"]]}
    return row, [{"feature_count": n, "feature": feature, "endpoint": endpoint["name"]} for feature in features]


@dataclass(frozen=True)
class _LooTaskContext:
    data: InputData
    phase: str
    cache_dir: Path | None
    run_id: str | None
    counts: tuple[int, ...]
    engine: str = "exact"
    backend: _CoxBackend | None = None


_PROCESS_LOO_CONTEXT: _LooTaskContext | None = None
_PROCESS_PID_QUEUE: Any | None = None
_PROCESS_THREAD_LIMIT: Any | None = None

# One flattened per-count LOO result: (score_row, selected_features, provenance).
_FoldRecord = tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]


def _run_fold_task(context: _LooTaskContext, fold: tuple[int, int]) -> tuple[list[_FoldRecord], int]:
    """Run one held-out fold across the whole feature-count grid.

    The count-independent ranking is computed at most once per fold (skipped entirely
    when every count is already cached), then reused for every feature count. Per-count
    cache files keep the exact legacy layout so ``resume_run_id`` stays compatible.
    """
    endpoint_index, heldout = fold
    cached: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    missing: list[int] = []
    caches: dict[int, Path | None] = {}
    for n in context.counts:
        cache = (context.cache_dir / f"n{n}_endpoint{endpoint_index}_fold{heldout}.json"
                 if context.cache_dir else None)
        caches[n] = cache
        if cache and cache.is_file():
            saved = json.loads(cache.read_text())
            cached[n] = (saved["score"], saved["selected"])
        else:
            missing.append(n)
    counter = FitCounter()
    if missing:
        ranking, normalization_matrix = _fold_ranking(context.data, endpoint_index, heldout,
                                                       counter, context.engine, context.backend)
        for n in missing:
            row, selected = _fold_score_row(context.data, endpoint_index, heldout, n,
                                            ranking, normalization_matrix)
            cache = caches[n]
            if cache:
                temporary = cache.with_suffix(".tmp")
                temporary.write_text(json.dumps({"score": row, "selected": selected}, default=float))
                os.replace(temporary, cache)
            cached[n] = (row, selected)
    resumed = set(context.counts) - set(missing)
    records: list[_FoldRecord] = []
    for n in context.counts:
        row, selected = cached[n]
        provenance = {"phase": context.phase, "feature_count": n, "endpoint_index": endpoint_index,
                      "heldout_index": heldout, "resumed": n in resumed, "run_id": context.run_id,
                      "ranking_statistic": "abs_ordinary_cox_z", "scaling": "fold_median_iqr"}
        records.append((row, selected, provenance))
    return records, counter.value


def _init_process_worker(pid_queue: Any) -> None:
    """Register a forked worker and prevent nested BLAS/OpenMP parallelism."""
    global _PROCESS_PID_QUEUE, _PROCESS_THREAD_LIMIT
    _PROCESS_PID_QUEUE = pid_queue
    from threadpoolctl import threadpool_limits
    _PROCESS_THREAD_LIMIT = threadpool_limits(limits=1)
    _PROCESS_THREAD_LIMIT.__enter__()
    pid_queue.put(os.getpid())


def _run_forked_fold_task(fold: tuple[int, int]) -> tuple[list[_FoldRecord], int]:
    if _PROCESS_LOO_CONTEXT is None:
        raise RuntimeError("forked metaCpG worker was not initialized with task context")
    return _run_fold_task(_PROCESS_LOO_CONTEXT, fold)


def _matrix_payload_bytes(data: InputData) -> int:
    return int(data.matrix.memory_usage(index=True, deep=True).sum())


def _worker_memory_estimate(data: InputData) -> int:
    phenotype_bytes = int(data.phenotype.memory_usage(index=True, deep=True).sum())
    # A fold temporarily materializes matrix slices and per-feature fit results. Keep a
    # deliberately conservative floor so small tables do not create an unlimited pool.
    return max(256 * _MEBIBYTE, 4 * _matrix_payload_bytes(data) + 2 * phenotype_bytes)


def _process_worker_plan(data: InputData, requested: int, task_count: int,
                         memory_budget: int | None) -> dict[str, int | str | None]:
    cpu_count = os.cpu_count() or 1
    requested = max(1, int(requested))
    worker_cap = min(requested, max(1, task_count), cpu_count)
    parent_memory, parent_metric = _process_memory(os.getpid())
    parent_memory = parent_memory or 0
    per_worker = _worker_memory_estimate(data)
    if memory_budget is not None:
        budget_cap = int((memory_budget - parent_memory) // per_worker)
        if budget_cap < 1:
            required = parent_memory + per_worker
            raise ValueError(
                f"memory budget {_format_bytes(memory_budget)} cannot run one process worker; "
                f"the conservative minimum is {_format_bytes(required)}"
            )
        worker_cap = min(worker_cap, budget_cap)
    return {"requested_workers": requested, "permitted_workers": worker_cap,
            "cpu_count": cpu_count, "task_count": task_count, "parent_memory": parent_memory,
            "parent_metric": parent_metric, "matrix_payload": _matrix_payload_bytes(data),
            "per_worker": per_worker, "memory_budget": memory_budget,
            "estimated_total": parent_memory + worker_cap * per_worker}


def _format_memory_plan(plan: dict[str, int | str | None]) -> str:
    budget = plan["memory_budget"]
    budget_text = _format_bytes(int(budget)) if budget is not None else "not set (advisory estimate only)"
    return ("Local process memory plan: "
            f"parent {plan['parent_metric']}={_format_bytes(int(plan['parent_memory']))}; "
            f"matrix={_format_bytes(int(plan['matrix_payload']))}; "
            f"per worker≈{_format_bytes(int(plan['per_worker']))}; "
            f"workers {plan['requested_workers']}→{plan['permitted_workers']}; "
            f"estimated total≈{_format_bytes(int(plan['estimated_total']))}; budget={budget_text}")


def preflight_resources(config: dict[str, Any] | str | Path, data: InputData | None = None,
                       feature_count: int | None = None) -> dict[str, Any]:
    """Validate inputs and estimate local execution resources without fitting Cox models."""
    cfg = read_config(config) if not isinstance(config, dict) else config
    data = data or validate_inputs(cfg)
    if cfg.get("methodology", "legacy_exact") != "legacy_exact":
        raise ValueError("only methodology: legacy_exact is currently supported")
    search = cfg.get("search", {})
    if search.get("cross_validation", "loo") != "loo":
        raise ValueError("legacy_exact requires leave-one-out cross-validation")
    counts = list(range(int(search.get("min_features", 5)), int(search.get("max_features", 60)) + 1,
                        int(search.get("step", 1))))
    if not counts:
        raise ValueError("search.max_features must be at least search.min_features")
    # One LOO task per held-out fold (the count grid is evaluated within each fold).
    search_tasks = sum(len(_valid_endpoint_rows(data, endpoint)) for endpoint in data.endpoints)
    memory_value = cfg.get("execution", {}).get("memory_budget")
    memory_budget = parse_memory_budget(memory_value) if memory_value is not None else None
    requested_workers = int(cfg.get("execution", {}).get("workers", 1))
    use_processes = sys.platform.startswith("linux")
    plan = _process_worker_plan(data, requested_workers, search_tasks, memory_budget) if use_processes else None
    configured_count = cfg.get("selection", {}).get("feature_count", cfg.get("search", {}).get("feature_count_override"))
    selected = feature_count if feature_count is not None else configured_count
    extraction = None
    if selected is not None:
        selected = int(selected)
        extraction_tasks = sum(len(_valid_endpoint_rows(data, endpoint)) for endpoint in data.endpoints)
        extraction = {"feature_count": selected, "loo_tasks": extraction_tasks,
                      "cox_feature_fits": len(data.matrix) * len(data.endpoints) * (1 + extraction_tasks)}
    engine = _resolve_engine(cfg)
    result: dict[str, Any] = {
        "valid": True,
        "engine": engine,
        "executor": ("serial (CUDA batched kernel)" if engine == "gpu"
                     else "processes" if use_processes else "threads"),
        "data": {"features": len(data.matrix), "samples": len(data.matrix.columns),
                 "endpoints": len(data.endpoints)},
        "search": {"min_features": counts[0], "max_features": counts[-1], "step": int(search.get("step", 1)),
                   "candidate_counts": len(counts), "loo_tasks": search_tasks,
                   "cox_feature_fits": len(data.matrix) * len(data.endpoints) + _loo_fit_total(data, counts)},
        "extraction": extraction,
        "memory_budget_is_advisory": True,
    }
    if plan is not None:
        result["memory"] = plan
        result["memory_summary"] = _format_memory_plan(plan)
    else:
        result["memory"] = {"memory_budget": memory_budget,
                            "note": "process memory planning is available on Linux local runs"}
    return result


def _assemble_loo(records: list[_FoldRecord], data: InputData) -> dict[str, pd.DataFrame]:
    """Build the scores / inclusion-frequency / provenance frames from per-fold records.

    Output rows are sorted by ``(feature_count, endpoint_index, heldout_index)`` so the
    result is deterministic and independent of task-completion order, and identical
    whether the folds ran in the process pool (exact) or serially (batched/GPU).
    """
    scores = pd.DataFrame([row for row, _, _ in records])
    scores = scores.sort_values(["feature_count", "endpoint_index", "heldout_index"]).reset_index(drop=True)
    selected_rows = [row for _, selected, _ in records for row in selected]
    frequency = (pd.DataFrame(selected_rows).groupby(["feature_count", "feature"], sort=True).size()
                 .rename("fold_count").reset_index())
    totals = scores.groupby("feature_count").size().rename("total_folds")
    frequency = frequency.join(totals, on="feature_count")
    # Preserve the paired legacy denominator: pooled selections divided by primary-cohort folds.
    if any(endpoint.get("_group") is not None for endpoint in data.endpoints):
        # group_column meta: denominator is every cohort's folds so inclusion_percent stays 0-100%.
        denominator = sum(len(_valid_endpoint_rows(data, endpoint)) for endpoint in data.endpoints)
    else:
        denominator = len(_valid_endpoint_rows(data, data.endpoints[0]))
    frequency["stability_denominator"] = denominator
    frequency["inclusion_percent"] = 100 * frequency["fold_count"] / frequency["stability_denominator"]
    frequency = frequency.sort_values(["feature_count", "inclusion_percent", "feature"], ascending=[True, False, True])
    provenance = pd.DataFrame([p for _, _, p in records])
    if not provenance.empty:
        provenance = provenance.sort_values(["feature_count", "endpoint_index", "heldout_index"]).reset_index(drop=True)
    return {"scores": scores, "inclusion_frequency": frequency, "task_provenance": provenance}


def _exact_loo(data: InputData, counts: list[int], phase: str, workers: int,
               cfg: dict[str, Any], counter: FitCounter, allow_resume: bool,
               progress: ProgressReporter | None = None,
               engine: str = "exact", backend: _CoxBackend | None = None) -> dict[str, pd.DataFrame]:
    global _PROCESS_LOO_CONTEXT
    folds = [(endpoint_index, int(heldout)) for endpoint_index, endpoint in enumerate(data.endpoints)
             for heldout in _valid_endpoint_rows(data, endpoint)]
    resume_id = cfg.get("execution", {}).get("resume_run_id") if allow_resume else None
    cache_dir = Path(cfg["output_dir"]) / "cache" / str(resume_id) / _fingerprint(cfg) / phase if resume_id else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    context = _LooTaskContext(data=data, phase=phase, cache_dir=cache_dir, run_id=cfg.get("_run_id"),
                              counts=tuple(counts), engine=engine, backend=backend)
    task_units = len(data.matrix) * len(data.endpoints)
    records: list[_FoldRecord] = []

    # The CUDA engine runs folds serially in the main process: a GPU context cannot be
    # shared across forked workers, and the batched kernel already parallelizes across all
    # features within a fold. The exact and CPU-batched engines both fan out over the
    # fork-based process pool (numpy is fork-safe), one fold per task.
    if engine == "gpu":
        for fold in folds:
            fold_records, executed = _run_fold_task(context, fold)
            records.extend(fold_records)
            counter.add(executed)
            if progress is not None:
                progress.advance(task_units)
        return _assemble_loo(records, data)

    memory_value = cfg.get("execution", {}).get("memory_budget")
    memory_budget = parse_memory_budget(memory_value) if memory_value is not None else None
    use_processes = sys.platform.startswith("linux")
    plan = _process_worker_plan(data, workers, len(folds), memory_budget) if use_processes else None
    max_workers = int(plan["permitted_workers"]) if plan is not None else min(max(1, int(workers)), max(1, len(folds)))
    if progress is not None and plan is not None:
        progress.stage(_format_memory_plan(plan))

    fold_results: list[tuple[list[_FoldRecord], int] | None] = [None] * len(folds)
    pid_queue: Any | None = None
    worker_pids: set[int] = set()
    memory_warning_level = 0

    def collect_worker_pids() -> None:
        if pid_queue is None:
            return
        while True:
            try:
                worker_pids.add(int(pid_queue.get_nowait()))
            except queue.Empty:
                return

    def memory_status() -> str | None:
        nonlocal memory_warning_level
        collect_worker_pids()
        pids = [os.getpid(), *sorted(worker_pids)]
        samples = [_process_memory(pid) for pid in pids]
        present = [(value, metric) for value, metric in samples if value is not None]
        if not present:
            return None
        total = sum(value for value, _ in present)
        metric = "PSS" if all(kind == "PSS" for _, kind in present) else "mixed PSS/RSS"
        message = f"Memory: {metric}={_format_bytes(total)} across {len(present)} process(es)"
        if memory_budget is not None:
            message += f"; advisory budget={_format_bytes(memory_budget)}"
            ratio = total / memory_budget
            if ratio >= 1 and memory_warning_level < 2:
                memory_warning_level = 2
                message += " [WARNING: advisory budget exceeded]"
            elif ratio >= .8 and memory_warning_level < 1:
                memory_warning_level = 1
                message += " [WARNING: above 80% of advisory budget]"
        return message

    heartbeats_paused = False
    if progress is not None and use_processes:
        progress.pause_heartbeats()
        heartbeats_paused = True
    try:
        if use_processes:
            _PROCESS_LOO_CONTEXT = context
            process_context = mp.get_context("fork")
            pid_queue = process_context.Queue()
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=process_context,
                                     initializer=_init_process_worker, initargs=(pid_queue,)) as executor:
                futures = {executor.submit(_run_forked_fold_task, fold): index for index, fold in enumerate(folds)}
                if progress is not None:
                    progress.set_memory_observer(memory_status)
                    progress.resume_heartbeats()
                    heartbeats_paused = False
                for future in as_completed(futures):
                    index = futures[future]
                    fold_results[index] = future.result()
                    _, executed = fold_results[index]
                    counter.add(executed)
                    if progress is not None:
                        progress.advance(task_units)
                    collect_worker_pids()
            collect_worker_pids()
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_run_fold_task, context, fold): index for index, fold in enumerate(folds)}
                for future in as_completed(futures):
                    index = futures[future]
                    fold_results[index] = future.result()
                    _, executed = fold_results[index]
                    counter.add(executed)
                    if progress is not None:
                        progress.advance(task_units)
    finally:
        _PROCESS_LOO_CONTEXT = None
        if progress is not None:
            progress.set_memory_observer(None)
            if heartbeats_paused:
                progress.resume_heartbeats()
    for item in fold_results:
        if item is not None:
            records.extend(item[0])
    return _assemble_loo(records, data)


def _harrell_c(time_v: np.ndarray, status: np.ndarray, score: np.ndarray) -> float:
    concordant = comparable = 0.0
    for i in range(len(time_v)):
        for j in range(i + 1, len(time_v)):
            if time_v[i] == time_v[j]:
                continue
            early, late = (i, j) if time_v[i] < time_v[j] else (j, i)
            if status[early] != 1:
                continue
            comparable += 1
            concordant += 1 if score[early] > score[late] else 0.5 if score[early] == score[late] else 0
    return concordant / comparable if comparable else math.nan


def _binary_auc_at_time(time_v: np.ndarray, status: np.ndarray, score: np.ndarray, horizon: float) -> float:
    cases = (time_v <= horizon) & (status == 1)
    controls = time_v > horizon
    if not cases.any() or not controls.any():
        return math.nan
    ranks = rankdata(np.r_[score[cases], score[controls]])
    n1, n0 = cases.sum(), controls.sum()
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _survival_roc_nne_auc(time_v: np.ndarray, status: np.ndarray, marker: np.ndarray,
                          horizon: float) -> float:
    """Port of survivalROC's NNE/span/symmetric estimator used by the legacy R code."""
    valid = np.isfinite(time_v) & np.isfinite(status) & np.isfinite(marker)
    time_v, status, marker = time_v[valid], status[valid], marker[valid]
    if not len(time_v):
        return math.nan
    order = np.argsort(time_v, kind="stable")
    time_v, status, marker = time_v[order], status[order], marker[order]
    unique_marker = np.sort(np.unique(marker)); n = len(time_v)
    if len(unique_marker) < 2:
        return math.nan
    span = .25 * n ** (-.20)
    event_times = np.sort(np.unique(time_v[(status == 1) & (time_v <= horizon)]))
    conditional_survival = np.zeros(len(unique_marker))
    for j, value in enumerate(unique_marker):
        distance = np.sort(marker - value)
        index0_r = int(np.sum(distance < 0)) + 1
        index1_r = min(n, index0_r + math.trunc(n * span + .5))
        bandwidth = abs(distance[index1_r - 1])
        weights = ((marker - value >= -bandwidth) & (marker - value <= bandwidth)).astype(int)
        survival = 1.0
        for event_time in event_times:
            at_risk = int(np.sum(weights * (time_v >= event_time)))
            events = int(np.sum(weights * (time_v == event_time) * (status == 1)))
            if at_risk > 0:
                survival *= 1 - events / at_risk
        conditional_survival[j] = survival
    survival_by_subject = conditional_survival[np.searchsorted(unique_marker, marker)]
    marginal = float(np.mean(survival_by_subject))
    if marginal <= 0 or marginal >= 1:
        return math.nan
    sensitivity, specificity = [], []
    for cut in unique_marker[:-1]:
        above = marker > cut
        p1 = float(np.mean(above)); sx = float(np.sum(survival_by_subject[above]) / n)
        sensitivity.append((p1 - sx) / (1 - marginal))
        specificity.append(1 - sx / marginal)
    sensitivity.append(0.0); specificity.append(1.0)
    fp = 1 - np.r_[0.0, specificity]
    tp = np.r_[1.0, sensitivity]
    return float(np.sum((fp[:-1] - fp[1:]) * (tp[:-1] + tp[1:]) / 2))


def _neglog10(p: float) -> float:
    return -math.log10(p) if p is not None and np.isfinite(p) and p > 0 else math.nan


def _legacy_performance(time_v: np.ndarray, status: np.ndarray, score: np.ndarray,
                        horizon: float) -> tuple[float, float, float]:
    valid = np.isfinite(time_v) & np.isfinite(status) & np.isfinite(score)
    time_v, status, score = time_v[valid], status[valid], score[valid]
    beta, se = _cox_fit(time_v, status, score)
    linear_predictor = score * beta[0]
    p = 2 * norm.sf(abs(beta[0] / se[0]))
    return (_survival_roc_nne_auc(time_v, status, linear_predictor, horizon),
            _harrell_c(time_v, status, linear_predictor), p)


def _legacy_performance_adjusted(time_v: np.ndarray, status: np.ndarray, score: np.ndarray,
                                 zcov: np.ndarray, strata: np.ndarray | None,
                                 horizon: float) -> tuple[float, float, float]:
    """Score + configured covariates, jointly fit — the legacy 'CV Metagene + Clinical' comparison.

    The reported AUC/C-index use the fit's full linear predictor (score and covariate terms
    together), while the p-value is the Wald test on the score's own coefficient — mirroring
    ``calc.auc``'s ``met.cv.clin`` branch in the legacy R script.
    """
    valid = np.isfinite(time_v) & np.isfinite(status) & np.isfinite(score)
    if zcov.size:
        valid &= np.isfinite(zcov).all(axis=1)
    time_v, status, score = time_v[valid], status[valid], score[valid]
    zcov = zcov[valid] if zcov.size else zcov
    strata = None if strata is None else strata[valid]
    design = np.column_stack([score, zcov]) if zcov.size else score[:, None]
    beta, se = _cox_fit(time_v, status, design, strata)
    linear_predictor = design @ beta
    p = 2 * norm.sf(abs(beta[0] / se[0]))
    return (_survival_roc_nne_auc(time_v, status, linear_predictor, horizon),
            _harrell_c(time_v, status, linear_predictor), p)


def search_features(config: dict[str, Any] | str | Path, data: InputData | None = None,
                    progress: ProgressReporter | None = None) -> dict[str, pd.DataFrame]:
    cfg = read_config(config) if not isinstance(config, dict) else config
    data = data or validate_inputs(cfg)
    if cfg.get("methodology", "legacy_exact") != "legacy_exact":
        raise ValueError("only methodology: legacy_exact is currently supported")
    search = cfg.get("search", {})
    counts = list(range(int(search.get("min_features", 5)), int(search.get("max_features", 60)) + 1, int(search.get("step", 1))))
    if search.get("cross_validation", "loo") != "loo":
        raise ValueError("legacy_exact requires leave-one-out cross-validation")
    engine = _resolve_engine(cfg)
    backend = _make_backend(engine)
    counter = FitCounter()
    if progress is not None:
        progress.start_phase("Search", len(data.matrix) * len(data.endpoints) + _loo_fit_total(data, counts))
    try:
        stats = []
        endpoint_matrices = []
        for endpoint in data.endpoints:
            rows = _valid_endpoint_rows(data, endpoint)
            endpoint_matrices.append(data.matrix.iloc[:, rows])
            stats.append(_endpoint_stats(data.matrix.iloc[:, rows], data.phenotype.iloc[rows],
                                         endpoint, data.covariates, counter, progress, engine, backend))
        combined_matrix = pd.concat(endpoint_matrices, axis=1)
        combined = stats[0] if len(stats) == 1 else _random_effects(stats, combined_matrix)
        ranking = _rank(combined)
        cv = _exact_loo(data, counts, "search", cfg.get("execution", {}).get("workers", 1),
                        cfg, counter, allow_resume=True, progress=progress, engine=engine, backend=backend)
    finally:
        if progress is not None:
            progress.finish_phase()
    scores, frequency = cv["scores"], cv["inclusion_frequency"]
    has_covariates = bool(data.covariates)
    if has_covariates:
        zcov_all, strata_all = _design(data.phenotype, data.covariates)
    metrics = []
    for n, group in scores.groupby("feature_count", sort=True):
        time_v = pd.to_numeric(group["time"], errors="coerce").to_numpy(float)
        status = pd.to_numeric(group["status"], errors="coerce").to_numpy(float)
        score = group["risk_score"].to_numpy(float)
        horizons = [float(ep.get("prediction_time", 1095)) for ep in data.endpoints]
        horizon = horizons[0] if len(set(horizons)) == 1 else min(horizons)
        auc, c_index, p = _legacy_performance(time_v, status, score, horizon)
        row = {"feature_count": int(n), "auc": auc, "c_index": c_index, "neglog10_p": _neglog10(p)}
        if has_covariates:
            heldout = group["heldout_index"].to_numpy()
            zcov = zcov_all[heldout]
            strata = None if strata_all is None else strata_all[heldout]
            auc_adj, c_adj, p_adj = _legacy_performance_adjusted(time_v, status, score, zcov, strata, horizon)
            row.update({"auc_adjusted": auc_adj, "c_index_adjusted": c_adj,
                        "neglog10_p_adjusted": _neglog10(p_adj)})
        metrics.append(row)
    logical_search_fits = len(data.endpoints) * len(data.matrix) * (1 + len(scores))
    return {"ranking": ranking, "scores": scores, "metrics": pd.DataFrame(metrics),
            "endpoint_stats": pd.concat(stats), "inclusion_frequency": frequency,
            "task_provenance": cv["task_provenance"], "search_cox_fit_count": int(logical_search_fits),
            "search_cox_fits_executed": counter.value}


def select_feature_count(metrics: pd.DataFrame, override: int | None = None,
                         mode: str = "manual") -> int:
    if override is not None:
        if override not in set(metrics["feature_count"]):
            raise ValueError("feature-count override is outside the search range")
        return int(override)
    if mode == "manual":
        raise ValueError("manual feature selection requires an explicit feature-count override")
    if mode != "max_auc":
        raise ValueError("selection.mode must be manual or max_auc")
    valid = metrics.dropna(subset=["auc"])
    if valid.empty:
        raise ValueError("no finite AUC values are available for feature selection")
    best = valid["auc"].max()
    return int(valid.loc[valid["auc"] == best, "feature_count"].min())


def _extract_signature(cfg: dict[str, Any], data: InputData, selected: int,
                       progress: ProgressReporter | None = None,
                       resume: bool = False) -> dict[str, Any]:
    """Run the extraction phase for a chosen feature count and return its result keys.

    Extraction is deliberately a fresh, complete leave-one-out run at ``selected`` plus a
    full-data weight refit under ``final_model.weight_refit``; it never consults the search caches. This is the
    self-contained half of :func:`fit_metacpg`, also used directly by :func:`extract_signature`
    so a feature count can be extracted after a separate search without re-running it.
    """
    engine = _resolve_engine(cfg)
    backend = _make_backend(engine)
    weight_refit = _validate_weight_refit(cfg)
    refit_covariates = data.covariates if weight_refit == "covariate_aware" else []
    extraction_counter = FitCounter()
    if data.covariates and not _weight_refit_explicit(cfg) and progress is not None:
        progress.stage(
            "WARNING: final_model.weight_refit now defaults to covariate_aware; "
            "LOO selection is unchanged, but exported weights/fixed scores differ from legacy runs. "
            "Rerun downstream validation or set legacy_unadjusted explicitly."
        )
    if progress is not None:
        progress.start_phase("Extraction", _loo_fit_total(data, [selected]) + len(data.matrix) * len(data.endpoints))
    try:
        extraction = _exact_loo(data, [selected], "extraction", cfg.get("execution", {}).get("workers", 1),
                                cfg, extraction_counter, allow_resume=resume, progress=progress,
                                engine=engine, backend=backend)
        frequency = extraction["inclusion_frequency"]
        chosen = frequency[frequency["feature_count"] == selected].head(selected)["feature"].tolist()
        full_stats = []
        endpoint_matrices = []
        for endpoint in data.endpoints:
            rows = _valid_endpoint_rows(data, endpoint)
            endpoint_matrices.append(data.matrix.iloc[:, rows])
            full_stats.append(_endpoint_stats(data.matrix.iloc[:, rows], data.phenotype.iloc[rows],
                                              endpoint, refit_covariates, extraction_counter,
                                              progress, engine, backend))
    finally:
        if progress is not None:
            progress.finish_phase()
    combined_matrix = pd.concat(endpoint_matrices, axis=1)
    final_stats = full_stats[0] if len(full_stats) == 1 else _random_effects(full_stats, combined_matrix)
    # Score every phenotype-matched sample, but freeze centre/IQR using only the
    # endpoint-eligible observations that supplied the final fitted statistics.
    scores, selected_features, scaling = _legacy_full_scores(
        data.matrix, final_stats, chosen, scaling_matrix=combined_matrix)
    selected_features.insert(1, "rank", np.arange(1, len(selected_features) + 1))
    annotation_cfg = cfg.get("annotation") or {}
    if annotation_cfg.get("path"):
        ann = _read_table(annotation_cfg["path"], annotation_cfg.get("separator"))
        selected_features = selected_features.merge(ann, how="left", left_on="feature", right_on=annotation_cfg.get("feature_id_column", "feature"))
    extraction_scores = extraction["scores"]
    t = pd.to_numeric(extraction_scores["time"], errors="coerce").to_numpy(float)
    s = pd.to_numeric(extraction_scores["status"], errors="coerce").to_numpy(float)
    cv_score = extraction_scores["risk_score"].to_numpy(float)
    horizon = min(float(ep.get("prediction_time", 1095)) for ep in data.endpoints)
    auc, c_index, p = _legacy_performance(t, s, cv_score, horizon)
    performance = {"feature_count": selected, "c_index": c_index, "auc": auc, "neglog10_p": _neglog10(p)}
    if data.covariates:
        heldout = extraction_scores["heldout_index"].to_numpy()
        zcov_all, strata_all = _design(data.phenotype, data.covariates)
        zcov = zcov_all[heldout]
        strata = None if strata_all is None else strata_all[heldout]
        auc_adj, c_adj, p_adj = _legacy_performance_adjusted(t, s, cv_score, zcov, strata, horizon)
        performance.update({"auc_adjusted": auc_adj, "c_index_adjusted": c_adj,
                            "neglog10_p_adjusted": _neglog10(p_adj)})
    return {"selected_features": selected_features, "risk_scores": scores,
            "scaling": scaling, "performance": performance, "weight_refit": weight_refit,
            "extraction_scores": extraction_scores,
            "extraction_inclusion_frequency": frequency,
            "extraction_task_provenance": extraction["task_provenance"],
            "extraction_cox_fit_count": int(len(data.endpoints) * len(data.matrix) *
                (1 + len(extraction_scores))),
            "extraction_cox_fits_executed": extraction_counter.value}


def fit_metacpg(config: dict[str, Any] | str | Path, search_result: dict[str, pd.DataFrame] | None = None,
                 feature_count: int | None = None, data: InputData | None = None,
                 _resume_extraction: bool = False,
                 progress: ProgressReporter | None = None) -> dict[str, Any]:
    cfg = read_config(config) if not isinstance(config, dict) else config
    data = data or validate_inputs(cfg)
    result = search_result or search_features(cfg, data, progress)
    configured = cfg.get("selection", {}).get("feature_count", cfg.get("search", {}).get("feature_count_override"))
    mode = cfg.get("selection", {}).get("mode", "manual")
    selected = select_feature_count(result["metrics"], feature_count if feature_count is not None else configured, mode)
    result = dict(result)
    result.update(_extract_signature(cfg, data, selected, progress, resume=_resume_extraction))
    return result


def _fingerprint(cfg: dict[str, Any]) -> str:
    clean = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
    clean.setdefault("final_model", {})["weight_refit"] = _validate_weight_refit(cfg)
    h = hashlib.sha256(json.dumps(clean, sort_keys=True).encode())
    for section in ("matrix", "phenotype"):
        p = Path(cfg[section]["path"])
        st = p.stat()
        h.update(f"{p}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _signature_model(cfg: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    """Assemble the portable meta-CpG model (features, weights, and training center/scale).

    Returns ``None`` for a search-only result. The dict is everything needed to score a new
    cohort: applied by :func:`apply_model` and self-describing enough to reimplement elsewhere.
    """
    selected = result.get("selected_features")
    scaling = result.get("scaling")
    if not isinstance(selected, pd.DataFrame) or not isinstance(scaling, dict):
        return None
    has_rank = "rank" in selected.columns
    signature = [{"feature": str(row["feature"]), "weight": float(row["standardized_coef"]),
                  "rank": int(row["rank"]) if has_rank else index + 1}
                 for index, (_, row) in enumerate(selected.iterrows())]
    mcfg = cfg.get("matrix", {})
    endpoints = [{"name": ep.get("name"), "time": ep.get("time"), "status": ep.get("status"),
                  "prediction_time": ep.get("prediction_time")} for ep in cfg.get("endpoints", [])]
    return {"schema_version": SCHEMA_VERSION, "methodology": cfg.get("methodology", "legacy_exact"),
            "model_version": 1, "analysis_id": cfg.get("analysis_id"),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fingerprint": _fingerprint(cfg),
            "input": {"scale": mcfg.get("scale", "beta"),
                      "orientation": mcfg.get("orientation", "features_by_samples"),
                      "feature_id_column": mcfg.get("feature_id_column", "first")},
            "endpoints": endpoints, "feature_count": len(signature),
            "weight_refit": result.get("weight_refit", _weight_refit_mode(cfg)),
            "scaling": {"center": float(scaling.get("center")), "scale": float(scaling.get("scale"))},
            "score": ("raw = mean_over_features(Mvalue(x[feature]) * weight[feature]); "
                      "risk_score = (raw - center) / scale; beta->M via log2(b/(1-b))"),
            "signature": signature}


def _write_results(cfg: dict[str, Any], result: dict[str, Any], elapsed: float) -> Path:
    out = Path(cfg["output_dir"]) / "python"
    out.mkdir(parents=True, exist_ok=True)
    for key in ("ranking", "scores", "metrics", "endpoint_stats", "inclusion_frequency",
                "task_provenance", "selected_features", "risk_scores", "extraction_scores",
                "extraction_inclusion_frequency", "extraction_task_provenance"):
        if isinstance(result.get(key), pd.DataFrame):
            result[key].to_csv(out / f"{key}.csv", index=False)
    model = _signature_model(cfg, result)
    if model is not None:
        (out / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
        pd.DataFrame(model["signature"]).to_csv(out / "signature_model.csv", index=False)
    provenance = {"analysis_id": cfg["analysis_id"], "engine": "python", "schema_version": SCHEMA_VERSION,
                  "methodology": cfg.get("methodology", "legacy_exact"), "run_id": cfg.get("_run_id"),
                  "fingerprint": _fingerprint(cfg), "elapsed_seconds": elapsed, "python": sys.version,
                  "platform": platform.platform(), "performance": result.get("performance", {}),
                  "scaling": result.get("scaling", {}),
                  "weight_refit": result.get("weight_refit", _weight_refit_mode(cfg)),
                  "cox_fit_counts": {"search": result.get("search_cox_fit_count"),
                                     "extraction": result.get("extraction_cox_fit_count"),
                                     "search_executed_in_process": result.get("search_cox_fits_executed"),
                                     "extraction_executed_in_process": result.get("extraction_cox_fits_executed")}}
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        return value
    (out / "result.json").write_text(json.dumps(safe(provenance), indent=2, allow_nan=False), encoding="utf-8")
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "metacpg-matplotlib"))
        import matplotlib.pyplot as plt
        metrics = result.get("metrics")
        if isinstance(metrics, pd.DataFrame) and not metrics.empty:
            has_adjusted = "auc_adjusted" in metrics.columns
            # Two stacked panels sharing the feature-count axis rather than a twin y-axis:
            # AUC/C-index are bounded [0,1] but -log10(p) is not, so a shared scale would
            # invent an arbitrary alignment between the two kinds of series.
            fig, (ax, ax_p) = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                                           gridspec_kw={"height_ratios": [2, 1]})
            ax.plot(metrics.feature_count, metrics.auc, label="AUC (LOO CV)", color="#2a78d6")
            ax.plot(metrics.feature_count, metrics.c_index, label="C-index (LOO CV)", color="#eb6834")
            if has_adjusted:
                ax.plot(metrics.feature_count, metrics.auc_adjusted,
                       label="AUC (LOO CV, +covariates)", color="#1baf7a", linestyle="--")
                ax.plot(metrics.feature_count, metrics.c_index_adjusted,
                       label="C-index (LOO CV, +covariates)", color="#eda100", linestyle="--")
            ax.set(ylabel="Performance", title=f"{cfg['analysis_id']} — leave-one-out cross-validated")
            ax_p.plot(metrics.feature_count, metrics.neglog10_p, label="-log10(p) (LOO CV)", color="#e87ba4")
            if has_adjusted and "neglog10_p_adjusted" in metrics.columns:
                ax_p.plot(metrics.feature_count, metrics.neglog10_p_adjusted,
                         label="-log10(p) (LOO CV, +covariates)", color="#e87ba4", linestyle="--")
            ax_p.set(xlabel="Feature count", ylabel="-log10(p)")
            # Vertical dotted gridline at every evaluated feature count so a peak can be
            # traced down to its count; the auto major ticks stay labelled and a touch darker.
            counts = metrics.feature_count.to_numpy()
            for panel in (ax, ax_p):
                panel.set_xticks(counts, minor=True)
                panel.grid(axis="x", which="minor", linestyle=":", linewidth=0.5, color="0.75")
                panel.grid(axis="x", which="major", linestyle=":", linewidth=0.9, color="0.5")
            ax.legend(); ax_p.legend(); fig.tight_layout()
            fig.savefig(out / "search_metrics.png", dpi=150); plt.close(fig)
        risk_scores = result.get("risk_scores")
        if isinstance(risk_scores, pd.DataFrame) and not risk_scores.empty:
            fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(risk_scores.risk_score, bins=20)
            ax.set(xlabel="Risk score", ylabel="Samples", title="Final risk-score distribution"); fig.tight_layout()
            fig.savefig(out / "risk_scores.png", dpi=150); plt.close(fig)
    except Exception as exc:
        (out / "plot_warning.txt").write_text(f"Diagnostic plots were not generated: {exc}\n")
    return out


def run_pipeline(config: dict[str, Any] | str | Path, backend: str = "local", dry_run: bool = False,
                 feature_count: int | None = None,
                 progress: ProgressReporter | None = None) -> dict[str, Any]:
    cfg = read_config(config) if not isinstance(config, dict) else config
    if backend == "slurm":
        return submit_slurm(cfg, dry_run=dry_run)
    if backend != "local":
        raise ValueError("backend must be local or slurm")
    owns_progress = progress is None
    progress = progress or ProgressReporter(heartbeat_seconds=heartbeat_seconds_from_config(cfg),
                                             show_progress=progress_enabled_from_config(cfg))
    if dry_run:
        try:
            progress.stage("Validating inputs")
            validate_inputs(cfg)
            progress.stage("Validation complete")
            return {"valid": True, "output_dir": cfg["output_dir"]}
        finally:
            if owns_progress:
                progress.close()
    cfg["_run_id"] = cfg.get("execution", {}).get("resume_run_id") or uuid.uuid4().hex
    start = time.monotonic()
    try:
        progress.stage("Validating inputs")
        data = validate_inputs(cfg)
        progress.stage("Validation complete")
        search = search_features(cfg, data, progress)
        if _manual_without_count(cfg, feature_count):
            # No feature count to extract yet: write the per-count search summary and stop
            # cleanly (rather than raising) so the user can inspect it and then run `extract`.
            progress.stage("Writing search summary")
            result = _search_summary_result(cfg, search, time.monotonic() - start)
            progress.stage("Search summary written")
            return result
        result = fit_metacpg(cfg, search, feature_count, data, progress=progress)
        progress.stage("Writing results")
        result["output_dir"] = str(_write_results(cfg, result, time.monotonic() - start))
        progress.stage("Results written")
        return result
    finally:
        if owns_progress:
            progress.close()


def _manual_without_count(cfg: dict[str, Any], feature_count: int | None) -> bool:
    """True when no feature count is available to extract (manual mode, none configured/given)."""
    if feature_count is not None:
        return False
    configured = cfg.get("selection", {}).get("feature_count", cfg.get("search", {}).get("feature_count_override"))
    if configured is not None:
        return False
    return cfg.get("selection", {}).get("mode", "manual") == "manual"


def _search_summary_result(cfg: dict[str, Any], search: dict[str, Any], elapsed: float) -> dict[str, Any]:
    """Write the search-only summary outputs and return them with next-step guidance."""
    out = _write_results(cfg, dict(search), elapsed)
    cfg_path = cfg.get("_config_path", "<config>")
    next_step = f"metacpg-py extract --config {cfg_path} --feature-count N"
    result = dict(search)
    result.update({"output_dir": str(out), "phase": "search", "next_step": next_step,
                   "weight_refit": _weight_refit_mode(cfg),
                   "message": ("Search complete — inspect metrics.csv and search_metrics.png in "
                               f"{out}, then extract a chosen count with: {next_step}")})
    return result


def _resolve_extraction_count(cfg: dict[str, Any], feature_count: int | None) -> int:
    """Resolve the feature count to extract, validating it against the search grid.

    Extraction does not run the search, so the count comes from ``--feature-count`` (or a
    configured ``selection.feature_count``); ``max_auc`` mode reads a prior search's
    ``metrics.csv`` from the output directory if present.
    """
    search = cfg.get("search", {})
    grid = list(range(int(search.get("min_features", 5)), int(search.get("max_features", 60)) + 1,
                      int(search.get("step", 1))))
    configured = cfg.get("selection", {}).get("feature_count", search.get("feature_count_override"))
    override = feature_count if feature_count is not None else configured
    if override is not None:
        override = int(override)
        if override not in grid:
            raise ValueError(f"feature-count {override} is outside the search range "
                             f"[{grid[0]}..{grid[-1]} step {int(search.get('step', 1))}]")
        return override
    if cfg.get("selection", {}).get("mode", "manual") == "max_auc":
        metrics_path = Path(cfg["output_dir"]) / "python" / "metrics.csv"
        if not metrics_path.is_file():
            raise ValueError("max_auc extraction needs a prior search summary (metrics.csv); run "
                             "`metacpg-py search` (or `run`) first, or pass --feature-count")
        return select_feature_count(pd.read_csv(metrics_path), None, "max_auc")
    raise ValueError(f"extract requires an explicit --feature-count (one of {grid[0]}..{grid[-1]})")


def search_only(config: dict[str, Any] | str | Path, data: InputData | None = None,
                progress: ProgressReporter | None = None) -> dict[str, Any]:
    """Run the search phase, write the per-count summary, and return it with next-step guidance.

    Unlike ``run_pipeline`` this never attempts extraction, so it cannot fail for lack of a
    feature count. Use it (or ``run`` without a count) to survey the feature-count range.
    """
    cfg = read_config(config) if not isinstance(config, dict) else config
    cfg["_run_id"] = cfg.get("execution", {}).get("resume_run_id") or uuid.uuid4().hex
    owns_progress = progress is None
    progress = progress or ProgressReporter(heartbeat_seconds=heartbeat_seconds_from_config(cfg),
                                             show_progress=progress_enabled_from_config(cfg))
    start = time.monotonic()
    try:
        progress.stage("Validating inputs")
        data = data or validate_inputs(cfg)
        progress.stage("Validation complete")
        search = search_features(cfg, data, progress)
        progress.stage("Writing search summary")
        result = _search_summary_result(cfg, search, time.monotonic() - start)
        progress.stage("Search summary written")
        return result
    finally:
        if owns_progress:
            progress.close()


def extract_signature(config: dict[str, Any] | str | Path, feature_count: int | None = None,
                      data: InputData | None = None,
                      progress: ProgressReporter | None = None) -> dict[str, Any]:
    """Extract the signature for one feature count and write the extraction outputs.

    This runs only the (independent, fresh) extraction phase — it never runs or reuses the
    search — so it can be invoked after a separate ``search``/``run`` to produce the final
    signature for a chosen count without repeating the search.
    """
    cfg = read_config(config) if not isinstance(config, dict) else config
    cfg["_run_id"] = cfg.get("execution", {}).get("resume_run_id") or uuid.uuid4().hex
    owns_progress = progress is None
    progress = progress or ProgressReporter(heartbeat_seconds=heartbeat_seconds_from_config(cfg),
                                             show_progress=progress_enabled_from_config(cfg))
    start = time.monotonic()
    try:
        progress.stage("Validating inputs")
        data = data or validate_inputs(cfg)
        progress.stage("Validation complete")
        selected = _resolve_extraction_count(cfg, feature_count)
        result = _extract_signature(cfg, data, selected, progress)
        progress.stage("Writing results")
        result["output_dir"] = str(_write_results(cfg, result, time.monotonic() - start))
        progress.stage("Results written")
        return result
    finally:
        if owns_progress:
            progress.close()


def apply_model(model: dict[str, Any], matrix: pd.DataFrame, allow_missing: bool = False) -> pd.DataFrame:
    """Score every sample in a new features-by-samples matrix with a saved meta-CpG model.

    Uses the same arithmetic as :func:`_legacy_full_scores` with the model's frozen weights and
    training center/scale, so applying a model to its own training matrix reproduces
    ``risk_scores.csv`` exactly. Signature CpGs must be present in ``matrix`` (strict by default);
    ``allow_missing`` scores over the present subset instead of raising.
    """
    signature = model["signature"]
    features = [str(entry["feature"]) for entry in signature]
    weights = np.array([float(entry["weight"]) for entry in signature], dtype=float)
    present = np.array([feature in matrix.index for feature in features], dtype=bool)
    if not present.all() and not allow_missing:
        missing = [feature for feature, ok in zip(features, present) if not ok]
        preview = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        raise ValueError(f"{len(missing)} of {len(features)} signature CpGs are missing from the "
                         f"matrix ({preview}); pass allow_missing to score over the "
                         f"{int(present.sum())} present CpGs")
    used_features = [feature for feature, ok in zip(features, present) if ok]
    if not used_features:
        raise ValueError("none of the signature CpGs are present in the matrix")
    used_weights = weights[present]
    values = matrix.loc[used_features].to_numpy(dtype=float)  # [features, samples]
    if model.get("input", {}).get("scale", "beta") == "beta":
        values = beta_to_m(values)
    raw = np.mean(values * used_weights[:, None], axis=0)
    center = float(model["scaling"]["center"])
    scale = float(model["scaling"]["scale"])
    risk = (raw - center) / scale if scale else np.full_like(raw, np.nan, dtype=float)
    frame = pd.DataFrame({"sample_id": matrix.columns.astype(str), "risk_score": risk, "raw_score": raw})
    if allow_missing:
        frame["features_used"] = len(used_features)
        frame["features_total"] = len(features)
    return frame


def apply_signature(model_path: str | Path, matrix_path: str | Path, orientation: str | None = None,
                    scale: str | None = None, feature_id_column: str | None = None,
                    separator: str | None = None, allow_missing: bool = False,
                    output: str | Path | None = None) -> dict[str, Any]:
    """Load a saved ``model.json``, score a new methylation matrix, and write ``applied_scores.csv``.

    The new matrix defaults to the input conventions recorded in the model (orientation, scale,
    feature-id column); pass the overrides when the new cohort's file is laid out differently.
    """
    model = json.loads(Path(model_path).read_text())
    recorded = model.get("input", {})
    matrix = _read_matrix(str(matrix_path),
                          orientation or recorded.get("orientation", "features_by_samples"),
                          feature_id_column if feature_id_column is not None else recorded.get("feature_id_column", "first"),
                          separator)
    if scale is not None:
        model = {**model, "input": {**recorded, "scale": scale}}
    scores = apply_model(model, matrix, allow_missing=allow_missing)
    destination = Path(output) if output else Path(matrix_path).with_name("applied_scores.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(destination, index=False)
    return {"output": str(destination), "samples": int(len(scores)),
            "features_used": int(scores["features_used"].iloc[0]) if "features_used" in scores else int(model["feature_count"]),
            "features_total": int(model["feature_count"]), "analysis_id": model.get("analysis_id"),
            "model_fingerprint": model.get("fingerprint")}


_PARITY_ARTIFACTS = {"ranking": "feature", "metrics": "feature_count",
                     "selected_features": "feature", "risk_scores": "sample_id"}
# Search-curve AUC / C-index are rank-based step functions of the LOO fold selections: a
# near-tie feature swapping in at a *non-selected* count nudges them without changing the
# final signature. They are reported but do not gate the verdict, which rests on the
# per-feature coefficients, the selected signature, and the risk scores.
_PARITY_ADVISORY_ARTIFACTS = frozenset({"metrics"})
# Integer position/count columns are derived from the (tolerance-compared) statistics; two
# features with |z| equal to within the reference solver's own tolerance may swap adjacent
# ranks, which is not a numeric disagreement, so positions are excluded from the diff.
_PARITY_SKIP_COLUMNS = frozenset({"rank"})
# The batched CPU and CUDA kernels run the identical float64 algorithm, so they must
# agree far more tightly than either does with the scipy reference solver.
_PARITY_CPU_GPU_TOL = 1e-9


def _artifact_diff(reference: pd.DataFrame, candidate: pd.DataFrame, key: str) -> dict[str, Any]:
    """Compare two result frames aligned on ``key`` and summarize numeric agreement."""
    ref = reference.set_index(key)
    cand = candidate.set_index(key)
    keys_match = set(ref.index) == set(cand.index)
    common = [k for k in ref.index if k in cand.index]
    ref = ref.loc[common]
    cand = cand.loc[common]
    numeric = [c for c in ref.columns if c in cand.columns and c not in _PARITY_SKIP_COLUMNS
               and np.issubdtype(ref[c].dtype, np.number)]
    max_abs = max_rel = 0.0
    nonfinite_match = True
    for col in numeric:
        a = ref[col].to_numpy(float)
        b = cand[col].to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            d = np.abs(a[finite] - b[finite])
            max_abs = max(max_abs, float(d.max()))
            max_rel = max(max_rel, float((d / np.maximum(np.abs(a[finite]), 1e-12)).max()))
        nonfinite_match = nonfinite_match and np.array_equal(~np.isfinite(a), ~np.isfinite(b))
    return {"rows": len(common), "columns": numeric, "keys_match": bool(keys_match),
            "nonfinite_match": bool(nonfinite_match), "max_abs": max_abs, "max_rel": max_rel}


def _run_engine_for_parity(base: dict[str, Any], engine: str, feature_count: int | None,
                           data: InputData, workers: int | None) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.setdefault("execution", {})["engine"] = engine
    if workers is not None:
        cfg["execution"]["workers"] = workers
    cfg["execution"]["resume_run_id"] = None  # never share caches between engines
    cfg["_run_id"] = uuid.uuid4().hex
    return fit_metacpg(cfg, feature_count=feature_count, data=data)


def verify_engine_parity(config: dict[str, Any] | str | Path, feature_count: int | None = None,
                         engines: Iterable[str] = ("exact", "batched", "gpu"),
                         tolerance: float = 1e-6, workers: int | None = None) -> dict[str, Any]:
    """Run the full signatures workflow under several engines and compare every output.

    The first engine (``exact`` by default) is the reference. Every other engine is
    compared artifact-by-artifact against it, and the two batched kernels (CPU ``batched``
    and CUDA ``gpu``) are additionally compared against each other under a strict
    tolerance to prove the CPU and GPU paths are numerically equivalent. Returns a report
    dict; ``overall_pass`` is True when every requested comparison is within tolerance and
    the selected signatures are identical.
    """
    base = read_config(config) if not isinstance(config, dict) else config
    data = validate_inputs(base)
    ordered = list(dict.fromkeys(engines))
    reference = ordered[0]
    results: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for engine in ordered:
        try:
            resolved = _resolve_engine({**base, "execution": {**base.get("execution", {}), "engine": engine}})
            _make_backend(resolved)  # surfaces "gpu requested but no CUDA" before a long run
            results[engine] = _run_engine_for_parity(base, engine, feature_count, data, workers)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the whole check
            skipped[engine] = f"{type(exc).__name__}: {exc}"

    def compare(ref_key: str, cand_key: str, tol: float) -> dict[str, Any] | None:
        if ref_key not in results or cand_key not in results:
            return None
        ref_res, cand_res = results[ref_key], results[cand_key]
        artifacts = {}
        ok = True
        for name, key in _PARITY_ARTIFACTS.items():
            if isinstance(ref_res.get(name), pd.DataFrame) and isinstance(cand_res.get(name), pd.DataFrame):
                diff = _artifact_diff(ref_res[name], cand_res[name], key)
                advisory = name in _PARITY_ADVISORY_ARTIFACTS
                diff["advisory"] = advisory
                within = diff["keys_match"] and diff["nonfinite_match"] and diff["max_abs"] <= tol
                diff["within_tolerance"] = bool(within)
                artifacts[name] = diff
                if not advisory:
                    ok = ok and within
        sel_ref = ref_res["selected_features"]["feature"].tolist()
        sel_cand = cand_res["selected_features"]["feature"].tolist()
        identical = sel_ref == sel_cand
        overlap = len(set(sel_ref) & set(sel_cand))
        worst = max((a["max_abs"] for a in artifacts.values() if not a["advisory"]), default=0.0)
        return {"tolerance": tol, "artifacts": artifacts, "selected_features_identical": identical,
                "selected_features_overlap": overlap, "selected_features_total": len(sel_ref),
                "max_abs": worst, "pass": bool(ok and identical)}

    comparisons: dict[str, dict[str, Any]] = {}
    for engine in ordered:
        if engine != reference and engine in results:
            comparisons[f"{reference}_vs_{engine}"] = compare(reference, engine, tolerance)
    if "batched" in results and "gpu" in results:
        comparisons["batched_vs_gpu"] = compare("batched", "gpu", _PARITY_CPU_GPU_TOL)

    overall = bool(comparisons) and reference in results
    overall = overall and all(c["pass"] for c in comparisons.values())
    return {"config": str(base.get("_config_path")), "feature_count": feature_count,
            "tolerance": tolerance, "cpu_gpu_tolerance": _PARITY_CPU_GPU_TOL,
            "reference": reference, "engines_run": list(results), "engines_skipped": skipped,
            "comparisons": comparisons, "overall_pass": overall}


def format_parity_report(report: dict[str, Any]) -> str:
    """Render :func:`verify_engine_parity` output as a human-readable pass/fail table."""
    lines = ["metaCpG engine parity report",
             f"  config           : {report['config']}",
             f"  feature_count    : {report['feature_count']}",
             f"  reference engine : {report['reference']}",
             f"  engines run      : {', '.join(report['engines_run']) or 'none'}",
             f"  coef tolerance   : {report['tolerance']:.1e}   (CPU-vs-GPU: {report['cpu_gpu_tolerance']:.1e})"]
    if report["engines_skipped"]:
        for engine, reason in report["engines_skipped"].items():
            lines.append(f"  skipped {engine:8s}: {reason}")
    header = f"  {'comparison':22s} {'artifact':18s} {'rows':>6s} {'max|abs|':>11s} {'max|rel|':>11s} {'keys':>5s} {'result':>7s}"
    lines += ["", header, "  " + "-" * (len(header) - 2)]
    for name, comp in report["comparisons"].items():
        if comp is None:
            continue
        for artifact, diff in comp["artifacts"].items():
            if diff["advisory"]:
                verdict = "advisory"
            else:
                verdict = "ok" if diff["within_tolerance"] else "DIFF"
            label = f"{artifact}*" if diff["advisory"] else artifact
            lines.append(f"  {name:22s} {label:18s} {diff['rows']:6d} {diff['max_abs']:11.3e} "
                         f"{diff['max_rel']:11.3e} {('yes' if diff['keys_match'] else 'no'):>5s} {verdict:>8s}")
        sel = "identical" if comp["selected_features_identical"] else \
              f"DIFFER ({comp['selected_features_overlap']}/{comp['selected_features_total']} overlap)"
        lines.append(f"  {name:22s} {'selected_features':18s} {'':>6s} {'':>11s} {'':>11s} {'':>5s} {sel}")
        lines.append(f"  {name:22s} -> {'PASS' if comp['pass'] else 'FAIL'}")
        lines.append("")
    lines.append("  * advisory: rank-based search-curve metric, reported but not gating the verdict")
    lines.append(f"  OVERALL: {'PASS' if report['overall_pass'] else 'FAIL'}")
    return "\n".join(lines)


def run_worker(config: dict[str, Any] | str | Path, phase: str, feature_count: int,
               run_id: str, progress: ProgressReporter | None = None) -> dict[str, Any]:
    cfg = read_config(config) if not isinstance(config, dict) else config
    if phase not in {"search", "extraction"}:
        raise ValueError("phase must be search or extraction")
    cfg.setdefault("execution", {})["resume_run_id"] = run_id
    cfg["_run_id"] = run_id
    owns_progress = progress is None
    progress = progress or ProgressReporter(heartbeat_seconds=heartbeat_seconds_from_config(cfg),
                                             show_progress=progress_enabled_from_config(cfg))
    try:
        progress.stage("Validating inputs")
        data = validate_inputs(cfg)
        progress.stage("Validation complete")
        progress.start_phase(phase.capitalize(), _loo_fit_total(data, [int(feature_count)]))
        try:
            counter = FitCounter()
            result = _exact_loo(data, [int(feature_count)], phase,
                                cfg.get("execution", {}).get("workers", 1), cfg, counter,
                                allow_resume=True, progress=progress)
        finally:
            progress.finish_phase()
        return {"phase": phase, "feature_count": int(feature_count), "run_id": run_id,
                "tasks": len(result["task_provenance"]), "cox_fits_executed": counter.value}
    finally:
        if owns_progress:
            progress.close()


def aggregate_run(config: dict[str, Any] | str | Path, run_id: str,
                  feature_count: int, progress: ProgressReporter | None = None) -> dict[str, Any]:
    cfg = read_config(config) if not isinstance(config, dict) else config
    cfg.setdefault("execution", {})["resume_run_id"] = run_id
    cfg["_run_id"] = run_id
    owns_progress = progress is None
    progress = progress or ProgressReporter(heartbeat_seconds=heartbeat_seconds_from_config(cfg),
                                             show_progress=progress_enabled_from_config(cfg))
    try:
        start = time.monotonic()
        progress.stage("Validating inputs")
        data = validate_inputs(cfg)
        progress.stage("Validation complete")
        search = search_features(cfg, data, progress)
        result = fit_metacpg(cfg, search, int(feature_count), data, _resume_extraction=True,
                             progress=progress)
        progress.stage("Writing results")
        result["output_dir"] = str(_write_results(cfg, result, time.monotonic() - start))
        progress.stage("Results written")
        return result
    finally:
        if owns_progress:
            progress.close()


def _slurm_directives(cfg: dict[str, Any], suffix: str, array: str | None = None) -> list[str]:
    slurm = cfg.get("execution", {}).get("slurm", {})
    directives = [f"#SBATCH --job-name={cfg['analysis_id']}-{suffix}", f"#SBATCH --cpus-per-task={slurm.get('cpus', 4)}",
                  f"#SBATCH --mem={slurm.get('memory', '16G')}", f"#SBATCH --time={slurm.get('time', '24:00:00')}"]
    if array:
        directives.append(f"#SBATCH --array={array}")
    for key in ("partition", "account", "mail-user", "mail-type"):
        cfg_key = key.replace("-", "_")
        if slurm.get(cfg_key):
            directives.append(f"#SBATCH --{key}={slurm[cfg_key]}")
    return directives


def submit_slurm(config: dict[str, Any] | str | Path, dry_run: bool = False) -> dict[str, Any]:
    cfg = read_config(config) if not isinstance(config, dict) else config
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    selected = cfg.get("selection", {}).get("feature_count", cfg.get("search", {}).get("feature_count_override"))
    if selected is None:
        raise ValueError("SLURM legacy_exact submission requires a manually selected feature count")
    search = cfg.get("search", {}); counts = list(range(int(search.get("min_features", 5)),
        int(search.get("max_features", 60)) + 1, int(search.get("step", 1))))
    run_id = cfg.get("execution", {}).get("resume_run_id") or uuid.uuid4().hex
    setup = cfg.get("execution", {}).get("slurm", {}).get("setup", [])
    prefix = ["#!/usr/bin/env bash", "set -euo pipefail"]
    search_script = out / "search-array.slurm.sh"
    search_script.write_text("\n".join([*prefix, *_slurm_directives(cfg,"search",f"0-{len(counts)-1}"),
        *setup, f"COUNTS=({' '.join(map(str,counts))})",
        f"metacpg-py worker --config {cfg['_config_path']} --phase search --feature-count ${{COUNTS[$SLURM_ARRAY_TASK_ID]}} --run-id {run_id}",""]))
    extraction_script = out / "extraction.slurm.sh"
    extraction_script.write_text("\n".join([*prefix,*_slurm_directives(cfg,"extract"),*setup,
        f"metacpg-py worker --config {cfg['_config_path']} --phase extraction --feature-count {selected} --run-id {run_id}",""]))
    aggregate_script = out / "aggregate.slurm.sh"
    aggregate_script.write_text("\n".join([*prefix,*_slurm_directives(cfg,"aggregate"),*setup,
        f"metacpg-py aggregate --config {cfg['_config_path']} --feature-count {selected} --run-id {run_id}",""]))
    scripts = [search_script,extraction_script,aggregate_script]
    for script in scripts:
        script.chmod(0o750)
    if dry_run:
        return {"scripts": [str(x) for x in scripts], "submitted": False, "run_id": run_id}
    search_job = subprocess.run(["sbatch","--parsable",str(search_script)],check=True,text=True,capture_output=True).stdout.strip()
    extraction_job = subprocess.run(["sbatch","--parsable",f"--dependency=afterok:{search_job}",str(extraction_script)],check=True,text=True,capture_output=True).stdout.strip()
    aggregate_job = subprocess.run(["sbatch","--parsable",f"--dependency=afterok:{extraction_job}",str(aggregate_script)],check=True,text=True,capture_output=True).stdout.strip()
    return {"scripts":[str(x) for x in scripts],"submitted":True,"run_id":run_id,
            "job_ids":{"search":search_job,"extraction":extraction_job,"aggregate":aggregate_job}}


def compare_results(reference_dir: str | Path, candidate_dir: str | Path,
                    engine: str = "python") -> dict[str, Any]:
    ref, cand = Path(reference_dir), Path(candidate_dir)
    rf, cf = pd.read_csv(ref / "selected_features.csv"), pd.read_csv(cand / "selected_features.csv")
    rs, cs = pd.read_csv(ref / "risk_scores.csv"), pd.read_csv(cand / "risk_scores.csv")
    shared = set(rf.feature) & set(cf.feature)
    overlap = len(shared) / max(1, len(set(rf.feature)))
    coefficient_rho = coefficient_sign = None
    weight_col = next((x for x in ("standardized_coef", "weights", "coef") if x in rf and x in cf), None)
    if weight_col and len(shared) >= 2:
        weights = rf[["feature", weight_col]].merge(cf[["feature", weight_col]], on="feature", suffixes=("_ref", "_candidate"))
        coefficient_rho = float(spearmanr(weights[f"{weight_col}_ref"], weights[f"{weight_col}_candidate"]).statistic)
        coefficient_sign = float(np.mean(np.sign(weights[f"{weight_col}_ref"]) == np.sign(weights[f"{weight_col}_candidate"])))
    merged = rs.merge(cs, on="sample_id", suffixes=("_ref", "_candidate"))
    rho = float(spearmanr(merged.risk_score_ref, merged.risk_score_candidate).statistic)
    rj, cj = json.loads((ref / "result.json").read_text()), json.loads((cand / "result.json").read_text())
    auc_delta = abs(rj["performance"]["auc"] - cj["performance"]["auc"])
    c_delta = abs(rj["performance"]["c_index"] - cj["performance"]["c_index"])
    thresholds = {"feature_overlap": .95, "risk_spearman": .99, "auc_delta": .01, "c_index_delta": .01} if engine == "r" else {
        "feature_overlap": .80, "risk_spearman": .95, "auc_delta": .03, "c_index_delta": .03}
    checks = {"feature_overlap": overlap >= thresholds["feature_overlap"], "risk_spearman": rho >= thresholds["risk_spearman"],
              "auc_delta": auc_delta <= thresholds["auc_delta"], "c_index_delta": c_delta <= thresholds["c_index_delta"]}
    return {"engine": engine, "metrics": {"feature_overlap": overlap, "risk_spearman": rho,
            "coefficient_spearman": coefficient_rho, "coefficient_sign_agreement": coefficient_sign,
            "auc_delta": auc_delta, "c_index_delta": c_delta}, "thresholds": thresholds,
            "checks": checks, "passed": all(checks.values())}
