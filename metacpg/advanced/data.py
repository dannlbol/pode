from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass
class OmicsBlock:
    name: str
    values: pd.DataFrame  # samples x original features; missing blocks are NaN rows
    present: pd.Series
    config: dict[str, Any]


@dataclass
class AdvancedData:
    sample_ids: pd.Index
    phenotype: pd.DataFrame
    blocks: dict[str, OmicsBlock]
    endpoints: list[dict[str, Any]]
    covariates: list[dict[str, Any]]
    cohort: pd.Series | None


def _resolve(base: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def read_advanced_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if cfg.get("schema_version") != 2:
        raise ValueError("advanced signatures require schema_version: 2")
    cfg["_config_path"] = str(source)
    cfg["_config_dir"] = str(source.parent)
    if "phenotype" in cfg:
        cfg["phenotype"]["path"] = _resolve(source.parent, cfg["phenotype"].get("path"))
    for block in cfg.get("omics", []):
        block["path"] = _resolve(source.parent, block.get("path"))
    for method_cfg in (cfg.get("methods", {}).get("config", {}) or {}).values():
        for key in ("graph_path", "pathway_path", "similarity_path"):
            if method_cfg.get(key):
                method_cfg[key] = _resolve(source.parent, method_cfg[key])
    output = Path(cfg.get("output_dir", "results")).expanduser()
    if not output.is_absolute():
        output = source.parent / output
    cfg["output_dir"] = str(output.resolve())
    validate_advanced_config(cfg)
    return cfg


def validate_advanced_config(cfg: dict[str, Any]) -> None:
    errors: list[str] = []
    for key in ("analysis_id", "phenotype", "omics", "endpoints", "methods", "output_dir"):
        if key not in cfg:
            errors.append(f"missing required key: {key}")
    phenotype = cfg.get("phenotype", {})
    if not phenotype.get("path") or not Path(phenotype["path"]).is_file():
        errors.append(f"phenotype file does not exist: {phenotype.get('path')}")
    mode = cfg.get("alignment", {}).get("mode", "sample_id")
    if mode not in {"sample_id", "row_order"}:
        errors.append("alignment.mode must be sample_id or row_order")
    if cfg.get("alignment", {}).get("sample_policy", "union") != "union":
        errors.append("alignment.sample_policy must be union")
    if mode == "sample_id" and not phenotype.get("sample_id_column"):
        errors.append("phenotype.sample_id_column is required for sample_id alignment")
    blocks = cfg.get("omics", [])
    if not isinstance(blocks, list) or not blocks:
        errors.append("omics must contain at least one block")
    names: list[str] = []
    for block in blocks if isinstance(blocks, list) else []:
        name = block.get("name")
        if not name:
            errors.append("every omics block requires a name")
        names.append(str(name))
        if not block.get("path") or not Path(block["path"]).is_file():
            errors.append(f"omics file does not exist: {block.get('path')}")
        if block.get("orientation", "samples_by_features") not in {
            "samples_by_features", "features_by_samples"
        }:
            errors.append(f"omics block {name}: invalid orientation")
        if block.get("transform", "none") not in {"none", "log1p", "beta_to_m"}:
            errors.append(f"omics block {name}: invalid transform")
        if mode == "sample_id" and block.get("orientation", "samples_by_features") == "samples_by_features" and not block.get("sample_id_column"):
            errors.append(f"omics block {name}: sample_id_column is required")
    if len(names) != len(set(names)):
        errors.append("omics block names must be unique")
    endpoints = cfg.get("endpoints", [])
    if not isinstance(endpoints, list) or not endpoints:
        errors.append("endpoints must contain at least one right-censored endpoint")
    for endpoint in endpoints if isinstance(endpoints, list) else []:
        for key in ("name", "time", "status"):
            if key not in endpoint:
                errors.append(f"endpoint missing required key: {key}")
    endpoint_names = [endpoint.get("name") for endpoint in endpoints if isinstance(endpoint, dict)]
    if len(endpoint_names) != len(set(endpoint_names)):
        errors.append("endpoint names must be unique")
    methods = cfg.get("methods", {}).get("enabled", [])
    if not isinstance(methods, list) or not methods:
        errors.append("methods.enabled must contain at least one method")
    else:
        from .registry import REGISTRY
        unknown = sorted(set(methods) - set(REGISTRY))
        if unknown:
            errors.append(f"unknown signature methods: {', '.join(unknown)}")
    for covariate in cfg.get("covariates", []) or []:
        if covariate.get("mode", "adjust") not in {"adjust", "strata"}:
            errors.append("covariate.mode must be adjust or strata")
        if covariate.get("penalized", False):
            errors.append("schema-v2 clinical covariates must be unpenalized")
    cohort_handling = phenotype.get("cohort_handling", "warn")
    if cohort_handling not in {"warn", "skip"}:
        errors.append("phenotype.cohort_handling must be warn or skip")
    evaluation = cfg.get("evaluation", {}) or {}
    cohort_validation = evaluation.get("cohort_validation", "stratified_kfold")
    if cohort_validation not in {"stratified_kfold", "leave_one_cohort_out"}:
        errors.append("evaluation.cohort_validation must be stratified_kfold or leave_one_cohort_out")
    elif cohort_validation == "leave_one_cohort_out" and not phenotype.get("cohort_column"):
        errors.append(
            "evaluation.cohort_validation: leave_one_cohort_out requires phenotype.cohort_column"
        )
    if not phenotype.get("cohort_column"):
        if "cohort_handling" in phenotype:
            errors.append("phenotype.cohort_handling requires phenotype.cohort_column")
    errors.extend(_missing_phenotype_columns(cfg, mode))
    if errors:
        raise ValueError("; ".join(errors))


def _missing_phenotype_columns(cfg: dict[str, Any], mode: str) -> list[str]:
    """Check the configured phenotype columns exist, reading only the header row.

    Doing this during validation means a mistyped or absent column is reported once, by
    ``pode signatures validate`` and before any method runs, instead of surfacing as an
    identical per-method failure in the benchmark leaderboard.
    """
    phenotype = cfg.get("phenotype", {})
    path = phenotype.get("path")
    if not path or not Path(path).is_file():
        return []
    try:
        available = list(_read_header(path, phenotype.get("separator")))
    except Exception:
        return []
    required: dict[str, str] = {}
    if mode == "sample_id" and phenotype.get("sample_id_column"):
        required[phenotype["sample_id_column"]] = "phenotype.sample_id_column"
    if phenotype.get("cohort_column"):
        required[phenotype["cohort_column"]] = "phenotype.cohort_column"
    for endpoint in cfg.get("endpoints", []) or []:
        if not isinstance(endpoint, dict):
            continue
        for key in ("time", "status"):
            if endpoint.get(key):
                required[endpoint[key]] = f"endpoints[{endpoint.get('name')}].{key}"
    for covariate in cfg.get("covariates", []) or []:
        if isinstance(covariate, dict) and covariate.get("column"):
            required[covariate["column"]] = "covariates[].column"
    errors = []
    for column, source in required.items():
        if column not in available:
            errors.append(
                f"{source}: column {column!r} is not in {Path(path).name}; "
                f"available columns are {', '.join(available)}"
            )
    return errors


def _separator(path: str, separator: str | None) -> str:
    if separator is not None:
        return separator
    return "\t" if path.lower().endswith((".tsv", ".txt")) else ","


def _read(path: str, separator: str | None) -> pd.DataFrame:
    return pd.read_csv(path, sep=_separator(path, separator))


def _read_header(path: str, separator: str | None) -> list[str]:
    return list(pd.read_csv(path, sep=_separator(path, separator), nrows=0).columns)


def _load_block(block: dict[str, Any], mode: str, phenotype_ids: pd.Index,
                phenotype_rows: int, row_order_keep: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.Index]:
    raw = _read(block["path"], block.get("separator"))
    orientation = block.get("orientation", "samples_by_features")
    if orientation == "samples_by_features":
        if mode == "sample_id":
            id_col = block["sample_id_column"]
            if id_col not in raw:
                raise ValueError(f"omics block {block['name']}: missing sample ID column {id_col}")
            ids = pd.Index(raw.pop(id_col).astype(str), name="sample_id")
        else:
            if row_order_keep is not None:
                raw = raw.iloc[row_order_keep]
            if len(raw) != phenotype_rows:
                raise ValueError(
                    f"omics block {block['name']}: row_order requires {phenotype_rows} rows, found {len(raw)}"
                )
            ids = phenotype_ids
        values = raw
    else:
        feature_col = block.get("feature_id_column", "first")
        if feature_col == "first":
            feature_col = raw.columns[0]
        if feature_col not in raw:
            raise ValueError(f"omics block {block['name']}: missing feature ID column {feature_col}")
        features = raw.pop(feature_col).astype(str)
        if features.duplicated().any():
            raise ValueError(f"omics block {block['name']}: duplicate feature IDs")
        values = raw.T
        values.columns = features
        if mode == "sample_id":
            ids = pd.Index(values.index.astype(str), name="sample_id")
        else:
            if row_order_keep is not None:
                values = values.iloc[row_order_keep]
            if len(values) != phenotype_rows:
                raise ValueError(
                    f"omics block {block['name']}: row_order requires {phenotype_rows} samples, found {len(values)}"
                )
            ids = phenotype_ids
    ids = pd.Index(ids.astype(str), name="sample_id")
    if ids.duplicated().any():
        raise ValueError(f"omics block {block['name']}: duplicate sample IDs")
    values.index = ids
    values.columns = values.columns.astype(str)
    if values.columns.duplicated().any():
        raise ValueError(f"omics block {block['name']}: duplicate feature IDs")
    values = values.apply(pd.to_numeric, errors="coerce")
    return values, ids


def load_advanced_data(config: dict[str, Any] | str | Path) -> AdvancedData:
    cfg = read_advanced_config(config) if not isinstance(config, dict) else config
    validate_advanced_config(cfg)
    pcfg = cfg["phenotype"]
    phenotype = _read(pcfg["path"], pcfg.get("separator"))
    mode = cfg.get("alignment", {}).get("mode", "sample_id")
    cohort_col = pcfg.get("cohort_column")
    removed_cohort_ids: set[str] = set()
    row_order_keep: np.ndarray | None = None
    if cohort_col and cohort_col in phenotype:
        # Empty CSV fields are read as NaN, but whitespace-only labels are
        # missing as well. Such samples cannot participate in cohort-aware
        # validation and are excluded from the analysis input.
        missing_cohort = (
            phenotype[cohort_col].isna()
            | phenotype[cohort_col].astype("string").str.strip().eq("").fillna(False)
        )
        if missing_cohort.any():
            row_order_keep = ~missing_cohort.to_numpy()
            id_col = pcfg.get("sample_id_column")
            if id_col and id_col in phenotype:
                removed_cohort_ids = set(phenotype.loc[missing_cohort, id_col].astype(str))
            phenotype = phenotype.loc[~missing_cohort].copy()
    if mode == "sample_id":
        id_col = pcfg["sample_id_column"]
        if id_col not in phenotype:
            raise ValueError(f"phenotype sample ID column is missing: {id_col}")
        ids = pd.Index(phenotype[id_col].astype(str), name="sample_id")
    else:
        ids = pd.Index([f"row_{i}" for i in range(len(phenotype))], name="sample_id")
    if ids.duplicated().any():
        raise ValueError("phenotype sample identifiers are not unique")
    phenotype.index = ids

    required = [x[k] for x in cfg["endpoints"] for k in ("time", "status")]
    required += [x["column"] for x in cfg.get("covariates", [])]
    if cohort_col:
        required.append(cohort_col)
    missing = [col for col in required if col not in phenotype]
    if missing:
        raise ValueError(f"phenotype columns are missing: {', '.join(sorted(set(missing)))}")

    blocks: dict[str, OmicsBlock] = {}
    phenotype_set = set(ids)
    for block_cfg in cfg["omics"]:
        values, block_ids = _load_block(block_cfg, mode, ids, len(phenotype), row_order_keep)
        if removed_cohort_ids:
            keep = ~block_ids.isin(removed_cohort_ids)
            values = values.loc[keep]
            block_ids = block_ids[keep]
        unknown = set(block_ids) - phenotype_set
        if unknown:
            raise ValueError(
                f"omics block {block_cfg['name']} has {len(unknown)} samples without phenotype rows"
            )
        present = pd.Series(ids.isin(block_ids), index=ids, name=block_cfg["name"])
        values = values.reindex(ids)
        blocks[block_cfg["name"]] = OmicsBlock(block_cfg["name"], values, present, block_cfg)

    if mode == "row_order" and any(not block.present.all() for block in blocks.values()):
        raise ValueError("row_order alignment does not support missing modality blocks")
    if not pd.concat([block.present for block in blocks.values()], axis=1).any(axis=1).all():
        raise ValueError("every phenotype sample must have at least one omics block")

    for endpoint in cfg["endpoints"]:
        times = pd.to_numeric(phenotype[endpoint["time"]], errors="coerce")
        status = pd.to_numeric(phenotype[endpoint["status"]], errors="coerce")
        if (times.dropna() <= 0).any():
            raise ValueError(f"endpoint {endpoint['name']} contains non-positive times")
        if not set(status.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"endpoint {endpoint['name']} status must be 0/1")

    if cohort_col and phenotype[cohort_col].isna().any():
        raise ValueError("phenotype cohort column contains missing values")
    cohort = phenotype[cohort_col].astype(str) if cohort_col else None
    return AdvancedData(ids, phenotype, blocks, cfg["endpoints"], cfg.get("covariates", []) or [], cohort)


@dataclass
class Preprocessor:
    block_names: list[str]
    feature_names: list[str]
    block_slices: dict[str, tuple[int, int]]
    centers: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    transforms: dict[str, str]

    @classmethod
    def fit(cls, data: AdvancedData, rows: np.ndarray) -> "Preprocessor":
        names: list[str] = []
        slices: dict[str, tuple[int, int]] = {}
        centers: dict[str, np.ndarray] = {}
        scales: dict[str, np.ndarray] = {}
        transforms: dict[str, str] = {}
        cursor = 0
        for name, block in data.blocks.items():
            values = block.values.iloc[rows].to_numpy(float)
            transform = block.config.get("transform", "none")
            values = _transform(values, transform, name)
            center = np.nanmean(values, axis=0)
            center = np.where(np.isfinite(center), center, 0.0)
            scale = np.nanstd(values, axis=0)
            scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
            if not block.config.get("standardize", True):
                center = np.zeros_like(center)
                scale = np.ones_like(scale)
            block_features = [f"{name}::{x}" for x in block.values.columns]
            names.extend(block_features)
            slices[name] = (cursor, cursor + len(block_features))
            cursor += len(block_features)
            centers[name], scales[name], transforms[name] = center, scale, transform
        return cls(list(data.blocks), names, slices, centers, scales, transforms)

    def transform(self, data: AdvancedData, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrices, masks = [], []
        for name in self.block_names:
            block = data.blocks[name]
            left, right = self.block_slices[name]
            expected = [value.split("::", 1)[1] for value in self.feature_names[left:right]]
            aligned = block.values.reindex(columns=expected)
            values = _transform(aligned.iloc[rows].to_numpy(float), self.transforms[name], name)
            values = (values - self.centers[name]) / self.scales[name]
            values = np.where(np.isfinite(values), values, 0.0)
            matrices.append(values)
            masks.append(block.present.iloc[rows].to_numpy(float)[:, None])
        return np.column_stack(matrices), np.column_stack(masks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_names": self.block_names,
            "feature_names": self.feature_names,
            "block_slices": {k: list(v) for k, v in self.block_slices.items()},
            "centers": {k: v.tolist() for k, v in self.centers.items()},
            "scales": {k: v.tolist() for k, v in self.scales.items()},
            "transforms": self.transforms,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Preprocessor":
        return cls(
            list(value["block_names"]), list(value["feature_names"]),
            {k: tuple(v) for k, v in value["block_slices"].items()},
            {k: np.asarray(v, float) for k, v in value["centers"].items()},
            {k: np.asarray(v, float) for k, v in value["scales"].items()},
            dict(value["transforms"]),
        )


def _transform(values: np.ndarray, transform: str, name: str) -> np.ndarray:
    values = np.asarray(values, float).copy()
    if transform == "log1p":
        if np.nanmin(values) < -1:
            raise ValueError(f"omics block {name}: log1p requires values >= -1")
        values = np.log1p(values)
    elif transform == "beta_to_m":
        finite = values[np.isfinite(values)]
        if finite.size and (finite.min() <= 0 or finite.max() >= 1):
            raise ValueError(f"omics block {name}: beta_to_m requires values strictly between 0 and 1")
        values = np.log2(values / (1 - values))
    return values


def strata_labels(data: AdvancedData, rows: np.ndarray) -> np.ndarray | None:
    columns = [cov["column"] for cov in data.covariates if cov.get("mode", "adjust") == "strata"]
    if not columns:
        return None
    frame = data.phenotype.iloc[rows][columns].astype(str)
    return frame.agg("|".join, axis=1).to_numpy()


def clinical_matrix(data: AdvancedData, rows: np.ndarray,
                    categories: dict[str, list[str]] | None = None) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    categories = {} if categories is None else dict(categories)
    columns: list[np.ndarray] = []
    names: list[str] = []
    for cov in data.covariates:
        if cov.get("mode", "adjust") == "strata":
            continue
        series = data.phenotype.iloc[rows][cov["column"]]
        if cov.get("type", "numeric") == "numeric":
            values = pd.to_numeric(series, errors="coerce").to_numpy(float)
            median = np.nanmedian(values)
            columns.append(np.where(np.isfinite(values), values, median)[:, None])
            names.append(f"clinical::{cov['column']}")
        else:
            known = categories.get(cov["column"])
            if known is None:
                known = sorted(series.dropna().astype(str).unique())
                categories[cov["column"]] = known
            for level in known[1:]:
                columns.append((series.astype(str).to_numpy() == level).astype(float)[:, None])
                names.append(f"clinical::{cov['column']}={level}")
    matrix = np.column_stack(columns) if columns else np.empty((len(rows), 0))
    return matrix, names, categories


@dataclass
class ClinicalPreprocessor:
    covariates: list[dict[str, Any]]
    medians: dict[str, float]
    categories: dict[str, list[str]]
    feature_names: list[str]

    @classmethod
    def fit(cls, data: AdvancedData, rows: np.ndarray) -> "ClinicalPreprocessor":
        medians: dict[str, float] = {}
        categories: dict[str, list[str]] = {}
        names: list[str] = []
        frame = data.phenotype.iloc[rows]
        for cov in data.covariates:
            if cov.get("mode", "adjust") == "strata":
                continue
            column = cov["column"]
            if cov.get("type", "numeric") == "numeric":
                values = pd.to_numeric(frame[column], errors="coerce")
                median = float(values.median()) if values.notna().any() else 0.0
                medians[column] = median
                names.append(f"clinical::{column}")
            else:
                levels = sorted(frame[column].dropna().astype(str).unique())
                categories[column] = levels
                names.extend(f"clinical::{column}={level}" for level in levels[1:])
        return cls(list(data.covariates), medians, categories, names)

    def transform(self, data: AdvancedData, rows: np.ndarray) -> np.ndarray:
        frame = data.phenotype.iloc[rows]
        columns: list[np.ndarray] = []
        for cov in self.covariates:
            if cov.get("mode", "adjust") == "strata":
                continue
            column = cov["column"]
            if cov.get("type", "numeric") == "numeric":
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
                columns.append(np.where(np.isfinite(values), values, self.medians[column])[:, None])
            else:
                values = frame[column].astype(str).to_numpy()
                for level in self.categories[column][1:]:
                    columns.append((values == level).astype(float)[:, None])
        return np.column_stack(columns) if columns else np.empty((len(rows), 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "covariates": self.covariates,
            "medians": self.medians,
            "categories": self.categories,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClinicalPreprocessor":
        return cls(list(value["covariates"]), dict(value["medians"]),
                   {k: list(v) for k, v in value["categories"].items()},
                   list(value["feature_names"]))
