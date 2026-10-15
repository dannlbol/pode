"""Per-split TVAE: fit on real training rows only, sample, check plausibility.

The generator is the existing :class:`~pode.generation.synthesizer.TVAEGenerator`
with the (already tuned/promoted) PODE synthesizer config; nothing is tuned
inside a split. Fitted models are cached per split so interrupted benchmarks
resume, and the cache is keyed on the exact training rows so a changed split
can never silently reuse a model that saw other rows.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PodeConfig
from ..imputation.base import resolve_columns, subset_schema
from ..schema.schema import DatasetSchema
from .config import AugmentationConfig


def generator_columns(df: pd.DataFrame, schema: DatasetSchema, cfg: AugmentationConfig) -> list[str]:
    """Columns the TVAE models: the schema columns minus ``generator.exclude_columns``,
    identifiers, ``id_column`` and the external-cohort flag."""
    from ..schema.column_types import ColumnRole

    cols = [c for c in schema.columns if c in df.columns]
    ids = set(schema.columns_of_role(ColumnRole.IDENTIFIER)) | {cfg.id_column, cfg.external.column}
    excluded = set(resolve_columns(cfg.generator.exclude_columns, cols)) | set(matrix_columns(df, cfg))
    return [c for c in cols if c not in excluded and c not in ids]


def matrix_columns(df: pd.DataFrame, cfg: AugmentationConfig) -> list[str]:
    """The omics matrix columns of a hybrid generator (empty for the plain TVAE)."""
    if cfg.generator.backend != "hybrid":
        return []
    return resolve_columns(cfg.generator.hybrid.matrix_columns, list(df.columns))


def _fingerprint(train: pd.DataFrame, synth_cfg, seed: int) -> str:
    h = hashlib.sha256()
    # Rounded, so float noise (e.g. hybrid latents recomputed on CPU instead of
    # GPU) does not invalidate a cached model trained on the same rows.
    h.update(pd.util.hash_pandas_object(train.round(4), index=False).values.tobytes())
    h.update(json.dumps(dataclasses.asdict(synth_cfg), sort_keys=True, default=str).encode())
    h.update(str(seed).encode())
    return h.hexdigest()[:16]


class SplitGenerator:
    """A TVAE fitted on one split's real training rows."""

    def __init__(self, gen, columns: list[str], schema: DatasetSchema, fit_seconds: float, cached: bool):
        self.gen = gen
        self.columns = columns
        self.schema = schema
        self.fit_seconds = fit_seconds
        self.cached = cached

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        schema: DatasetSchema,
        pode_config: PodeConfig,
        cfg: AugmentationConfig,
        seed: int,
        cache_dir: Path | None = None,
    ) -> "SplitGenerator":
        from ..generation.synthesizer import TVAEGenerator
        from ..imputation.decoder_imputer import _fit_sdv_on_cpu, _load_generator

        columns = generator_columns(train, schema, cfg)
        sub = subset_schema(schema, columns)
        synth_cfg = dataclasses.replace(pode_config.synthesizer)
        if cfg.generator.epochs is not None:
            synth_cfg.epochs = cfg.generator.epochs
        data = train[columns].reset_index(drop=True)
        key = _fingerprint(data, synth_cfg, seed)

        if cache_dir is not None and cfg.generator.cache:
            meta_path = cache_dir / "meta.json"
            if meta_path.is_file() and json.loads(meta_path.read_text()).get("fingerprint") == key:
                gen = _load_generator(cache_dir / "model.pkl", sub, cfg.device)
                return cls(gen, columns, sub, json.loads(meta_path.read_text())["fit_seconds"], True)

        t0 = time.perf_counter()
        if cfg.device == "cpu":
            gen = TVAEGenerator(sub, synth_cfg)
            gen.synthesizer = _fit_sdv_on_cpu(sub, synth_cfg, data, seed)
        else:
            gen = TVAEGenerator(sub, synth_cfg).fit(data, seed=seed)
        fit_s = time.perf_counter() - t0
        if cache_dir is not None and cfg.generator.cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            gen.save(cache_dir / "model.pkl")
            (cache_dir / "meta.json").write_text(json.dumps(
                {"fingerprint": key, "fit_seconds": fit_s, "rows": len(data), "columns": columns,
                 "epochs": synth_cfg.epochs, "seed": seed}, indent=2))
        return cls(gen, columns, sub, fit_s, False)

    @classmethod
    def from_model(cls, model_path: str | Path, schema: DatasetSchema, cfg: AugmentationConfig) -> "SplitGenerator":
        """Wrap a pre-trained model (external-cohort evaluation only)."""
        from ..imputation.decoder_imputer import _load_generator

        gen = _load_generator(model_path, schema, cfg.device)
        meta = gen.synthesizer.get_metadata().to_dict()
        cols_meta = meta.get("columns") or next(iter(meta.get("tables", {}).values()), {}).get("columns", {})
        columns = [c for c in cols_meta if c in schema.columns]
        return cls(gen, columns, subset_schema(schema, columns), 0.0, True)

    def view(self, df: pd.DataFrame) -> pd.DataFrame:
        """``df`` in the generator's own column space (identity for the plain TVAE)."""
        return df

    def sample(self, n: int, conditions=None, seed: int | None = None) -> tuple[pd.DataFrame, int]:
        """``n`` synthetic rows (conditioned if given) and the conditional shortfall."""
        if n <= 0:
            return pd.DataFrame(columns=self.columns), 0
        # Under-generation is returned as a count rather than warned about.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                df = self.gen.sample(n, conditions=conditions or None, seed=seed)
            except RuntimeError as exc:
                if "no rows" not in str(exc):
                    raise
                df = pd.DataFrame(columns=self.columns)
        df = df.reset_index(drop=True)
        return df, max(0, n - len(df))


# ---- hybrid: matrix -> latents -> TVAE -> decode ------------------------------------
def compressor_params(cfg: AugmentationConfig) -> dict:
    """Compressor settings: a tuned selected_config.json's best_config, then overrides."""
    from synthvae.cli.hybrid import BASE_COMPRESSOR_CONFIG

    h = cfg.generator.hybrid
    params = dict(BASE_COMPRESSOR_CONFIG)
    if h.compressor_config:
        blob = json.loads(Path(h.compressor_config).expanduser().read_text())
        params.update(blob.get("best_config", blob))
    params.update(h.compressor)
    params["matrix_kind"] = h.matrix_kind
    return params


class HybridSplitGenerator(SplitGenerator):
    """Compressor fitted on the split's real matrix rows + a TVAE on clinical + latents.

    ``columns`` / ``schema`` describe the TVAE's space (clinical + latents), which
    is where fidelity and privacy are measured (:meth:`view` maps real rows into
    it). :meth:`sample` returns clinical + latent + decoded matrix columns.
    """

    def __init__(self, tvae: SplitGenerator, pre, compressor, matrix_cols: list[str], latent_cols: list[str],
                 resid_sd: np.ndarray | None, fit_seconds: float, cached: bool):
        super().__init__(tvae.gen, tvae.columns, tvae.schema, fit_seconds, cached)
        self.tvae, self.pre, self.compressor = tvae, pre, compressor
        self.matrix_cols, self.latent_cols, self.resid_sd = matrix_cols, latent_cols, resid_sd

    @classmethod
    def fit(cls, train, schema, pode_config, cfg, seed, cache_dir=None, endpoint=None) -> "HybridSplitGenerator":
        import pickle

        from synthvae.cli.hybrid import build_compressor, build_preprocessor

        h = cfg.generator.hybrid
        params = compressor_params(cfg)
        params["seed"] = int(params.get("seed", 0)) + seed
        if cfg.device == "cpu":
            params["device"] = "cpu"
        mcols = matrix_columns(train, cfg)
        time_col, event_col = endpoint
        X_raw = train[mcols].to_numpy(float)
        key = hashlib.sha256(np.ascontiguousarray(X_raw).tobytes()
                             + train[[time_col, event_col]].to_numpy(float).tobytes()
                             + json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16]
        use_cache = cache_dir is not None and cfg.generator.cache
        meta_path = cache_dir / "compressor_meta.pkl" if cache_dir is not None else None
        cached = False
        if use_cache and meta_path.is_file():
            meta = pickle.loads(meta_path.read_bytes())
            if meta["key"] == key:
                pre, resid_sd, comp_s = meta["pre"], meta["resid_sd"], meta["seconds"]
                comp = _compressor_class(params["compressor"]).load(str(cache_dir / "compressor.pt"))
                cached = True
        if not cached:
            t0 = time.perf_counter()
            pre = build_preprocessor(params)
            X = pre.fit_transform(X_raw)
            comp = build_compressor(params["compressor"], params)
            comp.fit(X, train[time_col].to_numpy(float), train[event_col].to_numpy(float))
            resid_sd = (X - comp.inverse_transform(comp.transform(X))).std(axis=0)
            comp_s = time.perf_counter() - t0
            if use_cache:
                cache_dir.mkdir(parents=True, exist_ok=True)
                comp.save(str(cache_dir / "compressor.pt"))
                meta_path.write_bytes(pickle.dumps({"key": key, "pre": pre, "resid_sd": resid_sd, "seconds": comp_s}))
        latent_cols = [f"{h.latent_prefix}{i}" for i in range(comp.n_latent)]
        train_lat = encode_matrix(train, pre, comp, mcols, latent_cols)
        tvae = SplitGenerator.fit(train_lat, _latent_schema(schema, train_lat, latent_cols), pode_config, cfg, seed,
                                  cache_dir)
        return cls(tvae, pre, comp, mcols, latent_cols, resid_sd if h.residual_noise else None,
                   comp_s + tvae.fit_seconds, cached and tvae.cached)

    def view(self, df: pd.DataFrame) -> pd.DataFrame:
        """Real rows with their compressor latents (posterior means) appended."""
        return encode_matrix(df, self.pre, self.compressor, self.matrix_cols, self.latent_cols)

    def decode(self, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        X = self.compressor.inverse_transform(np.asarray(Z, float))
        if self.resid_sd is not None:
            X = X + rng.normal(0.0, 1.0, X.shape) * self.resid_sd
        return self.pre.inverse_transform(X)

    def sample(self, n: int, conditions=None, seed: int | None = None) -> tuple[pd.DataFrame, int]:
        rows, short = self.tvae.sample(n, conditions, seed)
        if rows.empty:
            return rows, short
        M = self.decode(rows[self.latent_cols].to_numpy(float), np.random.default_rng(seed))
        return pd.concat([rows, pd.DataFrame(M, columns=self.matrix_cols)], axis=1), short


def encode_matrix(df: pd.DataFrame, pre, compressor, matrix_cols: list[str], latent_cols: list[str]) -> pd.DataFrame:
    """``df`` with the compressor latents of its matrix columns appended."""
    Z = compressor.transform(pre.transform(df[matrix_cols].to_numpy(float)))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(Z, columns=latent_cols)], axis=1)


def _compressor_class(kind: str):
    if kind == "survival_vae":
        from synthvae.compressors.survival_vae import SurvivalVAE

        return SurvivalVAE
    from synthvae.compressors.pca import PCACompressor

    return PCACompressor


def _latent_schema(schema: DatasetSchema, df: pd.DataFrame, latent_cols: list[str]) -> DatasetSchema:
    from ..schema.column_types import ColumnRole, ColumnType

    keep = [c for c in schema.columns if c in df.columns]
    out = subset_schema(schema, keep)
    for c in latent_cols:
        out.types[c] = ColumnType.CONTINUOUS
        out.roles[c] = ColumnRole.FEATURE
    return out


# ---- plausibility rules ------------------------------------------------------
def apply_constraints(df: pd.DataFrame, constraints: dict) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop synthetic rows that break ``constraints``; return kept rows and drop counts."""
    if df.empty or not constraints:
        return df, {}
    keep = pd.Series(True, index=df.index)
    counts: dict[str, int] = {}

    def _drop(name: str, bad: pd.Series) -> None:
        nonlocal keep
        bad = bad.fillna(False) & keep
        if bad.any():
            counts[name] = counts.get(name, 0) + int(bad.sum())
        keep &= ~bad

    for col, b in (constraints.get("ranges") or {}).items():
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            lo, hi = b.get("min", -np.inf), b.get("max", np.inf)
            _drop(f"range:{col}", (v < lo) | (v > hi))
    for col in constraints.get("positive") or []:
        if col in df.columns:
            _drop(f"positive:{col}", pd.to_numeric(df[col], errors="coerce") <= 0)
    for a, b in constraints.get("ordered") or []:
        if a in df.columns and b in df.columns:
            va, vb = pd.to_numeric(df[a], errors="coerce"), pd.to_numeric(df[b], errors="coerce")
            _drop(f"ordered:{a}<={b}", va.notna() & vb.notna() & (va > vb))
    for i, combo in enumerate(constraints.get("forbid") or []):
        if all(c in df.columns for c in combo):
            m = pd.Series(True, index=df.index)
            for c, v in combo.items():
                m &= df[c].astype(str) == str(v)
            _drop(f"forbid:{i}", m)
    return df[keep].reset_index(drop=True), counts


def with_survival_rules(schema: DatasetSchema, constraints: dict) -> dict:
    """``constraints`` plus the rule every synthetic row must satisfy: survival time > 0."""
    out = dict(constraints or {})
    times = [p.time_col for p in schema.survival_pairs]
    out["positive"] = list(dict.fromkeys([*(out.get("positive") or []), *times]))
    return out
