"""A shared column encoder built from per-site summaries, never from pooled rows.

Every site in a federated TVAE must encode its rows identically: same one-hot
vocabularies, same Gaussian-mixture modes per continuous column (ctgan's
mode-specific normalisation). ctgan fits those from the training table. Here
the encoder only ever needs **per-column marginals**, so it is fitted on a
pseudo-sample drawn independently, column by column, from merged site
summaries:

1. **Profile** (``site_column_profile``): per column, the missing count; for a
   categorical column its category counts, with categories seen in fewer than
   ``min_cell`` patients at that site pooled into ``__rare__`` (small-cell
   suppression); for a numeric column two robust quantiles that set the shared
   range (or public ``bounds`` from the config instead).
2. **Histograms** (``site_histograms``): per numeric column, counts on the
   agreed bin edges.
3. The coordinator merges the summaries, draws the pseudo-sample and fits the
   ctgan ``DataTransformer``; the fitted encoder is broadcast to every site.

Under DP, sites add Laplace noise to every count they send (the summaries'
budget is ``dp.transformer_epsilon``, split evenly over columns and the two
steps). Data-derived ranges and category vocabularies are *not* covered by that
noise; the encoder lists such columns in ``dp_notes`` so the report can say so
(supply ``transformer.bounds`` / ``transformer.categories`` to remove them).

Missing values: a numeric column with any missing values gets a ``<col>__missing``
indicator column (and a median fill); a categorical one gets a ``__NA__`` level.
Both are turned back into missing values on decoding.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ctgan.data_transformer import DataTransformer

from ..imputation.base import category_key
from .config import DPConfig, TransformerConfig

NA_LEVEL = "__NA__"
MISSING_SUFFIX = "__missing"
SITE_COLUMN = "__site__"


class SyncDataTransformer(DataTransformer):
    """ctgan ``DataTransformer`` that never spawns its own all-cores joblib pool."""

    def transform(self, raw_data):
        cols = self._synchronous_transform(raw_data, self._column_transform_info_list)
        return np.concatenate(cols, axis=1).astype(float)


# ---- site operations -----------------------------------------------------------
def _is_number(key: str) -> bool:
    try:
        float(key)
        return True
    except (TypeError, ValueError):
        return False


def site_column_profile(data: pd.DataFrame, columns: list[str], discrete: list[str], min_cell: int,
                        rare_label: str, range_quantiles: list[float], dp_scale: float | None = None,
                        seed: int = 0) -> dict:
    """Per-column summaries a site discloses for the shared encoder (see module docstring)."""
    rng = np.random.default_rng(seed)
    noise = (lambda v: float(v + rng.laplace(0, dp_scale))) if dp_scale else (lambda v: float(v))
    out = {}
    for c in columns:
        x = data[c]
        miss = int(x.isna().sum())
        if c in discrete:
            keys = [k for k in category_key(x.to_numpy()) if k is not None]
            counts = Counter(keys)
            shown = {k: v for k, v in counts.items() if v >= min_cell}
            rare = sum(v for v in counts.values() if v < min_cell)
            if rare:
                shown[rare_label] = rare
            out[c] = {"kind": "discrete", "missing": noise(miss),
                      "counts": {k: noise(v) for k, v in shown.items()},
                      "numeric_values": bool(keys) and all(_is_number(k) for k in counts)}
        else:
            v = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
            q = [float(np.quantile(v, p)) for p in range_quantiles] if len(v) else None
            out[c] = {"kind": "numeric", "missing": noise(miss), "quantiles": q,
                      "integer": bool(len(v)) and bool(np.all(np.isclose(v, np.round(v))))}
    return out


def site_histograms(data: pd.DataFrame, edges: dict[str, np.ndarray], dp_scale: float | None = None,
                    seed: int = 0) -> dict[str, np.ndarray]:
    """Counts of each numeric column on the agreed bin edges (values clipped into range)."""
    rng = np.random.default_rng(seed)
    out = {}
    for c, e in edges.items():
        v = pd.to_numeric(data[c], errors="coerce").dropna().to_numpy(float)
        h = np.histogram(np.clip(v, e[0], e[-1]), bins=e)[0].astype(float)
        if dp_scale:
            h = h + rng.laplace(0, dp_scale, size=h.shape)
        out[c] = h
    return out


# ---- the encoder ---------------------------------------------------------------
@dataclass
class ColumnSpec:
    name: str
    kind: str  # numeric | discrete
    categories: list[str] = field(default_factory=list)
    numeric_values: bool = False
    edges: np.ndarray | None = None
    hist: np.ndarray | None = None
    integer: bool = False
    missing_rate: float = 0.0

    @property
    def lo(self) -> float:
        return float(self.edges[0])

    @property
    def hi(self) -> float:
        return float(self.edges[-1])

    @property
    def has_missing(self) -> bool:
        return self.missing_rate > 0

    def fill_value(self) -> float:
        """Median of the merged histogram (used to fill missing numeric cells)."""
        h = np.clip(self.hist, 0, None)
        if h.sum() <= 0:
            return 0.5 * (self.lo + self.hi)
        cdf = np.cumsum(h) / h.sum()
        i = int(np.searchsorted(cdf, 0.5))
        return float(0.5 * (self.edges[i] + self.edges[i + 1]))


class FederatedEncoder:
    """Encodes rows to the TVAE's input space and decodes samples back, identically at every site."""

    def __init__(self, specs: list[ColumnSpec], transformer: DataTransformer, site_names: list[str] | None,
                 dp_notes: list[str], rare_label: str):
        self.specs = specs
        self.transformer = transformer
        self.site_names = site_names
        self.dp_notes = dp_notes
        self.rare_label = rare_label

    @property
    def columns(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def output_info_list(self):
        return self.transformer.output_info_list

    @property
    def output_dimensions(self) -> int:
        return self.transformer.output_dimensions

    def model_frame(self, df: pd.DataFrame, site: str | None = None) -> pd.DataFrame:
        """``df`` in the encoder's model columns (indicators, fills, string categories)."""
        out = {}
        for s in self.specs:
            x = df[s.name] if s.name in df.columns else pd.Series([np.nan] * len(df), index=df.index)
            if s.kind == "numeric":
                v = pd.to_numeric(x, errors="coerce")
                if s.has_missing:
                    out[s.name + MISSING_SUFFIX] = np.where(v.isna(), "1", "0")
                out[s.name] = v.fillna(s.fill_value()).clip(s.lo, s.hi).to_numpy(float)
            else:
                cats = set(s.categories)
                fallback = self.rare_label if self.rare_label in cats else s.categories[0]
                keys = category_key(x.to_numpy())
                out[s.name] = [NA_LEVEL if k is None and NA_LEVEL in cats else
                               (k if k in cats else fallback) for k in keys]
        if self.site_names:
            out[SITE_COLUMN] = [str(site) if site is not None else self.site_names[0]] * len(df)
        return pd.DataFrame(out, index=df.index)[self._model_columns()]

    def _model_columns(self) -> list[str]:
        cols = []
        for s in self.specs:
            if s.kind == "numeric" and s.has_missing:
                cols.append(s.name + MISSING_SUFFIX)
            cols.append(s.name)
        if self.site_names:
            cols.append(SITE_COLUMN)
        return cols

    def _reset(self) -> None:
        """rdt transformers keep their own advancing random state: reset it so that
        encoding depends only on the rows (and is identical at every site)."""
        for info in self.transformer._column_transform_info_list:
            if hasattr(info.transform, "reset_randomization"):
                info.transform.reset_randomization()

    def encode(self, df: pd.DataFrame, site: str | None = None) -> np.ndarray:
        self._reset()
        return self.transformer.transform(self.model_frame(df, site)).astype(np.float32)

    def decode(self, samples: np.ndarray, sigmas: np.ndarray | None = None) -> pd.DataFrame:
        self._reset()
        raw = self.transformer.inverse_transform(np.asarray(samples, np.float64), sigmas)
        out = {}
        for s in self.specs:
            if s.kind == "numeric":
                v = pd.to_numeric(raw[s.name], errors="coerce").clip(s.lo, s.hi)
                if s.integer:
                    v = v.round()
                if s.has_missing:
                    v = v.mask(raw[s.name + MISSING_SUFFIX].astype(str) == "1")
                out[s.name] = v.to_numpy()
            else:
                v = raw[s.name].astype(str)
                v = v.mask(v == NA_LEVEL)
                if s.numeric_values:
                    num = pd.to_numeric(v, errors="coerce")
                    if (num.notna() | v.isna()).all():  # keep strings if e.g. __rare__ was drawn
                        v = num
                out[s.name] = v.to_numpy()
        frame = pd.DataFrame(out)
        if self.site_names:
            frame[SITE_COLUMN] = raw[SITE_COLUMN].astype(str).to_numpy()
        return frame


def _pseudo_sample(specs: list[ColumnSpec], rows: int, site_names: list[str] | None,
                   seed: int) -> pd.DataFrame:
    """Independent per-column draws from the merged summaries (+ one row per category)."""
    rng = np.random.default_rng(seed)
    data = {}
    extra = max([len(s.categories) for s in specs if s.kind == "discrete"] + [2])
    if site_names:
        extra = max(extra, len(site_names))
    total = rows + extra
    for s in specs:
        if s.kind == "numeric":
            p = np.clip(s.hist, 0, None)
            p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
            b = rng.choice(len(p), size=total, p=p)
            v = rng.uniform(s.edges[b], s.edges[b + 1])
            if s.integer:
                v = np.clip(np.round(v), s.lo, s.hi)
            data[s.name] = v
            if s.has_missing:
                ind = (rng.random(total) < s.missing_rate).astype(int).astype(str)
                ind[:2] = ["0", "1"]
                data[s.name + MISSING_SUFFIX] = ind
        else:
            counts = np.clip(np.array([s_c for s_c in s.hist], float), 0, None)
            p = counts / counts.sum() if counts.sum() > 0 else np.full(len(counts), 1 / len(counts))
            v = np.asarray(s.categories, dtype=object)[rng.choice(len(p), size=total, p=p)]
            k = len(s.categories)
            v[:extra] = [s.categories[i % k] for i in range(extra)]
            data[s.name] = v
    if site_names:
        data[SITE_COLUMN] = np.asarray([site_names[i % len(site_names)] for i in range(total)], dtype=object)
    return pd.DataFrame(data)


def federated_encoder(
    sites,
    columns: list[str],
    discrete: list[str],
    cfg: TransformerConfig,
    dp: DPConfig | None = None,
    site_in_model: bool = False,
    seed: int = 0,
) -> FederatedEncoder:
    """Run the two-step summary protocol over ``sites`` and fit the shared encoder."""
    dp_on = bool(dp is not None and dp.enabled)
    n_cols = max(1, len(columns))
    # Each patient touches every column's summary in both steps: split the budget.
    scale = (2.0 * n_cols / dp.transformer_epsilon) if dp_on and dp.transformer_epsilon > 0 else None
    profiles = [s.run(site_column_profile, columns, discrete, cfg.min_cell, cfg.rare_label,
                      cfg.range_quantiles, scale, seed + i, label="encoder_profile")
                for i, s in enumerate(sites)]
    n_total = float(sum(s.n for s in sites))
    # Under DP a missing count is only believed when it clears the noise
    # (about two Laplace scales of the summed site noise); a post-processing
    # decision, so it costs no extra privacy.
    miss_floor = 2.0 * scale * np.sqrt(len(sites)) if scale else 0.5
    specs, notes, edges = [], [], {}
    for c in columns:
        parts = [p[c] for p in profiles]
        miss = max(0.0, sum(p["missing"] for p in parts))
        if miss < miss_floor:
            miss = 0.0
        if c in discrete:
            totals: Counter = Counter()
            for p in parts:
                for k, v in p["counts"].items():
                    totals[k] += v
            public = cfg.categories.get(c)
            if public is not None:
                cats = [str(k) for k in category_key(np.asarray(public, dtype=object))]
                if cfg.rare_label in totals and cfg.rare_label not in cats:
                    cats.append(cfg.rare_label)
            else:
                cats = [k for k, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))]
                if dp_on:
                    notes.append(f"category vocabulary of '{c}' is data-derived (not covered by DP)")
            if miss > 0:
                cats.append(NA_LEVEL)
            if not cats:
                cats = [NA_LEVEL]
            counts = np.array([totals.get(k, miss if k == NA_LEVEL else 0.0) for k in cats], float)
            spec = ColumnSpec(c, "discrete", categories=cats, hist=np.clip(counts, 0, None) + 1e-9,
                              numeric_values=any(p["numeric_values"] for p in parts),
                              missing_rate=min(1.0, miss / n_total) if n_total else 0.0)
        else:
            if c in cfg.bounds:
                lo, hi = (float(b) for b in cfg.bounds[c])
            else:
                qs = [p["quantiles"] for p in parts if p["quantiles"] is not None]
                if not qs:
                    raise ValueError(f"Numeric column '{c}' has no observed values at any site.")
                lo, hi = min(q[0] for q in qs), max(q[1] for q in qs)
                if dp_on:
                    notes.append(f"range of '{c}' is data-derived from site quantiles (not covered by DP); "
                                 "set transformer.bounds")
            if hi <= lo:
                lo, hi = lo - 0.5, hi + 0.5
            edges[c] = np.linspace(lo, hi, cfg.bins + 1)
            spec = ColumnSpec(c, "numeric", edges=edges[c], integer=all(p["integer"] for p in parts),
                              missing_rate=min(1.0, miss / n_total) if n_total else 0.0)
        specs.append(spec)
    if edges:
        hists = [s.run(site_histograms, edges, scale, seed + 1000 + i, label="encoder_histograms")
                 for i, s in enumerate(sites)]
        for spec in specs:
            if spec.kind == "numeric":
                spec.hist = np.sum([h[spec.name] for h in hists], axis=0)
    site_names = [s.name for s in sites] if site_in_model else None
    pseudo = _pseudo_sample(specs, cfg.pseudo_rows, site_names, seed)
    from ..reproducibility.seeds import set_all_seeds

    set_all_seeds(seed)
    tr = SyncDataTransformer(max_clusters=cfg.max_clusters)
    disc = [c for c in pseudo.columns if c in discrete or c.endswith(MISSING_SUFFIX) or c == SITE_COLUMN]
    tr.fit(pseudo, disc)
    return FederatedEncoder(specs, tr, site_names, notes, cfg.rare_label)
