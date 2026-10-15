"""Data-assembly glue for the joint multimodal VAE.

Turns a ``(clinical, expression, survival)`` trio into the tensors and
``ColumnSpec``s that :class:`JointMultimodalVAE` consumes, and maps decoded
outputs back to DataFrames for generation.

Three fitted transformers:

* :class:`TabularTransformer` -- encodes the clinical table into a dense encoder
  input ``x_tab`` ``[standardized continuous | one-hot categorical]`` plus a
  per-head ``tab_target`` (continuous block + one long tensor per categorical
  column) and the ordered ``column_specs`` the :class:`heads.TabularHead` expects.
* :class:`ExpressionTransformer` -- standardizes expression for the gaussian
  likelihood, or ``log1p``-normalizes the encoder input and keeps raw counts as
  the head target (plus per-cell size factors) for ``nb``/``zinb``.
* the survival :class:`~synthvae.survival.discretize.TimeGrid` (built here).

The categorical-typing rule is deliberately *replicated* from
``pipeline.latents.build_metadata`` rather than imported, so the hybrid code path
stays frozen.

Requires torch; import-guarded so the rest of the package imports without it.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..survival.discretize import TimeGrid

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Subset

    from .heads import ColumnSpec

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def infer_categorical(clinical: pd.DataFrame, categorical=None) -> list:
    """Return the clinical columns to treat as categorical.

    Mirrors the rule in ``pipeline.latents.build_metadata`` (named-categorical /
    object|bool dtype / low-cardinality integer). Replicated locally on purpose.
    """
    named = {str(c).casefold() for c in (categorical or [])}
    cats = []
    for c in clinical.columns:
        col = clinical[c]
        is_named = str(c).casefold() in named
        is_dtype = (
            col.dtype == object
            or str(col.dtype).startswith("category")
            or str(col.dtype) == "bool"
        )
        is_low_card_int = (
            pd.api.types.is_integer_dtype(col) and col.nunique(dropna=True) <= 10
        )
        if is_named or is_dtype or is_low_card_int:
            cats.append(c)
    return cats


if _HAS_TORCH:

    class TabularTransformer:
        """Encode/decode the clinical table for the joint VAE."""

        def __init__(self, categorical=None):
            self._categorical_hint = categorical
            self.cont_cols: list = []
            self.cat_cols: list = []
            self.categories: dict = {}      # col -> ordered list of categories
            self.mean_ = None
            self.std_ = None

        # -- fit / transform ------------------------------------------------
        def fit(self, clinical: pd.DataFrame) -> "TabularTransformer":
            cat_cols = infer_categorical(clinical, self._categorical_hint)
            self.cat_cols = list(cat_cols)
            self.cont_cols = [c for c in clinical.columns if c not in self.cat_cols]
            for c in self.cat_cols:
                # stable, sorted category ordering => reproducible codes
                self.categories[c] = sorted(clinical[c].dropna().unique().tolist())
            if self.cont_cols:
                cont = clinical[self.cont_cols].to_numpy(dtype=float)
                self.mean_ = cont.mean(axis=0, keepdims=True)
                self.std_ = cont.std(axis=0, keepdims=True) + 1e-8
            else:
                self.mean_ = np.zeros((1, 0))
                self.std_ = np.ones((1, 0))
            return self

        def _codes(self, clinical: pd.DataFrame) -> list:
            codes = []
            for c in self.cat_cols:
                mapping = {v: i for i, v in enumerate(self.categories[c])}
                col = clinical[c].map(mapping)
                # unseen / NaN -> category 0 (kept in-range for cross-entropy)
                col = col.fillna(0).to_numpy(dtype=np.int64)
                codes.append(np.clip(col, 0, len(self.categories[c]) - 1))
            return codes

        def transform(self, clinical: pd.DataFrame):
            """Return ``(x_tab, tab_cont, cat_codes)`` as numpy arrays.

            * ``x_tab``     -- (n, D_tab) encoder input [cont_std | one-hot cats]
            * ``tab_cont``  -- (n, C_cont) standardized continuous head target
            * ``cat_codes`` -- list of (n,) int arrays, one per categorical column
            """
            n = len(clinical)
            if self.cont_cols:
                cont = clinical[self.cont_cols].to_numpy(dtype=float)
                cont_std = (cont - self.mean_) / self.std_
            else:
                cont_std = np.zeros((n, 0))
            cat_codes = self._codes(clinical)
            onehots = []
            for c, codes in zip(self.cat_cols, cat_codes):
                k = len(self.categories[c])
                oh = np.zeros((n, k), dtype=float)
                oh[np.arange(n), codes] = 1.0
                onehots.append(oh)
            x_tab = np.concatenate([cont_std] + onehots, axis=1) if (
                self.cont_cols or onehots
            ) else np.zeros((n, 0))
            return x_tab.astype(np.float32), cont_std.astype(np.float32), cat_codes

        # -- specs / inverse ------------------------------------------------
        @property
        def column_specs(self) -> list:
            specs = [ColumnSpec(name=c, kind="continuous") for c in self.cont_cols]
            for c in self.cat_cols:
                specs.append(
                    ColumnSpec(name=c, kind="categorical",
                               n_categories=len(self.categories[c]))
                )
            return specs

        @property
        def tabular_dim(self) -> int:
            return len(self.cont_cols) + sum(
                len(self.categories[c]) for c in self.cat_cols
            )

        def inverse(self, cont_std: np.ndarray, cat_codes: list) -> pd.DataFrame:
            data = {}
            if self.cont_cols:
                cont = np.asarray(cont_std, dtype=float) * self.std_ + self.mean_
                for j, c in enumerate(self.cont_cols):
                    data[c] = cont[:, j]
            for c, codes in zip(self.cat_cols, cat_codes):
                cats = self.categories[c]
                codes = np.clip(np.asarray(codes, dtype=int), 0, len(cats) - 1)
                data[c] = [cats[i] for i in codes]
            return pd.DataFrame(data)

    class ExpressionTransformer:
        """Standardize expression (gaussian) or normalize for count likelihoods."""

        def __init__(self, likelihood: str = "gaussian"):
            self.likelihood = likelihood
            self.mean_ = None
            self.std_ = None
            self.columns: list | None = None

        def _counts(self, X):
            # counts are non-negative; clamp so the log1p encoder input is finite
            # even if the caller passes a continuous matrix by mistake.
            return np.clip(np.asarray(X, dtype=float), 0.0, None)

        def fit(self, X: np.ndarray, columns=None) -> "ExpressionTransformer":
            X = np.asarray(X, dtype=float)
            self.columns = list(columns) if columns is not None else None
            base = X if self.likelihood == "gaussian" else np.log1p(self._counts(X))
            self.mean_ = base.mean(axis=0, keepdims=True)
            self.std_ = base.std(axis=0, keepdims=True) + 1e-8
            return self

        def transform(self, X: np.ndarray):
            """Return ``(x_expr, expr_target, library)``.

            gaussian: encoder input == target == standardized X, library = 1.
            nb/zinb : encoder input = standardized log1p(X); target = raw counts;
                      library = per-cell size factor (row sum / mean row sum).
            """
            X = np.asarray(X, dtype=float)
            if self.likelihood == "gaussian":
                Xs = (X - self.mean_) / self.std_
                lib = np.ones((X.shape[0], 1))
                return (Xs.astype(np.float32), Xs.astype(np.float32),
                        lib.astype(np.float32))
            Xc = self._counts(X)
            enc = (np.log1p(Xc) - self.mean_) / self.std_
            row_sum = Xc.sum(axis=1, keepdims=True)
            lib = row_sum / (row_sum.mean() + 1e-8)
            return (enc.astype(np.float32), Xc.astype(np.float32),
                    lib.astype(np.float32))

        def inverse(self, out: np.ndarray) -> np.ndarray:
            """Map a head-space expression sample back to raw feature space."""
            out = np.asarray(out, dtype=float)
            if self.likelihood == "gaussian":
                return out * self.std_ + self.mean_
            return out  # count means are already in raw space

    class MethylationTransformer:
        """Encode/decode methylation beta-values for the joint VAE.

        Beta-values in [0, 1] are clipped and logit-transformed to the (unbounded)
        M-value scale, then standardized -- so a Gaussian likelihood on the head is
        symmetric and well-behaved. The logit/sigmoid math matches
        ``synthvae.data.preprocessing.ExpressionPreprocessor(matrix_kind="methylation")``.
        Mirrors ``ExpressionTransformer``'s gaussian path (encoder input == target).
        """

        def __init__(self, beta_epsilon: float = 1e-6):
            if not (0.0 < beta_epsilon < 0.5):
                raise ValueError("beta_epsilon must be between 0 and 0.5")
            self.beta_epsilon = float(beta_epsilon)
            self.mean_ = None
            self.std_ = None
            self.columns: list | None = None

        def _to_mvalues(self, X):
            X = np.asarray(X, dtype=float)
            if not np.isfinite(X).all():
                raise ValueError("methylation matrix contains non-finite values")
            if np.any((X < 0.0) | (X > 1.0)):
                raise ValueError("methylation beta-values must be in [0, 1]")
            eps = self.beta_epsilon
            X = np.clip(X, eps, 1.0 - eps)
            return np.log(X / (1.0 - X))

        def fit(self, X: np.ndarray, columns=None) -> "MethylationTransformer":
            self.columns = list(columns) if columns is not None else None
            M = self._to_mvalues(X)
            self.mean_ = M.mean(axis=0, keepdims=True)
            self.std_ = M.std(axis=0, keepdims=True) + 1e-8
            return self

        def transform(self, X: np.ndarray):
            """Return ``(x_meth, meth_target)`` -- both the standardized M-values
            (encoder input == gaussian head target)."""
            Ms = (self._to_mvalues(X) - self.mean_) / self.std_
            Ms = Ms.astype(np.float32)
            return Ms, Ms

        def inverse(self, out: np.ndarray) -> np.ndarray:
            """Map standardized-M head samples back to beta-values in [0, 1]."""
            M = np.asarray(out, dtype=float) * self.std_ + self.mean_
            return np.clip(1.0 / (1.0 + np.exp(-M)), 0.0, 1.0)

    class BinaryTransformer:
        """Validate and retain a 0/1 molecular matrix for Bernoulli decoding."""

        def __init__(self):
            self.columns: list | None = None

        def fit(self, X: np.ndarray, columns=None) -> "BinaryTransformer":
            X = np.asarray(X, dtype=float)
            if not np.isfinite(X).all() or not np.isin(X, (0.0, 1.0)).all():
                raise ValueError("binary modality values must be finite 0/1 values")
            self.columns = list(columns) if columns is not None else None
            return self

        def transform(self, X: np.ndarray):
            X = np.asarray(X, dtype=float)
            if not np.isfinite(X).all() or not np.isin(X, (0.0, 1.0)).all():
                raise ValueError("binary modality values must be finite 0/1 values")
            return X.astype(np.float32), X.astype(np.float32), np.ones((len(X), 1), np.float32)

        def inverse(self, out: np.ndarray) -> np.ndarray:
            return (np.asarray(out, dtype=float) >= 0.0).astype(float)

    def make_modality_transformer(spec: dict):
        """Create a fitted-transform compatible object for a registry entry."""
        kind = str(spec.get("kind", "continuous"))
        likelihood = str(spec.get("likelihood", "gaussian"))
        if kind == "methylation":
            return MethylationTransformer()
        if kind == "binary":
            return BinaryTransformer()
        if kind in ("continuous", "count"):
            return ExpressionTransformer(likelihood if kind == "count" else "gaussian")
        raise ValueError(f"unknown modality kind {kind!r}")

    @dataclass
    class FittedTransforms:
        tabular: TabularTransformer
        expression: ExpressionTransformer
        grid: TimeGrid
        column_specs: list
        methylation: "MethylationTransformer | None" = None
        modalities: dict | None = None

    class MMVAEDataset(Dataset):
        """Per-row tensors for the joint VAE (see module docstring)."""

        def __init__(self, x_tab, x_expr, expr_target, library,
                     tab_cont, cat_codes, surv_idx, surv_event,
                     x_meth=None, meth_target=None, modalities=None):
            self.x_tab = torch.as_tensor(x_tab, dtype=torch.float32)
            self.x_expr = torch.as_tensor(x_expr, dtype=torch.float32)
            self.expr_target = torch.as_tensor(expr_target, dtype=torch.float32)
            self.library = torch.as_tensor(library, dtype=torch.float32)
            # Optional methylation modality; None keeps the batch byte-identical to
            # the expression-only path.
            self.has_methylation = x_meth is not None
            if self.has_methylation:
                self.x_meth = torch.as_tensor(x_meth, dtype=torch.float32)
                self.meth_target = torch.as_tensor(meth_target, dtype=torch.float32)
            # Registry representation.  Keep expression/methylation aliases above
            # so historical callers and serialized models remain usable.
            modalities = modalities or {"expression": (x_expr, expr_target, library)}
            if self.has_methylation and "methylation" not in modalities:
                modalities["methylation"] = (x_meth, meth_target, np.ones((len(x_tab), 1)))
            self.modalities = {
                name: {
                    "x": torch.as_tensor(values[0], dtype=torch.float32),
                    "target": torch.as_tensor(values[1], dtype=torch.float32),
                    "library": torch.as_tensor(values[2], dtype=torch.float32),
                }
                for name, values in modalities.items()
            }
            self.tab_cont = torch.as_tensor(tab_cont, dtype=torch.float32)
            # (n, K_cat) long; empty (n, 0) when no categorical columns
            if cat_codes:
                cat = np.stack(cat_codes, axis=1).astype(np.int64)
            else:
                cat = np.zeros((len(self.x_tab), 0), dtype=np.int64)
            self.tab_cat = torch.as_tensor(cat, dtype=torch.long)
            self.surv_idx = torch.as_tensor(surv_idx, dtype=torch.long)
            self.surv_event = torch.as_tensor(surv_event, dtype=torch.float32)

        def __len__(self):
            return len(self.x_tab)

        def __getitem__(self, i):
            item = {
                "x_tab": self.x_tab[i],
                "x_expr": self.x_expr[i],
                "expr_target": self.expr_target[i],
                "library": self.library[i],
                "tab_cont": self.tab_cont[i],
                "tab_cat": self.tab_cat[i],
                "surv_idx": self.surv_idx[i],
                "surv_event": self.surv_event[i],
            }
            if self.has_methylation:
                item["x_meth"] = self.x_meth[i]
                item["meth_target"] = self.meth_target[i]
            item["modalities"] = {
                name: {key: value[i] for key, value in values.items()}
                for name, values in self.modalities.items()
            }
            return item

    def collate(rows):
        """Stack per-row dicts into the batch structure the model expects."""
        stack = lambda k: torch.stack([r[k] for r in rows])
        tab_cont = stack("tab_cont")
        tab_cat = stack("tab_cat")               # (B, K_cat)
        categorical = [tab_cat[:, j] for j in range(tab_cat.shape[1])]
        batch = {
            "x_tab": stack("x_tab"),
            "x_expr": stack("x_expr"),
            "expr_target": stack("expr_target"),
            "library": stack("library"),
            "tab_target": {
                "continuous": tab_cont if tab_cont.shape[1] else None,
                "categorical": categorical,
            },
            "surv_target": {"idx": stack("surv_idx"), "event": stack("surv_event")},
        }
        batch["modalities"] = {
            name: {key: torch.stack([r["modalities"][name][key] for r in rows])
                   for key in rows[0]["modalities"][name]}
            for name in rows[0]["modalities"]
        }
        if "x_meth" in rows[0]:
            batch["x_meth"] = stack("x_meth")
            batch["meth_target"] = stack("meth_target")
        return batch

    def move_batch(batch, device):
        """Move a collated batch (with nested dicts / lists) onto ``device``."""
        def mv(v):
            if torch.is_tensor(v):
                return v.to(device)
            if isinstance(v, dict):
                return {k: mv(x) for k, x in v.items()}
            if isinstance(v, list):
                return [mv(x) for x in v]
            return v
        return {k: mv(v) for k, v in batch.items()}

    def make_loaders(dataset, cfg):
        """Seeded train/val split -> two DataLoaders (val may be ``None``)."""
        n = len(dataset)
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(n)
        n_val = int(cfg.val_fraction * n)
        val_ix, tr_ix = perm[:n_val], perm[n_val:]
        train_loader = DataLoader(
            Subset(dataset, tr_ix.tolist()), batch_size=cfg.batch_size,
            shuffle=True, collate_fn=collate,
        )
        val_loader = None
        if n_val > 0:
            val_loader = DataLoader(
                Subset(dataset, val_ix.tolist()), batch_size=max(cfg.batch_size, n_val),
                shuffle=False, collate_fn=collate,
            )
        return train_loader, val_loader

    def build_from_frames(clinical, expression, survival, cfg, categorical=None,
                          methylation=None, omics=None):
        """Fit transformers, set ``cfg`` dims, return dataset + fitted transforms.

        Returns ``(dataset, transforms, cfg)`` where ``transforms`` is a
        :class:`FittedTransforms`. ``methylation`` is optional: when supplied it
        adds a 4th modality (sets ``cfg.methylation_dim``); when ``None`` the
        dataset and cfg are identical to the expression-only build.
        """
        has_columns = hasattr(expression, "columns")
        expr_columns = list(expression.columns) if has_columns else None
        X = expression.to_numpy(dtype=float) if hasattr(expression, "to_numpy") \
            else np.asarray(expression, dtype=float)
        time = survival["time"].to_numpy(dtype=float)
        event = survival["event"].to_numpy(dtype=float)

        tab = TabularTransformer(categorical).fit(clinical)
        x_tab, tab_cont, cat_codes = tab.transform(clinical)

        expr_tf = ExpressionTransformer(cfg.expression_likelihood).fit(X, expr_columns)
        x_expr, expr_target, library = expr_tf.transform(X)

        omics = dict(omics or {})
        if methylation is not None:
            omics.setdefault("methylation", methylation)
        specs = cfg.resolved_modalities(omics.keys())
        unknown = set(omics) - set(specs)
        if unknown:
            raise ValueError("undeclared modality input(s): " + ", ".join(sorted(unknown)))
        missing = set(cfg.modalities or {}) - {"expression"} - set(omics)
        if missing:
            raise ValueError("configured modality input(s) missing: " + ", ".join(sorted(missing)))
        modality_values = {"expression": (x_expr, expr_target, library)}
        modality_transforms = {"expression": expr_tf}
        modality_dims = {"expression": X.shape[1]}
        meth_tf = None
        x_meth = meth_target = None
        for name, frame in omics.items():
            columns = list(frame.columns) if hasattr(frame, "columns") else None
            Xm = frame.to_numpy(dtype=float) if hasattr(frame, "to_numpy") else np.asarray(frame, dtype=float)
            if len(Xm) != len(X):
                raise ValueError(f"{name} must share row count with expression")
            spec = specs[name]
            if not np.isfinite(Xm).all():
                raise ValueError(f"{name} matrix contains non-finite values")
            if spec.get("kind") == "count" and np.any(Xm < 0):
                raise ValueError(f"{name} count matrix must be non-negative")
            tf = make_modality_transformer(spec).fit(Xm, columns)
            transformed = tf.transform(Xm)
            # ExpressionTransformer returns x,target,library; methylation's legacy
            # transformer returns x,target; normalize both contracts here.
            if len(transformed) == 2:
                xm, target = transformed
                lib = np.ones((len(xm), 1), dtype=np.float32)
            else:
                xm, target, lib = transformed
            modality_values[name] = (xm, target, lib)
            modality_transforms[name] = tf
            modality_dims[name] = Xm.shape[1]
            if name == "methylation":
                meth_tf, x_meth, meth_target = tf, xm, target
                cfg.methylation_dim = Xm.shape[1]

        grid = TimeGrid.from_times(time, event, cfg.n_time_bins)
        surv_idx = grid.index(time)

        cfg.tabular_dim = tab.tabular_dim
        cfg.expression_dim = X.shape[1]
        cfg.modality_dims = modality_dims
        cfg.active_modalities = list(modality_values)
        cfg.n_time_bins = grid.n_bins

        dataset = MMVAEDataset(
            x_tab, x_expr, expr_target, library, tab_cont, cat_codes,
            surv_idx, event, x_meth=x_meth, meth_target=meth_target,
            modalities=modality_values,
        )
        transforms = FittedTransforms(
            tabular=tab, expression=expr_tf, grid=grid,
            column_specs=tab.column_specs, methylation=meth_tf,
            modalities=modality_transforms,
        )
        return dataset, transforms, cfg
