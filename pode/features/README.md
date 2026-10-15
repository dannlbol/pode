# `pode.features` — biomarker / feature-extraction hooks

This subpackage is the **planned home of the biomarker layer** (metagene
signatures, risk scores, feature selection). Today it contains only the
*contracts* — Python `Protocol`s in [`base.py`](base.py) — that future modules
implement so they integrate with the rest of PODE without any core changes.

The guiding principle: **a biomarker computed on real data and the same
biomarker computed on PODE-generated synthetic data should run through identical
code.** That is what makes "use synthetically generated variables as part of a
metagene or other biomarker package" a drop-in, rather than a rewrite.

---

## 1. The two protocols

```python
from pode.features import FeatureExtractor, FeatureSelector
```

### `FeatureExtractor`
Derives **new columns** from a dataset (e.g. a metagene score from CpG columns).

```python
class FeatureExtractor(Protocol):
    def fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "FeatureExtractor": ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

### `FeatureSelector`
Selects an **informative subset** of existing columns (biomarker discovery).

```python
class FeatureSelector(Protocol):
    def fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "FeatureSelector": ...
    def selected_features(self) -> list[str]: ...
```

Both are `@runtime_checkable`, so `isinstance(obj, FeatureExtractor)` works via
structural typing — your class does **not** need to subclass anything, it just
needs matching method signatures. This keeps the biomarker code decoupled from
PODE internals.

Both receive a [`DatasetSchema`](../schema/schema.py). Use it instead of
hard-coding column names:

```python
schema.feature_columns          # covariates (non-survival, non-id)
schema.numerical_columns        # continuous + survival-time columns
schema.primary_survival_pair()  # -> SurvivalPair(time_col, event_col)
schema.columns_of_role(ColumnRole.TARGET)
```

---

## 2. Why synthetic variables "just work"

A `FeatureExtractor.transform()` takes a dataframe and returns a dataframe. It
has no knowledge of whether the rows are real or synthetic. So the flow is:

```
real cohort ──▶ extractor.fit(...).transform()  ──▶ metagene on real data
                                  ▲
                                  │ (same fitted extractor)
PODE synthetic cohort ──▶ extractor.transform()  ──▶ metagene on synthetic data
```

Typical usage in a downstream analysis:

```python
import pandas as pd
import pode
from pode.config import load_config

config = load_config("configs/example_cohort.yaml")
real, schema = pode.prepare_data(pd.read_csv("examples/cohort.csv"), config)

# 1) generate synthetic variables with PODE
synthetic, _ = pode.generate_synthetic(real, config, num_samples=1000, seed=42)

# 2) derive the biomarker on BOTH, with the same extractor
extractor = MetageneExtractor(weights=my_weights).fit(real, schema)
real_scored      = extractor.transform(real)
synthetic_scored = extractor.transform(synthetic)

# 3) now you have a 'metagene' column on synthetic data to analyse / model
```

---

## 3. Reflecting a biomarker in synthetic-data fidelity

If you want the synthetic data to be *scored* on how faithfully it preserves
your biomarker, register a penalty metric (lower = better) with the objective
registry. The composite objective then includes it whenever a non-zero weight is
configured — no edits to `objective.py` or the tuning loop.

```python
from pode.evaluation.objective import register_metric
from pode.evaluation.metrics import compute_hazard_ratios

@register_metric("metagene_hr_preservation")
def _metric(real_df, synth_df, schema, **ctx):
    pair = schema.primary_survival_pair()
    rm = extractor.fit(real_df, schema).transform(real_df)
    sm = extractor.transform(synth_df)
    hr_r = compute_hazard_ratios(rm, [extractor.name], schema, pair.time_col, pair.event_col)
    hr_s = compute_hazard_ratios(sm, [extractor.name], schema, pair.time_col, pair.event_col)
    if extractor.name in hr_r and extractor.name in hr_s:
        return abs(hr_r[extractor.name] - hr_s[extractor.name])
    return float("nan")
```

```yaml
# in your PODE config
objective:
  weights:
    metagene_hr_preservation: 2.0
```

A metric receives `**ctx` (currently `timepoint`) for extra context and must
return a float where **smaller means more faithful**. Return `float("nan")` when
the metric can't be computed; NaN values are skipped in the weighted sum.

---

## 4. Suggested future layout

```
pode/features/
├── base.py            # protocols (exists today)
├── metagene.py        # MetageneExtractor: weights, beta->M conversion, scaling
└── selection.py       # concrete FeatureSelector implementations
```

When porting the original `metagene_extractor_robust.py`, wrap it as a
`MetageneExtractor` implementing `FeatureExtractor`, load its CoxBoost-derived
weights in `fit()`, and compute the score in `transform()`. Register a metric
(section 3) if you want it to steer generation/tuning. That is the entire
integration surface — everything else in PODE stays untouched.
