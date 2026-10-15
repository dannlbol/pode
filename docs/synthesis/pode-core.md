# Synthesis theme — PODE core reference

> This is the detailed reference for the **Synthesis** theme's core engine (the
> `pode` package + TVAE workflow). For the suite overview and the other themes see
> the root [README](../../README.md). Two things changed when this became part of
> the umbrella suite: the standalone `python scripts/*.py` synthvae commands are now
> `pode synthesis …` (see [WORKFLOWS.md](WORKFLOWS.md)); and the "future biomarker
> package" sketched near the end of this document has since landed as the
> **Signatures** (`metacpg`) and **Prognosis** (`prognostk`) themes.

# PODE — Pediatric Oncology Discovery Engine

PODE is a modular toolkit for pediatric-oncology data science. Its first
capability is **data-agnostic synthetic clinical-data generation** built on a
TVAE (Tabular Variational AutoEncoder) backend. Later phases (biomarker
discovery, feature selection) plug into the same schema and metric abstractions.

This package is a clean, reusable refactor of a set of research notebooks that
trained TVAE models, tuned their hyperparameters, generated conditional
synthetic data, and verified its fidelity against the originating data.

## Key ideas

- **Schema-driven, not column-name-driven.** A [`DatasetSchema`](../../pode/schema/schema.py)
  auto-detects each column's type (continuous, binary, ordinal, categorical,
  survival time/event) and role, with YAML/dict overrides. Every other module
  consumes the schema, so new cohorts with *similar kinds* of columns work
  without code changes.
- **Identifiers handled automatically.** Unique row tags (e.g. `Sample_ID`,
  `Basename`, `sentrix.id`) are detected and **dropped before modelling**, so
  they neither bloat the synthesizer nor poison survival/Cox analysis. Control
  this via the `schema.detect_identifiers` / `identifier_patterns` config keys.
- **Config over forks.** A whole run is a [`PodeConfig`](../../pode/config.py) YAML
  file. The original `_final` / `_late` / `_shh` notebook variants become three
  config files, not three code copies.
- **Configurable fidelity objective.** The composite score is a weighted sum of
  pluggable penalty metrics (HR preservation, distribution TVD, survival
  agreement, SDV quality). Cohort-specific metrics can be registered later via
  `register_metric` without touching the core.

## Architecture

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Schema | `pode/schema/` | Column-type taxonomy, auto-detection, `DatasetSchema`, SDV metadata bridge |
| Data | `pode/data/` | Load/save, wrangling, administrative censoring, NaN imputation |
| Generation | `pode/generation/` | TVAE train/sample wrapper, conditional sampling, post-processing |
| Adjustment | `pode/adjustment/` | Survival hazard-ratio matching to a target population |
| Evaluation | `pode/evaluation/` | Fidelity metrics, configurable objective + registry, plots |
| Tuning | `pode/tuning/` | Search-space expansion, stability-aware grid search |
| Reproducibility | `pode/reproducibility/` | Seed tracking and RNG control |
| Features | `pode/features/` | Hooks (protocols) for the deferred biomarker phase |
| CLI | `pode/cli/` | `pode synthesis train | generate | tune | evaluate | schema | seeds` |

## The synthvae omics front end

Alongside the `pode` package, this repo ships **`synthvae`** — a survival-aware
generative toolkit for high-dimensional cancer omics (gene expression, DNA
methylation beta-values) with matched clinical + survival data. It was previously
a standalone repo (`synth.vae`) and is the upstream/downstream wrapper around
PODE's TVAE. It offers two approaches:

- **Hybrid (two-stage):** compress each omics matrix under survival supervision
  into low-dimensional `expr_latent_*` / `methylation_latent_*` tables
  (`scripts/run_hybrid.py`), let PODE's TVAE model the joint
  `[clinical | latents | time,event]` distribution, then decode sampled latents
  back to full matrices.
- **Joint multimodal VAE (mmvae):** one shared-latent PoE/MoE/MoPoE model over
  all modalities (`scripts/run_mmvae.py`), for cross-modal / missing-modality
  generation.

| Layer | Location | Responsibility |
|-------|----------|----------------|
| Package | `synthvae/` | compressors, survival losses, synthesizers, evaluation, mmvae, handoff pipeline |
| Scripts | `scripts/run_hybrid.py`, `scripts/run_mmvae.py`, `scripts/benchmark.py`, `scripts/make_toy_data.py` | hybrid / joint CLIs + benchmark + toy data |
| Handoff | `scripts/prepare_hybrid_bundle.py`, `scripts/decode_hybrid_samples.py`, `scripts/promote_best_config.py` | bridge synthvae latents ↔ the PODE TVAE |
| R validation | `R/scripts/*.R` | build cohort CSVs; real-vs-synthetic UMAP / heatmaps |
| Configs | `configs/synthesis/*.yaml` | hybrid + mmvae config and sweep specs |
| Docs | `docs/synthesis/`, `docs/synthesis/WORKFLOWS.md` | design/reference guide + end-to-end pipelines |

The full copy-paste pipelines (expression, methylation, joint, and mmvae) are in
[`docs/synthesis/WORKFLOWS.md`](WORKFLOWS.md); the design rationale is in
[`docs/synthesis/`](.).

## Setup

PODE needs Python **3.10+** (developed and tested on 3.12). The heavy backend
(`sdv`, `torch`, `sdmetrics`) is only imported lazily by the generation/tuning
code, so the schema and data-wrangling utilities still work in a minimal
environment — but a full install is recommended.

### 1. Create and activate a virtual environment

Keep the environment local to the project (the `.venv/` folder is git-ignored):

```bash
cd pode                       # the repository root (this folder)

# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
# py -3.12 -m venv .venv
# .venv\Scripts\Activate.ps1
```

Your shell prompt should now be prefixed with `(.venv)`.

### 2. Install dependencies

```bash
python -m pip install --upgrade pip

# Pinned, reproducible versions (matches the original research environment):
python -m pip install -r requirements.txt

# Install PODE itself (editable, so source edits take effect immediately).
# --no-deps avoids re-resolving the pins you just installed.
python -m pip install -e . --no-deps
```

Alternatively, a single editable install will resolve everything from
`pyproject.toml` (unpinned, latest-compatible versions):

```bash
python -m pip install -e ".[dev]"   # [dev] adds pytest
```

> **Note on first install:** `torch` and `sdv` are large; the first
> `pip install` can take several minutes while wheels download.

### 3. Verify the installation

```bash
python -c "import pode; print('PODE', pode.__version__)"
pode --help            # the CLI entry point
pytest -q              # 22 unit tests should pass
```

### Deactivating / removing

```bash
deactivate             # leave the environment
rm -rf .venv           # remove it entirely (safe; nothing else depends on it)
```

## Toy dataset

The repo ships a **fully synthetic, made-up** cohort at
[`examples/cohort.csv`](../../examples/cohort.csv) (200 rows, no real patient data) so
you can try everything immediately. It is medulloblastoma-shaped and exercises
every column type plus every option in the example config: continuous (`Age`),
binary (`Sex`, `ConsensusMYC`), ordinal (`CSI_dose`, `M_stage`), categorical
(`Histology`, `MNP_Subgroup`), a survival pair (`OS_Time`/`OS_Status`),
methylation columns (`cg0001..cg0010`), columns the config drops
(`PFS_*`, `MethGroup`), and a high-missingness column the quality filter
removes (`SubtotalResection`).

Regenerate it any time with:

```bash
python examples/make_toy_cohort.py        # rewrites examples/cohort.csv
```

## CLI usage

The commands below use the bundled toy data; swap in your own CSV path when ready.

```bash
# Inspect what PODE detected about your columns
pode synthesis schema --data examples/cohort.csv --config configs/example_cohort.yaml

# Train once and save a reusable model + prepared schema
pode synthesis train --data examples/cohort.csv --config configs/example_cohort.yaml \
    --output-dir ./trained --seed 42

# Generate from the saved model
pode synthesis generate --model ./trained/model.pkl --schema ./trained/schema.yaml \
    --num-samples 500 --output-dir ./out --seed 42

# Backward-compatible shortcut: train + generate + evaluate in one command
pode synthesis generate --data examples/cohort.csv --config configs/example_cohort.yaml \
    --num-samples 500 --seed 42 --evaluate --output-dir ./out

# Conditional generation from the saved model (50% MYC-amplified, 50% not)
pode synthesis generate --model ./trained/model.pkl --schema ./trained/schema.yaml \
    --conditions "0.5:ConsensusMYC=1;0.5:ConsensusMYC=0" --num-samples 1000

# Hyperparameter grid search
pode synthesis tune --data examples/cohort.csv --config configs/example_cohort.yaml \
    --output-dir ./tuning
```

### Practical CLI guide

The six CLI subcommands fit into a simple workflow:

1. `schema`: check how PODE interprets your dataset.
2. `train`: fit a reusable TVAE model and save its prepared schema.
3. `generate`: sample synthetic rows, either from a saved model or by using the
   older train-and-sample shortcut.
4. `evaluate`: score an existing synthetic CSV against the real cohort.
5. `tune`: search over synthesizer hyperparameters.
6. `seeds`: list saved seed logs so runs can be traced and repeated.

#### `pode synthesis schema`

Use `schema` first on any new cohort. It is a dry-run inspection step: PODE
loads the CSV, applies schema auto-detection, and prints the resulting schema as
JSON. This lets you verify that the config is classifying columns the way you
expect before any model training starts.

```bash
pode synthesis schema --data examples/cohort.csv --config configs/example_cohort.yaml
```

What to look for in the output:

- `types`: whether each column was detected as `continuous`, `binary`,
  `ordinal`, `categorical`, `survival_time`, or `survival_event`.
- `roles`: whether a column is treated as a modelling `feature`, `survival`
  column, or `identifier`.
- `survival_pairs`: whether time/event columns were paired correctly, e.g.
  `OS_Time` with `OS_Status`.

If a dataset does not follow the default `*_Time` / `*_Status` naming
convention, add explicit pairs in the config:

```yaml
schema:
  survival_pairs:
    - time_col: OS
      event_col: OS.status
    - time_col: Efs
      event_col: EFS.status
    - time_col: DFS
      event_col: DFS.status
```

In the bundled example, this is mainly checking that:

- `cg0001`..`cg0010` are forced to continuous because the config sets
  `schema.continuous_prefixes: ["cg"]`.
- `OS_Time` and `OS_Status` are detected as the primary survival pair.
- `Sample_ID` is recognised as an identifier role.

By default, this command is for raw detection inspection, not full preprocessing.
It does not show the final training table after the data-cleaning pipeline drops
columns such as `PFS_*` or `MethGroup`; those removals happen later in
`train`, `generate`, `evaluate`, and `tune`.

If detection is wrong, adjust the YAML config and rerun `schema` until it looks
right. You can also save the detected schema for review:

```bash
pode synthesis schema --data examples/cohort.csv \
    --config configs/example_cohort.yaml \
    --out ./detected_schema.yaml
```

`detected_schema.yaml` can be passed back with `--schema`, but only when it
matches the data being modelled. If you want a schema that is safe to pair with
a trained model, save the prepared schema instead:

```bash
pode synthesis schema --prepared --data examples/cohort.csv \
    --config configs/example_cohort.yaml \
    --out ./prepared_schema.yaml
```

The prepared schema is built after configured column drops, identifier removal,
quality filtering, and survival handling. `pode synthesis train` writes this same kind of
schema automatically to `schema.yaml`.

If survival times are stored in days but your censoring/evaluation settings are
in years, rescale the survival-time columns during preparation:

```yaml
data:
  survival_time_scale: 0.0027378508  # 1 / 365.25
  censor_time_threshold: 10.0

objective:
  survival_timepoint: 5.0
```

`survival_time_scale` is applied to every detected survival-time column before
censoring, training, and evaluation.

#### `pode synthesis train`

Use `train` when you want to fit the TVAE once and keep the model for repeated
synthetic sampling.

```bash
pode synthesis train --data examples/cohort.csv --config configs/example_cohort.yaml \
    --output-dir ./trained --seed 42
```

What happens during `train`:

- The raw CSV is cleaned using the config.
- Identifier columns are dropped before modelling.
- Low-quality variables can be filtered out.
- Survival times can be administratively censored.
- A TVAE is trained on the prepared table.
- The fitted model is written to `./trained/model.pkl`.
- The prepared schema is written to `./trained/schema.yaml`.
- A seed log is saved in the output directory for reproducibility.

You can override the default file names:

```bash
pode synthesis train --data examples/cohort.csv --config configs/example_cohort.yaml \
    --model-out ./models/mb_tvae.pkl --schema-out ./models/mb_schema.yaml
```

#### `pode synthesis generate`

Use `generate` when you want to sample synthetic rows. The recommended workflow
is to generate from a saved model:

```bash
pode synthesis generate --model ./trained/model.pkl --schema ./trained/schema.yaml \
    --num-samples 500 --seed 42 --output-dir ./out
```

In model mode, `--schema` is required. SDV can reload the fitted model from
`model.pkl`, but PODE needs its own prepared schema to know which columns are
survival columns, identifiers, categorical variables, and numeric variables.

`generate` still supports the older train-and-sample shortcut:

```bash
pode synthesis generate --data examples/cohort.csv --config configs/example_cohort.yaml \
    --num-samples 500 --seed 42 --output-dir ./out
```

That shortcut trains a temporary model and immediately samples from it, but it
does not save a reusable model file.

When to use the main options:

- `--num-samples`: control how many synthetic rows to create.
- `--seed`: force a specific RNG seed instead of letting PODE mint one.
- `--output-dir`: choose where the synthetic CSV and seed log are written.
- `--model`: load a saved model from `pode synthesis train` instead of retraining.
- `--schema`: load the prepared schema that matches the saved model.
- `--impute`: fill missing values in the synthetic output based on the real data.
- `--evaluate`: immediately run the fidelity metrics on the generated data too.

Example with evaluation:

```bash
pode synthesis generate --data examples/cohort.csv --config configs/example_cohort.yaml \
    --num-samples 500 --seed 42 --evaluate --output-dir ./out
```

That produces:

- `./out/synthetic_data.csv`
- `./out/seed_log_*.json`
- `./out/fidelity_scores.json` and evaluation plots if `--evaluate` is set

Conditional generation is useful when you want the synthetic output to match a
target subgroup mix. Inline conditions use this grammar:

```text
proportion:constraint[,constraint];proportion:constraint[,constraint]
```

Rules:

- Semicolon separates groups.
- Each group is `proportion:constraints`.
- Proportions must sum to `1.0`.
- Comma separates multiple constraints inside a group.
- Equality constraints use `column=value`.
- Numeric range constraints use `column:min:max`.

```bash
pode synthesis generate --model ./trained/model.pkl --schema ./trained/schema.yaml \
    --conditions "0.5:ConsensusMYC=1;0.5:ConsensusMYC=0" --num-samples 1000
```

That asks PODE to sample roughly half the rows with `ConsensusMYC=1` and half
with `ConsensusMYC=0`.

You can combine equality and range constraints:

```bash
pode synthesis generate --model ./trained/model.pkl --schema ./trained/schema.yaml \
    --conditions "0.3:ConsensusMYC=1,Age:3:10;0.7:Age:10:18" \
    --num-samples 1000
```

For more readable conditional generation, use a YAML condition config:

```yaml
groups:
  - proportion: 0.5
    equals:
      ConsensusMYC: 1
  - proportion: 0.5
    equals:
      ConsensusMYC: 0
```

Then pass it with `--conditions-config`:

```bash
pode synthesis generate --model ./trained/model.pkl --schema ./trained/schema.yaml \
    --conditions-config ./conditions.yaml --num-samples 1000
```

Mixed equality and range conditions are also supported:

```yaml
groups:
  - proportion: 0.3
    equals:
      ConsensusMYC: 1
    ranges:
      Age:
        min: 3
        max: 10
  - proportion: 0.7
    ranges:
      Age:
        min: 10
        max: 18
```

How conditional generation works:

- Equality constraints are attempted through SDV conditional sampling
  (`sample_from_conditions`).
- If SDV conditional sampling fails for a group, PODE falls back to generating
  oversized batches and filtering rows that match the equality constraints.
- Range constraints are implemented by rejection sampling: PODE generates
  candidate rows and keeps rows inside the requested numeric range.
- Restrictive conditions may produce fewer rows than requested; PODE warns when
  a condition group under-generates.

You can reuse the same saved model to generate many different datasets:

```bash
pode synthesis generate --model ./trained/model.pkl --schema ./trained/schema.yaml \
    --conditions "1.0:ConsensusMYC=1" --num-samples 250 --output-dir ./out_myc_amp

pode synthesis generate --model ./trained/model.pkl --schema ./trained/schema.yaml \
    --conditions "1.0:ConsensusMYC=0" --num-samples 250 --output-dir ./out_myc_non_amp
```

#### `pode synthesis evaluate`

Use `evaluate` when you already have a synthetic CSV and want to score it
against the real cohort without retraining anything.

```bash
pode synthesis evaluate --data examples/cohort.csv --config configs/example_cohort.yaml \
    --synthetic ./out/synthetic_data.csv --output-dir ./eval
```

This is useful when:

- you generated data in an earlier run and only want the metrics now;
- you want to compare multiple synthetic datasets against the same real cohort;
- you have synthetic data produced elsewhere but want to score it with PODE's
  schema and objective settings.

`evaluate` writes the composite fidelity report to `fidelity_scores.json` and
also creates diagnostic plots such as marginals, correlation comparisons, and a
Kaplan-Meier plot if a survival pair is present.

#### `pode synthesis tune`

Use `tune` when you want PODE to try a grid of synthesizer settings and score
each one, rather than committing to a single training configuration.

```bash
pode synthesis tune --data examples/cohort.csv --config configs/example_cohort.yaml \
    --output-dir ./tuning
```

This is the command to run when the default generator is not good enough and
you need to improve fidelity systematically. The parameter grid comes from the
config file under `tuning.param_grid`, and the results are scored with the same
objective used by `evaluate`.

Typical workflow:

1. Start with `schema` to confirm the cohort is interpreted correctly.
2. Optionally write `pode synthesis schema --prepared --out ./prepared_schema.yaml`.
3. Run `generate` once as a baseline or `train` if you know you will reuse the model.
4. Run `tune` if fidelity is poor or unstable.
5. Take the best settings from tuning and run `train` again.
6. Reuse `model.pkl` and `schema.yaml` with `generate --model`.

Hybrid-handoff workflow (see [`docs/synthesis/WORKFLOWS.md`](WORKFLOWS.md) for the
full end-to-end pipelines):

PODE is the downstream TVAE for the in-repo `synthvae` latent handoff flow. The
synthvae side (`scripts/run_hybrid.py export-latents`) exports a bundle:

* `clinical_full.csv`
* `latents_full.csv`
* `survival_full.csv`
* `handoff_metadata.json`

These files are row-aligned and are intended to be combined into a single PODE
training table containing:

```text
[ clinical columns | latent columns | survival columns ]
```

Expression bundles use `expr_latent_*`; methylation bundles use
`methylation_latent_*`. PODE reads the expected prefix from
`handoff_metadata.json`.

To automate that preparation, a helper script now lives at
`scripts/prepare_hybrid_bundle.py`. It:

* concatenates the aligned hybrid bundle CSVs into one wide training table,
* renames survival columns to an explicit pair such as `OS_Time` and
  `OS_Status`,
* writes a starter PODE YAML with the survival pair declared under
  `schema.survival_pairs`,
* forces the bundle's latent columns, such as `expr_latent_*` or
  `methylation_latent_*`, to `continuous`,
* lets you force clinical columns such as `Age` to `continuous` and columns
  such as `Sex`, `M._versus_M.`, `ConsensusMYC`, and `WCA` to `categorical`.

Example:

```bash
pode synthesis bundle \
    --bundle-dir runs/final_handoff \
    --out-dir ./hybrid_prep \
    --continuous-cols Age \
    --categorical-cols Sex M._versus_M. ConsensusMYC WCA
```

Then train PODE on the generated table:

```bash
pode synthesis train \
    --data ./hybrid_prep/training_table.csv \
    --config ./hybrid_prep/pode_config.yaml \
    --output-dir ./trained_hybrid \
    --seed 42
```

This keeps the responsibilities clean:

* `synthvae` learns and exports survival-aware expression or methylation latents,
* PODE models the joint tabular distribution of clinical + latent + survival,
* the saved hybrid compressor later decodes sampled latent rows back into the
  full original expression or methylation matrix.

The reverse helper for that final step now lives at
`scripts/decode_hybrid_samples.py`. It expects a wide generated table from PODE
containing:

```text
[ clinical columns | latent columns | survival columns ]
```

and it writes:

* `sampled_latents.csv`
* `synthetic_clinical.csv`
* `synthetic_survival.csv`
* `synthetic_expression.csv` or `synthetic_methylation.csv`

by splitting the PODE-wide table and decoding the latent block with the saved
hybrid compressor bundle.

Example:

```bash
pode synthesis decode \
    --bundle-dir runs/final_handoff \
    --pode-table ./generated/synthetic_data.csv \
    --out-dir ./decoded_hybrid
```

#### `pode synthesis seeds`

Use `seeds` to find the seed logs written by previous generation runs.

```bash
pode synthesis seeds --dir ./out
```

This command lists files matching `seed_log_*.json` in the chosen directory. It
is mainly a bookkeeping tool: it helps you locate the logs that record which
seed was used for each run.

Typical use cases:

- trace which seed produced a synthetic dataset you want to keep;
- confirm that repeated runs are actually using different seeds;
- find the latest saved seed log in an output directory.

> **Tip:** the example config trains for 1500 epochs (realistic, but a minute or
> two on the toy data). For a quick smoke run, lower it — e.g. add
> `--config` overrides or edit `synthesizer.epochs` in a copy of the YAML.

PODE suppresses benign third-party warnings (SDV/ctgan deprecations, lifelines
KM approximation) by default. Add `--show-warnings` to any subcommand to see
them, e.g. `pode synthesis generate --show-warnings ...`.

## Programmatic API

```python
import pandas as pd
import pode
from pode.config import load_config

config = load_config("configs/example_cohort.yaml")
df = pd.read_csv("examples/cohort.csv")

real, schema = pode.prepare_data(df, config)
synthetic, generator = pode.generate_synthetic(real, config, num_samples=500, seed=42)
```

## Extending PODE: biomarker / feature hooks

PODE is built so that a **later biomarker package** (e.g. a metagene score, a
risk signature, a feature selector) can consume PODE's synthetically generated
variables *without modifying the core*. There are three deliberate extension
points. See [`pode/features/README.md`](../../pode/features/README.md) for the full
walk-through; the essentials are below.

### The three hooks

| Hook | Where | Purpose |
|------|-------|---------|
| `FeatureExtractor` protocol | [`pode/features/base.py`](../../pode/features/base.py) | Derive new columns (e.g. a metagene score) from a dataset — real *or* synthetic |
| `FeatureSelector` protocol | [`pode/features/base.py`](../../pode/features/base.py) | Pick an informative subset of columns (biomarker discovery) |
| `register_metric(...)` | [`pode/evaluation/objective.py`](../../pode/evaluation/objective.py) | Add a fidelity metric so the synthetic data is scored on how well it preserves your biomarker |

All three speak the shared [`DatasetSchema`](../../pode/schema/schema.py) vocabulary
(`ColumnType`, `ColumnRole`), so a new module can ask *"which columns are
features? which is the survival pair?"* without re-parsing anything.

### How synthetic variables flow into a biomarker module

```
real data ──▶ PODE generate ──▶ synthetic data (clinical + e.g. CpG columns)
                                      │
                                      ▼
                         your FeatureExtractor.transform()
                                      │  (same code runs on real & synthetic)
                                      ▼
                         synthetic data + derived "metagene" column
                                      │
                                      ▼
                 register_metric() ──▶ composite fidelity score
                 (does the metagene keep its prognostic value?)
```

Because an extractor's `transform()` is agnostic to data provenance, the *same*
biomarker code runs on the real cohort and on any PODE-generated synthetic
cohort. That is the whole point of the abstraction.

### Worked example: a metagene extractor + a fidelity metric

A metagene is a weighted combination of input columns (in the original work,
CpG methylation sites). Implementing it as a `FeatureExtractor` is all that is
needed for it to work on synthetic data:

```python
import numpy as np
import pandas as pd
from pode.features import FeatureExtractor   # Protocol (structural typing)
from pode.schema import DatasetSchema

class MetageneExtractor:
    """Weighted-sum signature over a set of (synthetic or real) columns."""
    def __init__(self, weights: dict[str, float], name: str = "metagene"):
        self.weights = weights
        self.name = name

    def fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "MetageneExtractor":
        # Nothing to learn here; a real signature might standardise inputs.
        self._cols = [c for c in self.weights if c in df.columns]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        w = np.array([self.weights[c] for c in self._cols])
        out[self.name] = df[self._cols].to_numpy() @ w
        return out

# `isinstance` works because FeatureExtractor is a runtime_checkable Protocol:
assert isinstance(MetageneExtractor({"cg_0001": 0.5}), FeatureExtractor)
```

Now wire it into PODE's fidelity scoring so the grid search / evaluation will
*reward synthetic data that keeps the metagene's prognostic signal*. You only
register a metric and reference it by name in config — no core edits:

```python
from pode.evaluation.objective import register_metric
from pode.evaluation.metrics import compute_hazard_ratios
from pode.schema import ColumnType

extractor = MetageneExtractor(weights={"cg_0001": 0.7, "cg_0002": 0.3})

@register_metric("metagene_hr_preservation")
def metagene_hr_penalty(real_df, synth_df, schema, **ctx):
    """|HR_real - HR_synth| for the derived metagene (lower = better)."""
    pair = schema.primary_survival_pair()
    real_m = extractor.fit(real_df, schema).transform(real_df)
    synth_m = extractor.transform(synth_df)
    # Make the Cox model aware of the derived column.
    schema.types.setdefault(extractor.name, ColumnType.CONTINUOUS)
    hr_real = compute_hazard_ratios(real_m, [extractor.name], schema,
                                    pair.time_col, pair.event_col)
    hr_synth = compute_hazard_ratios(synth_m, [extractor.name], schema,
                                     pair.time_col, pair.event_col)
    if extractor.name not in hr_real or extractor.name not in hr_synth:
        return float("nan")
    return abs(hr_real[extractor.name] - hr_synth[extractor.name])
```

Then activate it through config weights (YAML or `ObjectiveConfig`):

```yaml
objective:
  weights:
    hr_mae: 1.0
    tvd_mean: 1.0
    metagene_hr_preservation: 2.0   # your new metric, weighted into the score
```

Metrics with weight `0` (or absent) are skipped, so adding metrics never slows
existing runs unless you opt in. This is exactly how cohort-specific terms
(MNP subgroups, MYC status, etc.) from the original notebooks would be
re-introduced later — as registered plugins, not hard-coded branches.

### Where a future biomarker package would live

A natural layout, mirroring the existing structure:

```
pode/features/
├── base.py            # protocols (exists today)
├── metagene.py        # MetageneExtractor + model loading  (future)
└── selection.py       # FeatureSelector implementations     (future)
```

Each new module imports `DatasetSchema` for column semantics and, if it wants
its biomarker reflected in synthetic-data quality, registers a metric. Nothing
in `pode/generation/`, `pode/tuning/`, or `pode/evaluation/objective.py` needs
to change.

## Status

Synthetic-data generation, conditional sampling, survival adjustment, grid
search, and fidelity evaluation are implemented and tested. The metagene /
biomarker extractor itself is intentionally deferred — only the extension
*contracts* exist today (see the section above).
