# Generative imputation with TVAE

`pode synthesis impute` turns a PODE TVAE into a **multiple-imputation** engine and
ships a **masking benchmark** to measure how good its imputations are against
classical and deep baselines. Code lives in `pode/imputation/`; the benchmark
config template is [`configs/imputation/benchmark.example.yaml`](../../configs/imputation/benchmark.example.yaml).

The feature is additive: `train / generate / tune / evaluate` and PODE configs are
unchanged. Imputation settings live in their own YAML (`--impute-config`).

**What we found when benchmarking** on a clinical + expression + methylation
hybrid table:
- Use **`hybrid`**: MICE for clinical gaps, the mask-aware TVAE for patients
  missing an omics modality.
- Expect good recovery of clinical values, moderate recovery of an absent
  methylation block, and little recovery of an absent expression block. The
  gain is keeping patients in the analysis without bias, not reconstructing
  their omics.
- Never single-impute for inference.

## Engines

| Engine | What it is | Draws | Trains on incomplete data? |
|---|---|---|---|
| `tvae-decoder` | Conditions an **existing** `model.pkl` from `pode synthesis train` | multiple | no (complete rows only) |
| `tvae-masked` | Mask-aware TVAE trained for imputation (`pode synthesis impute train`) | multiple | **yes** |
| `mice` | Chained equations: Bayesian ridge / logistic regression, bootstrap parameter draws | multiple | yes |
| `gain` | GAIN (Yoon et al. 2018) with hint mechanism | multiple (noise) | yes |
| `missforest` | Iterative ExtraTrees, mtry = √p | single | yes |
| `knn` | k-nearest neighbours on standardised + one-hot data | single | yes |
| `mean_mode` | Column mean / mode | single | yes |
| `hybrid` | Routes rows: `mice` for scattered gaps, `tvae-masked` for rows with an absent block | multiple | yes |

### Why two TVAE engines

* **Saved PODE models have no encoder.** ctgan discards the TVAE encoder after
  training, so `model.pkl` holds only the decoder and transformers.
  `tvae-decoder` therefore conditions the latent code on a row's observed cells
  through the decoder alone. It targets p(z | x_obs) ∝ N(z; 0, I) · p(x_obs | z)^λ:
  * **SIR**: K prior samples, importance-weighted by the observed-cell
    likelihood, then M resampled draws.
  * **Tempered SMC**: for rows whose SIR effective sample size (ESS) is below
    M (`method: auto`), the likelihood is annealed in from the prior. Particles
    are resampled at each temperature and moved with **pCN** (preconditioned
    Crank–Nicolson) MCMC, which is prior-reversible and mixes in the 128-d
    latent space. This matters for well-trained models: with many observed
    columns, the likelihood is sharp enough that one-shot SIR collapses to a
    single particle.

  λ defaults to the model's `loss_factor`, the tempering it was trained with.
  Missing cells are then *sampled* from the decoder: scalars from N(tanh(μ), σ),
  modes and categories from the softmax. Repeated draws therefore reflect model
  uncertainty.
* **SDV mean-fills numeric NaNs before training.** A synthesis TVAE trained on
  incomplete data learns the artificial fill values. `tvae-masked` avoids this:
  * Per-column normalisers are fitted on observed values only.
  * The encoder sees `[x with missing spans zeroed, column mask]`.
  * Each batch hides extra observed cells (MCAR at a rate ~ U(0, `p_max`), plus
    whole `block_groups` with probability `block_prob`), and those hidden
    cells are up-weighted in the loss.

  It keeps its encoder, so inference is MIWAE-style SIR with q(z | x_obs, m) as
  the proposal. Low-ESS rows get pCN refinement of their SIR draws. Architecture hyperparameters come
  from the PODE config's `synthesizer:` block, so tuned values carry over.

## Commands

```bash
# Impute with an existing PODE model (no retraining) -> M completed datasets
pode synthesis impute apply \
    --model trained_hybrid_tuned/model.pkl \
    --data incomplete.csv --m 20 --output-dir out/imputed --seed 42
# --schema defaults to schema.yaml next to the model

# Train a mask-aware TVAE directly on an incomplete cohort, then apply it
pode synthesis impute train --data incomplete.csv --config pode_config.yaml \
    --impute-config impute.yaml --output-dir out/masked_tvae --seed 42
pode synthesis impute apply --model out/masked_tvae --data incomplete.csv \
    --m 20 --output-dir out/imputed_masked --seed 42

# Fit any engine on the incomplete cohort itself (no pre-trained model needed)
pode synthesis impute apply --engine mice --data incomplete.csv \
    --m 20 --output-dir out/imputed_mice --seed 42

# Hybrid: MICE for scattered gaps, mask-aware TVAE for rows missing a whole block
# (needs block_groups in the imputation YAML)
pode synthesis impute apply --engine hybrid --data incomplete.csv \
    --config pode_config.yaml --impute-config configs/imputation/benchmark.example.yaml \
    --m 20 --output-dir out/imputed_hybrid --seed 42

# Prediction-safe: fit on a training cohort, impute new patients without
# letting their outcomes (or their observed values) shape the imputation model
pode synthesis impute apply --engine hybrid --reference train.csv --data test.csv \
    --config pode_config.yaml --impute-config configs/imputation/benchmark.example.yaml \
    --outcome-policy hide --fit-scope reference --m 20 --output-dir out/imputed_test

# Benchmark: hide known values in a complete table and score all engines
pode synthesis impute benchmark \
    --data runs/final_handoff/pode_prep/training_table.csv \
    --config runs/final_handoff/pode_prep/pode_config.yaml \
    --impute-config configs/imputation/benchmark.example.yaml \
    --output-dir out/impute_bench            # add --quick for a smoke run
```

`apply` writes `imputed_001..M.csv` and `imputed_long.csv` (stacked, with `.imp`
and `.id` columns in the R `mice` long format, where `.imp = 0` is the original).
It also writes `diagnostics.json` and, for `tvae-decoder`, a per-row
`row_diagnostics.csv` with ESS, whether SMC/pCN refined the row, the MCMC
acceptance rate, and the number of SMC stages.

**Read the diagnostics.** An `ess_median` near 1 together with
`frac_refined` near 1 means SIR alone would have returned copies of one
particle. That is expected for long-trained models; SMC then does the work.

**Columns.**
- Columns the model doesn't know are passed through, and identifier columns are
  never imputed.
- Model columns absent from the input are imputed as fully missing, which is
  how an absent modality block is handled.
- Input values must be in the same units as the training table.

With `--engine`, the schema comes from `--schema` or is detected from the
reference (or `--data`) using the PODE config's schema settings.

Imputed latent blocks (`expr_latent_*`, `methylation_latent_*`) can go through
`pode synthesis decode --bundle-dir ... --pode-table imputed_001.csv` to recover
expression or methylation.

## Controlling what the imputer sees

Every engine honours the same settings: global defaults in the imputation YAML,
per-engine overrides in an engine's `params`, and CLI flags on
`train`/`apply`/`benchmark`. `diagnostics.json` and the benchmark's "Leakage
settings" report section record what each engine was allowed to see.

| Setting (flag) | Values | Effect |
|---|---|---|
| `outcome_policy` (`--outcome-policy`) | `use` (default) | Outcomes (default: survival time/event; override with `outcome_columns`) are ordinary predictors. |
| | `hide` | Outcomes stay in the **training** data, but the rows being imputed have theirs masked from the engine and restored afterwards, so a row's own outcome never informs its covariates. |
| | `exclude` | Outcome columns never reach the engine at all. |
| `exclude_columns` (`--exclude-columns`) | globs | Columns the imputer never sees and never imputes, e.g. post-baseline measurements, response to treatment, or anything recorded after the prediction time point. |
| `fit_scope` (`--fit-scope`) | `stacked` (default) | kNN / MICE / missForest learn from the training rows **and** the observed values of the rows being imputed. This is standard MICE practice and uses all available information. |
| | `reference` | They learn from the training/reference rows only. Rows being imputed are only predicted, never learned from, which is how a deployed model meets new patients. TVAE engines and GAIN always behave this way. |

**When to use which:**

| Goal | Recommended settings |
|---|---|
| Inference on one cohort (pooled Cox, effect estimates) | `use`. Leaving the outcome out of the imputation model biases its associations toward the null. |
| Developing a prediction model: training rows | `use` |
| Developing a prediction model: validation/test rows, or the benchmark's held-out folds | `hide` + `fit_scope: reference`, fitting on the training rows (`--reference` in `apply`; automatic in k-fold benchmarks) |
| Deployment / external validation | `--reference training_cohort.csv --outcome-policy hide --fit-scope reference`. At deployment the outcome is unknown anyway; this reproduces that situation. |
| One imputer reused for several endpoints | `exclude` |

**Safeguards and caveats:**
- `hide` only means something when the engine is fitted on other rows. If
  `apply` has no `--reference`, or the benchmark is in transductive mode, the
  engine is fitted on the rows being imputed, so `hide` is upgraded to
  `exclude` with a note.
- A pre-trained model (`--model`) cannot drop columns. For it, `exclude`
  degrades to `hide` with a warning, and `--exclude-columns` that name model
  columns are refused (retrain without them).
- To *measure* the cost of prediction-safe imputation, put both variants in a
  benchmark, e.g. `mice` and
  `{name: mice, label: mice-hide-outcome, params: {outcome_policy: hide}}`.
  Expect the hidden variant's pooled hazard ratios to sit closer to 1.

## Benchmark design

**Masking scenarios** (`scenarios:`) hide observed cells; survival time and event
columns are protected by default.

**Blocks are all-or-nothing** (`atomic_blocks: true`, the default). Latents
derived from one omics assay are present or absent together, so:
- cell-level kinds (`mcar`, `mar`, `mnar_self`, `tail`, `rare_category`) never
  touch `block_groups` columns;
- only `block` hides them, a whole group at a time;
- a mask that would split a block is an error;
- the mask-aware TVAE's training masks follow the same rule;
- `apply` warns (and records `partial_block_rows` in `diagnostics.json`) if
  real input has partly missing blocks.

Turn it off only for data where partial blocks genuinely occur. Hiding single
latents makes imputation look far better than it is, because latents within a
block predict each other almost perfectly.

| kind | hides | key options |
|---|---|---|
| `mcar` | each cell independently | `rate` |
| `mar` | cells with probability driven by observed `drivers` | `rate`, `drivers` |
| `block` | a whole column group in a fraction of rows (absent modality) | `group`, `row_frac` |
| `tail` | values beyond a quantile | `columns`, `quantile`, `side`, `rate` |
| `rare_category` | levels with frequency below `max_freq` | `columns`, `max_freq`, `rate` |
| `mnar_self` | cells with probability driven by their own value (stress test) | `rate`, `strength` |
| `union` | the OR of sub-scenarios, e.g. scattered gaps plus an absent block | `parts` |

**Leakage control.**
- `mode: kfold` (default): every engine is fitted on the training folds and
  imputes the masked held-out fold, so no model sees a row it imputes. The
  folds are then reassembled into M completed copies of the full table.
- `mask_training_rows: true` masks the training folds too, simulating an
  incomplete training cohort.
- `mode: transductive` fits on the whole masked table. `tvae-decoder` then
  trains on its complete rows only, so it fails when high missingness leaves
  too few. A failure is recorded and the run continues.

**Metrics** are computed only on hidden cells, using all M draws.

| Property | Continuous | Discrete |
|---|---|---|
| Point accuracy | `nmae`, `nrmse` (÷ column SD), `bias_norm` | `accuracy`, `balanced_accuracy` |
| Probabilistic | `crps_norm` (CRPS ÷ SD) | `log_score`, `brier` (Laplace-smoothed draw frequencies) |
| Uncertainty | `coverage50`, `coverage90`, `width90_norm` | – |
| Distribution | `w1_norm` (Wasserstein-1 ÷ SD) | `tvd` |
| Dependence | `spearman_mad` (mean \|ΔSpearman\| over all pairs) | `cramers_v_mad` |

**Downstream inference** (`cox:`):
- Univariate or multivariable Cox models are fitted on each completed dataset.
- Estimates are pooled with Rubin's rules: Q̄, Ū, B, T = Ū + (1 + 1/M)B,
  Barnard–Rubin df and the fraction of missing information (FMI).
- Pooled estimates are compared with the fit on the unmasked data: log-HR bias,
  absolute HR error, whether the pooled CI covers the full-data estimate, CI
  width ratio, and agreement on significance.
- Single-imputation engines have B = 0 by construction. Their low coverage is
  the expected symptom of treating imputations as observed.

**MNAR sensitivity** (`delta:`):
- Imputed continuous cells shift by δ × SD.
- Binary cells have their per-cell log-odds shifted by δ and are re-sampled.
- The pooled Cox is re-fitted at each δ.
- `delta_tipping.csv` reports the smallest |δ| at which a term changes
  significance or direction. Terms marked robust over the grid don't flip.

**Outputs:**
- `summary.csv` (one row per scenario × engine)
- `cell_metrics.csv`, `dependence.csv`, `rubin_cox.csv`, `delta_sensitivity.csv`,
  `delta_tipping.csv`, `diagnostics.csv` (timing, ESS, pCN acceptance)
- `failures.csv` (if any), `report.md`, `report.json`, `imputation_config_used.yaml`
- `plots/`: metric panels, coverage calibration, delta trajectories, and
  imputed-vs-true distributions for the most-masked continuous column

### Reading the results

- Prefer engines with low **CRPS** and **log score** *and* near-nominal
  **coverage**. A mean-like imputer can win on NRMSE while being badly
  over-confident.
- Check `spearman_mad` and the Cox columns. An imputer that distorts
  associations is unsafe for inference, even with good cell-level accuracy.
- Block scenarios measure cross-modality borrowing: how well clinical data and
  the other omics block predict an absent block.
- At n ≈ 259, MICE or missForest may beat both TVAEs. That's a legitimate
  finding, and the reason the harness exists.

## Running on a shared server

- **CPU.** `n_jobs` (config) or `--n-jobs` caps the cores used: worker
  processes, BLAS/OpenMP threads, torch CPU threads and MICE chains. The
  default is 4; `-1` means all cores. Engines also apply the cap inside
  `fit`/`impute` when used from Python, so library use is just as polite.
- **GPU.** `device` (config) or `--device auto|cpu|cuda`:
  - **`auto`** (default): uses the GPU when available. If a saved model can't
    be loaded onto it, for instance because another user's job is holding the
    memory, it falls back to CPU with a warning.
  - **`cpu`**: forces CPU for loading *and* training (including the SDV TVAE
    inside `tvae-decoder` benchmarks).
  - **`cuda`**: never falls back.
  - Check with `nvidia-smi`. An idle Jupyter kernel keeps its GPU memory until
    it is shut down.
- The imputation test suite runs on CPU by default for the same reason. Set
  `PODE_TEST_GPU=1` to exercise the GPU path.

```bash
pode synthesis impute apply --model trained_hybrid_tuned/model.pkl \
    --data incomplete.csv --m 20 --output-dir out/ --device cpu --n-jobs 4
```

## Python API

Everything the CLI does is available programmatically. The examples below assume
a trained hybrid model and its `pode_prep/` hand-off directory.

**1. Impute with an existing PODE model**

```python
import numpy as np
import pandas as pd
from pode.schema.schema import DatasetSchema
from pode.imputation.config import LatentConfig
from pode.imputation.decoder_imputer import TVAEDecoderImputer

model_dir = "trained_hybrid_tuned"
imp = TVAEDecoderImputer.from_model(
    f"{model_dir}/model.pkl",
    DatasetSchema.from_yaml(f"{model_dir}/schema.yaml"),
    LatentConfig(k=2000),
    device="cpu",   # or "auto": GPU if it has room, else CPU
)
incomplete = pd.read_csv("incomplete.csv")
draws = imp.impute(incomplete, m=20, seed=42)   # list of 20 completed DataFrames
print(imp.diagnostics())                        # ESS, SMC stages, timings
```

**2. Any engine, with leakage controls (here: prediction-safe hybrid)**

```python
from pode.config import load_config
from pode.imputation import EngineSpec, ImputationConfig, build_imputer

prep = "runs/final_handoff/pode_prep"
pode_cfg = load_config(f"{prep}/pode_config.yaml")
table = pd.read_csv(f"{prep}/training_table.csv")
schema = DatasetSchema.from_dataframe(table, pode_cfg.schema)
train, test = table.iloc[:200], table.iloc[200:]   # test may contain gaps / absent blocks

icfg = ImputationConfig(
    block_groups={"expr": ["expr_latent_*"], "meth": ["methylation_latent_*"]},
    outcome_policy="hide",     # test rows never see their own OS_Time/OS_Status
    fit_scope="reference",     # MICE learns from training rows only
    n_jobs=4, device="auto",
)
hybrid = build_imputer(EngineSpec("hybrid"), pode_cfg, icfg, seed=1).fit(train, schema)
completed = hybrid.impute(test, m=20, seed=1)
print(hybrid.leakage_settings())   # recorded for your methods section
print(hybrid.diagnostics())        # rows routed to each sub-engine
```

`EngineSpec` takes any engine name plus `params`, e.g.
`EngineSpec("mice", params={"max_iter": 10, "outcome_policy": "use"})` or
`EngineSpec("tvae-masked", params={"epochs": 500, "p_max": 0.6})`.

**3. Rubin-pooled Cox on the completed datasets**

```python
from pode.imputation.metrics import pooled_cox

pooled = pooled_cox(completed, ["Age", "Sex", "M._versus_M."], schema,
                    "OS_Time", "OS_Status", ref=train)
print(pooled[["term", "coef", "se", "ci_lo", "ci_hi", "p", "fmi"]])
```

`coef` is the pooled log-HR, `se` the total-variance standard error,
`ci_lo`/`ci_hi` use Barnard–Rubin degrees of freedom, and `fmi` is the fraction
of missing information. `ref` fixes the categorical reference levels. Use
`mode="multivariable"` for one joint model.

**4. Score an imputer yourself on hidden values**

```python
from pode.imputation.masking import MaskSpec, apply_mask, make_mask
from pode.imputation.metrics import cell_metrics

holdout = table.iloc[200:].reset_index(drop=True)
mask = make_mask(holdout, MaskSpec("gaps", "mcar", rate=0.2), schema, np.random.default_rng(0))
mice = build_imputer(EngineSpec("mice"), pode_cfg, ImputationConfig(n_jobs=4), seed=0).fit(train, schema)
draws = mice.impute(apply_mask(holdout, mask), m=10, seed=0)
scores = pd.DataFrame(cell_metrics(holdout, draws, mask, schema))
print(scores[["column", "n_masked", "nrmse", "crps_norm", "coverage90", "accuracy"]])
```

**5. The whole benchmark from Python**

```python
from pode.imputation.benchmark import run_benchmark
from pode.imputation.config import load_imputation_config
from pode.pipeline import prepare_data

data, schema = prepare_data(table, pode_cfg)
icfg = load_imputation_config("configs/imputation/benchmark.example.yaml").quick()
result = run_benchmark(data, schema, pode_cfg, icfg, "out/bench_py")
print(result["tables"]["summary"].head())
```

**6. Pool in R with `mice`**

`imputed_long.csv` from `apply` is in the `mice` long format, where `.imp = 0`
is the original incomplete data:

```r
library(mice); library(survival)
long <- read.csv("out/imputed_long.csv", check.names = FALSE, na.strings = c("", "NA"))
imp  <- as.mids(long, .imp = ".imp", .id = ".id")
fit  <- with(imp, coxph(Surv(OS_Time, OS_Status) ~ Age + factor(Sex) + factor(`M._versus_M.`)))
summary(pool(fit), conf.int = TRUE, exponentiate = TRUE)
```

Keep `na.strings`: empty cells in text columns would otherwise become a
spurious `""` factor level.

## Worked scenarios

**A. A clinical + omics cohort with gaps, analysed for survival associations
(inference).**

```bash
# 1. Check which engine suits this kind of missingness on a complete subset.
pode synthesis impute benchmark --data complete_subset.csv --config pode_config.yaml \
    --impute-config my_impute.yaml --output-dir out/bench --n-jobs 4
# 2. Impute the real cohort with the winner (outcome used, as inference requires).
pode synthesis impute apply --engine hybrid --data cohort.csv --config pode_config.yaml \
    --impute-config my_impute.yaml --m 20 --output-dir out/imputed --seed 42
# 3. Pool the analysis across imputed_001..020.csv (Python example 3, or R above).
```

**B. Developing and validating a prognostic model (prediction).**

```bash
# Training split: impute using the outcome (it is known at development time).
pode synthesis impute apply --engine mice --data train.csv \
    --m 20 --output-dir out/train_imputed --seed 42
# Test split: fit on training data only, hide test outcomes, and never let the
# test rows shape the imputation model.
pode synthesis impute apply --engine mice --reference train.csv --data test.csv \
    --outcome-policy hide --fit-scope reference --m 20 --output-dir out/test_imputed --seed 42
```

**C. Patients missing an omics modality entirely.** Put the modality's
columns in `block_groups` and use `tvae-masked` or `hybrid`. Imputed latent
blocks can be decoded back to expression or methylation with
`pode synthesis decode` (see WORKFLOWS.md, section H).

## Caveats

- No imputer resolves MAR vs MNAR from the data alone. Include auxiliary
  variables that predict missingness, and report the delta analysis.
- `tvae-decoder` uses private SDV/ctgan attributes (`_data_processor`,
  `_model.decoder`, `_model.transformer`); `pode/imputation/sdv_bridge.py`
  validates them. It was built against sdv 1.25 / ctgan 0.12.
- Runtime grows with folds × scenarios × engines. The TVAE engines and GAIN use
  the GPU when present; scoring (Cox pooling, delta refits) and the classical
  baselines run on CPU. Scoring runs in parallel processes, and MICE draws run
  as parallel chains.
- CPU and GPU limits: see "Running on a shared server" above.
- Use `--quick` (M = 5, 3 folds, ≤ 50 epochs) to iterate, and `--engines` to
  subset.
- For prediction pipelines, nest imputation inside resampling. Fit the imputer on
  training folds only, as the benchmark's `kfold` mode does.
