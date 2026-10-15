# metaCpG-func

For schema-v2 multi-omics, machine-learning, deep-learning, and multi-study
methods, see [advanced.md](advanced.md). Everything below documents the preserved
schema-v1 `legacy_exact` workflow.

`metaCpG-func` is a standalone, configuration-driven refactor of the workflows previously launched by `metagene_run.sh` and `extractRun.sh`. It contains independent R and Python packages, a common YAML contract, local and SLURM launchers, portable results, and regression tools.

## What is consolidated

One parameterized workflow replaces the separate single-endpoint, meta-endpoint, stratified, non-stratified, and numeric-covariate wrappers. It supports:

- one or two survival endpoints;
- categorical covariates as Cox strata or ordinary adjustment terms;
- numeric adjustment terms;
- beta matrices or precomputed M-values;
- manual feature-count selection by default, with explicitly requested maximum-AUC selection available;
- local execution and `sbatch` submission;
- optional CpG annotation from an ordinary CSV/TSV table.

`methodology: legacy_exact` deliberately preserves the original leave-one-out calculation structure. For each held-out sample it refits all CpG Cox models, ranks by absolute ordinary Cox z, uses the legacy standardized-coefficient weights, and normalizes the held-out score with that fold's median and IQR. The ranking is computed once per fold and reused across every candidate feature count (see [Per-fold ranking reuse](#per-fold-ranking-reuse-the-dedup)); it is never reused between the search and extraction phases. Final extraction performs a second, fresh leave-one-out run at the chosen feature count and selects CpGs by inclusion frequency. By default the exported final weights are refitted with the configured covariates and strata; `final_model.weight_refit: legacy_unadjusted` restores the original wrappers' unadjusted export.

Every reported `auc`/`c_index` (in `metrics.csv`, `search_metrics.png`, and the final extraction `performance`) is a leave-one-out cross-validated estimate — never a full-data/in-sample fit — and each is accompanied by `neglog10_p`, the `-log10` Wald p-value of the cross-validated score's own Cox coefficient. When `covariates:` is configured, a second comparison metric is also reported — `auc_adjusted`/`c_index_adjusted` (and `neglog10_p_adjusted`) — from a joint Cox fit of the score plus every configured covariate (respecting each covariate's `strata`/`adjust` mode), mirroring the legacy R script's "CV Metagene + Clinical" comparison curve. These adjusted columns are informational only: they are never used for feature-count selection (`selection.mode: max_auc` still picks by the unadjusted `auc` column), and they are simply absent from `metrics.csv` when no covariates are configured.

The per-CpG Cox scan can run under a choice of **compute engines** (the exact scipy solver, a vectorized CPU solver, or a CUDA/GPU solver) that all return the same result to numerical tolerance — see [Compute engines](#compute-engines-cpugpu-acceleration).

Stability is measured across leave-one-out perturbations of the cohort. Repeating the whole command recalculates the same deterministic folds; it is not bootstrap or repeated random cross-validation. Resume caching is disabled unless `execution.resume_run_id` is explicitly set, is keyed by phase/count/endpoint/fold, and is never shared with the fresh extraction phase.

## Layout

- `python/`: native Python package.
- `R/`: R reference package.
- `schema/config.schema.json`: shared versioned configuration contract.
- `config/`: generic and read-only legacy examples.
- `bin/`: source-tree CLI launchers.
- `scripts/`: legacy conversion, parity validation, and fit-count benchmark.

## Installation

Python (prefer a fresh virtual environment):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e .
```

R:

```bash
Rscript -e 'install.packages("renv"); renv::restore(project="R")'
R CMD INSTALL R
```

The root launchers can also use the source trees directly after their dependencies are installed:

```bash
bin/metacpg-py validate --config config/example.yml
bin/metacpg-r validate --config config/example.yml
```

## Configuration

Copy `config/example.yml`, then set the matrix and phenotype paths. Matrix columns and phenotype rows are joined explicitly through `phenotype.sample_id_column`; order is never assumed. By default unmatched samples, duplicate IDs, invalid event statuses, non-positive times, missing matrix values, and beta values outside `(0,1)` are rejected before modelling.

Covariate examples:

```yaml
covariates:
  - column: molecular_subgroup
    type: categorical
    mode: strata
  - column: age
    type: numeric
    mode: adjust
final_model:
  weight_refit: covariate_aware  # default; use legacy_unadjusted for historical exports
```

Both covariate modes — `strata` (Cox strata / cohort-specific baselines) and `adjust`
(numeric or categorical adjustment terms) — run on the batched CPU and GPU engines and are
congruent with the exact solver (verified by `verify-parity`). A strata covariate no longer
forces the slow per-CpG path.

`final_model.weight_refit` affects only the full-data weights exported after feature selection.
It does not change the LOO search curve, inclusion frequencies, chosen feature count, or selected
CpG membership. Changing it does change `selected_features.csv` coefficients, the portable model,
fixed scores and downstream validation, so those artifacts must be regenerated together.

### Two-cohort meta-analysis from one table

The random-effects meta-analysis is normally expressed as **two** `endpoints`, each pointing
to its own time/status columns (one cohort per column-pair). If instead your two cohorts are
**stacked in one table** — a single `time`, a single `status`, and a `cohort` column — add
`group_column` to a single endpoint and the tool splits it into the per-cohort meta:

```yaml
endpoints:
  - name: pooled
    time: OS_Time
    status: OS_Status
    prediction_time: 1095
    group_column: cohort      # split into per-cohort Cox, pooled by random effects
```

This fits each cohort's per-CpG Cox separately (on the batched/GPU engine) and pools the
effect sizes with the empirical-Bayes random-effects model. The cohort column may have **two or
more** non-null levels — the pooling generalizes to any number of cohorts. Any covariates still
apply, so each cohort's Cox may itself be stratified or adjusted.

For two cohorts, the ranking and selected features are identical to writing the two endpoints
by hand in wide format; the two forms differ only in the per-fold risk-score normalization
(`group_column` normalizes each held-out score against all cohorts' training samples, whereas
the explicit wide form reproduces the legacy two-endpoint normalization).

The full schema is in `schema/config.schema.json`.

## Running

Run the complete search and extraction locally:

```bash
metacpg-py run --config analysis.yml --backend local
metacpg-r run --config analysis.yml --backend local
```

Set persistent local concurrency in the YAML, or override it for one Python
invocation without editing the file:

```yaml
execution:
  workers: 8
  memory_budget: 8G  # advisory process-pool budget
  heartbeat_seconds: 300  # periodic progress/memory update interval; 0 disables it
  progress: false  # suppress Cox fit-count progress rendering
  engine: exact  # exact (default) | batched (vectorized CPU) | gpu (CUDA) | auto
```

```bash
metacpg-py run --config analysis.yml --backend local --workers 8
# --cores 8 is an alias for --workers 8

# Apply an advisory process-memory budget for just this invocation.
metacpg-py run --config analysis.yml --backend local --cores 20 --memory-budget 8G

# The default heartbeat is 300 seconds; override or disable it for one run.
metacpg-py run --config analysis.yml --backend local --heartbeat-seconds 900
metacpg-py run --config analysis.yml --backend local --no-heartbeat

# Suppress Cox fit progress bars/count updates while retaining phase and memory messages.
metacpg-py run --config analysis.yml --backend local --no-progress

# Load and validate the cohort, then print the process and memory plan without fitting Cox models.
metacpg-py run --config analysis.yml --backend local --cores 20 --memory-budget 8G --preflight
```

The local Python engine uses this value as the number of concurrent
leave-one-out tasks. Linux uses a process pool, while other platforms retain a
thread fallback. It is not a guarantee that exactly that many CPU cores will be
saturated. Local runs report validation,
search, extraction, and result-writing progress to stderr: interactive terminals
show Cox-fit progress bars, while redirected logs receive periodic plain-text
updates. The final JSON remains on stdout for scripts.

On Linux, local leave-one-out tasks run in a forked process pool, so the loaded
matrix is inherited copy-on-write rather than serialized for every worker. Set
`execution.memory_budget` or `--memory-budget` to cap the permitted process
count from a conservative private-memory estimate. The runner reports that
estimate before each LOO phase and periodically reports aggregate PSS (falling
back to RSS if PSS is unavailable). This is an **advisory** budget, not a hard
operating-system memory limit; use a scheduler allocation or an
administrator-enabled cgroup for an enforceable job-wide cap.

Use `--preflight` to make this choice before a fit starts. It performs input
validation only, then writes JSON with the feature/sample counts, expected LOO
fit count, requested and permitted worker counts, and the parent, matrix,
per-worker, and total memory estimates. It creates no output files or cache.

`--workers` / `--cores`, `--memory-budget`, heartbeat options, and `--preflight` are local-only. For SLURM, set
`execution.slurm.cpus` and `execution.slurm.memory` for the scheduler allocation
and keep `execution.workers` at or below the allocated CPU count for the worker
process running inside each job.

Provide or override the manually selected count:

```bash
metacpg-py run --config analysis.yml --feature-count 19
```

### Two-step workflow: survey the range, then extract a count

In `manual` selection mode you usually don't know the feature count you want until you
have seen the search curve. Run the search first — with no count it writes the per-count
summary and **stops cleanly** instead of failing:

```bash
# Step 1: search across the feature-count range and write the summary.
metacpg-py run --config analysis.yml --engine auto --cores 20
# (equivalently: metacpg-py search --config analysis.yml --engine auto --cores 20)
```

This writes `metrics.csv` (leave-one-out cross-validated AUC / C-index / `neglog10_p` per
feature count, plus `auc_adjusted` / `c_index_adjusted` / `neglog10_p_adjusted` when
`covariates:` is configured), `search_metrics.png`, `ranking.csv`, and the other search
tables to `<output_dir>/python/`, and prints the next step. Nothing fails for lack of a
count. `search_metrics.png` is a two-panel figure (performance metrics on top, `-log10(p)`
below, sharing the feature-count axis) with every series labeled "LOO CV" so it can't be
mistaken for a full-data/in-sample fit.

Inspect `metrics.csv` / `search_metrics.png`, choose a count, then extract just that
signature:

```bash
# Step 2: extract one feature count (writes selected_features.csv, risk_scores.csv, ...).
metacpg-py extract --config analysis.yml --engine auto --cores 20 --feature-count 25
```

`extract` runs **only** the extraction phase — a fresh leave-one-out pass at the chosen
count plus the full-data weight fit — into the same `<output_dir>/python/`, leaving the
search summary in place. It does not re-run (or reuse) the search; extraction is
deliberately a separate, fresh phase, so `extract --feature-count 25` produces exactly the
signature that a one-shot `run --feature-count 25` would. Its cost is one LOO pass
(comparable to the search, ~30–40 s on the GPU for a ≈12,957-CpG × 259-sample matrix). The feature count
must lie within the search grid (`search.min_features … max_features`); an out-of-range or
missing count is rejected with a clear message.

If `selection.mode: max_auc` is set instead, `run` needs no count — it auto-selects the
best-AUC count and completes search and extraction in one command; a standalone `extract`
in that mode reuses the `metrics.csv` written by a prior search.

Render a SLURM job without submitting it, inspect it, then submit:

```bash
metacpg-py submit --config analysis.yml --dry-run
metacpg-py submit --config analysis.yml
```

SLURM module/activation commands and resources live under `execution.slurm`, so no cluster paths are embedded in code. Submission creates a candidate-count search array, a dependent extraction job using a separate phase cache, and a final aggregation job. Each array element performs the same per-count/per-fold recalculations as local `legacy_exact`; task distribution does not change the statistics. SLURM submission requires a manual `selection.feature_count` because extraction is intentionally a distinct second phase.

## Compute engines (CPU/GPU acceleration)

The runtime of `legacy_exact` is dominated by one operation: a **univariate Cox
proportional-hazards fit for every CpG**, repeated for every leave-one-out fold.
The original method performs each of these fits independently — one call to a
scipy BFGS optimizer per CpG, inside a Python loop over the rows of the
methylation matrix. On a test matrix of ≈12,957 CpGs × 259
samples, a single full-matrix scan takes ~585 s, and a full search/extraction
run performs one such scan per fold.

`--engine` (or `execution.engine` in the YAML) selects **how** those Cox models
are fit. Every engine solves the *same* Efron partial likelihood and returns the
same coefficients, standard errors, z-scores, ranking, selected features, and
risk scores to numerical tolerance; they differ only in speed.

| `--engine` | Solver | Where it runs | Notes |
|---|---|---|---|
| `exact` (default) | scipy BFGS, one fit per CpG | fork process pool over folds | Unchanged reference behaviour. |
| `batched` | vectorized Newton–Raphson, all CpGs at once (numpy) | fork process pool over folds | No GPU required; typically the fastest at ~10⁴ CpGs because 20-way fold parallelism beats a single serial GPU. |
| `gpu` | the same batched Newton on CUDA (`torch`, float64) | folds run serially in the main process | Requires a CUDA-capable `torch`; a GPU context cannot be shared across forked workers, so folds are serial. Its margin grows as the CpG count grows. |
| `auto` | `gpu` if a CUDA device is available, else `batched` | — | Recommended default when you want acceleration without thinking about hardware. |

```bash
# Vectorized CPU engine (no GPU needed)
metacpg-py run --config analysis.yml --engine batched --feature-count 30 --cores 20

# CUDA engine
metacpg-py run --config analysis.yml --engine gpu --feature-count 30

# Pick the best available automatically
metacpg-py run --config analysis.yml --engine auto --feature-count 30 --cores 20
```

`--engine` is a local-only option (schema-v1 legacy runs). The SLURM worker path
continues to use the exact engine.

### What "batched" mode is, and why it differs from the original

Every CpG is a *single-covariate* Cox model that shares the **same** survival
outcome (time, status) across all features, and therefore the **same risk sets
and tie structure**. The original method ignores this: it treats each CpG as an
independent optimization problem and hands it to a general-purpose BFGS solver,
one CpG at a time.

Stratified and covariate-adjusted models share the structure too: strata just
split the risk sets (summed per stratum), and adjustment terms add a small shared
design block, so the batched engine fits them as a stratified multivariate Newton
over all CpGs. Covariate configs (`mode: strata` and/or `mode: adjust`) are
therefore GPU/batched-accelerated as well, and congruent with the exact solver.

The batched engine exploits the shared structure. It builds the survival
"skeleton" (event times, tie groups, risk-set boundaries) **once**, then fits
**all** CpGs simultaneously with a Newton–Raphson iteration expressed as a few
vectorized array operations over the whole `samples × CpGs` block. Instead of
hundreds of thousands of small Python/scipy calls, each Newton step is a handful
of large array reductions — which is exactly the shape of computation a
multi-core CPU (numpy/BLAS) or a GPU (`torch`) executes efficiently.

Two properties make this safe:

- **Same mathematics.** The batched Newton is applied to the *identical* Efron
  partial likelihood implemented in the exact solver, term for term. It
  converges to the same maximum-likelihood estimate, so coefficients and
  standard errors match the scipy solver to ~1e-6 (that residual is the exact
  BFGS solver's own convergence tolerance; the batched Newton is actually the
  tighter of the two).
- **CPU and GPU are identical.** The CPU (`batched`) and CUDA (`gpu`) paths run
  the *same* float64 kernel, so they agree to machine precision (~1e-14).

Non-identifiable features (e.g. a perfectly constant CpG, whose Cox model has no
information) are frozen at coefficient 0 and sort to the bottom of the ranking,
where a genuinely non-informative feature belongs.

Measured on a ≈12,957-CpG × 259-sample test matrix, a single full-matrix Cox scan:

| Engine | Time | Speed-up vs exact |
|---|---:|---:|
| exact (scipy, per CpG) | 585 s | 1× |
| batched (numpy, CPU) | 1.27 s | ~460× |
| gpu (CUDA) | 0.33 s | ~1,780× |

End to end (search + extraction) the two accelerated engines complete the full
12,957-CpG workflow in ~1–2 minutes versus hours for the exact engine. At this
CpG count the CPU `batched` engine is competitive with — often faster than — the
serial `gpu` engine, because it runs many folds in parallel across cores while
the GPU processes folds one at a time; the GPU's advantage grows as the number
of CpGs increases (e.g. full 450k/850k arrays) or if folds are later batched
onto the device.

<a name="per-fold-ranking-reuse-the-dedup"></a>
### Per-fold ranking reuse (the "dedup")

This is a **restructuring of redundant work, not a change to any result** — no
data, fold, or feature is dropped.

In the original method, the leave-one-out search was organised as one task per
`(candidate feature count, held-out sample)` pair. For each such task it refit
the entire per-CpG Cox scan and re-ranked the features. But the ranking depends
**only on the held-out sample** — not on the candidate count. The count is used
solely at the very end, to take the top *N* features of the ranking
(`ranking.head(n)`). So a fold with a count grid of, say, 5→60 recomputed the
identical full Cox scan **56 times**, once per count, and threw away 55 of them.

The change ("dedup") computes the ranking **once per held-out fold** and reuses
it for every candidate count in that fold. Because the ranking is provably
independent of the count, the scores, selected features, inclusion frequencies,
metrics, and risk scores are **bit-for-bit the same** as before — the existing
worker-parallelism regression test confirms the exact engine's output is
unchanged. What changes is only the amount of redundant computation: the search
phase now performs `CpGs × folds` Cox scans instead of `CpGs × folds × counts`,
a reduction equal to the size of the count grid (≈56× on a 12,957-CpG test config).
Per-count cache files (`n{count}_endpoint{e}_fold{h}.json`) and task provenance
keep the same layout, so `execution.resume_run_id` remains compatible. This
speed-up is independent of the engine and applies even to `--engine exact`.

#### The leave-one-out re-ranking is preserved

There are two distinct kinds of repetition in the original search, and only one
of them is removed:

| Repetition | Original | After dedup | Status |
|---|---|---|---|
| Re-rank per **held-out sample** (the LOO reassessment) | yes | **yes** | preserved — this is the statistics |
| Re-rank per **feature count** within a fold | yes (identical every time) | no — computed once, then sliced per count | removed — it was recomputing the same numbers |

The count-level repetition was pure waste: the ranking is mathematically
independent of how many top features are later kept, so all *k* recomputations
in a fold produced the identical ranking. The **sample-level** repetition is the
opposite — it is the mechanism by which stability is assessed. A CpG's inclusion
frequency (selected in some folds, not others) is only meaningful *because* the
ranking is recomputed each time a different patient is held out, and that is
fully intact.

Concretely, per held-out sample `h`, [`_fold_ranking`](../../metacpg/core.py)
(1) drops that sample from its endpoint (`rows = rows[rows != heldout]`),
(2) refits **all** CpG Cox models on the remaining N−1 patients via
`_endpoint_stats`, and (3) produces a fresh `_rank(...)`. The task list in
`_exact_loo` contains one such fold per valid held-out sample.

Empirical check on a test matrix (300-CpG subset, 259 patients), using
`_fold_ranking` directly:

- **259 folds — exactly one per patient.**
- Of 30 sampled folds, **all 30 produced a distinct top-30 ranking**; if the
  ranking were computed once and reused globally they would be identical
  (e.g. holding out patient 0 vs patient 1 agrees down to rank 1, then swaps the
  rank-2 and rank-3 CpGs).
- The fit counter reports **9,000 = 300 CpGs × 30 folds** Cox fits, confirming
  the entire feature scan is genuinely re-run for every held-out patient.

This is verifiable directly:

```python
from metacpg import core
cfg = core.read_config("analysis.yml"); data = core.validate_inputs(cfg)
folds = [int(h) for h in core._valid_endpoint_rows(data, data.endpoints[0])]
print(len(folds))  # one fold per valid held-out patient
be = core._make_backend("batched"); counter = core.FitCounter()
tops = {h: tuple(core._fold_ranking(data, 0, h, counter, engine="batched",
                                    backend=be)[0]["feature"].head(30))
        for h in folds[:30]}
print(len(set(tops.values())))  # distinct rankings across folds (expect ~all)
```

### Verifying the engines agree (`verify-parity`)

`verify-parity` runs the whole workflow under several engines and reports, per
output artifact, the maximum absolute/relative difference, whether the selected
signature is identical, and an overall PASS/FAIL against a tolerance. It exits
non-zero on failure, so it is usable in CI.

```bash
# Prove the exact, CPU, and GPU engines produce the same result, and
# additionally that CPU and GPU are numerically identical.
metacpg-py verify-parity \
  --config analysis.yml --feature-count 30 \
  --engines exact,batched,gpu --workers 20 \
  --output parity_report.json
```

The first engine in `--engines` is the reference (default `exact`). Every other
engine is compared against it within `--tolerance` (default `1e-6` on
coefficients/z-scores/risk scores, with identical selected features required),
and the two batched kernels are additionally compared against each other under a
strict `1e-9`. Search-curve AUC/C-index are reported but marked *advisory* (a
`*`): they are rank-based step functions that can shift at a non-selected count
if a near-tie feature swaps, without changing the final signature.

Observed on the same test matrix: CPU vs GPU agree to ~1e-14 end to end with an
identical selected signature; both match the exact reference within 1e-6, with
identical selected features and risk scores agreeing to ~1e-9. (A full exact
reference run takes hours, so the exact-vs-accelerated comparison is practical
to run on a CpG subset; the CPU-vs-GPU comparison runs quickly on the full
matrix.)

## Outputs

Each analysis directory contains separate `r/` and `python/` subdirectories, so the engines never overwrite one another. They contain CSV tables for endpoint statistics, fold-normalized search scores, search and extraction inclusion frequencies, task provenance, fresh extraction scores, selected features, and final scaled patient risk scores. `result.json` records methodology, run ID, Cox-refit counts, median/IQR scaling, performance, runtime, configuration/input fingerprint, runtime version, and analysis ID.

### What `extract` produces, and the saved model

After `extract` (or a one-shot `run --feature-count N`), `<output_dir>/python/` holds the
trained signature and a portable model you can apply to new cohorts:

| File | Contents |
|---|---|
| `selected_features.csv` | the chosen CpGs, their `rank`, and their weights (`standardized_coef`) |
| `risk_scores.csv` | the per-patient meta-CpG score for every phenotype-matched input sample (`risk_score`, `raw_score`) |
| `model.json` | the **portable model**: signature CpGs + weights, the training `center`/`scale`, the input scale convention, endpoints, and provenance |
| `signature_model.csv` | the same signature as a tidy `feature, rank, weight` table |
| `result.json` | methodology, scaling, performance, fingerprint, run ID |

The meta-CpG score for a patient is:

```
raw        = mean over signature CpGs of ( M-value(x[cpg]) * weight[cpg] )   # beta -> M: log2(b/(1-b))
risk_score = (raw - center) / scale                                          # center = median, scale = IQR of endpoint-eligible TRAINING raw scores
```

`center` and `scale` are frozen from the endpoint-eligible training observations into
`model.json`; samples with incomplete endpoint/covariate data can be scored but never influence
those fitted scale parameters. Scores from a new cohort are therefore on the same scale and
directly comparable.

### Applying a saved model to new samples

Use `apply` to score a new methylation matrix (same CpG probe IDs) with a saved `model.json`:

```bash
metacpg-py extract --config analysis.yml --feature-count 25            # writes model.json
metacpg-py apply \
  --model results/analysis/python/model.json \
  --matrix new_cohort_methylation.csv \
  --output new_cohort_scores.csv
```

This writes `sample_id, risk_score, raw_score` per patient. The new matrix defaults to the
orientation, scale, and feature-id column recorded in the model; override with
`--orientation`, `--scale`, `--feature-id-column`, or `--separator` if the new file is laid
out differently. By default every signature CpG must be present (the score is a mean over the
exact trained set); `--allow-missing` scores over the CpGs that are present and reports how
many were used. Applying a model to its own training matrix reproduces `risk_scores.csv`
exactly, so the saved model *is* the pipeline's scorer.

## Testing and regression validation

Fast synthetic tests exercise conversion, alignment, selection, endpoint/covariate modes, both engines, output creation, and SLURM rendering:

```bash
make test
```

The full G3/G4 leave-one-out regressions are intentionally long-running and are not part of the fast unit suite. Run the desired engine with `config/legacy_g34_single_n36.yml` and `config/legacy_g34_meta_n19.yml`, then convert the saved legacy RDS fixtures and compare results with `scripts/validate_legacy.sh` or the `compare` API/CLI.

Acceptance thresholds are:

| Engine | CpG overlap | Risk-score Spearman | Max absolute C-index/AUC delta |
|---|---:|---:|---:|
| R | 0.95 | 0.99 | 0.01 |
| Python | 0.80 | 0.95 | 0.03 |

The comparison command exits non-zero when a threshold fails, making it suitable for a batch validation job or CI.

## Reproducibility notes

Feature-ranking ties preserve input feature order, matching the legacy R ordering. When `selection.mode: max_auc` is explicitly used, tied maximum AUCs select the smallest model. Differences can still arise between R and Python Cox optimizers, empirical-Bayes implementations, time-dependent ROC implementations, BLAS, and package versions; the R golden tests require the legacy calculations to match within floating-point tolerance.
