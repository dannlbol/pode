# Changelog

## Unreleased

**Exported schema-v1 weights are covariate-aware by default.**

- The final full-data CpG refit now honours configured `adjust` covariates and Cox `strata`,
  matching the models used for stability selection. The search curve, LOO scores, inclusion
  frequencies, selected feature count and CpG membership are unchanged.
- Set `final_model.weight_refit: legacy_unadjusted` to reproduce the historical unadjusted
  exported coefficients. The effective mode is fingerprinted and recorded in result/model
  provenance.
- Final score centre/IQR are estimated only from endpoint- and covariate-eligible training
  observations, even when additional phenotype-matched samples are scored.

## 0.8.1

**`pode signatures benchmark` now compiles the metrics it was already computing.**

- The benchmark previously kept three pooled means per method and discarded the rest. Every
  fold-level metric `_metrics` produces — `time_dependent_auc`, `integrated_brier_score`,
  `clinical_c_index`, `incremental_c_index`, `metric_warning` — and the split between the
  `native` model and the `reduced_signature`, and the split across endpoints, were all
  computed and then dropped. They are now retained.
- `leaderboard.csv` keeps every column it had and gains `rank`, `n_folds`, `n_endpoints`,
  `endpoints`, `mean_`/`sd_`/`ci_low_`/`ci_high_` for each metric, `native_mean_c_index`,
  `native_mean_ipcw_c_index`, `median_feature_count`, `feature_count_range`,
  `n_metric_warnings` and `selected_parameters_modal`. Existing readers are unaffected.
- Four new tables: `benchmark_metrics.csv` (tidy method × endpoint × model_type × metric),
  `fold_metrics.csv` (all methods' per-fold rows pooled), `method_comparison.csv` (paired
  per-fold differences, win counts and a Wilcoxon signed-rank test between every method
  pair), and `signature_overlap.csv` (Jaccard agreement between selected feature sets).
- New `benchmark_report.md` renders the ranking, per-endpoint tables, non-completed methods,
  the largest paired differences and any warnings for reading.
- `result.json` gains `primary_metric`, `ranking`, `endpoints`, `fold_scheme` and `artifacts`;
  the CLI JSON summary gains `primary_metric`, `ranking` and `artifacts`.
- `evaluation.primary_metric` is now actually honoured: it selects the ranking metric, and
  `integrated_brier_score` correctly ranks ascending.
- No extra model fitting: this is pure aggregation of results already held in memory.

## 0.8.0

**Behaviour change — schema-v2 `phenotype.cohort_column` no longer defines a train/test split.**

- `cohort_column` previously switched outer validation to **leave-one-cohort-out** for every
  method, silently overriding `evaluation.outer_folds`. That measured cross-cohort transfer,
  not performance on the pooled training population, and was applied whether or not the method
  could model cohorts at all. It now means what it means in the schema-v1 metaCpG method: the
  training samples come from two or more sources (trials, eras, batches) whose underlying
  survival rates may differ. All cohorts train together.
- Methods that route through the PODE Cox loss (`elastic_net_cox`, `coxboost`,
  `stability_coxboost`, `pls_cox_multiblock`, `net_cox`, and the deep family) now cross cohort
  into the **Cox strata**: coefficients are shared, each cohort gets its own unspecified
  baseline hazard, and risk sets never cross cohorts. This is the single-joint-fit analogue of
  the per-cohort fits pooled by empirical-Bayes random effects in schema v1.
- `hierarchical_cox` additionally receives cohort strata (its per-cohort deviations act on the
  coefficients, not the baseline). `frailty_cox` and `maml_cox` keep their existing explicit
  handling and are not stratified, which would make their cohort terms unidentifiable.
- `random_survival_forest`, `gradient_boosted_survival` and `survival_svm` cannot represent
  cohorts. They are no longer silently pooled: results carry `cohort_handling: ignored` and a
  `cohort_warning` in `leaderboard.csv`, `result.json`, each endpoint `manifest.json`, and on
  the CLI during the run. New `phenotype.cohort_handling: skip` records them as `skipped`.
- Outer and inner folds are now balanced on **cohort × event status** (degrading to event-only,
  then plain K-fold, when cells are too small). Every cohort is represented in every training
  fold, so `hierarchical_cox`/`frailty_cox`/`maml_cox` now work with exactly **two** cohorts;
  they previously required three because leave-one-cohort-out consumed one.
- Leave-one-cohort-out is preserved as an explicit opt-in:
  `evaluation.cohort_validation: leave_one_cohort_out` (default `stratified_kfold`). It
  requires `cohort_column` and still ignores `outer_folds`.
- New per-fold metric `c_index_within_cohort`: the event-weighted mean of the per-cohort
  concordance, since pooled Harrell's C is contaminated when baseline rates differ.
  `integrated_brier_score` still uses an unstratified Breslow baseline and is an approximation
  when cohort strata are active.
- Schema: added `phenotype.cohort_handling` and `evaluation.cohort_validation`; documented
  `phenotype.cohort_column`; corrected `covariates[].mode` to `adjust | strata`, which the code
  has accepted since 0.6.0.
- **Migration**: results produced before this release with a `cohort_column` set are not
  comparable with results produced after it, and neither are SLURM per-fold resume caches (the
  config fingerprint covers config values, not split semantics). Start a fresh `run_id`. To
  reproduce the old numbers exactly, set
  `evaluation.cohort_validation: leave_one_cohort_out`.

## 0.7.0

- `metrics.csv`/`search_metrics.png` now report a leave-one-out cross-validated `neglog10_p`
  (Wald p-value of the LOO score's Cox coefficient) alongside AUC/C-index, and, when
  `covariates:` is configured, comparison columns `auc_adjusted`/`c_index_adjusted`
  (`neglog10_p_adjusted`) from a joint score+covariates LOO fit (respecting each covariate's
  `strata`/`adjust` mode) — reproducing the legacy R script's "CV Metagene + Clinical"
  comparison curve. Adjusted columns/series are only present when covariates are configured.
  Feature-count selection (`selection.mode: max_auc`) is unaffected — it still uses the
  unadjusted `auc` column. `search_metrics.png` is now a two-panel figure (performance metrics
  above, `-log10(p)` below, sharing the feature-count axis instead of a dual y-axis), and its
  title/legend explicitly label every series as leave-one-out cross-validated. The same
  `neglog10_p`/adjusted keys are also added to the final extraction `performance` dict
  (`result.json`).

## 0.6.0

- The batched/GPU Cox engine now handles **covariates**: `mode: strata` (per-stratum risk sets)
  and `mode: adjust` (numeric/categorical, via a batched multivariate Newton with per-CpG solves),
  and combinations. Covariate configs no longer fall back to the slow exact per-CpG solver; results
  are congruent with exact (verified) and CPU vs CUDA are bit-identical. Fixes the slow strata runs.
- Added `endpoints[].group_column`: express a two-cohort random-effects meta-analysis from a single
  stacked table (one time, one status, one cohort column) instead of two wide column-pairs. It
  expands internally to per-cohort endpoints pooled by random effects (any number of cohorts, two
  or more), inheriting the batched/GPU speed-up. For two cohorts the ranking/selected features match
  the hand-written wide form (they differ only in the per-fold risk-score normalization).
- Schema: added optional `endpoints[].group_column`.

## 0.5.0

- `extract`/`run` now write a portable `model.json` (plus `signature_model.csv`): the signature
  CpGs and weights, the training center/scale, the input scale convention, endpoints, and a
  provenance fingerprint — everything needed to score new samples.
- Added the `apply` subcommand and `apply_model`/`apply_signature`: score a new methylation
  matrix with a saved model to obtain per-patient meta-CpG scores (`sample_id, risk_score,
  raw_score`). New matrices default to the model's recorded input conventions (overridable);
  missing signature CpGs error by default, with `--allow-missing` to score the present subset.
  Applying a model to its own training matrix reproduces `risk_scores.csv` exactly.
- Factored the matrix reader into `_read_matrix`, shared by `validate_inputs` and `apply`.

## 0.4.0

- `run` in `manual` selection mode with no feature count now writes the per-count search
  summary (`metrics.csv`, `search_metrics.png`, `ranking.csv`, …) and stops cleanly instead
  of raising, so the range can be surveyed before choosing a count.
- The `search` subcommand now writes the search summary (previously printed only shapes);
  the `extract` subcommand now writes the extraction outputs (previously returned only
  performance) and runs **only** the extraction phase for a chosen `--feature-count`,
  without running or reusing the search. `extract --feature-count N` produces exactly the
  signature a one-shot `run --feature-count N` would, and validates the count against the
  search grid. Enables a two-step survey-then-extract workflow.
- Refactored the extraction half of `fit_metacpg` into `_extract_signature` (behavior
  preserved) and added `search_only` / `extract_signature` entry points.

## 0.3.0

- Added selectable Cox compute engines via `--engine` / `execution.engine`: `exact`
  (default, unchanged scipy per-CpG solver), `batched` (vectorized Newton–Raphson over
  all CpGs on numpy/CPU), `gpu` (the same batched Newton on CUDA via `torch`, float64),
  and `auto`. All engines solve the identical Efron partial likelihood and agree with the
  exact solver to ~1e-6; CPU and GPU are numerically identical (~1e-14).
- Restructured the leave-one-out search so the per-fold feature ranking is computed once
  per held-out fold and reused across all candidate feature counts (it is count-independent).
  Results are bit-for-bit unchanged; search Cox scans drop from `CpGs × folds × counts` to
  `CpGs × folds` (≈56× fewer on a 12,957-CpG test config). Per-count cache layout and provenance
  are preserved, so `execution.resume_run_id` remains compatible. The speed-up applies to
  every engine, including `exact`.
- Added the `verify-parity` subcommand: runs the workflow under several engines and prints
  a per-artifact pass/fail report (max abs/rel difference, selected-feature identity),
  exiting non-zero on failure. CPU vs GPU are checked under a strict tolerance; search-curve
  AUC/C-index are reported as advisory.
- Measured on a ≈12,957-CpG × 259-sample test matrix: a single Cox scan runs in
  585 s (exact) vs 1.27 s (batched, ~460×) vs 0.33 s (GPU, ~1,780×); full search+extraction
  completes end to end in ~1–2 minutes on the accelerated engines.

## 0.2.0

- Restored literal legacy per-count/per-fold Cox recalculation, ordinary-z ranking, fold median/IQR normalization, and fresh extraction.
- Added phase-separated SLURM execution, explicit run-scoped resume behavior, refit-count provenance, and direct golden tests against the read-only legacy R functions.
- Corrected Python Efron tie handling and ported the legacy survivalROC NNE estimator.

## 0.1.0

- Initial independent R and Python packages.
- Shared configuration schema and local/SLURM CLIs.
- Consolidated single/meta endpoint and covariate modes.
- Initial consolidated leave-one-out workflow.
- Portable outputs, provenance, synthetic tests, legacy fixtures, and parity tooling.
# Advanced signature framework

- Added schema v2 multi-omics inputs with masked-union and explicit row-order alignment.
- Added stable and experimental prognostic method registry, nested evaluation,
  portable reduced-Cox artifacts, prediction, benchmarking, and advanced SLURM manifests.
- Preserved schema-v1 `legacy_exact` Python and R behavior.
