# Advanced prognostic signatures (schema v2)

Schema v2 adds multi-omics and cohort-aware prognostic modelling without changing
the schema-v1 `legacy_exact` workflow. The advanced implementations are canonical
PODE adaptations: they implement the named statistical objective or architecture,
but do not claim exact numerical parity with the authors' software.

> **Detailed reference:** [methods.md](methods.md) documents every method's tunable
> parameters and defaults, how clinical covariates and multiple training cohorts
> are handled, how to run a parameter grid search, and every SLURM option. The
> fully commented example is
> [../../configs/signatures/example.advanced.yml](../../configs/signatures/example.advanced.yml).

## Quick start

Generate the example inputs, validate them, list available methods, and run one
method:

```bash
python examples/signatures/generate_multiomics.py
pode signatures methods
pode signatures validate --config configs/signatures/example.advanced.yml
pode signatures run --config configs/signatures/example.advanced.yml \
  --method elastic_net_cox
```

Benchmarking is budgeted explicitly:

```bash
pode signatures benchmark --config configs/signatures/example.advanced.yml --preset quick
pode signatures benchmark --config configs/signatures/example.advanced.yml --preset balanced
pode signatures benchmark --config configs/signatures/example.advanced.yml --preset exhaustive
```

`quick` runs configured stable methods with reduced folds/epochs; `balanced` and
`exhaustive` include configured experimental methods. Failures, unavailable
optional dependencies, and preset skips are retained in `leaderboard.csv`.
Benchmark, run, prediction, and SLURM commands report method, endpoint, and
fold stages to stderr while the final JSON summary remains on stdout.

A benchmark writes six artifacts to `<output_dir>/python/benchmark/`. They are all
aggregations of the per-fold metrics each method already produces, so nothing is
refitted to build them:

| File | Grain | What it is for |
| --- | --- | --- |
| `leaderboard.csv` | one row per method | The ranked scoreboard. `rank` is 1 = best on `evaluation.primary_metric`. Every metric carries `mean_`, `sd_`, `ci_low_`, `ci_high_` (normal-approximation 95% CI over folds), alongside `n_folds`, `median_feature_count`, `n_metric_warnings` and the modal tuned parameters. `native_mean_c_index` shows what the full model scored before reduction to a signature. |
| `benchmark_metrics.csv` | method × endpoint × model_type × metric | Tidy long form. This is where per-endpoint detail and the `native` vs `reduced_signature` split live, which is why the leaderboard can stay one row per method. |
| `fold_metrics.csv` | one row per method × endpoint × fold × model_type | Every method's raw per-fold rows in one table, for your own re-analysis. `inner_scores` is dropped here; it is kept in full in each per-method directory. |
| `method_comparison.csv` | ordered method pair × endpoint × metric | Paired per-fold differences: `mean_diff`, `wins_a`/`wins_b`/`ties`, and a Wilcoxon signed-rank test. Both orderings are kept so the table can be joined on `(method_a, method_b)`. |
| `signature_overlap.csv` | ordered method pair × endpoint | Jaccard agreement between the methods' selected feature sets, plus the shared feature names. Two methods scoring the same on very different features is worth knowing. |
| `benchmark_report.md` | — | The same content rendered for reading. |

Pairing is valid because outer folds are seeded from the endpoint alone, not the
method, so every method in one benchmark sees identical splits.

**Read `n_pairs` before reading `wilcoxon_p`.** The number of paired folds is small
by construction — with 5 outer folds the smallest attainable two-sided signed-rank
p-value is 0.0625, so no pairwise result can reach p < 0.05 no matter how large the
gap. Treat `mean_diff` and the win counts as the primary evidence and the p-value as
a tiebreak.
The shipped advanced example enables every registered method. Its input generator
also writes the graph and pathway fixtures needed by `net_cox` and `cox_nnet`.

Install the optional GPL survival estimators when RSF, gradient boosting, or SVM
is required:

```bash
pip install -e '.[signatures-advanced]'
```

This pins scikit-survival 0.24.1, the release compatible with the suite's pinned
scikit-learn 1.6.1. PODE remains MIT licensed and the GPL package is not part of
the default dependency set.

## Data contract

`alignment.mode: sample_id` is the default. Phenotype and every samples-by-features
block must name a sample-ID column. Samples may omit entire blocks: PODE aligns
blocks to phenotype IDs, fills the standardized missing block with zero (its
training mean), and supplies explicit block-presence masks. Block-only IDs without
a phenotype row are rejected.

`alignment.mode: row_order` is an explicit compatibility mode for ID-less files
such as some public-cohort exports. Every phenotype and omics table must have the
same number and order of rows; missing blocks are not allowed in this mode.

Transforms (`none`, `log1p`, or `beta_to_m`), imputation, scaling, clinical
encoding, ranking, and selection are fitted within training folds. Each entry in
`covariates` takes a `mode` of `adjust` (default) or `strata`. `mode: adjust`
covariates are mandatory/unpenalized adjustment terms: they get their own
coefficient and never count toward the omics signature. `mode: strata`
covariates get no coefficient at all — they partition samples into groups that
each get their own implicit baseline hazard, mirroring how `mode: strata`
behaves in the legacy `legacy_exact` schema. Schema v2 does not silently
reinterpret one mode as the other. Strata support is implemented for every
method family except the scikit-survival-backed ones (`random_survival_forest`,
`gradient_boosted_survival`, `survival_svm`), which have no notion of a
stratified Cox baseline hazard. Those three may still be listed in
`methods.enabled` alongside a `mode: strata` covariate (e.g. in a shared config
used for several `--method` invocations or a `benchmark` run); only actually
building/fitting one of them raises an error, scoped to that method — `pode
signatures validate` and every other method are unaffected. `benchmark` records
the incompatible methods as `failed` in its leaderboard rather than aborting.

`phenotype.cohort_column` is a separate concept from `mode: strata` covariates:
it names the source (trial, era, batch) each training sample came from, so that
methods which support it can give each source its own baseline hazard. It does
not define a train/test split. The same three scikit-survival methods cannot
honour it either; by default they run pooled and are flagged with
`cohort_handling: ignored` plus a `cohort_warning`, and `phenotype.cohort_handling:
skip` records them as `skipped` instead. See
[methods.md](methods.md#cohorts-and-multiple-training-cohorts).

Multiple endpoints are supported as separate right-censored prediction tasks.
The deep methods use the same block-aware architecture for each endpoint; the
saved artifact contains one endpoint-specific head/signature per endpoint.
Cause-coded competing risks are not implemented.

## Methods

| CLI name | Status | Notes / required inputs |
|---|---|---|
| `elastic_net_cox` | stable | Sparse Cox partial likelihood; mandatory clinical terms |
| `coxboost` | stable | Component-wise updates plus mandatory covariate updates |
| `stability_coxboost` | stable | Complementary half-sample selection frequencies |
| `pls_cox_multiblock` | stable | PODE event-weighted multiblock PLS-Cox adaptation |
| `random_survival_forest` | stable | Optional scikit-survival extra |
| `gradient_boosted_survival` | stable | Optional scikit-survival extra |
| `survival_svm` | stable | Optional scikit-survival extra |
| `deepsurv` | stable | Cox-loss MLP |
| `autoencoder_cox` | stable | Joint reconstruction and Cox losses |
| `net_cox` | experimental | `graph_path` CSV with `source,target[,weight]` |
| `cox_nnet` | experimental | `pathway_path` CSV with `feature_index,pathway_index` |
| `surv_transformer` | experimental | Block-token transformer (SurvTRACE-style) |
| `prototype_contrastive_survival` | experimental | Block fusion plus learned prototypes (PCLSurv-style) |
| `moli_survival` | experimental | Modality-specific late integration |
| `hierarchical_cox` | experimental | Shared effect plus per-cohort deviations; requires at least two values in `cohort_column` |
| `maml_cox` | experimental | First-order cohort-task meta-updates; requires cohorts |
| `frailty_cox` | experimental | Penalised cohort random-effect adaptation; requires cohorts |

Experimental methods are operational CLI methods, not placeholders. Their output
manifests carry an adaptation notice because external benchmark parity has not
been established.

### Parameters and grid search

Each method has its own tunable parameters (with documented defaults) set under
`methods.config.<name>`; see the per-method tables in
[methods.md](methods.md#per-method-parameters). Any key you omit uses its default.

A single hand-picked parameter point makes the benchmark leaderboard a poor
contest, so a method config may instead declare a `search_space:` grid that is
tuned per outer fold by inner cross-validation:

```yaml
methods:
  config:
    elastic_net_cox:
      max_iter: 200
      search_space:
        l1: [0.001, 0.005, 0.02]   # nested keys like params.n_estimators tune sksurv estimators
        l2: [0.001, 0.01]
```

Presets truncate each axis (`quick` → 2 values, `balanced` → 3, `exhaustive` →
full grid). Full details in [methods.md](methods.md#grid-search-search_space).

## Evaluation and extracted signatures

With two or more cohorts, outer validation holds out one cohort at a time;
otherwise it uses event-stratified K-fold validation. Candidate feature counts
are compared in inner folds. The native model is evaluated independently from a
portable reduced Cox signature refitted on the top-ranked original omics features.
Mandatory clinical terms and technical block masks are retained in the reduced
model but excluded from its feature count.

Every method directory contains:

- `fold_predictions.csv` and `fold_metrics.csv` for native and reduced models;
- `feature_rankings.csv`, `selected_features.csv`, and optional stability scores;
- `signature_coefficients.csv` and full-data `risk_scores.csv`;
- `result.json` with method maturity, fingerprint, runtime, and adaptation notice;
- `model/manifest.json` plus endpoint manifests containing preprocessing and
  portable coefficients.

Apply the portable signature to another schema-v2 dataset with:

```bash
pode signatures predict \
  --model-dir results/advanced_multiomics_example/python/elastic_net_cox/model \
  --config external-validation.yml \
  --output-dir external-scores
```

Unknown extra features are ignored, required training features absent from a
block are mean-imputed, and unseen categorical levels use the stored reference
encoding. Model artifacts must be treated as trusted analysis inputs.

## SLURM

`pode signatures submit --config config.yml --dry-run` writes a method task
manifest and an array script (one task per method × endpoint × outer fold) plus a
dependent aggregation job; drop `--dry-run` to submit both. Each task uses the same
`run` implementation and fingerprinted output layout as local execution.

`execution.slurm` accepts `cpus`, `memory`, `time`, optional `partition` and
`gres` (e.g. `gpu:1` for the deep methods), and a `setup:` list of shell lines
(module loads, virtualenv activation) emitted before each worker command. Set
`execution.resume_run_id` to reuse a previous run's per-fold cache after a partial
failure. See [methods.md](methods.md#slurm-execution) for the full option list and
worker/aggregate workflow. Legacy schema-v1 search/extraction arrays are unchanged.
