# Signature methods & configuration reference (schema v2)

This is the detailed companion to [advanced.md](advanced.md). It documents every
registered method, its tunable parameters and defaults, how clinical covariates
and multiple training cohorts are handled, how to run a parameter grid search, and
every SLURM option. A fully commented configuration lives at
[../../configs/signatures/example.advanced.yml](../../configs/signatures/example.advanced.yml).

All advanced methods are **canonical PODE adaptations**: they implement the named
statistical objective or architecture but do not claim exact numerical parity with
the original authors' software. [References and provenance](#references-and-provenance)
gives the original citation, the canonical software, and a pointer to the exact PODE
class for every method so the adaptation can be read and compared against the source.

- [Method catalogue](#method-catalogue)
- [References and provenance](#references-and-provenance)
- [Per-method parameters](#per-method-parameters)
- [Clinical covariates](#clinical-covariates)
- [Cohorts and multiple training cohorts](#cohorts-and-multiple-training-cohorts)
- [Grid search (`search_space`)](#grid-search-search_space)
- [SLURM execution](#slurm-execution)

---

## Method catalogue

`pode signatures methods` prints this table as JSON, including whether each
method's optional dependency is installed on your machine.

| CLI name | Family | Maturity | Required inputs | What it is |
|---|---|---|---|---|
| `elastic_net_cox` | penalised | stable | — | Sparse Cox partial-likelihood with L1+L2 penalties on omics coefficients. |
| `coxboost` | boosting | stable | — | Component-wise Cox boosting; one omics coefficient updated per step, covariates updated jointly. |
| `stability_coxboost` | stability | stable | — | Complementary half-sample selection frequencies around CoxBoost, then an unpenalised refit of stable features. |
| `pls_cox_multiblock` | latent | stable | — | Event-weighted multiblock PLS-Cox: PLS components of the omics blocks feed a Cox model. |
| `random_survival_forest` | ensemble | stable | scikit-survival | Random survival forest (permutation/impurity importance). |
| `gradient_boosted_survival` | ensemble | stable | scikit-survival | Gradient-boosted survival trees. |
| `survival_svm` | kernel | stable | scikit-survival | Fast (ranking) survival SVM. |
| `deepsurv` | deep | stable | — | Cox-loss MLP over the concatenated blocks. |
| `autoencoder_cox` | deep | stable | — | Autoencoder with a joint reconstruction + Cox loss. |
| `net_cox` | penalised | experimental | `graph_path` | Sparse Cox with a graph-Laplacian penalty coupling connected features. |
| `cox_nnet` | deep | experimental | `pathway_path` | Pathway-masked Cox neural network (first layer masked to pathway membership). |
| `surv_transformer` | deep | experimental | — | Block-token transformer fusion (SurvTRACE-style). |
| `prototype_contrastive_survival` | deep | experimental | — | Block fusion with learned prototypes (PCLSurv-style). |
| `moli_survival` | deep | experimental | — | Modality-specific late integration (MOLI-style). |
| `hierarchical_cox` | multi-study | experimental | ≥2 cohorts | Shared effect plus penalised per-cohort deviation coefficients. |
| `maml_cox` | meta-learning | experimental | ≥2 cohorts | First-order MAML/Reptile meta-updates across cohort tasks. |
| `frailty_cox` | multi-study | experimental | ≥2 cohorts | Cox with penalised per-cohort random-effect (shared-frailty) terms. |

`stable` methods run under every benchmark preset; `experimental` methods are
excluded from `--preset quick` but included in `balanced` and `exhaustive`. They
are fully operational, not placeholders — their manifests carry an adaptation
notice because external benchmark parity has not been established.

---

## References and provenance

Every advanced method is an **independent reimplementation** in this repository, not
a wrapper around the original authors' code (the three scikit-survival methods are
the exception — they call scikit-survival directly). The table below names the
method each PODE implementation is based on, the canonical software you would check
against, and the PODE source file/class that implements the adaptation. Read the
PODE class alongside the original to see exactly what is and is not reproduced.

> **How to use this:** the citations point to the method PODE follows in spirit.
> They are provided for traceability, not to claim equivalence — verify each
> reference against its source, and expect the PODE version (especially the deep
> methods) to be a simplified, deterministic adaptation rather than a line-for-line
> port. Where the PODE design deliberately departs from the source, it is noted
> under [Adaptation notes](#adaptation-notes).

| Method | Based on | Canonical software | PODE implementation |
|---|---|---|---|
| `elastic_net_cox` | Simon, Friedman, Hastie & Tibshirani (2011), *J. Stat. Softw.* 39(5), regularization paths for Cox | R `glmnet` | `LinearCoxModel` in `metacpg/advanced/models.py` |
| `coxboost` | Binder & Schumacher (2008), *BMC Bioinformatics* 9:14, boosting with mandatory covariates | R `CoxBoost` | `CoxBoostModel` in `metacpg/advanced/models.py` |
| `stability_coxboost` | Meinshausen & Bühlmann (2010), *JRSS-B* 72(4), stability selection; complementary pairs: Shah & Samworth (2013), *JRSS-B* 75(1) | R `stabs` (concept) | `StabilityBoostModel` in `metacpg/advanced/models.py` |
| `pls_cox_multiblock` | Bastien (2008), *Chemom. Intell. Lab. Syst.* 91, deviance-residual PLS-Cox; multiblock integration: Rohart et al. (2017) mixOmics | R `plsRcox`, `mixOmics` | `PLSCoxModel` in `metacpg/advanced/models.py` |
| `random_survival_forest` | Ishwaran, Kogalur, Blackstone & Lauer (2008), *Ann. Appl. Stat.* 2(3) | `scikit-survival` `RandomSurvivalForest` (called directly) | `SksurvModel` in `metacpg/advanced/models.py` |
| `gradient_boosted_survival` | Friedman (2001) gradient boosting + Cox loss; Hothorn et al. (2006), *Biostatistics* 7(3), survival ensembles | `scikit-survival` `GradientBoostingSurvivalAnalysis` (called directly) | `SksurvModel` |
| `survival_svm` | Pölsterl, Navab & Katouzian (2015/2016), fast survival SVM; Van Belle et al. (2011), *Artif. Intell. Med.* 53 | `scikit-survival` `FastSurvivalSVM` (called directly) | `SksurvModel` |
| `deepsurv` | Katzman et al. (2018), *BMC Med. Res. Methodol.* 18:24 | `DeepSurv`, `pycox` | `DeepCoxModel` (MLP head) in `metacpg/advanced/models.py` |
| `autoencoder_cox` | Chaudhary et al. (2018), *Clin. Cancer Res.* 24(6), multi-omics autoencoder + Cox | (paper code) | `DeepCoxModel` (autoencoder head) |
| `net_cox` | Zhang et al. (2013), *PLoS Comput. Biol.* 9(3), network-based (graph-Laplacian) Cox | Net-Cox (MATLAB) | `LinearCoxModel` with `graph_edges` in `metacpg/advanced/models.py` |
| `cox_nnet` | Ching, Zhu & Garmire (2018), *PLoS Comput. Biol.* 14(4); pathway masking follows Cox-PASNet, Hao et al. (2018), *BMC Med. Genomics* 11(Suppl 3) | `cox-nnet`, `Cox-PASNet` | `DeepCoxModel` + `_apply_pathway_mask` |
| `surv_transformer` | Wang & Sun (2022), SurvTRACE, *ACM-BCB* | `SurvTRACE` | `DeepCoxModel` (`BlockNetwork`, transformer fusion) |
| `prototype_contrastive_survival` | Prototypical + contrastive survival learning, "PCLSurv" family — **verify citation** (see notes) | — | `DeepCoxModel` (`BlockNetwork` + prototypes) |
| `moli_survival` | Sharifi-Noghabi et al. (2019), *Bioinformatics* 35(14), MOLI late integration (originally drug response) | `MOLI` | `DeepCoxModel` (`BlockNetwork`, late fusion) |
| `hierarchical_cox` | Shared-effect + per-study deviation penalty; cf. Gross & Tibshirani (2016), *JCGS* 25(4), data-shared lasso; multi-study learning | — | `HierarchicalCoxModel` in `metacpg/advanced/models.py` |
| `maml_cox` | Finn, Abbeel & Levine (2017), MAML, *ICML*; first-order variant Nichol, Achiam & Schulman (2018), Reptile | (paper code) | `DeepCoxModel` with `cohort_aware` meta-loop |
| `frailty_cox` | Shared-frailty Cox; Therneau & Grambsch (2000), *Modeling Survival Data*; Rondeau et al. (2012), *J. Stat. Softw.* 47(4) | R `survival` (`frailty`), `frailtypack` | `FrailtyCoxModel` in `metacpg/advanced/models.py` |

The factory that maps each CLI name to its class and default parameters is
`metacpg/advanced/registry.py`; the Cox partial-likelihood objective shared by the
linear methods is `cox_loss_gradient` / `cox_loss_torch` in
`metacpg/advanced/base.py`.

### Adaptation notes

Where the PODE version is known to diverge from the source, so you know what *not*
to expect parity on:

- **Deep methods** (`deepsurv`, `autoencoder_cox`, `cox_nnet`, `surv_transformer`,
  `prototype_contrastive_survival`, `moli_survival`, `maml_cox`) share one small,
  fixed backbone (two hidden layers, dropout 0.1, Adam weight-decay 1e-4, seeded)
  rather than each paper's full architecture, and derive feature importance from
  mean absolute input gradients. They reproduce the *training objective* (Cox loss,
  plus reconstruction / prototype / late-fusion terms) of the named method, not its
  exact network or hyper-parameters.
- **`pls_cox_multiblock`** uses an event-weighted negative-log-time PLS target
  (documented in the code) rather than Bastien's iterative deviance-residual
  reweighting; blocks are concatenated, not modelled with a mixOmics-style block
  covariance.
- **`moli_survival`** adapts MOLI from its original drug-response classification
  objective to a Cox survival objective.
- **`net_cox`** applies a quadratic graph-Laplacian penalty over the supplied edge
  list; it is not the exact Net-Cox optimiser.
- **`hierarchical_cox`** / **`frailty_cox`** implement the shared-effect-plus-
  per-cohort structure as ℓ1/ℓ2-penalised augmented Cox designs, not REML/EM
  frailty estimation.
- **`prototype_contrastive_survival`** is the one entry whose exact upstream
  citation is uncertain in this documentation: it implements a prototypical +
  contrastive survival objective in the spirit of the "PCLSurv" line of work.
  Treat it as a PODE-defined method and confirm the intended reference before
  citing it in a manuscript. If you have a specific source paper in mind, tell us
  and we will pin the citation.

All persisted method manifests also record
`"adaptation_notice": "Canonical PODE adaptation; not an exact upstream research
implementation."` for provenance in downstream analyses.

---

## Per-method parameters

Set these under `methods.config.<name>`. **Any key you omit uses the default
shown.** Turn a scalar into a tuned grid by moving it under a `search_space:`
block (see [Grid search](#grid-search-search_space)).

### `elastic_net_cox`
| Key | Meaning | Default |
|---|---|---|
| `l1` | L1 (lasso / sparsity) penalty on omics coefficients | `0.002` |
| `l2` | L2 (ridge) penalty on omics coefficients | `0.01` |
| `max_iter` | Max proximal-gradient iterations | `500` |
| `learning_rate` | Initial step size (backtracking line search) | `0.2` |

### `coxboost`
| Key | Meaning | Default |
|---|---|---|
| `steps` | Number of boosting iterations | `100` |
| `learning_rate` | Shrinkage applied per boosting step | `0.05` |

### `stability_coxboost`
| Key | Meaning | Default |
|---|---|---|
| `repetitions` | Complementary half-sample splits | `20` |
| `threshold` | Selection-frequency cutoff for keeping a feature | `0.6` |
| `steps` | CoxBoost steps per subsample | `60` |
| `learning_rate` | CoxBoost shrinkage per step | `0.05` |

### `pls_cox_multiblock`
| Key | Meaning | Default |
|---|---|---|
| `components` | Number of PLS latent components (clipped to `n-1` and the feature count) | `3` |

### `net_cox`
| Key | Meaning | Default |
|---|---|---|
| `graph_path` | **Required.** CSV with `source,target[,weight]` edges over omics features | — |
| `l1` | L1 penalty on omics coefficients | `0.001` |
| `l2` | L2 penalty on omics coefficients | `0.01` |
| `graph_strength` | Weight of the graph-Laplacian smoothing penalty | `0.1` |
| `max_iter` | Max proximal-gradient iterations | `500` |

Feature names in the graph match the block-prefixed omics names (e.g.
`expression::GENE1`); a bare feature suffix is accepted when it is unique.

### scikit-survival methods (`random_survival_forest`, `gradient_boosted_survival`, `survival_svm`)
These wrap scikit-survival estimators. Estimator arguments go under a **nested
`params:`** mapping (or `params.<arg>` inside `search_space`). Any argument the
underlying estimator accepts is forwarded.

| Method | Estimator | Notable defaults |
|---|---|---|
| `random_survival_forest` | `RandomSurvivalForest` | `n_estimators=100`, `min_samples_leaf=3`, `n_jobs=1` |
| `gradient_boosted_survival` | `GradientBoostingSurvivalAnalysis` | `n_estimators=100`, `learning_rate=0.05`, `max_depth=2` |
| `survival_svm` | `FastSurvivalSVM` | `alpha=1.0`, `rank_ratio=1.0`, `max_iter=100` |

```yaml
random_survival_forest:
  params:
    n_estimators: 300
    max_features: sqrt
```

### Deep methods (`deepsurv`, `autoencoder_cox`, `surv_transformer`, `prototype_contrastive_survival`, `moli_survival`, `maml_cox`, `cox_nnet`)
All share the same three knobs. Dropout (`0.1`) and Adam weight decay (`1e-4`) are
fixed and not configurable.

| Key | Meaning | Default |
|---|---|---|
| `epochs` | Training epochs | `80` |
| `hidden` | Hidden width | `32` |
| `learning_rate` | Adam learning rate | `0.001` |

`cox_nnet` additionally requires `pathway_path` — a CSV with `feature_index,
pathway_index` columns that masks the first layer to pathway membership.

### `hierarchical_cox`
| Key | Meaning | Default |
|---|---|---|
| `l1` | L1 penalty on the shared omics effect | `0.001` |
| `deviation_penalty` | Shrinkage of per-cohort deviation coefficients toward the shared effect | `0.1` |

### `frailty_cox`
| Key | Meaning | Default |
|---|---|---|
| `frailty_penalty` | Penalty on the per-cohort random-effect (frailty) indicator terms | `1.0` |

---

## Clinical covariates

Declare covariates once under the top-level `covariates:` list; they apply to
**every** enabled method:

```yaml
covariates:
  - column: age
    type: numeric        # median-imputed, used as-is
    penalized: false
  - column: subgroup
    type: categorical    # one-hot encoded; first level is the reference
    penalized: false
```

- `penalized: false` is **mandatory** — the validator rejects penalised clinical
  terms.
- **Linear / boosting / latent / multi-study family** (`elastic_net_cox`,
  `coxboost`, `stability_coxboost`, `pls_cox_multiblock`, `net_cox`,
  `hierarchical_cox`, `frailty_cox`): covariates are **mandatory, unpenalised
  adjustment terms**. They are always in the model, never selected out, and never
  count toward the omics `feature_counts`.
- **Deep family** and **scikit-survival family**: covariates enter as ordinary
  input features (importance is still reported for omics only).
- The portable reduced signature keeps the covariates (and technical block-presence
  masks) but reports the incremental C-index over a covariates-only baseline, so
  you can see what the omics features add on top of the clinical model.

---

## Cohorts and multiple training cohorts

`cohort_column` names the **source** each training sample came from — different
trials, recruitment eras, or processing batches:

```yaml
phenotype:
  sample_id_column: sample_id
  cohort_column: cohort
```

It says *"these samples are pooled for training, but the sources may have
different underlying survival rates."* It is **not** a train/test split. All
cohorts train together, and cohort structure is accounted for while fitting.

This is the schema-v2 counterpart of what the schema-v1 metaCpG method does with
`endpoints[].group_column`, which fits Cox estimates per cohort and pools them
with an empirical-Bayes random-effects estimator. Here the equivalent is achieved
inside a single joint fit rather than by pooling separate fits.

### How each method accounts for cohorts

| `cohort_handling` | Methods | What happens |
|---|---|---|
| `stratified` | `elastic_net_cox`, `coxboost`, `stability_coxboost`, `pls_cox_multiblock`, `net_cox`, `deepsurv`, `autoencoder_cox`, `cox_nnet`, `surv_transformer`, `prototype_contrastive_survival`, `moli_survival` | Cohort becomes an extra stratum in the Cox partial likelihood: coefficients are shared across cohorts, but each cohort gets its own unspecified baseline hazard. Risk sets never cross cohorts, so a cohort's overall survival rate cannot bias the effect estimates. |
| `explicit` | `hierarchical_cox`, `frailty_cox`, `maml_cox` | The model represents cohorts directly — a shared effect plus penalised per-cohort deviations (`hierarchical_cox`, exported per cohort in `signature_coefficients.csv`), penalised per-cohort random-effect terms (`frailty_cox`), or first-order meta-learning across cohorts (`maml_cox`). These three **require** at least two cohorts. |
| `ignored` | `random_survival_forest`, `gradient_boosted_survival`, `survival_svm` | scikit-survival has no stratified Cox and no random effects, so these **cannot** honour `cohort_column`. Samples are pooled and the difference in baseline survival is not accounted for. |

`hierarchical_cox` also receives cohort strata, because its per-cohort deviations
act on the coefficients rather than the baseline. `frailty_cox` and `maml_cox` do
not: `frailty_cox` puts cohort indicators on the linear predictor, where a
stratified baseline would absorb them, and `maml_cox` already partitions its
inner loop by cohort.

### The methods that cannot comply

The three scikit-survival methods are flagged rather than silently pooled. Every
result they produce carries `cohort_handling: ignored` and a `cohort_warning` —
in `leaderboard.csv`, in `result.json`, in each endpoint `manifest.json`, and on
the CLI while the run is in progress. To drop them instead:

```yaml
phenotype:
  cohort_column: cohort
  cohort_handling: skip     # default: warn
```

They are then recorded as `status: skipped` in the leaderboard.

### Validation

Outer folds are balanced on **cohort × event status**, so every cohort is
represented in every training fold and no cohort is accidentally held out. Both
`evaluation.outer_folds` and `evaluation.inner_folds` apply as usual. Balancing
degrades to event-only stratification, then to plain K-fold, when the smallest
`cohort × event` cell is too small to spread across the folds.

Cross-cohort external validation is available, but must be asked for:

```yaml
evaluation:
  cohort_validation: leave_one_cohort_out   # default: stratified_kfold
```

This trains on all but one cohort and tests on the held-out one, ignoring
`outer_folds`. It answers a different and much harsher question — how well the
signature *transfers* to an unseen cohort — and is not a measure of performance
on the pooled training population. Every method run this way carries a
`cohort_warning` saying so.

### Reported metrics

Alongside `c_index`, each fold reports `c_index_within_cohort`: the
event-weighted mean of the per-cohort concordance. Pooled Harrell's C is
contaminated when cohorts have different baseline rates, because samples from a
low-risk cohort outlive those from a high-risk one for reasons the signature does
not model. Compare within-cohort values when cohorts differ substantially.

`integrated_brier_score` uses an unstratified Breslow baseline and is therefore
an approximation when cohort strata are active.

Remove `cohort_column` to pool all samples with no cohort handling at all; the
three cohort-only methods are then unavailable.

---

## Grid search (`search_space`)

Each method's config holds a single parameter point by default, which makes a
benchmark leaderboard sensitive to how well you guessed. Add a `search_space:`
block to tune parameters instead of comparing hand-picked values:

```yaml
methods:
  config:
    elastic_net_cox:
      max_iter: 200            # fixed base value
      search_space:
        l1: [0.001, 0.005, 0.02]
        l2: [0.001, 0.01]
```

- Values are non-empty lists; the engine evaluates the **full Cartesian product**
  of the axes.
- For each outer fold, the best `(parameter combination × feature_count)` is
  chosen by **inner cross-validation** on Harrell's C, so tuning never sees the
  outer test data. The full-data refit uses the modal winning configuration.
- Keys not in `search_space` keep their base value (e.g. `max_iter: 200` above).
- **Nested parameters use dotted keys**, which is how you tune scikit-survival
  estimators:

  ```yaml
  random_survival_forest:
    search_space:
      params.n_estimators: [50, 100, 200]
      params.min_samples_leaf: [3, 5]
  ```

- Benchmark **presets truncate** each axis to keep runtime bounded:
  `--preset quick` keeps the first 2 values per axis (and stable methods only),
  `--preset balanced` the first 3, and `--preset exhaustive` uses the **full grid**
  with full epochs. Ordering the most promising values first lets `quick`/`balanced`
  still explore a sensible subset.

---

## SLURM execution

Resources and environment live entirely under `execution.slurm` so no cluster
paths are baked into code:

```yaml
execution:
  seed: 42
  # resume_run_id: my-run-2024-06   # reuse a previous run's per-fold cache
  slurm:
    cpus: 4                 # -> #SBATCH --cpus-per-task
    memory: 16G             # -> #SBATCH --mem
    time: "24:00:00"        # -> #SBATCH --time
    partition: gpu          # -> #SBATCH --partition (optional)
    gres: "gpu:1"           # -> #SBATCH --gres (optional; useful for deep methods)
    setup:                  # shell lines emitted before each worker command
      - module load python/3.12
      - source /path/to/.venv/bin/activate
```

Workflow:

```bash
# 1. Render the array + aggregate scripts without submitting, and inspect them.
pode signatures submit --config analysis.yml --dry-run

# 2. Submit. This queues an array job (one task per method x endpoint x outer fold)
#    plus a dependent aggregation job that finalises models once the array finishes.
pode signatures submit --config analysis.yml
```

`submit` writes `<output_dir>/slurm/tasks-<run_id>.csv`,
`advanced-<run_id>.sh` (the `--array` worker script), and `aggregate-<run_id>.sh`.
Each worker (`pode signatures worker`) computes one fold and writes a JSON cache
entry; the aggregate step (`pode signatures aggregate`) reuses that cache to refit
the full-data models — so the statistics are identical to a local `run`, only the
work is distributed.

Set `execution.resume_run_id` to reuse an earlier run's cache (skip already-computed
folds) when re-submitting after a partial failure.
