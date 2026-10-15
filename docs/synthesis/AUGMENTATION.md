# TVAE data augmentation for prognostic models

`pode synthesis augment` asks one question: **does adding TVAE-generated
patients to a prognostic model's training data make it better on real patients
it has never seen?** And if so, at what dose, at what cost, and without what
side effects?

It follows the safeguards in the data-augmentation notes this feature was built
from:
- split first, synthesise second;
- fit the generator inside each resample;
- evaluate only on untouched real patients;
- audit fidelity, calibration, subgroups and privacy;
- report against cheap baselines.

```
pode synthesis augment benchmark   # the evaluation harness (this page)
pode synthesis augment generate    # write one augmented training table
pode synthesis augment report      # recompute summaries from saved raw tables, re-render report.md / plots
```

The feature is additive. Its own YAML (`--augment-config`) sits next to the usual
PODE config (`--config`, which supplies the TVAE architecture). No existing
command or default changes.

Expect a sober answer: on small real cohorts, augmentation often gives no
reliable gain, and some strategies (e.g. forcing rare outcome groups) clearly
hurt. The benchmark exists so that question is asked properly, with paired
confidence intervals, before any synthetic rows reach a model.

## Applying it to a new study

Everything study-specific lives in the YAML; no code changes are needed.

1. **One table, one row per patient**, holding:
   - a survival time + event;
   - candidate features: clinical columns and/or an omics matrix (e.g. `cg*` betas);
   - columns used only for splitting, strata, subgroups or the external flag
     (e.g. `Cohort`), plus an id.

   An external cohort can sit in the same table (`external: {column, values}`)
   or come as a separate file (`--external other.csv`).
2. **Copy the template.**
   [`configs/augmentation/benchmark.example.yaml`](../../configs/augmentation/benchmark.example.yaml)
   lists every option, commented.
   - **Essentials:** `endpoint`, `generator.exclude_columns` (ids, cohort labels,
     post-outcome or treatment-response variables), `features`, `strategies`,
     `models`.
   - **Worked example:** `toy.example.yaml` (runs in about 30 s on
     `examples/cohort.csv`).
3. **Choose a TVAE architecture** with `--config`: a PODE config, ideally one
   tuned and promoted for this table (`pode synthesis tune` / `promote`). Omit it
   for defaults.
4. **Dry run.** `--dry-run` prints the resolved endpoint, row counts, splits,
   generator columns, per-model feature counts and the workload, then exits
   without fitting anything. Fix any surprises, run once with `--quick`, then
   run in full.
5. **Read** `report.md` and `scorecard.csv`. Deploy only a configuration
   that earned `RECOMMENDED`, using `generate` (see
   [Using a result](#using-a-result)).

### Command-line reference

```
pode synthesis augment benchmark --data TABLE [--config PODE_CFG] --augment-config AUG.yaml --output-dir DIR
    [--external EXTERNAL.csv]            # external cohort as a separate table (test only)
    [--generator-model MODEL.pkl]        # plain TVAE, external cohort only (internal splits always refit)
    [--strategies a,b] [--models a,b]    # run a subset of the configured labels
    [--ratios 0.25,0.5] [--draws N] [--folds K] [--repeats R]
    [--n-jobs 4] [--device auto|cpu|cuda] [--seed S] [--quick] [--dry-run]

pode synthesis augment generate --data TABLE [--config PODE_CFG] --augment-config AUG.yaml
    --strategy LABEL --ratio R [--benchmark-dir DIR] --output-dir OUT

pode synthesis augment report --output-dir DIR
```

### Long runs

- **Launch detached:** `cd ~ && setsid nohup <command> > run.log 2>&1 < /dev/null &`.
  This keeps the run alive after the session that started it ends.
- **Use a separate checkout.** A run imports code from the checkout it was
  started in, so switching branches in that checkout mid-run can break it.
  For runs over an hour, use a git worktree:
  - `git worktree add --detach ../pode_worktrees/run HEAD`
  - then run with `PYTHONPATH=../pode_worktrees/run`.
- **Re-running resumes.** Every finished split is checkpointed
  (`checkpoints/`), so re-running the same command after a crash or interruption
  picks up where it stopped. Per-split generators are cached too
  (`generators/`).

## Quick start (toy data)

```bash
pode synthesis augment benchmark --data examples/cohort.csv \
    --augment-config configs/augmentation/toy.example.yaml \
    --output-dir runs/augment_toy --quick
```

For real studies, runs take hours; launch detached (see [Long runs](#long-runs)).

Read `report.md` first. The `scorecard` table gives a verdict for every
model × strategy × ratio.

## What happens

```
data ──► drop rows missing endpoint/strata ──► external cohort set aside (never fitted on)
          │
          └► development rows ──► repeated stratified K-fold (default 5 × 5)
                 for each split:
                   1. TVAE fitted on the split's REAL training rows only (cached per split)
                   2. one pool per strategy  ──► fidelity + privacy checks vs held-out patients
                   3. per model: real-only fit, learning-curve fits (50/75% of real rows),
                      and every (strategy, ratio, draw) augmented fit
                   4. every fit scored on the split's REAL held-out patients
          └► once more with all development rows as training and the external cohort as test
```

Leakage controls, all covered by tests:
- **Generators never see test or external patients.** Each internal split fits its own generator. The test suite records the patient ids every generator is trained on and checks this.
- **A pre-trained model is used only for the external cohort.** You can pass one with `--generator-model`, but a model trained on all development rows has seen every internal test patient, so internal splits never use it.
- **Encoding comes from real rows only.** Imputation medians, one-hot levels, standardisation and the MetaCPG median/IQR scaling are all estimated on the **real** training rows. Synthetic rows can change what a model learns, but not how real patients are encoded.
- **Synthetic rows are tagged.** Every row carries a `_source` tag, and synthetic rows never reach a test set.
- **Synthetic patients get their own baseline stratum** (`synthetic_strata: own`). They then never share a cohort's baseline hazard in stratified Cox models.
- **Split membership is recorded.** With `id_column` set, it is written to `splits.csv`.

## Strategies

Sizes are **ratios of the split's real training rows**. For example, 0.5 means
"add half as many rows as there are real patients". Each split samples one pool
per strategy, and every draw subsamples it.

| kind | what it adds |
|---|---|
| `tvae_natural` | unconditional TVAE samples (the generator's own case mix) |
| `tvae_targeted` | TVAE samples meeting `target`, e.g. `{equals: {Metastases: "Yes"}}`, optionally with `ranges` (e.g. `OS_Time ≥ 24`). The target uses the same grammar as `generate --conditions-config`, including multi-group `{groups: [...]}`. |
| `tvae_subgroup` | TVAE samples spread over the levels of `params.columns`, in proportion to each level's shortfall against the largest level (or `params.target_count`) |
| `duplicate` | target rows resampled with replacement (random oversampling) |
| `jitter` | duplicates with Gaussian noise (`params.sd_frac` × training SD) on continuous non-outcome features. Outcomes and categories are copied, so this is the survival-safe SMOTE analogue. |
| `weight` | no new rows: target rows get case weights worth the same count. Only `cox_ph` and `logistic_landmark` accept weights. |

Under-generation of rare conditional targets is recorded as a `shortfall` (see
the pools table in the report). Rows that break `generator.constraints` are
dropped and counted. The survival time must always be > 0. Other rules you can
add are ranges, `ordered: [[PFS_Time, OS_Time]]`, and `forbid` combinations.

## Models

| kind | model |
|---|---|
| `metacpg_fixed` | MetaCPG's legacy score on a fixed CpG list: per-CpG stratified univariable Cox weights refitted on real + synthetic rows, then a weighted mean of M-values, centred and scaled by the real rows' median/IQR. **Real-only on the full development cohort, this reproduces a frozen `pode signatures` `model.json` exactly** (weights, scaling and all risk scores to ~1e-16). |
| `metacpg_search` | MetaCPG's own CpG **selection** inside every split, on real + synthetic rows. See [MetaCPG-native augmentation](#metacpg-native-augmentation-hybrid-backend--metacpg_search). |
| `coxnet` | elastic-net Cox (scikit-survival). The penalty is chosen by inner K-fold C-index, scored on real rows only. |
| `cox_ph` | ridge Cox (lifelines), with strata and case weights |
| `logistic_landmark` | L2 logistic regression for survival past the horizon |

High-dimensional inputs: `coxnet`, `cox_ph`, `logistic_landmark` and
`metacpg_fixed` take these optional `params`:
- `m_value_columns: ["cg*"]` converts methylation betas to M-values.
- `top_variance: 2000` (with `top_variance_columns: ["cg*"]`) keeps the 2,000
  most variable of those columns, chosen on **each fit's real training rows**.
  This unsupervised filter therefore never sees held-out or synthetic patients.
  Other numeric columns, such as Age, are always kept.

Absolute survival probabilities come from a Breslow fit of each model's score
on its own training rows (real + synthetic). So an augmentation that distorts the
case mix shows up as miscalibration.

## Metrics, with reference points

All metrics are on real held-out patients only.

| metric | reading it |
|---|---|
| `c_harrell`, `c_uno` | Probability that the model orders two patients correctly. 0.5 = coin toss; 0.7 = a useful prognostic score. A **ΔC of +0.01–0.02 is small but real**. |
| `auc_h` | Like C, but only "dies before the horizon" vs "survives past it". |
| `brier_h`, `ibs` | Squared error of predicted survival at the horizon, and integrated up to it. Lower is better. |
| `ipa_h` | Brier skill, an R²-like scale: 0 = no better than the Kaplan–Meier average; 0.1–0.2 is good for clinical data. |
| `cal_slope` | Ideal 1. Below 1 means predictions are too extreme (overfitting); above 1 means too timid. `cal_slope_err` = \|slope − 1\|. |
| `oe_ratio` | Observed / expected events by the horizon. Ideal 1. |
| `cal_abs_err` | Mean \|predicted − observed\| risk over risk quintiles. |
| `nb_*`, `nb_mean` | Decision-curve net benefit at the `net_benefit_thresholds`. |
| `auprc_minority` | Precision–recall AUC for the rarer landmark class (death or survival). |
| subgroup C / O/E | Per level of each `subgroups` column, when it has at least `min_subgroup_n` patients and `min_subgroup_events` events. |

**Paired comparisons.** Each augmented arm is compared with real-only *within
the same split*. The confidence intervals use the Nadeau–Bengio correction,
because repeated-CV training sets overlap and naive CIs are far too narrow. On
the external cohort, the comparison is a paired bootstrap of the ensemble
(mean over draws) against real-only.

## Efficiency, four ways

1. **Real-data equivalent** (`efficiency.csv`, `learning_curve_slopes.csv`).
   The real-only learning curve (50/75/100% of real training rows) is
   summarised as `C ≈ a + b·log(n)`. The slope `b` is estimated *within*
   splits, so differences in difficulty between splits do not pose as a size
   effect, and it comes with a standard error.

   An arm that beats real-only by ΔC is then worth `n_equiv = n·exp(ΔC/b)` real
   patients, and

   `efficiency = (n_equiv − n) / n_synthetic`

   This is the number of real patients each synthetic row is worth: 0.1 means
   ten synthetic rows buy one more real patient, and a negative value means the
   rows cost performance.

   The conversion divides by `b`, so it is only defined when the curve rises
   distinguishably from flat (one-sided 95%). Its CI combines the ΔC CI with the
   slope's uncertainty. On a shallow curve a tiny ΔC turns into hundreds of
   "real patients", so read the CI, not the point estimate. The report says
   whether each CI shows a real gain, a real loss or no detectable effect.
   `learning_curve_slopes.csv` shows what doubling the real data is worth for
   each model; it is the yardstick for every equivalent.
2. **Gated scorecard** (`scorecard.csv`). Gates, all configurable under `gates:`:
   - **utility:** improves (CI above 0), non-inferior (CI above −margin), worse (whole CI below 0), or inconclusive;
   - **calibration:** the calibration slope moves at most 0.1 further from 1, and
     ΔIBS ≤ 0.005. Improvements never fail; an overfit model whose slope rises
     towards 1 passes;
   - **subgroups:** no subgroup loses more than 0.05 C;
   - **fidelity** (TVAE only): discriminator AUC ≤ 0.8 and target-stratum TVD ≤ 0.15;
   - **privacy** (TVAE only): no exact copies, membership AUC ≤ 0.6, DCR p5 ratio ≥ 0.8.

   Verdicts: `UNSAFE` (any privacy failure), `RECOMMENDED` (improves and every
   other gate passes), `NON-INFERIOR`, `NOT RECOMMENDED`. Internal and external
   verdicts sit side by side.
3. **Dose-response** (`dose_response.csv`). The best ratio per strategy, and
   whether the gain is still rising, has plateaued, or reverses beyond the optimum.
4. **Compute cost** (`compute_cost.csv`, `timings.csv`). Wall time per fit,
   generator fit and pool sampling, relative to real-only, plus peak GPU memory.

## Fidelity and privacy

Fidelity compares each TVAE pool with the real training rows of its split:
- marginal TVD and KS;
- correlation gap;
- real-vs-synthetic discriminator AUC (0.5 = indistinguishable);
- KM difference at the horizon, log-rank test, and event-time KS (does survival time stay coupled to events?);
- for targeted pools, the same checks within the target stratum.

Privacy is computed on **internal splits only**. It uses each split's held-out
patients as the never-seen reference, because they come from the same
population as the generator's training rows. An external cohort is not a valid
reference: it differs systematically from the development cohorts, which would
make synthetic rows look spuriously close and inflate membership AUC.

The privacy checks are:
- **DCR:** the Gower distance from each synthetic row to its closest training patient, compared with the same distance for genuinely new patients;
- exact and near copies;
- **NNDR:** the nearest-to-second-nearest distance ratio;
- a distance-based **membership-inference AUC** (0.5 = no signal);
- **rare-combination leaks:** synthetic rows that reproduce a categorical combination seen in at most `rare_k` training patients.

Duplicate, jitter and weight are copies of real patients by design. They are not
privacy-gated, and like every training table here they must never leave the
pipeline.

## Outputs

| file | content |
|---|---|
| `report.md`, `report.json` | design, scorecard, efficiency in words, dose-response, fidelity, privacy, cost, external results, caveats |
| `scorecard.csv` | verdict and reasons per model × strategy × ratio |
| `paired_differences.csv` | every metric: real-only, augmented, Δ, corrected CI, P(better) |
| `results_by_fold.csv` | every single fit and its metrics (the raw data behind everything) |
| `external_results.csv`, `external_paired.csv` | external cohort: per-arm metrics and paired bootstrap ΔC |
| `efficiency.csv`, `learning_curve.csv`, `learning_curve_slopes.csv`, `dose_response.csv`, `compute_cost.csv`, `timings.csv` | efficiency views |
| `fidelity.csv`, `privacy.csv`, `pools.csv` | per split and strategy |
| `subgroups.csv` | per-fit subgroup metrics |
| `splits.csv` | patient ids in each split's train/test (with `id_column`) |
| `generators/<split>/` | cached per-split TVAEs (`model.pkl`, `meta.json` fingerprinted on the exact training rows) |
| `plots/` | dose-response, learning curve with equivalents, gain vs calibration, fidelity/privacy, subgroups |
| `augmentation_config_used.yaml`, `run_manifest.json` | config, git commit, package versions, seeds |

## Using a result

```bash
pode synthesis augment generate --data TABLE --config PODE_CFG --augment-config AUG.yaml \
    --strategy tvae_natural --ratio 0.5 --benchmark-dir BENCH_DIR --output-dir OUT
```

This fits one generator on all development rows (external rows are left out)
and writes `augmented_training_table.csv`. The output carries:
- `_source`, marking real vs synthetic rows;
- `_stratum`, when `strata` are set;
- `_weight`, for the weight strategy.

With `--benchmark-dir`, the command prints the benchmarked verdict for that
configuration and warns unless it is `RECOMMENDED`.

## Caveats

- **Optimistic internal estimates with a fixed signature.** If the signature's CpGs were chosen on all development patients, internal estimates are optimistic for every arm alike. Paired comparisons stay fair; absolute values do not. The external cohort is unaffected. `metacpg_search` (below) reselects inside every split and has no such bias.
- **No censoring model in the TVAE.** Survival time and event are ordinary TVAE columns. Check `km_diff_h` and `event_time_ks` before trusting survival-targeted strategies.
- **Synthetic rows are training auxiliaries only.** They give no evidence about incidence, prevalence or treatment effects, and they cannot repair confounding or validate transportability. Only real external patients can do that.

## MetaCPG-native augmentation (hybrid backend + `metacpg_search`)

The table-level benchmark refits models on a fixed feature list, so it cannot
change *which* CpGs a signature uses. The hybrid backend makes augmentation
reach MetaCPG's CpG selection:

```
per split:  real CpG betas ────────► survival-VAE compressor (fitted on the split's real rows)
                                            │ latents (8)
            clinical + latents ──► TVAE ──► synthetic clinical + latents
                                            │ decoder + residual noise
                                            ▼
            synthetic patients with every CpG ──────► MetaCPG ranking on real + synthetic
                                                        ──► top-n CpGs ──► score real held-out patients
```

Set it up in the augmentation config, and pass a tuned hybrid TVAE as `--config`:

```yaml
generator:
  backend: hybrid
  hybrid:
    matrix_columns: ["cg*"]          # never given to the TVAE
    matrix_kind: methylation          # betas; logit-transformed for the compressor
    compressor_config: path/to/tuned_compressor/selected_config.json   # its best_config
    residual_noise: true              # see below
models:
  - {kind: metacpg_search, label: metacpg_search_35, features: ["cg*"], params: {n: 35}}
reference_signature: path/to/signature/python/model.json   # optional
```

How the hybrid backend works:
- **Leakage controls.** The compressor and the TVAE are both fitted inside each
  split on its real training rows only, and cached per split, keyed on the
  exact rows.
- **Residual noise.** Decoded matrices are decoder means, much smoother than
  real methylation. With `residual_noise`, each CpG gets Gaussian noise with the
  training reconstruction residual SD (on the logit scale).
  `matrix_sd_ratio` in the fidelity table shows how much spread survives.

How `metacpg_search` scores:
- **Ranking.** Every CpG gets MetaCPG's univariable Cox fit, stratified by
  `strata`, with synthetic patients in their own stratum. The top `n` by |z|
  form the score, weighted by their standardised coefficients and
  centred/scaled on real rows only.
- **Parity.** Real-only, it selects exactly the CpGs, weights and scaling that
  `pode signatures` extracts at the same `n` and strata.
- **Cost.** One fit (~7,000 CpGs, a few hundred patients) takes ~0.3 s on the CPU.

Extra outputs:
- **`selection_stability.csv`.** Three overlaps per arm:
  - `jaccard_vs_real_only`: overlap with the same split's real-only selection;
  - `across_split_jaccard`: how stable the selection is to resampling;
  - `reference_overlap`: the share of the reference signature that is recovered.
- **`selection_frequency.csv`.** How often each CpG is selected, per arm.
- **Matrix fidelity.** Discriminator AUC, SD ratio, and mean and correlation
  gaps on the 500 highest-variance CpGs.

Privacy is measured on clinical + latents. Per-CpG distances on thousands of noisy
features say little about re-identification.
