# synthvae — survival-aware omics synthesis

> **Part of PODE.** `synthvae` is the omics front end that lives alongside the
> `pode` package in this repo (`synthvae/` at the repo root). It was previously a
> standalone project named `synthvae-hybrid`; that name was a misnomer because it
> covers **both** the two-stage hybrid **and** the joint multimodal VAE. For the
> up-to-date, copy-paste pipelines that bridge synthvae with the PODE TVAE, see
> [`docs/synthesis/WORKFLOWS.md`](WORKFLOWS.md). Configs live in `configs/synthesis/`,
> scripts in `scripts/`. This document is the design + reference guide. For
> generative (multiple) imputation with the PODE TVAE, see
> [`docs/synthesis/IMPUTATION.md`](IMPUTATION.md); for TVAE data augmentation of
> prognostic models, see [`docs/synthesis/AUGMENTATION.md`](AUGMENTATION.md). To assess
> whether synthetic data preserves patient privacy (memorisation, membership, linkage and
> disclosure checks with a release gate), see [`docs/synthesis/PRIVACY.md`](PRIVACY.md).
> To train the TVAE (or MetaCPG signatures) across sites without pooling patient rows,
> see [`docs/federated/FEDERATED.md`](../federated/FEDERATED.md).

Survival-aware synthesis of joint **clinico-pathological + high-dimensional
molecular matrix** data, including expression and methylation beta-values.

Two approaches live here:

1. **The recommended hybrid** (`synthvae/`, runnable): a *survival-supervised
   matrix compressor* whose latents are handed to an SDV **TVAE** alongside the
   clinical columns, then decoded back to the original matrix. This keeps your existing
   TVAE workflow, adds a compressor that is explicitly pushed to retain prognostic
   signal, and ships with a TSTR / Kaplan–Meier / hazard-ratio validation harness.

2. **The joint multimodal VAE** (`synthvae/mmvae/` + `docs/SPECIFICATION.md`):
   a shared-latent PoE/MoE/MoPoE model with three decoder heads (tabular /
   expression / survival) and data-dependent gating that learns *one* latent for a
   patient across all modalities. This is now fully implemented and runnable
   (training loop, generation, gaussian + negative-binomial + zero-inflated-NB
   expression likelihoods, and fixed / uncertainty / GradNorm task weighting), with
   its own CLI (`scripts/run_mmvae.py`) and a benchmark harness
   (`scripts/benchmark.py`) that scores it head-to-head against the hybrid. Prefer
   the hybrid as the baseline; reach for the joint model when you need cross-modal
   or missing-modality generation (encode one modality, generate another). See
   `docs/ARCHITECTURE.md` §2 for the equivalence argument and when to escalate.

## Why this design

Reconstruction-optimal expression latents are the *worst* behaved for downstream
generation, and vanilla VAEs are documented to drift on survival (KM curves
diverge in mid-to-late follow-up; no C-index benefit). The fix is to (a) compress
expression under survival supervision so prognostic variance is retained, (b) let a
flexible density model (your TVAE) model the joint latent+clinical+survival table,
and (c) validate on survival-preserving metrics, not just marginal fidelity. See
`docs/ARCHITECTURE.md` for the full rationale and the two-stage/joint equivalence.

## Install

```bash
# core only (runs the torch/SDV-free demo + all evaluation)
pip install -e .

# production path
pip install -e ".[all]"     # torch + sdv + lifelines
```

Core deps: numpy, pandas, scikit-learn, pyyaml. Optional: torch (expression VAE),
sdv (TVAE synthesis), lifelines (preferred Cox; a numpy fallback is built in).

## Quickstart

Torch/SDV-free demo — runs anywhere, exercises the whole pipeline on a toy cohort
with genuine latent structure:

```bash
pode synthesis hybrid run --compressor pca --synthesizer gmm --n-latent 16
```

Production path:

```bash
pode synthesis hybrid \
    --compressor survival_vae --synthesizer sdv \
    --n-latent 32 --epochs 300 --n-top-genes 2000
```

Bring your own expression data (wide, one gene per column; survival with
`time,event`; `clinical.csv` is optional):

```bash
pode synthesis hybrid run --compressor survival_vae --synthesizer sdv \
    --clinical clinical.csv --expression expression.csv --survival survival.csv
```

Expression + survival only also works:

```bash
pode synthesis hybrid run --compressor survival_vae --synthesizer sdv \
    --expression expression.csv --survival survival.csv
```

Methylation beta-values use explicit methylation input, latent, and output names:

```bash
pode synthesis hybrid export-latents \
    --methylation methylation_beta.csv \
    --clinical clinical.csv \
    --survival survival.csv \
    --out-dir runs/methylation_handoff \
    --include-input
```

DNA copy number is segment-based, so first collapse the segments into an aligned
`samples x bins` matrix with
`synthvae.data.copy_number.segments_to_matrix` (fixed-width genomic bins,
length-weighted mean `Seg.CN`), then pass the matrix with `--copy-number`:

```python
from synthvae.data.copy_number import convert_csv
convert_csv("segments.csv", "cnv.csv", bin_width=10_000_000)  # representation="continuous"
```

```bash
pode synthesis hybrid export-latents \
    --copy-number cnv.csv \
    --survival survival.csv \
    --out-dir runs/cnv_handoff --include-input
```

Copy-number bundles use `cnv_latent_*` columns and write `synthetic_copy_number.csv`.
Add `--copy-number-representation discrete` to model ordinal `{-1,0,1}` gain/loss
calls instead of continuous log2 ratios (produced by
`segments_to_matrix(..., representation="discrete")`); decoded values snap back to
`{-1,0,1}` and outputs use the `copy_number_discrete` / `cnv_call_latent_` names.

## CSV contract

The pipeline expects `survival.csv`, plus exactly one high-dimensional matrix:
`--expression expression.csv`, `--methylation methylation.csv`, or
`--copy-number cnv.csv`. `clinical.csv`
is optional. Any files you provide must have the same row count and the same row
order. There is no join key in the current driver; row `i` across the provided
CSVs is treated as the same sample.

`clinical.csv`

* Optional.
* One row per sample.
* Columns are the clinical features you want the synthesizer to model.
* Numeric columns are treated as continuous by default.
* String, categorical, boolean, and low-cardinality integer columns are treated
  as categorical automatically in the synthesis metadata.
* Columns listed in `--categorical` are also forced to categorical, which is
  useful when you want to override dtype-based inference.
* Do not include expression/methylation columns, `time`, or `event` here.

Example:

```csv
age,stage,sex
63,2,F
71,3,M
```

`expression.csv`

* One row per sample, aligned to `clinical.csv` by row order.
* One gene per column.
* Values should already be in the numeric scale you want the compressor to learn
  from, for example log-expression, log-CPM, or another continuous
  approximately Gaussian scale.
* Gene column names are preserved and reused when decoded expression is written.

Example:

```csv
TP53,EGFR,MYC
5.21,2.14,3.78
4.98,1.87,4.10
```

`methylation.csv`

* One row per sample, aligned to `clinical.csv` by row order.
* One CpG/site/feature per column.
* Values must be beta-values on `[0, 1]`.
* Internally, beta-values are clipped by `--beta-epsilon` and modeled on
  logit/M-value scale; decoded outputs are written back as beta-values.
* Methylation bundles use `methylation_latent_*` columns and write
  `synthetic_methylation.csv`.

Example:

```csv
cg000001,cg000002,cg000003
0.12,0.81,0.44
0.09,0.76,0.51
```

`survival.csv`

* One row per sample, aligned to the other two CSVs by row order.
* Must contain `time` and `event`.
* `time` must be numeric and positive.
* `event` is expected as `0/1`; internally it is normalized to boolean for the
  synthesizer and back to integer on output.

Example:

```csv
time,event
14.2,1
25.7,0
```

Important practical notes:

* Missing values are not handled explicitly in the current driver. Impute or
  clean them before running the pipeline.
* Any row filtering or sample reordering must be applied consistently across all
  provided CSVs before you run the script.
* If you use `--n-top-genes`, only the selected high-variance matrix features are
  modeled directly by the compressor, but decoded outputs are still written with
  the full original column set.
* If `clinical.csv` is omitted, the pipeline runs on latents + survival only.
  That affects tuning and evaluation, but it does not break export or decode.

## Controlled latent handoff workflow

The hybrid path is deliberately two-stage:

```
expression -> SurvivalVAE/PCA compressor -> expr_latent_* table
methylation -> SurvivalVAE/PCA compressor -> methylation_latent_* table
latents + clinical + survival -> your TVAE or the bundled SDV TVAE
sampled latents -> saved compressor decoder -> reconstructed matrix
```

The train/test split in `run_hybrid.py` is for model selection. It estimates how
well a compressor reconstructs the held-out matrix and whether the latent space
still preserves survival signal. Once you choose parameters, refit on all rows and
export the complete latent table for your own TVAE.

If `n_top_genes` selects only high-variance features, the decoder learns those
features. Decoded output is still written with the full original matrix column
set; features outside the selected set are filled with their training-set mean
in the preprocessed matrix space.

Tune compressor settings from a YAML sweep:

```bash
pode synthesis hybrid tune \
    --sweep configs/synthesis/hybrid_sweep.yaml \
    --expression expression.csv --survival survival.csv \
    --out-dir runs/tune
```

This writes `sweep_results.csv` and `selected_config.json`. If the sweep varies
`seed`, the tuner also writes `sweep_results_aggregated.csv` and ranks configs by
their mean score across seeds rather than by a single lucky run. The default
ranking is a composite score using reconstruction MSE, concordance gap, KM
distance, latent covariance gap, and hazard-ratio correlation penalty; inspect
the raw and aggregated metrics before accepting the winner.

You can choose how tuning ranks configurations:

* `full_joint` -- use clinical + latents + survival for scoring. This is the
  default when you want the best end-to-end joint model.
* `latents_survival` -- score using latents + survival only, even if clinical
  columns are present. Use this when you do not want strong clinical predictors
  to mask weak expression latents.
* `reconstruction_only` -- rank only by held-out expression reconstruction MSE.

Set this in the sweep YAML with `tuning_objective:` or override it with
`--tuning-objective`.

Recent workflow changes:

* `run_hybrid.py` now exposes explicit `run`, `tune`, `export-latents`, and
  `sample-decode` subcommands instead of relying only on the legacy single-shot
  entrypoint.
* `clinical.csv` is optional. Expression + survival runs are supported without
  breaking tuning, export, or decode.
* When `seed` varies in the sweep, tuning writes
  `sweep_results_aggregated.csv` and selects the winner by mean score across
  seeds rather than by a single run.
* `latents_survival` is available as a tuning objective when you want to judge
  the latent handoff mainly on survival preservation.
* Cox evaluation now uses a fixed feature mask derived from the training split,
  so sweep candidates remain comparable even if some folds contain constant or
  near-constant clinical columns.
* The survival VAE sweep space now includes `free_bits` and
  `kl_warmup_epochs`, which are useful when larger latent spaces start to
  collapse.

Export full-data latents for an external TVAE:

```bash
pode synthesis hybrid export-latents \
    --config runs/tune/selected_config.json \
    --clinical clinical.csv --expression expression.csv --survival survival.csv \
    --out-dir runs/final_handoff
```

The handoff directory contains:

* `latents_full.csv` -- one row per original sample, columns `expr_latent_0...`
* `clinical_full.csv` and `survival_full.csv` -- aligned rows for TVAE training
* `compressor.pkl` or `compressor.pt` -- decoder/checkpoint for reconstruction
* `handoff_metadata.json` -- selected genes, fill values, latent order, and config

`latents_full.csv` is the main handoff file for a separate TVAE. Its structure is
simple: one row per original sample and one column per latent dimension, named
`expr_latent_0`, `expr_latent_1`, and so on. Those column names matter because
`sample-decode` looks for that exact latent naming scheme when reconstructing
expression.

Example:

```csv
expr_latent_0,expr_latent_1,expr_latent_2
0.128,-1.442,0.337
-0.221,-0.918,1.104
```

After your separate TVAE generates new latent rows, decode them:

```bash
pode synthesis hybrid sample-decode \
    --bundle-dir runs/final_handoff \
    --latents sampled_latents.csv \
    --out-dir runs/synthetic
```

This writes `synthetic_expression.csv` from the sampled `expr_latent_*` columns
and keeps a copy of `sampled_latents.csv` in the output directory. If you also
have sampled clinical/survival CSVs, pass `--clinical` and `--survival` to copy
them into the same output bundle.

The main output CSVs have these roles:

* `synthetic_clinical.csv` -- generated clinical table when you run the bundled
  synthesizer or when you copy your external TVAE outputs into the decode bundle.
* `synthetic_survival.csv` -- generated `time,event` table in the same row order
  as the synthetic clinical and latent rows.
* `sampled_latents.csv` -- latent samples produced by the bundled synthesizer or
  copied in from your external TVAE.
* `synthetic_expression.csv` -- decoded expression matrix in the original gene
  column order from `expression.csv`.

When HVG selection is active, `synthetic_expression.csv` still has the full gene
set. The selected genes come from the decoder; unselected genes are filled from
the saved training-set means recorded in `handoff_metadata.json`.

For a local external TVAE workflow, the exported bundle can now be converted
directly into a single PODE-ready training table. The helper script
`scripts/prepare_hybrid_bundle.py` concatenates:

* `clinical_full.csv`
* `latents_full.csv`
* `survival_full.csv`

and also writes a starter PODE YAML that marks the latent columns as
continuous, preserves an explicit survival pair (`OS_Time`, `OS_Status` by
default), and lets you force clinical columns such as `Age` to continuous and
`Sex`, `M._versus_M.`, `ConsensusMYC`, and `WCA` to categorical.
By default, the emitted PODE YAML now contains a non-trivial tuning grid for
`batch_size`, `embedding_dim`, `compress_dims`, `decompress_dims`, `l2scale`,
and `loss_factor`, so `pode synthesis tune` explores more than one combination. If you
explicitly want the old one-combination behavior, pass
`--singleton-tuning-grid`.

After PODE generates a new wide table, the reverse helper
`scripts/decode_hybrid_samples.py` goes the other way. It expects a table
containing:

* clinical columns,
* `expr_latent_*`,
* a survival pair such as `OS_Time`, `OS_Status`.

It writes:

* `sampled_latents.csv`
* `synthetic_clinical.csv`
* `synthetic_survival.csv`
* `synthetic_expression.csv`

and uses the saved hybrid compressor from `export-latents` to decode the
sampled latent rows back into the full original expression matrix.

Example:

```bash
pode synthesis decode \
    --bundle-dir runs/final_handoff \
    --pode-table path/to/pode_generated.csv \
    --out-dir runs/decoded_from_pode
```

### Joint multi-omic handoff (expression + methylation)

Both `prepare_pode_training.py` and `decode_pode_samples.py` (and their `pode`-repo
twins `prepare_hybrid_bundle.py` / `decode_hybrid_samples.py`) accept **multiple**
`--bundle-dir`s. Export each modality separately as usual, then concatenate their
latent blocks into a **single** PODE training table so one downstream TVAE learns the
joint distribution over both `expr_latent_*` and `methylation_latent_*` (plus clinical
and survival) — capturing cross-modal correlation the two independent runs cannot.
The bundles must be row-aligned (same patients/order) and use distinct latent prefixes;
`prepare` validates both.

```bash
# one PODE-ready table from an expression bundle + a methylation bundle
pode synthesis bundle \
    --bundle-dir runs/final_handoff runs/final_handoff_methylation \
    --out-dir runs/joint_pode_prep \
    --continuous-cols Age \
    --categorical-cols Sex M._versus_M. ConsensusMYC WCA MNP_Subgroup

# ... train/tune/generate one PODE model on runs/joint_pode_prep/training_table.csv ...

# split the generated table and decode BOTH blocks back to their matrices
pode synthesis decode \
    --bundle-dir runs/final_handoff runs/final_handoff_methylation \
    --pode-table path/to/pode_generated.csv \
    --out-dir runs/joint_decoded
```

Decode writes `synthetic_clinical.csv`, `synthetic_survival.csv`, `sampled_latents.csv`,
and one `synthetic_<matrix_kind>.csv` per bundle (`synthetic_expression.csv` **and**
`synthetic_methylation.csv`). Passing a single `--bundle-dir` reproduces the original
single-modality behavior unchanged.

## Joint multimodal VAE (`scripts/run_mmvae.py`)

The joint model (`synthvae/mmvae/`) learns a shared latent `z` that
jointly generates the clinical table, the expression matrix, and survival —
and, optionally, any number of configured molecular heads sharing the same
latent. Methylation remains available through its dedicated compatibility flag.
Unlike the two-stage hybrid, everything trains together and you can
drop a modality at inference — encode clinical only and generate expression, or
vice versa. Use it when you specifically need that cross-modal / missing-modality
generation; for plain survival-preserving synthesis the hybrid is the stronger,
simpler default (see `docs/ARCHITECTURE.md` §2).

It requires torch (`pip install -e ".[all]"`). Same CSV contract as the hybrid
(see [CSV contract](#csv-contract)); expression should be on a continuous scale for
the default gaussian likelihood, or raw counts for `nb` / `zinb`. Methylation is
supplied as a beta-value matrix in `[0, 1]` (features as columns) via
`--methylation`; it is logit-transformed to M-values internally and decoded back to
beta-values on generation.

Toy cohort:

```bash
pode synthesis mmvae run --toy --config configs/synthesis/mmvae.yaml --epochs 150
```

Bring your own data:

```bash
pode synthesis mmvae run --config configs/synthesis/mmvae.yaml \
    --clinical clinical.csv --expression expression.csv --survival survival.csv
```

Joint expression **and** methylation (both heads share the latent) — just add
`--methylation`:

```bash
pode synthesis mmvae run --config configs/synthesis/mmvae.yaml \
    --clinical clinical.csv --expression expression.csv \
    --methylation methylation.csv --survival survival.csv --out runs/mmvae_joint
```

Add a DNA copy-number head with `--omic` once the `cnv` block is uncommented in
the config (build `cnv.csv` with `segments_to_matrix`, as above). It joins the
shared latent as a Gaussian head and is written to `synthetic_cnv.csv`:

```bash
pode synthesis mmvae run --config configs/synthesis/mmvae.yaml \
    --clinical clinical.csv --expression expression.csv \
    --survival survival.csv --omic cnv=cnv.csv --out runs/mmvae_cnv
```

`run` trains on a train split, samples a synthetic cohort, and prints the same
TSTR / TRTR / KM / hazard-ratio report the hybrid uses. Add `--out DIR` to write
`synthetic_{clinical,expression,survival}.csv` (plus `synthetic_methylation.csv`
when `--methylation` was supplied). Omitting `--methylation` leaves the model,
training, and outputs identical to the expression-only path. To sweep the knobs
below, train the winner on all data, and reuse the saved model, see
[Tuning + saving a joint model](#tuning--saving-a-joint-model).

Key levers (each overrides `configs/synthesis/mmvae.yaml`):

* `--fusion poe|moe|mopoe` — how the per-modality posteriors combine. **PoE**
  (precision-weighted product, default) is confident and cheap; **MoE** is a
  mixture; **MoPoE** mixes over every modality subset (most robust to a missing
  modality, `2^M` cost).
* `--likelihood gaussian|nb|zinb` — expression likelihood. `gaussian` for
  log-CPM/VST/continuous; `nb` / `zinb` for raw counts (zero-inflated for sparse
  single-cell-like data).
* `--weighting fixed|uncertainty|gradnorm` — how the reconstruction terms are
  balanced. `fixed` (config weights, bias survival up), `uncertainty` (learned
  Kendall & Gal, default), or `gradnorm` (dynamic gradient-norm balancing). With
  `--methylation` the methylation term is balanced alongside the others.
* `--methylation FILE` — optional methylation beta-value matrix; adds a fourth
  modality (a third *encoded* one) with its own Gaussian-on-M-value head. Its
  head size / likelihood come from `architecture.methylation_head_hidden` and
  `likelihoods.methylation` in the config.
* `--omic NAME=PATH` — repeatable generic molecular input. `NAME` must be
  declared under `modalities:` in the MMVAE config; use it for CNV, mutation,
  ATAC, proteomic, or other supported matrices.
* `--latent-dim`, `--epochs`, `--device`, `--seed`. Runs are reproducible for a
  given `--seed` (weight init and training are both seeded).

`configs/synthesis/mmvae.yaml` mirrors `MMVAEConfig` and holds the full option set plus a
`sweeps:` grid (consumed by the benchmark below). One field worth knowing:
`objective.recon_reduction: mean_per_feature` puts the high-dimensional expression
term on a per-gene scale so it does not swamp the low-dimensional survival and
clinical terms in the shared latent — the key knob for retaining prognostic signal.

### Additional molecular heads

Use one config for expression-only and multi-omic runs. The active heads are
determined by supplied inputs: expression-only tuning scores expression and
survival; joint runs add per-head geometry checks and cross-omic checks for every
active pair. Do not create a separate sweep merely because a head is absent.

Declare generic heads in `modalities:` and pass their CSVs with `--omic NAME=PATH`:

```yaml
modalities:
  cnv:
    kind: continuous       # standardized Gaussian model; log2 ratios are typical
  mutations:
    kind: binary           # Bernoulli model; values must be exactly 0 or 1
  atac:
    kind: count
    likelihood: zinb       # `nb` is also supported
```

All matrices need one row per patient in exactly the same order as `clinical.csv`,
`expression.csv`, and `survival.csv`. Continuous/CNV matrices must be finite,
count matrices non-negative, methylation beta-values in `[0, 1]`, and binary
matrices only `0`/`1`.

```bash
pode synthesis mmvae tune --config configs/synthesis/mmvae.yaml \
    --sweep configs/synthesis/mmvae_sweep.yaml \
    --clinical clinical.csv --expression expression.csv --survival survival.csv \
    --omic cnv=cnv_log2_ratio.csv --omic mutations=gene_mutations.csv \
    --out-dir runs/mmvae_multi_tune

pode synthesis mmvae train --config runs/mmvae_multi_tune/selected_config.json \
    --clinical clinical.csv --expression expression.csv --survival survival.csv \
    --omic cnv=cnv_log2_ratio.csv --omic mutations=gene_mutations.csv \
    --out-dir runs/mmvae_multi_final
```

Generation writes `synthetic_<name>.csv` for each generic head, in addition to
the established `synthetic_expression.csv` and `synthetic_methylation.csv` names.
Gaussian heads use `generation.observation_sampler`. New joint configurations
default to `residual_factor`: reconstruction residuals are decomposed into
per-head real-fitted PCA factors, their score covariance is shrunk and sampled
jointly across heads, and only the small per-feature remainder is independent.
This avoids restoring marginal variance with covariance-destroying white noise.
`mean` and the historical scalar-noise `independent` sampler remain available.
The internal `auto` default resolves to `independent` for expression-only models
and `residual_factor` when two or more molecular heads are active, so older config
files gain the joint fix without changing expression-only behaviour.
`generation.observation_noise_scale` calibrates each head independently. Count
heads continue to sample NB/ZINB and binary heads Bernoulli probabilities.

### Opt-in shared/private latents

> **Development status: paused.** The broad shared/private experiment and its
> low-noise residual-factor calibration both failed quantitative and fixed-real
> visual promotion. The calibration reduced manifold recall and within-subgroup
> dispersion while worsening expression and methylation boundary penalties.
> Use the hybrid expression/methylation generator for production. The commands
> below remain documented only for reproducibility of the failed experiment;
> do not treat them as the recommended workflow or start another rank/noise
> calibration without a new observation-sampling design.

The compatibility default remains `dims.latent_structure: shared`. To let each
molecular decoder retain modality-specific structure, opt in explicitly:

```yaml
dims:
  latent_structure: shared_private
  private_latent_dims: {default: 8, methylation: 16}
objective:
  private_kl_weight: 0.5
```

Each molecular head decodes from the fused shared latent plus its own private
posterior. Clinical and survival remain shared-only. The ex-post GMM is fitted
jointly over shared and private draws, retaining paired cross-omic sampling.
Setting `latent_structure: shared` is the rollback switch; old checkpoints and
expression-only configs do not construct private modules.

Use the separate staged sweep rather than changing the established v2 sweep:

```bash
pode synthesis mmvae tune --config configs/synthesis/mmvae.yaml \
    --sweep configs/synthesis/mmvae_shared_private_sweep.yaml \
    --clinical clinical.csv --expression expression.csv \
    --methylation methylation.csv --survival survival.csv \
    --out-dir runs/mmvae_shared_private_tune

# Continue an interrupted search; completed rows are skipped.
pode synthesis mmvae tune --config configs/synthesis/mmvae.yaml \
    --sweep configs/synthesis/mmvae_shared_private_sweep.yaml \
    --clinical clinical.csv --expression expression.csv \
    --methylation methylation.csv --survival survival.csv \
    --out-dir runs/mmvae_shared_private_tune --resume
```

The four stages screen latent structure, cross shortlisted structures with
fusion/objective choices, calibrate the GMM and residual sampler, then refit
three finalists across three seeds. Stage CSVs are written incrementally and
`selected_config.json` contains the resolved 1500-epoch finalist configuration.

### Tuning + saving a joint model

The joint model has the same three-step controlled workflow as the hybrid: **tune**
a grid to pick a winner, **train** that winner on all your data and save it, then
**generate** synthetic cohorts from the saved model whenever you need them — no
retraining.

```bash
# 1. Sweep the grid, rank by the composite objective, pick the winner
pode synthesis mmvae tune --config configs/synthesis/mmvae.yaml \
    --sweep configs/synthesis/mmvae_sweep.yaml \
    --clinical clinical.csv --expression expression.csv --survival survival.csv \
    --out-dir runs/mmvae_tune

# 2. Train the winning config on ALL data and save the model
pode synthesis mmvae train --config runs/mmvae_tune/selected_config.json \
    --clinical clinical.csv --expression expression.csv --survival survival.csv \
    --out-dir runs/mmvae_final

# 3. Generate synthetic cohorts from the saved model, any time later
pode synthesis mmvae generate --model runs/mmvae_final/model.pt \
    --n 1000 --out-dir runs/mmvae_synth
```

For the joint **expression + methylation** model, pass `--methylation` to `tune`
and `train` (it is baked into the saved `model.pt`, so `generate` needs no extra
flag and writes `synthetic_methylation.csv` alongside the rest):

```bash
# 1. Tune with both molecular modalities
pode synthesis mmvae tune --config configs/synthesis/mmvae.yaml \
    --sweep configs/synthesis/mmvae_sweep.yaml \
    --clinical clinical.csv --expression expression.csv \
    --methylation methylation.csv --survival survival.csv \
    --out-dir runs/mmvae_meth_tune

# 2. Train the winner on ALL data (methylation head included)
pode synthesis mmvae train --config runs/mmvae_meth_tune/selected_config.json \
    --clinical clinical.csv --expression expression.csv \
    --methylation methylation.csv --survival survival.csv \
    --out-dir runs/mmvae_meth_final

# 3. Generate — synthetic_methylation.csv is written automatically
pode synthesis mmvae generate --model runs/mmvae_meth_final/model.pt \
    --n 1000 --out-dir runs/mmvae_meth_synth
```

* **`tune`** expands `configs/synthesis/mmvae_sweep.yaml` (a standalone `grid`/`fixed` spec with
  dotted MMVAE keys, e.g. `fusion.strategy`, `objective.beta_max`, `dims.latent_dim`),
  evaluates the same three subgroup/event-stratified folds for every architecture,
  then calibrates generation-only presets on the fitted network without neural
  retraining. Three generated replicates stabilize stochastic metrics. Survival
  and molecular-fidelity block ranks each contribute 50%; within the molecular
  block every active head and the cross-omic block receive equal weight. Guards
  reject collapsed variance, poor neighbour mixing, and covariance-destroying
  white-noise solutions. Sweep several `optimisation.seed` values to aggregate
  per-config mean/std across seeds. Outputs land in `--out-dir`: `sweep_results_folds.csv`
  (fold/replicate detail), `sweep_results.csv`
  (one row per candidate), `sweep_results_aggregated.csv` (per-config across seeds),
  and `selected_config.json` (the complete resolved winner). The
  `generation.preset` list samples useful combinations of residual rank, head
  noise scales, GMM components, and posterior draws without their full Cartesian
  product.
  TODO: some sweep candidates trigger Cox convergence problems during evaluation,
  usually from sparse or near-separating clinical one-hot covariates, or from
  having more Cox features than stable events in the held-out split. Those cases
  now fall back from `lifelines` to the bundled numpy Cox scorer so tuning can
  continue, but that means unstable candidates are not scored with a perfectly
  identical Cox backend. The ranking is still useful for triage, but relative
  ordering among fallback-heavy candidates is not perfectly apples-to-apples.
  A future cleanup is to use one Cox backend consistently for sweep scoring, or
  to flag or penalize candidates that required fallback.
* **`train`** reads the winner (its `--config` accepts either `selected_config.json`
  or a plain `configs/synthesis/mmvae.yaml`), fits `JointVAESynthesizer` on the whole cohort,
  and writes `model.pt` + `train_metadata.json` to `--out-dir`. The model file bundles
  the config, feature transformers, weights, and the latent density, so it is
  self-contained.
* **`generate`** reloads `model.pt` and writes `synthetic_{clinical,expression,`
  `survival}.csv`, plus `synthetic_methylation.csv` for the legacy methylation
  head and `synthetic_<name>.csv` for every generic head. A fresh process needs
  only the model, not the training data.

* **`evaluate`** compares any number of generated cohort directories using
  original-feature and real-fitted PCA metrics:

  ```bash
  pode synthesis mmvae evaluate \
      --clinical clinical.csv --expression expression.csv \
      --methylation methylation.csv --survival survival.csv \
      --synthetic joint=runs/mmvae_meth_synth \
      --synthetic joint_v2=runs/mmvae_joint_v2_synth \
      --synthetic hybrid=runs/generated_hybrid/decoded \
      --synthetic expression_only=runs/mmvae_expression_synth \
      --joint-baseline joint_v2 \
      --out-dir runs/mmvae_quality
  ```

  `quality_metrics.csv` and `quality_summary.json` report marginal variance,
  shrinkage covariance, PCA eigenspectra, source-neighbour mixing, real coverage,
  subgroup dispersion and marker effects, cross-modal association/canonical
  spectra, survival utility, duplicates, closest-real distance, and NN distance
  ratio. Source overlap is also checked at 512 and up to 5000 real-variable
  features using subgroup-conditional mixing, symmetric source AUC, radial
  Wasserstein discrepancy, and manifold precision/recall. Worst-subgroup radial
  median/p90 ratios, local real-manifold density, source AUC, and subgroup
  precision/recall prevent a large subgroup from hiding a smaller source shell;
  the worse resolution drives acceptance. Missing heads are recorded and skipped, allowing an expression-only
  reference in the same comparison. When cohorts are named `joint`, `hybrid`,
  and `expression_only`, the JSON also records the 10% expression and 15%
  methylation/cross-omic non-inferiority gates. A failed shared/private candidate
  recommends the separate low-noise calibration sweep rather than recursively
  recommending shared/private latents. A quantitative pass still requires visual
  review and never overwrites an existing model.

The R UMAP scripts now fit feature selection, scaling, and the primary UMAP on
real samples and project synthetic samples into that fixed reference. Pass
`--embedding-mode both` to write suffixed real-reference and combined plots for
visual sign-off; existing single-mode filenames remain unchanged. DE/DM heatmaps
likewise use real-only means and standard deviations for their colour scale.

## Benchmarking joint vs hybrid (`scripts/benchmark.py`)

`benchmark.py` runs **both** architectures across their sweep grids on the same
cohort and the same train/test split (per seed), scores them with the shared
evaluation harness, and reports the winner on each metric.

```bash
python scripts/benchmark.py --toy --quick          # fast smoke (small grid + GMM)
python scripts/benchmark.py --toy --seeds 0 1 2     # capped grid, 3 seeds
python scripts/benchmark.py --toy --full            # the entire grid (slow!)
```

The joint grid comes from `configs/synthesis/mmvae.yaml`'s `sweeps:` block; the hybrid grid
from `configs/synthesis/hybrid_sweep.yaml`. The hybrid arm defaults to the production path
(`survival_vae` compressor + `sdv` TVAE) and is driven through the exact same code
as `run_hybrid.py`.

Guardrails (the full grids are ~432 configs per arm, and each SDV fit is slow):

* `--max-configs N` — cap configs per architecture (default 8; deterministic
  even subset). `--full` removes the cap.
* `--quick` — tiny built-in grid + few epochs + the torch-free GMM synthesizer.
* `--joint-only` / `--hybrid-only`, `--seeds`, `--epochs` (override both arms),
  `--hybrid-compressor`, `--hybrid-synthesizer`, `--out-dir` (default `runs/benchmark`).

Outputs land in `--out-dir`: `benchmark_results.csv` (one row per arch × config ×
seed), `benchmark_aggregated.csv` (mean/std per config when multiple seeds), and
`benchmark_summary.md` — a side-by-side table of the best joint vs best hybrid
config with a per-metric winner. On the toy cohort the hybrid is expected to lead
on raw survival metrics; the joint model's advantage is cross-modal generation, not
this benchmark.

Tests:

```bash
python -m pytest tests/ -q        # or: python tests/test_smoke.py
```

## Repo layout

```
synthvae/
  data/          toy cohort generator, HVG preprocessing + batch-correction hook
  compressors/   Compressor interface; PCACompressor; SurvivalVAE (the front end)
  survival/      time discretization + discrete-time (logistic-hazard) loss
  losses/        gaussian NLL, free-bits KL, uncertainty weighting
  synthesizers/  Synthesizer interface; SdvTvaeSynthesizer; GaussianMixtureFallback
  pipeline/      latent table assembly + HybridPipeline orchestrator
  evaluation/    concordance, Kaplan–Meier, Cox (lifelines + numpy), report
  mmvae/         joint multimodal VAE (deliverable 2): encoders, experts (fusion),
                 heads, objectives, data (transformers/Dataset), model (fit/sample),
                 generate (JointVAESynthesizer orchestrator)
scripts/         run_hybrid.py, run_mmvae.py, benchmark.py, make_toy_data.py
config/          hybrid.yaml, hybrid_sweep.yaml, mmvae.yaml, mmvae_sweep.yaml (settings + sweep grids)
docs/            SPECIFICATION.md, ARCHITECTURE.md, TASKS.md
tests/           test_smoke.py (numpy-only), test_mmvae.py (torch joint model)
```

## What is verified vs. what needs your stack

The numeric core (discrete-time survival math, concordance, KM, numpy Cox,
PCA round-trip, and the full PCA→GMM→decode→evaluate pipeline) is unit-tested and
runs with numpy/sklearn/pandas alone. The torch paths are now runtime-tested too:
the whole joint `mmvae/` package (fit, all three fusion strategies, gaussian and
NB/ZINB likelihoods, generation, GradNorm) is exercised by `tests/test_mmvae.py`,
and both `run_mmvae.py` and `benchmark.py` run end-to-end on the toy cohort. You
still need `pip install -e ".[all]"` for the torch (`SurvivalVAE`, `mmvae`) and SDV
(`sdv` TVAE) paths — the `mmvae` tests skip automatically when torch is absent, and
the SDV-heavy `benchmark.py --hybrid-synthesizer sdv` runs need the full extras.
