# Mantis — MANTIS-MB survival-model inference wrapper

**Read this before running anything on a new tumour type.** MANTIS-MB's
released weights were trained exclusively on medulloblastoma. Applying them
to a different tumour type is an unvalidated,
exploratory use of the model — the predictions it produces have not been
shown to mean anything for a population it was never trained or tested on.
Treat any output on such a cohort as a curiosity to investigate further, not
a survival estimate to act on or report as validated.

Generic input preparation and subprocess orchestration around **MANTIS-MB**,
a published, frozen deep-learning survival model for medulloblastoma
(Stefan/Hovestadt, Dana-Farber — "AI-based Precision Prognostics and Therapy
Personalization for Childhood Brain Tumors"). This theme does **not**
reimplement or retrain the model: per the source project's own governing
principle, the reference implementation
([`sstefan01/methylation-survival`](https://github.com/sstefan01/methylation-survival))
is never modified, only driven via subprocess from your own checkout of it,
in its own isolated environment.

| Produces | From |
| --- | --- |
| A reindexed, imputed beta matrix in MANTIS's exact probe order | `pode mantis prepare-beta` |
| CNV gene-level features via conumee2 | `pode mantis prepare-cnv` |
| A MANTIS-format survival/clinical CSV | `pode mantis prepare-survival` |
| A correctly-shaped reference `config.yaml` for your cohort | `pode mantis generate-config` |
| Survival predictions from the released, frozen model | `pode mantis infer` |
| C-index / time-dependent AUC / Brier score | `pode mantis evaluate` |

## Required external setup

This theme is inert without all of the following in place — there is
deliberately no bundled/vendored copy of any of it (see Provenance).

**1. Three separate environments, cleanly separated:**

| Environment | Used for | Notes |
| --- | --- | --- |
| PODE's own (this repo) | `pode mantis prepare-beta`/`prepare-survival`/`generate-config` | Ordinary PODE deps only (pandas/numpy/pyyaml) |
| An **isolated** MANTIS Python env | `pode mantis infer`/`evaluate`/`plot` (subprocess) | Pinned deps incompatible with PODE's own — see below |
| R + Bioconductor | `pode mantis prepare-cnv` (subprocess to `Rscript`) | `minfi` + `conumee2`, no Python equivalent exists |

Create the isolated MANTIS environment (this exact recipe is proven working):

```bash
conda create -n mantis python=3.10.12 -y
conda activate mantis
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.2.2
python -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
    -f https://pytorch-geometric.com/whl/torch-2.2.2+cpu.html
python -m pip install sparselinear==0.0.5
python -m pip install -r requirements.txt   # from your reference-repo checkout, see below
```

Verify: `python -c "import sparselinear, torch, sksurv, lifelines"` must succeed.
The reference implementation forces CPU internally — a GPU is not used or needed.

Do **not** try to install MANTIS's dependencies into PODE's own environment:
`scikit-survival==0.23.1` (required here) conflicts with the `0.24.1` pin PODE
uses elsewhere, and `sparselinear`/`torch-scatter`/`torch-sparse` are
narrow-compatibility compiled extensions.

**2. Clone the reference implementation:**

```bash
git clone https://github.com/sstefan01/methylation-survival.git
cd methylation-survival && git checkout v1.0.0
```

This is `paths.reference_repo` in your `pode_mantis` config. Never edit files
inside it.

**3. Download the Zenodo release assets** (record 20818365) into a directory
of your choosing (`paths.zenodo_assets_dir`):

| Asset | Used by |
| --- | --- |
| `conn_mat_beta.csv` | the reference implementation's model construction |
| `mantismb_model_weights.pt` | `inference.model_weights_path` |
| The released 357,692-CpG order | `beta.cpg_order_path` |
| The released 16,067-gene order | `cnv.gene_order_path` (derivable from `cnv_hovestadt.csv`'s row labels if not shipped separately) |
| `training_probe_means.csv` | `beta.training_means_path` (a fixed constant of the published model — derive once from the Zenodo training-cohort betas, reuse for every new cohort; not something to rebuild per cohort) |

**4. Install R + Bioconductor:**

```r
install.packages("BiocManager")
BiocManager::install(c("minfi", "conumee2"))
install.packages("digest")
```

## Which path should I take?

1. **`prepare-beta`** your cohort's beta matrix (any orientation/format) to
   MANTIS's exact probe order.
2. **`prepare-cnv`** your cohort's Msets against a normal-tissue control
   reference (see the CNV control-reference note below — this has no
   default, deliberately).
3. **`prepare-survival`** your clinical data into the required format,
   always passing `--sample-order` from step 1 so misalignment is caught
   immediately, not silently.
4. **`generate-config`** to assemble a reference-repo config for your
   cohort — pass `--endpoint os` or `--endpoint pfs`.
5. **`infer`** to run the released model (subprocess into the isolated env).
6. **`evaluate`** (optional) for C-index / AUC / Brier score.

## Usage

```bash
pode mantis prepare-beta --config my_config.yaml

pode mantis prepare-cnv --config my_config.yaml

pode mantis prepare-survival --config my_config.yaml \
    --sample-order outputs/mantis/beta/beta_sample_order.txt

pode mantis generate-config --config my_config.yaml \
    --cohort my_cohort \
    --beta outputs/mantis/beta/beta_numeric.csv \
    --cnv  outputs/mantis/cnv/cnv_my_cohort_numeric.csv \
    --surv outputs/mantis/survival/survival.csv \
    --endpoint os

pode mantis infer --config my_config.yaml \
    --mantis-config outputs/mantis/mantis_config.yaml \
    --beta-sample-order outputs/mantis/beta/beta_sample_order.txt \
    --survival outputs/mantis/survival/survival.csv

pode mantis evaluate --config my_config.yaml \
    --mantis-config outputs/mantis/mantis_config.yaml \
    --pred-path outputs/mantis/predictions.csv \
    --survival outputs/mantis/survival/survival.csv
```

See `configs/mantis/config.template.yaml` for every field, or
`configs/mantis/config.schema.md` for the same schema as a reference table.

## The CNV control reference has no default — this is deliberate

A medulloblastoma cohort would typically use a cerebellum-derived
normal-tissue reference, which makes sense for a posterior-fossa tumour.
Other tumour types may not share that location, so silently reusing that same
reference would be a biological assumption nobody actually made for your
data. Pick a control Mset appropriate for your tumour type/location
deliberately, and record why.

## The noise threshold has no default either

An a priori "typical conumee" guess (`0.3`) can exclude almost every sample
in a cohort, because observed fit noise varies a lot between arrays,
batches and tissue types. A sensible threshold is the Tukey outlier fence
(`Q3 + 1.5*IQR`) over the cohort's own pooled noise distribution. Do the same
for yours: run `prepare-cnv` once, inspect the
`conumee_noise` column in the resulting `cnv_qc.csv`, and derive a threshold
from what you actually observe — never reuse a value tuned for a different
cohort.

## Correctness: the #1 failure mode this theme guards against

Nothing in the reference `inference.py`/`evaluate.py` checks that the beta
matrix's sample columns, the CNV matrix's sample columns, and the survival
CSV's row order actually agree with each other — they trust positional
convention alone, and a misaligned input produces plausible-looking,
confidently wrong predictions with no internal check. `prepare-survival`
asserts this when given `--sample-order`; `infer` re-checks it pre-flight
when given `--beta-sample-order`/`--survival`; `evaluate` checks the
predictions' row order, column count, and NaN-row explanation post-flight
when given `--survival`. Always pass these flags.

## Verification

This theme's wrapper (config generation, subprocess dispatch, alignment
checks) was verified against real data: the Zenodo-released `hovestadt`
cohort (n=100, MANTIS's own calibration cohort — see the caveat below)
reproduces the published-pipeline's own previously-recorded results exactly
through this wrapper: **PFS C-index 0.6705, OS C-index 0.7761**.

Note `hovestadt`/`jones`/`ICGC` is MANTIS's *calibration* cohort (used to
select the beta+CNV+metastasis architecture and hyperparameters), not a
clean held-out test set for the released model — a good positive control
for "is my environment/config correct," not a claim about generalisation.

## Generating the toy data

```bash
python examples/mantis/generate_toy_mantis_inputs.py
```

Exercises `prepare-beta`/`prepare-survival`/`generate-config`'s own logic
(reindexing, imputation, alignment assertions, fixed-section copying) with
`configs/mantis/config.example.yaml`, without needing R/Bioconductor or the
external MANTIS environment. It does **not** exercise real CNV calling —
there's no meaningful synthetic substitute for that; the toy CNV input is a
plain synthetic numeric matrix in the right shape, not conumee2 output.

## Provenance

Ported from an internal comparison study that adapted the published reference
implementation ([`sstefan01/methylation-survival`](https://github.com/sstefan01/methylation-survival),
MIT licensed) without modifying it. The reference implementation and Zenodo
assets are intentionally not vendored here (see "Required external setup"
above) — only the generic input-preparation, config-generation, subprocess-
orchestration, and correctness-assertion layer that drives them.

## License

MIT — see the repository's top-level `LICENSE`. The reference implementation
this theme wraps is separately MIT-licensed by its own authors (Stefan et al.).
