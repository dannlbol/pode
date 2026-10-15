# pode_mantis configuration schema

This describes **pode_mantis's own config** -- what drives every `pode mantis
<command>`. It is distinct from the MANTIS reference repo's own `config.yaml`,
which `generate-config` *produces* as an output artefact for
`infer`/`evaluate`/`plot` to hand to the reference implementation via
`--config`.

The loader validates internal consistency on load (e.g. `beta.orientation`
must be a recognised value, `cnv.noise_threshold` must be in a plausible
range if set). It deliberately does **not** require most fields up front --
each subcommand raises its own clear error if a value it specifically needs
is still unset when it runs, since (for example) you should be able to run
`prepare-beta` alone without having picked a CNV control reference yet.

---

## `paths` -- external setup locations (see `docs/mantis/README.md`)

| Key | Type | Required by | Meaning |
|-----|------|-------------|---------|
| `reference_repo` | string | `infer`, `evaluate`, `plot` | Where you cloned+checked-out `sstefan01/methylation-survival` (v1.0.0). Never modified, only invoked via subprocess. |
| `mantis_python` | string | `infer`, `evaluate`, `plot` | Interpreter inside the **isolated external MANTIS environment**. Never PODE's own interpreter -- MANTIS's pinned deps (`sparselinear`, `torch-scatter`, `torch-sparse`, `scikit-survival==0.23.1`) cannot coexist with PODE's own environment (PODE elsewhere pins `scikit-survival==0.24.1`). |
| `zenodo_assets_dir` | string | `prepare-beta`, `prepare-cnv`, `generate-config` (as defaults) | Directory holding the Zenodo record 20818365 assets: `conn_mat_beta.csv`, `mantismb_model_weights.pt`, the released CpG/gene order lists, `training_probe_means.csv`. |

## `beta` -- `pode mantis prepare-beta`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `cohort_path` | string \| null | `null` | Your cohort's beta matrix. CSV/TSV/Parquet, any of the two orientations below. Required by this command. |
| `orientation` | `samples_by_features`\|`features_by_samples` | `samples_by_features` | `samples_by_features`: rows=samples, columns=probes. `features_by_samples`: rows=probes, columns=samples. |
| `cpg_order_path` | string \| null | `null` | The released 357,692-CpG order (from `zenodo_assets_dir`). Required by this command. |
| `training_means_path` | string \| null | `null` | Per-probe means from MANTIS's own training cohorts (from `zenodo_assets_dir`), used to impute probes missing from your cohort. If unset, imputation falls back to your own cohort's per-probe median only (a weaker fallback -- a warning is printed). |
| `output_dir` | string | `outputs/mantis/beta` | Where `beta_numeric.csv` (the file `inference.py` actually reads), `beta_sample_order.txt` (the sidecar every other step cross-checks), and `probe_coverage.csv` are written. |

Sex-chromosome filtering, LD-pruning, etc. do **not** apply here -- unlike the
`foundation` theme's `preprocess`, this step's job is purely to align your
cohort's probes to a fixed, already-published feature set, not to select
features. A missing probe is always imputed, never dropped: the reference
implementation's `normalize_instance_wise` is per-sample and per-block, so
dropping a probe would silently shift every other value for that sample.

## `cnv` -- `pode mantis prepare-cnv` (R/Bioconductor subprocess)

No Python equivalent exists for CNV calling; this dispatches to
`R/scripts/prepare_mantis_cnv.R` via `Rscript` on `PATH`.

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `msets` | map{name: path} | `{}` | One or more query Msets (RDS files, from `minfi`). Required by this command. |
| `control_mset` | string \| null | `null` | **Required, no default.** A normal-tissue reference Mset appropriate for your tumour type/location. A medulloblastoma cohort would typically use a cerebellum reference (sensible for a posterior-fossa tumour); other tumour types may not share that location, so this is a deliberate judgement call, not something a generic tool should silently assume for you. |
| `gene_order_path` | string \| null | `null` | The released 16,067-gene order (from `zenodo_assets_dir`). Required by this command. |
| `noise_threshold` | number \| null | `null` | **Required, no default.** conumee2 fit-noise exclusion cutoff. An a priori "typical conumee" guess (`0.3`) can exclude almost every sample in a cohort. Derive yours from your own cohort's observed noise distribution: run `prepare-cnv` once, inspect the `conumee_noise` column in the resulting `cnv_qc.csv`, and compute (a starting point) the Tukey outlier fence, `Q3 + 1.5*IQR`, over the pooled distribution. Do not reuse a value tuned for a different cohort. |
| `array_type` | `450k`\|`EPIC` | `450k` | Never mix array types within one run -- build a separate common-probe annotation per array type if you have both. |
| `phenotype_path` | string \| null | `null` | Optional sample sheet with `ConsensusMYC`/`ConsensusMYCN` columns, enabling an optional MYC/MYCN positive-control sensitivity/specificity check. Skipped entirely if you don't have independent CNV ground truth -- not required for the pipeline to run. |
| `output_dir` | string | `outputs/mantis/cnv` | Where `cnv_<cohort>_numeric.csv`, `cnv_<cohort>_sample_order.txt`, `common_probes.txt`(+md5), `cnv_unmapped_genes.txt`, and `cnv_qc.csv` (per-sample noise/imputation/QC) are written. |

## `survival` -- `pode mantis prepare-survival`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `clinical_path` | string \| null | `null` | Your clinical/phenotype CSV. Required by this command. |
| `id_column` | string \| null | `null` | Required. |
| `time_column` | string \| null | `null` | OS time. Required. |
| `event_column` | string \| null | `null` | OS event, `1`=death. Required. |
| `pfs_time_column` | string \| null | `null` | Only needed if you'll use `reference_config.endpoint: pfs`. |
| `pfs_event_column` | string \| null | `null` | As above. |
| `metastasis_column` | string \| null | `null` | **Required.** Metastasis status is a required model input (`include_metastasis: True` upstream), not optional -- coded `0`/`1` in your data; MANTIS's generated config recodes `0`→`-1` itself, so do **not** pre-recode. Samples with missing metastasis status should be excluded from primary analysis (a warning is printed; M0-imputation is a sensitivity-only option, not a default). |
| `output_dir` | string | `outputs/mantis/survival` | Where `survival.csv` is written. |

If you pass `--sample-order` (the beta step's sidecar) to `prepare-survival`,
row order is asserted to exactly match before writing -- this is the single
most important check in the whole pipeline (see `docs/mantis/README.md`'s
"silent row misalignment" warning), so always pass it.

## `reference_config` -- `pode mantis generate-config`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `endpoint` | `os`\|`pfs` | `os` | Which survival endpoint to evaluate. Sets **both** `time_column`/`event_column` and `os_time_column`/`os_event_column` consistently, closing a real bug in the reference `evaluate.py` (it scores the test cohort's outcome using `os_time_column`/`os_event_column` regardless of nominal endpoint, falling back to `time_column`/`event_column` only if the `os_*` keys are absent entirely -- a hand-written "PFS config" that forgets this silently still evaluates on OS). |
| `fixed_sections_from` | string \| null | `null` | The reference repo's own shipped `config.yaml`. `model:`, `data.time_bins`, `run_setup.feature_types`, `run_setup.metastasis_recoding` are copied from here **verbatim** -- these are tied 1:1 to the released checkpoint's tensor shapes and must never be hand-edited or re-guessed. Required by this command. |
| `output_path` | string | `outputs/mantis/mantis_config.yaml` | The generated reference-repo-style config, passed to `infer`/`evaluate`/`plot` via `--mantis-config`. |

## `inference` -- `pode mantis infer`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `model_weights_path` | string \| null | `null` | The released checkpoint. Defaults to `<zenodo_assets_dir>/mantismb_model_weights.pt` if unset. |
| `output_predictions_path` | string | `outputs/mantis/predictions.csv` | Where the reference `inference.py` writes its output. |
| `batch_size` | int | `64` | Passed straight through to the reference config. |
