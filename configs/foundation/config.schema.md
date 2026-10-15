# pode_foundation configuration schema

Every `pode foundation <command>` takes its defaults from a YAML config loaded
with `pode_foundation.load_config()`. Only the command you're running reads
the sections it needs -- e.g. `preprocess` never looks at `survival`. Any
value can be overridden by passing an explicit CLI flag (`--x-path`,
`--out-dir`, ...) where one exists; the config just supplies the defaults.

The loader **validates** the config on load and raises a clear error for
internally inconsistent values (bad orientation/scale, `n_candidates` below
`n_keep`, a `mask_ratio` outside `(0, 1)`, an unknown `survival.mode`, ...).
Nothing is *required* at load time -- individual commands raise their own
clear errors if a value they specifically need (e.g. `matrix.path`) is still
unset when they run.

---

## `matrix` -- the beta (or M-value) matrix every other section reads

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `path` | string \| null | `null` | Wide CSV/TSV/TSV.gz or Parquet beta matrix, or a preprocessed `.npy` from `preprocess`. |
| `orientation` | `samples_by_features`\|`features_by_samples` | `samples_by_features` | `samples_by_features`: rows=samples, columns=probes (most exported CSVs). `features_by_samples`: rows=probes, columns=samples (minfi/raw array exports, e.g. `Mset_noob.parquet`-style). |
| `scale` | `beta`\|`m_value` | `beta` | Governs the sensible default for `model.output_activation` (`sigmoid` for beta, `linear` for M-values). |
| `feature_id_column` | string \| null | `null` | Header column name holding probe IDs, if the file isn't already indexed on them (e.g. a `probe_id` column in a Parquet export). |

## `preprocess` -- QC filter + genomic LD-pruning (`pode foundation preprocess`)

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `annotation_path` | string \| null | `null` | Probe annotation CSV with `Name`/`chr`/`pos` columns (auto-detects `IlmnID`/`chromosome`/`MAPINFO`-style aliases too). Required by this command. |
| `n_keep` | int | `50000` | Final probe count after pruning. Validated `> 0`. |
| `n_candidates` | int | `200000` | Top-N probes by SD considered as pruning candidates. Validated `>= n_keep`. |
| `r_thresh` | number | `0.5` | \|Pearson r\| above which a genomic neighbour is pruned. |
| `window_bp` | int | `1000` | Genomic window (bp) searched for correlated neighbours. |
| `epic_overlap_path` | string \| null | `null` | Optional one-column probe-ID CSV; candidates are restricted to this list first. |
| `cross_reactive_path` | string \| null | `null` | Optional one-column probe-ID CSV of cross-reactive probes to drop. |
| `output_dir` | string | `outputs/foundation/preprocess` | Where `X.npy`/`sample_ids.npy`/`feature_ids.npy`/`feature_stats.tsv`/`pruning_report.tsv` are written. |

Sex-chromosome probes (chrX/chrY) and any probe with a missing value in any
sample are always dropped -- this isn't configurable, matching the source
pipeline this theme was ported from.

## `model` -- the MAE architecture

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `encoder_dims` | list[int] | `[1024, 512]` | Encoder hidden-layer widths; the decoder mirrors them in reverse. |
| `latent_dim` | int | `256` | Embedding dimensionality. |
| `dropout` | number | `0.2` | Applied after every encoder/decoder hidden layer. |
| `output_activation` | `sigmoid`\|`linear` | `sigmoid` | `sigmoid` keeps reconstructions in `[0, 1]` (beta scale); use `linear` if `matrix.scale` is `m_value`. |

Read by `pretrain` when building a model from scratch; `finetune` and
`embed` instead read the architecture out of the checkpoint they're given,
so this section mostly matters for the first `pretrain` call.

## `training` -- `pode foundation pretrain` (also the Stage-1 trainer)

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `mask_ratio` | number | `0.15` | Fraction of probes hidden per training step. Validated in `(0, 1)`. |
| `batch_size` | int | `64` | |
| `epochs` | int | `100` | |
| `patience` | int | `10` | Early-stop after this many epochs with no validation-loss improvement. |
| `lr` | number | `1e-4` | |
| `weight_decay` | number | `1e-4` | |
| `val_frac` | number | `0.1` | Fraction of samples held out for validation. Validated in `(0, 1)`. |
| `val_mask_draws` | int | `3` | Mask realisations averaged per validation batch, for a less noisy validation loss. |
| `num_workers` | int | `0` | `DataLoader` worker processes. |
| `amp` | bool | `false` | Mixed precision; only takes effect when `device` resolves to CUDA. |
| `seed` | int | `42` | |
| `device` | `auto`\|`cpu`\|`cuda`\|`cuda:N` | `auto` | `auto` picks CUDA if available, else CPU. |
| `output_dir` | string | `outputs/foundation/pretrain` | Where `mae_best.pt`/`config.json`/`history.json` are written. |
| `prognostic_weight.path` | string \| null | `null` | Optional boolean `.npy` mask (aligned to the training feature set) up-weighting reconstruction error on a chosen probe subset. `null` = plain flat MSE. |
| `prognostic_weight.weight` | number | `1.0` | Weight applied to the masked-in subset; `1.0` is a no-op. |

## `pretrain_stage1` -- EXPERIMENTAL broad-reference corpus builder

Upstream marks this "written, not run": the alignment logic is exercised and
sound, but the full broad-reference pretraining run it enables was never
itself evaluated end to end. Off by default; skip entirely if you're
pretraining directly on your own cohort.

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `enabled` | bool | `false` | Documentation flag only -- `pode foundation combine` runs whenever it's invoked regardless; this just records intent in the config. |
| `combine.tcga_glob` | string \| null | `null` | Glob (e.g. `"TCGA-*.methylation450.tsv.gz"`) matched against `combine.tcga_dir` for wide TSV.gz sources. |
| `combine.tcga_dir` | string \| null | `null` | Directory searched with `tcga_glob`. |
| `combine.geo_dir` | string \| null | `null` | Directory of `<series>/betas.parquet` sources. |
| `combine.feature_ids_path` | string \| null | `null` | The fixed feature set (from `preprocess`) every source is aligned to; probes absent from a given source are filled with `0.5`. Required by `combine`. |
| `combine.chunk_size` | int | `25000` | Row-chunk size when streaming a wide TSV.gz source. |
| `output_dir` | string | `outputs/foundation/stage1` | Where `X_stage1.npy`/`sample_ids_stage1.npy`/`source_labels.tsv`/`probe_overlap.tsv` are written. |

## `finetune_stage2` -- `pode foundation finetune` (domain-adaptive fine-tune)

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `enabled` | bool | `false` | Documentation flag only, as above -- `finetune` runs whenever invoked. |
| `init_ckpt` | string \| null | `null` | Checkpoint from `pretrain` to warm-start from. Required by this command (or pass `--init-ckpt`). |
| `lr` | number | `1e-5` | Typically ~10x below `training.lr`. |
| `epochs` | int | `50` | |
| `patience` | int | `20` | |
| `output_dir` | string | `outputs/foundation/stage2` | |

Architecture (`encoder_dims`/`latent_dim`/`dropout`/`output_activation`) and
`mask_ratio` are inherited from the checkpoint, not re-read from `model`/
`training`, so a fine-tune always matches the model it's warm-starting from.

## `survival` -- optional Cox/C-index evaluation (`pode foundation survival`)

Off by default (`enabled: false`) -- your core output is a pretrained/
fine-tuned encoder and its embeddings, not a survival report. Turn this on
once you have a clinical/phenotype CSV to evaluate against.

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `enabled` | bool | `false` | Documentation flag only, as above. |
| `clinical_path` | string \| null | `null` | CSV with at least id/time/event columns. Required by this command. |
| `id_column` | string | `sample_id` | |
| `time_column` | string \| null | `null` | Required by this command. |
| `event_column` | string \| null | `null` | Required by this command; `1` = event, `0` = censored. |
| `fold_column` | string \| null | `null` | Optional pre-computed CV fold column; an auto stratified-K-fold (`n_folds`, stratified on `event_column`) is generated if unset. |
| `n_folds` | int | `5` | Used only when `fold_column` is unset. |
| `seed` | int | `42` | |
| `mode` | `frozen`\|`finetune` | `frozen` | `frozen`: fast, `lifelines`-based OOF Cox on a fixed embeddings CSV (`--embeddings`). `finetune`: torch encoder+head Cox fine-tune, OOF per fold, needs a checkpoint + the raw beta matrix (`--ckpt`/`--betas`), not just embeddings. |
| `penalizer` | number | `0.1` | `frozen` mode: `lifelines.CoxPHFitter` regularisation strength. |
| `l1_ratio` | number | `0.5` | `frozen` mode: elastic-net mix (`0`=ridge, `1`=lasso). |
| `finetune_mode` | `frozen`\|`partial`\|`full` | `partial` | `finetune` mode: which encoder layers train (`frozen`=head only, `partial`=all but the first encoder layer, `full`=everything). |
| `finetune_epochs` | int | `100` | `finetune` mode only. |
| `head_lr` | number | `1e-3` | `finetune` mode only. |
| `encoder_lr` | number | `1e-5` | `finetune` mode only. |
| `output_dir` | string | `outputs/foundation/survival` | Where the OOF risk CSV is written. |

Both modes report Harrell's C-index on held-out folds -- an *internal*
number. If the same cohort informed feature selection or model choice
upstream, treat it as optimistically biased and prefer an independent
validation cohort for a headline claim, if you have one.

## `embed` -- `pode foundation embed`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `feature_ids_path` | string \| null | `null` | The checkpoint's training feature set (from `preprocess`). Required by this command (or pass via `finetune_stage2`'s architecture, and the `survival` command's `finetune` mode, which reuse it too). |
| `output_path` | string | `outputs/foundation/embeddings.csv` | `sample_id x z000..zN` embeddings CSV. |

Probes present in `feature_ids_path` but missing from the input cohort are
imputed with that probe's per-cohort mean; a probe with *zero* support in
the cohort (no samples have it at all) falls back to `0.5`, the same neutral
prior the MAE itself uses for masked/unknown positions during training.
