# Foundation — methylation-array masked-autoencoder pretraining

Pretrain a masked autoencoder (MAE) on your own methylation-array cohort --
or on a broader public reference, then domain-adaptively fine-tune onto your
cohort -- and use the frozen encoder as a feature extractor for embeddings,
UMAP-style exploration, or downstream survival modelling. Config-driven, one
CLI (`pode foundation ...`), no institution-specific plumbing.

| Produces | From |
| --- | --- |
| A fixed, LD-pruned probe set | `pode foundation preprocess` |
| A pretrained MAE checkpoint | `pode foundation pretrain` |
| A domain-adapted checkpoint | `pode foundation finetune` |
| A `sample_id x z000..zN` embeddings CSV | `pode foundation embed` |
| An (optional) survival C-index + risk scores | `pode foundation survival` |

## Repository layout

```
pode_foundation/
  cli.py                  # `pode foundation ...` subcommands
  config.py                # YAML config loading/validation
  core/
    mae_model.py            # MethylationMAE architecture (generic, no cohort assumptions)
    mae_model_weighted.py   # optional per-feature-weighted reconstruction loss
    preprocess.py           # QC filter + genomic LD-pruning
    train.py                 # pretrain (also the Stage-1 trainer)
    finetune.py              # Stage-2 domain-adaptive fine-tune (DAPT)
    combine.py                # EXPERIMENTAL: broad-reference corpus builder
    embed.py                  # frozen-encoder inference
    survival.py                # optional Cox/C-index evaluation
configs/foundation/
  config.template.yaml     # every field, documented (inline YAML comments)
  config.schema.md          # the same schema as a reference table, section by section
  config.example.yaml      # runs end to end against the toy fixtures below
examples/foundation/
  generate_toy_methylation.py
tests/foundation/
```

## The config is the interface

Every command takes one `--config path/to/config.yaml`. Unset fields fall
back to the defaults in `pode_foundation/config.py`; only fill in what you
need for the command you're running:

```yaml
matrix:
  path: my_cohort_betas.csv
  orientation: samples_by_features   # rows=samples, columns=probes
  scale: beta

preprocess:
  annotation_path: probe_annotation.csv   # Name/chr/pos columns
  n_keep: 50000

training:
  mask_ratio: 0.15
  epochs: 100
  device: auto
```

See `configs/foundation/config.template.yaml` for the full, heavily-commented
YAML (architecture, training hyperparameters, the optional two-stage
pretrain/fine-tune split, and the optional survival-evaluation block), or
`configs/foundation/config.schema.md` for the same schema as a reference
table -- every key, its type, default, and what it does, section by section.

## Usage

```bash
# 1. QC filter + LD-prune your beta matrix to a fixed, ~independent probe set
pode foundation preprocess --config my_config.yaml

# 2a. Train directly on your own cohort ...
pode foundation pretrain --config my_config.yaml \
    --x-path outputs/foundation/preprocess/X.npy

# 2b. ... OR pretrain on a broader public reference first, then domain-adapt:
pode foundation pretrain  --config my_config.yaml --x-path broad_reference_X.npy --stage-tag 1
pode foundation finetune  --config my_config.yaml \
    --init-ckpt outputs/foundation/pretrain/mae_best.pt \
    --x-path    outputs/foundation/preprocess/X.npy

# 3. Extract embeddings for any cohort with the frozen encoder
pode foundation embed --config my_config.yaml \
    --ckpt outputs/foundation/pretrain/mae_best.pt \
    --betas my_cohort_betas.csv --out embeddings.csv

# 4. (optional) Benchmark the embeddings against survival
pode foundation survival --config my_config.yaml \
    --embeddings embeddings.csv --clinical my_clinical.csv
```

Also usable directly from Python (heavy names are exposed lazily, so
`import pode_foundation` stays fast without pulling in torch):

```python
from pode_foundation import load_config, run_preprocess, run_pretrain, run_embed

cfg = load_config("my_config.yaml")
run_preprocess(cfg)
run_pretrain(cfg, x_path="outputs/foundation/preprocess/X.npy")
run_embed(cfg, ckpt="outputs/foundation/pretrain/mae_best.pt", betas="my_cohort_betas.csv")
```

### Which path should I take?

- **Just my cohort, no broader reference available or wanted:** run
  `preprocess` then `pretrain` directly. This is a completely valid starting
  point -- see Provenance below for zero-shot evidence this generalises
  across tumour types even without any cohort-specific pretraining.
- **I have (or can source) a broader public reference corpus:** `pretrain`
  on the broad corpus first (`--stage-tag 1`), then `finetune` onto your own
  cohort (Stage-2 domain-adaptive pretraining, DAPT). `pode foundation
  combine` can help build that broad corpus from TCGA-style TSV.gz / GEO-style
  parquet sources, but it's marked **experimental** below -- read that note
  before relying on it.

## Generating the toy data

```bash
python examples/foundation/generate_toy_methylation.py
```

Writes `toy_beta.csv` (40 samples x 2000 CpGs), `toy_annotation.csv`
(chr/pos for LD-pruning), and `toy_phenotype.csv` (a synthetic survival
sidecar) beside the script. This is a smoke-test fixture -- it exercises
every pipeline stage quickly and deterministically, not a realistic dataset.
`configs/foundation/config.example.yaml` is pre-wired to run the whole
pipeline (`preprocess` -> `pretrain` -> `embed` -> `survival`) against it in
well under a minute on a CPU.

## Experimental: `combine` / two-stage pretraining

`pode foundation combine` and the Stage-1/Stage-2 pretrain-then-fine-tune
split are ported from a private research pipeline where this code path is
explicitly marked *"written, not run"* -- the alignment logic was exercised
on small inputs and is structurally sound, but the full broad-reference
pretraining run it enables has not itself been evaluated end to end. Treat
it as a documented starting point, not a validated result. If you don't have
(or don't want) a broader reference corpus, skip it entirely and pretrain
directly on your own cohort -- that's the fully supported path.

## Provenance

This theme's core (architecture, preprocessing, pretrain/fine-tune,
embedding extraction, survival evaluation) was ported from an internal
methylation-MAE research pipeline that pretrains on a public reference cohort
of 2,801 CNS tumours (Capper et al., GSE90496) and evaluates the frozen
encoder's embeddings against Cox survival models. The reference cohort's MNP
classifier spans many CNS tumour entities, so the same broad-reference
approach applies in principle to other CNS tumour cohorts.

Everything cohort/institution-specific in the source pipeline (a
disease-specific hand-crafted CpG risk score used as a comparator,
an external validation cohort crosswalk, institutional sample-ID matching)
was deliberately left behind in this port -- see the module docstrings in
`pode_foundation/core/` for what each piece does and why it's generic.

## License

MIT -- see the repository's top-level `LICENSE`.
