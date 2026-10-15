# PODE — Pediatric Oncology Discovery Engine

PODE is an umbrella suite for cohort-scale oncology data work. It began as a
synthetic-data engine and has grown to cover three interrelated **themes**, all
sharing one repository, one virtual environment, and one `pode` command:

| Theme | What it does | Command | Python pkg | R pkg |
| --- | --- | --- | --- | --- |
| **Synthesis** | data-agnostic synthetic clinical & omics data (TVAE + VAE front end) | `pode synthesis generate` / `pode synthesis …` | `pode`, `synthvae` | `R/scripts` |
| **Signatures** | prognostic CpG signature extraction (Cox + meta-analysis + LOO search) | `pode signatures …` | `metacpg` | `R/metacpg` |
| **Prognosis** | survival metrics, risk stratification & publication figures | `pode prognosis …` | `prognostk` | `R/prognostk` |
| **Federated** | federated TVAE, gated exchange, DP-SGD and MetaCPG signatures across sites | `pode federated …` | `pode.federated` | — |

Each theme keeps its own package and docs (so the distinct aims stay separate),
but you install and drive them together. Full docs: **[docs/README.md](docs/README.md)**.

### New features

| Feature | Command | Guide | Example config |
| --- | --- | --- | --- |
| Generative imputation (TVAE) | `pode synthesis impute <train\|apply\|benchmark>` | [IMPUTATION.md](docs/synthesis/IMPUTATION.md) | [configs/imputation/](configs/imputation/) |
| Augmentation for prognostic models | `pode synthesis augment <benchmark\|generate\|report>` | [AUGMENTATION.md](docs/synthesis/AUGMENTATION.md) | [configs/augmentation/](configs/augmentation/) |
| Privacy gate for synthetic data | `pode synthesis privacy <assess\|audit>` | [PRIVACY.md](docs/synthesis/PRIVACY.md) | [configs/privacy/](configs/privacy/) |
| Federated learning (TVAE, exchange, DP, signatures) | `pode federated …` | [FEDERATED.md](docs/federated/FEDERATED.md) | [configs/federated/](configs/federated/) |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# Pinned, reproducible versions (the original research environment):
python -m pip install -r requirements.txt
# Install the suite (editable). --no-deps avoids re-resolving the pins above.
python -m pip install -e . --no-deps        # add ".[dev]" for pytest
```

One dependency set covers all three Python packages. Theme install profiles
(`pip install -e ".[signatures]"`, `[prognosis]`, `[all]`) document each theme's
minimal footprint; a lean-core split that makes the heavy synthesis backends
(torch/sdv) optional is tracked in [docs/DEFERRED.md](docs/DEFERRED.md).

Verify:

```bash
python -c "import pode, synthvae, metacpg, prognostk; print('ok')"
pode --help
```

## The unified `pode` command

Everything is a packaged subcommand — no `python scripts/foo.py`:

```
pode synthesis generate | train | tune | evaluate | schema | seeds   # core synthesis
pode synthesis hybrid  <run|tune|export-latents|sample-decode>
pode synthesis mmvae   <run|tune|train|generate|evaluate>
pode synthesis bundle | promote | decode
pode synthesis impute  <train|apply|benchmark>                      # generative imputation
pode synthesis augment <benchmark|generate|report>                  # generative augmentation
pode synthesis privacy <assess|audit>                               # privacy gate for synthetic data
pode federated <tvae|exchange|benchmark|report|signature>           # federated learning across sites
pode signatures <run|search|extract|validate|submit|worker|aggregate|compare|methods|benchmark|predict>
pode prognosis  <metrics|cutoffs|validate|km|distribution|forest|calibration|oncoplot>
```

(The old `python scripts/run_hybrid.py …` entry points remain as thin deprecated
shims that forward to `pode synthesis …`.)

## Quickstart per theme

**Synthesis** — generate synthetic clinical data, or run the omics hybrid:

```bash
python examples/make_toy_cohort.py                 # writes a toy cohort CSV
pode synthesis generate --data examples/cohort.csv --num-samples 300 --evaluate
pode synthesis hybrid run --n 200 --n-genes 100 --out-dir runs/hybrid_toy
```

See [docs/synthesis/WORKFLOWS.md](docs/synthesis/WORKFLOWS.md) for the full
compressor → TVAE pipelines and [docs/synthesis/pode-core.md](docs/synthesis/pode-core.md)
for the detailed core reference. The joint shared/private residual-factor experiment is currently paused after
failed quantitative and fixed-real visual reviews; the hybrid generator remains
the recommended production path for paired expression/methylation synthesis.

**Synthesis: generative augmentation for prognostic models.** Does adding
TVAE-generated patients to a prognostic model's training data make it better
on real patients it has never seen? `pode synthesis augment benchmark` answers
that, split first, synthesise second:

1. For each split of the real development rows (repeated K-fold, optionally plus
   an external cohort), a TVAE is fitted on that split's real training rows only.
2. The training rows are augmented by each strategy, at several amounts.
3. Each model is refitted on the augmented rows.
4. Everything is scored on the split's **real held-out patients** only.

The report covers:
- **discrimination:** paired ΔC with corrected CIs;
- calibration, Brier and IPA, net benefit;
- a subgroup audit;
- fidelity and privacy of the synthetic rows;
- real-data-equivalent efficiency, dose-response and compute cost;
- a gated verdict per configuration (`RECOMMENDED` / `NON-INFERIOR` / `NOT RECOMMENDED` / `UNSAFE`).

```bash
# Toy run (~30 s): every moving part on examples/cohort.csv
pode synthesis augment benchmark --data examples/cohort.csv \
    --augment-config configs/augmentation/toy.example.yaml --output-dir runs/augment_toy --quick
# -> runs/augment_toy/report.md, scorecard.csv, plots/
```

A new study is configured entirely in YAML; no code changes are needed.

1. **Prepare one table.**
   - One row per patient, with a survival time and event column and the
     candidate features.
   - Add any columns used only for splitting, strata or subgroups (e.g. cohort).
   - Optionally add an external cohort, flagged by a column or passed as `--external other.csv`.
2. **Write the config.** Copy
   [`configs/augmentation/benchmark.example.yaml`](configs/augmentation/benchmark.example.yaml)
   (every option, commented) and set at least:
   - the endpoint and horizon;
   - `generator.exclude_columns`: ids, cohort labels, post-outcome variables;
   - `features`;
   - the `strategies` (who to synthesise, how many);
   - the `models` (what to refit).
3. **Check what it resolves to,** without fitting anything:
   ```bash
   pode synthesis augment benchmark --data TABLE --config PODE_CFG --augment-config MY.yaml --dry-run
   ```
   `--config` is a PODE config holding the TVAE architecture (e.g. a promoted
   `pode_config_final.yaml`). Omit it for defaults.
4. **Smoke test, then run.** Smoke test with `--quick`, then run in full. Vary a
   run without editing YAML using `--strategies`, `--models`, `--ratios`,
   `--draws`, `--folds`, `--repeats`, `--n-jobs` and `--device`. Long runs:
   launch detached (`setsid nohup … &`), ideally from a git worktree. Every
   finished split is checkpointed, and re-running the same command resumes.
5. **Read** `report.md` → `scorecard.csv`, then **deploy** a configuration that
   earned `RECOMMENDED`:
   ```bash
   pode synthesis augment generate --data TABLE --config PODE_CFG --augment-config MY.yaml \
       --strategy tvae_events --ratio 0.5 --benchmark-dir RUN_DIR --output-dir augmented/
   ```
   This writes `augmented_training_table.csv`, with real and synthetic rows
   tagged by `_source`. `pode synthesis augment report --output-dir RUN_DIR`
   re-renders a finished run.

What can be varied:

| | options |
| --- | --- |
| **Strategies** | `tvae_natural`; `tvae_targeted` (any `equals`/`ranges` condition, or a mixture of groups, e.g. M+ patients, 2-year survivors, events); `tvae_subgroup` (balance a column's levels); cheap baselines `duplicate`, `jitter`, `weight` |
| **Generators** | `tvae` (the table directly); `hybrid` for omics matrices: per-split survival-VAE/PCA compressor → TVAE on clinical + latents → decode to every feature |
| **Models** | `coxnet`, `cox_ph`, `logistic_landmark` (with optional `top_variance` / `m_value_columns` for high-dimensional inputs); `metacpg_fixed` (a fixed MetaCPG signature); `metacpg_search` (MetaCPG's own CpG selection inside every split) |
| **Design** | folds/repeats, stratification, grouped rows, external cohort, Cox strata, subgroup audit, horizon, gates |

Guide: [docs/synthesis/AUGMENTATION.md](docs/synthesis/AUGMENTATION.md).
Augmentation often does not help; the benchmark exists so the question can be
asked properly, with paired confidence intervals and a privacy gate, before any
synthetic rows are used.

Generative *imputation* (`pode synthesis impute`) is documented in
[docs/synthesis/IMPUTATION.md](docs/synthesis/IMPUTATION.md), and the privacy
gate for releasing synthetic data (`pode synthesis privacy`) in
[docs/synthesis/PRIVACY.md](docs/synthesis/PRIVACY.md).

**Federated learning** — build generators and signatures across centres
without pooling patient rows (sites are simulated today; each site answers only
logged *site operations*):

- `pode federated tvae train|sample` — a TVAE trained by FedAvg/FedProx with
  secure aggregation, optional personalisation and DP-SGD (`--target-epsilon`,
  needs `pip install -e '.[federated]'`);
- `pode federated exchange` — local TVAEs; synthetic rows released only if they
  pass the privacy gate;
- `pode federated signature verify|run|benchmark` — MetaCPG from per-site Cox
  statistics; reproduces a centrally fitted model exactly and adds
  leave-one-site-out validation;
- `pode federated benchmark` — local vs exchange vs federated vs DP vs pooled
  generators: fidelity, utility on each site's real held-out patients, privacy,
  leave-one-site-out.

Guide: [docs/federated/FEDERATED.md](docs/federated/FEDERATED.md).

**Signatures** — extract a prognostic CpG signature from a config:

```bash
pode signatures validate --config configs/signatures/example.yml
pode signatures run      --config configs/signatures/example.yml --backend local
```

Config schema: [configs/signatures/schema/config.schema.json](configs/signatures/schema/config.schema.json).
Details: [docs/signatures/README.md](docs/signatures/README.md).
Advanced multi-omics methods: [docs/signatures/advanced.md](docs/signatures/advanced.md).
Method / parameter / covariate / cohort / grid-search / SLURM reference: [docs/signatures/methods.md](docs/signatures/methods.md).

**Prognosis** — metrics and figures from one YAML config:

```bash
pode prognosis metrics --config configs/prognosis/config.example.yaml \
    --data examples/prognosis/toy_cohort.csv
pode prognosis km      --config configs/prognosis/config.example.yaml \
    --data examples/prognosis/toy_cohort.csv --outdir figures --out km.pdf
```

Details: [docs/prognosis/README.md](docs/prognosis/README.md).

## R engines

Signatures and Prognosis ship parallel R implementations under the single
[`R/`](R/) tree, plus loose R validation/plotting scripts in `R/scripts/`:

```bash
R CMD INSTALL R/metacpg      # package metaCpG
R CMD INSTALL R/prognostk    # package prognostk
```

Each R package has its own `testthat` suite.

## Repository layout

```
pode/ synthvae/ metacpg/ prognostk/   # importable Python packages (flat)
R/                                     # R packages (metacpg, prognostk) + R/scripts
configs/   {synthesis,imputation,augmentation,signatures,prognosis}/
docs/      README.md + {synthesis,signatures,prognosis}/ + DEFERRED.md
examples/  cohort.csv, make_toy_cohort.py, prognosis/ (toy data)
scripts/   operational scripts + deprecated shims
tests/     pode/synthvae (flat) + imputation/ + augmentation/ + signatures/ + prognosis/
```

## Testing

```bash
pytest                       # all Python themes
pytest tests/signatures tests/prognosis   # just the new themes
pytest tests/augmentation                 # one directory at a time if full collection clashes
```

## Status & roadmap

The suite integration co-located each component intact; shared-code consolidation
(a common survival/plotting/config layer), the lean-core dependency split, and a
unified R build are tracked in [docs/DEFERRED.md](docs/DEFERRED.md).
