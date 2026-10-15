# PODE documentation

PODE is an umbrella suite organised into four **themes**. Each theme is a
first-class component (its own package + docs) but they share one repo, one
virtual environment, and one `pode` command.

| Theme | Command | Python pkg | R pkg | What it does |
| --- | --- | --- | --- | --- |
| [Synthesis](synthesis/) | `pode synthesis generate` / `pode synthesis …` | `pode`, `synthvae` | (R plot/validate scripts) | synthetic clinical & omics data |
| [Signatures](signatures/) | `pode signatures …` | `metacpg` | `R/metacpg` | prognostic CpG signature extraction |
| [Prognosis](prognosis/) | `pode prognosis …` | `prognostk` | `R/prognostk` | metrics, risk stratification, figures |
| [Mantis](mantis/) | `pode mantis …` | `pode_mantis` | `R/scripts` | generic input prep + subprocess wrapper around the published MANTIS-MB survival model |
| [Foundation](foundation/) | `pode foundation …` | `pode_foundation` | — | methylation-array MAE pretraining, embeddings, optional survival fine-tuning |
| [Federated](federated/) | `pode federated …` | `pode.federated` | — | federated TVAE, gated exchange, DP-SGD and MetaCPG signatures across sites |

## Where things live

- **Synthesis** — [synthesis/README.md](synthesis/README.md),
  [synthesis/ARCHITECTURE.md](synthesis/ARCHITECTURE.md),
  [synthesis/SPECIFICATION.md](synthesis/SPECIFICATION.md), and the copy-paste
  end-to-end pipelines in [synthesis/WORKFLOWS.md](synthesis/WORKFLOWS.md).
  Generative imputation: [synthesis/IMPUTATION.md](synthesis/IMPUTATION.md) (guide
  and API). Augmentation for prognostic models:
  [synthesis/AUGMENTATION.md](synthesis/AUGMENTATION.md). Privacy gate for synthetic data: [synthesis/PRIVACY.md](synthesis/PRIVACY.md).
  Example configs: [../configs/imputation/](../configs/imputation/),
  [../configs/augmentation/](../configs/augmentation/), [../configs/privacy/](../configs/privacy/).
- **Signatures** — [signatures/README.md](signatures/README.md),
  [signatures/architecture.md](signatures/architecture.md),
  [signatures/verification_commands.md](signatures/verification_commands.md). Federated
  MetaCPG (signatures from per-site statistics, leave-one-site-out validation):
  [federated/FEDERATED.md](federated/FEDERATED.md#4-federated-metacpg-signatures).
- **Prognosis** — [prognosis/README.md](prognosis/README.md) and the config schema
  in [../configs/prognosis/config.schema.md](../configs/prognosis/config.schema.md).
- **Mantis** — [mantis/README.md](mantis/README.md) (start here — it has the required
  external setup checklist) and the config schema in
  [../configs/mantis/config.schema.md](../configs/mantis/config.schema.md).
- **Foundation** — [foundation/README.md](foundation/README.md) and the config schema
  in [../configs/foundation/config.schema.md](../configs/foundation/config.schema.md).

- **Federated** — [federated/FEDERATED.md](federated/FEDERATED.md) (guide: protocols,
  disclosure, DP, benchmark); configs in [../configs/federated/](../configs/federated/).

## R engines

Signatures and Prognosis ship parallel R implementations under the single
[`R/`](../R/) tree (`R/metacpg`, `R/prognostk`). Install and test them with
`R CMD INSTALL R/<pkg>` and each package's `testthat` suite. Mantis's CNV-calling
step and Synthesis's validation plots instead use loose, standalone scripts under
`R/scripts/` (not full installable packages) — see [mantis/README.md](mantis/README.md)
for Mantis's Bioconductor (`conumee2`/`minfi`) setup, which is heavier than the
CRAN-only deps of the two R packages above.

## Roadmap / deferred work

See [DEFERRED.md](DEFERRED.md) for the refactors intentionally parked during the
suite integration (shared `pode.common`, lean-core dependency split, config-schema
unification, unified R build).
