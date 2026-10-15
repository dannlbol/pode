# Deferred refactors

Work intentionally **parked** during the suite integration (metaCpG-func +
generic_figs_metrics_prognosis folded into PODE). Each item below is safe to pick
up independently; the integration co-located components intact rather than
rewiring shared code, so these are the follow-ups that pay down that decision.

## 1. Shared `pode.common` layer
Extract the duplicated primitives into one place and rewire all themes to it:
- **Survival / Cox stats** — duplicated across `pode/evaluation/survival.py`,
  `prognostk/metrics/survival.py`, and `metacpg/core.py`.
- **Plotting theme / palette** — `prognostk/utils/theme.py` vs the ad-hoc plotting
  in `pode/evaluation/plots.py` and `metacpg/core.py`.
- **YAML config loader/schema** — `pode/config.py`, `prognostk/config.py`, and
  `metacpg`'s `read_config` each parse+validate their own YAML.

## 2. Lean-core dependency split
Move the heavy synthesis backends (`torch`, `sdv`, `sdmetrics`) out of the core
`dependencies` in `pyproject.toml` into a `synth` extra, guarded by lazy imports
(synthvae's `mmvae` package already models the try/except pattern). Goal: a user of
the **Signatures** or **Prognosis** themes can `pip install pode[signatures]` /
`pode[prognosis]` without pulling a multi-hundred-MB deep-learning stack. The CLI
already degrades gracefully when a theme's imports fail
(`pode/cli/main.py:_mount_theme`), so this is mostly a packaging + import-guard job.

## 3. Config-schema unification
Three config dialects coexist: metacpg's `configs/signatures/schema/config.schema.json`
(versioned JSON Schema), prognostk's `configs/prognosis/config.schema.md`
(markdown contract), and pode's own config. Converge on one schema mechanism.

## 4. R build under one roof
The two R packages (`R/metacpg`, `R/prognostk`) currently build/test on their own
(`R CMD INSTALL`, testthat, renv) separately from the Python wheel. Add a unified
`make`/CI target that installs + tests both engines (Python and R) together, and
decide whether the R packages should be published from this monorepo.

## 5. R plotting consolidation
`R/prognostk` provides publication figures; `R/scripts/*.R` provides the standalone
UMAP/heatmap validation plots used by the synthesis workflows. These overlap and
could share a common R plotting/theme layer.

## 6. Imputation: MNAR modelling and modality-specific encoders
`pode/imputation/` (see `docs/synthesis/IMPUTATION.md`) handles MCAR/MAR and
reports MNAR *sensitivity* via delta adjustment. Parked follow-ups:
- A joint missingness model p(m | x, z) (not-MIWAE style) so the mask is an
  outcome rather than an input, for MNAR-plausible variables.
- Modality-specific encoders/decoders inside `tvae-masked` for wide omics blocks,
  instead of one shared MLP over all columns.
- Marginalising the continuous-column GMM mode in the decoder likelihood (ctgan
  samples one mode per value) for a sharper p(x_obs | z).
