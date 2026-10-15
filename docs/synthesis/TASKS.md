# Task backlog (for Claude Code / Codex)

Ordered, self-contained tasks. Each names the files to touch, the reference
implementation to mirror, and an acceptance check. Spec sections in
`SPECIFICATION.md`. Do the hybrid hardening first (T1–T2) if you want a stronger
baseline, then build out the joint model (T3–T8).

**Status:** T3–T7 (training loop, generation, NB/ZINB, GradNorm, fusion/gating test
matrix) are **done**; T8 (benchmark) is delivered as `scripts/benchmark.py`. T9
(tabular embeddings) and T10 (batch correction) remain optional/open.

Conventions: keep the torch import guards; keep public interfaces
(`Compressor`, `Synthesizer`) stable; add tests under `tests/`; run
`python -m pytest tests/ -q` after each task.

---

## T1 — Runtime-verify the torch + SDV paths
- **Files**: `compressors/survival_vae.py`, `synthesizers/sdv_tvae.py`, `tests/`.
- **Do**: install `.[all]`; run `scripts/run_hybrid.py --compressor survival_vae
  --synthesizer sdv` on the toy cohort; fix any version-specific SDV metadata
  issues (the one spot is `SdvTvaeSynthesizer._build_metadata`).
- **Done**: end-to-end run prints a report; add a `@pytest.mark.skipif(no torch)`
  test that fits `SurvivalVAE` for 2 epochs and checks latent shape + finite loss.

## T2 — Hybrid: keep a few raw genes alongside latents
- **Files**: `pipeline/latents.py`, `pipeline/hybrid.py`, `configs/synthesis/hybrid.yaml`.
- **Do**: allow a configurable list of "passthrough" genes included as raw columns
  in the synthesized table (so individually-important genes are modelled directly,
  not only through the latent). Decode: concat passthrough genes with
  inverse-transformed latents.
- **Done**: `--passthrough-genes gene_5 gene_12` round-trips; report unchanged API.

## T3 — MMVAE training loop  (SPEC §5, §9)
- **Files**: `mmvae/model.py` (`fit`).
- **Mirror**: `compressors/survival_vae.py::SurvivalVAE.fit`.
- **Do**: Adam over `model.parameters()`; per-epoch `beta_schedule`; inject `beta`
  into each batch; grad clip; validation objective at `beta_max`; early stopping;
  restore best state. Build a `Dataset`/`DataLoader` yielding
  `{x_tab, x_expr, tab_target, surv_target}`.
- **Done**: trains on the toy cohort; loss decreases; a smoke test runs 2 epochs.

## T4 — MMVAE generation  (SPEC §6)
- **Files**: `mmvae/model.py` (`sample`), new `mmvae/generate.py` if useful.
- **Do**: implement both latent sources (prior; GMM on encoded train `z`); decode
  all heads; sample `(time, event)` from discrete hazards; return a
  `SynthesisResult`-like dataclass.
- **Done**: `sample(n)` returns aligned modalities; feeds `evaluation.report.evaluate`.

## T5 — Count likelihoods  (SPEC §3.3)
- **Files**: `mmvae/heads.py` (`ExpressionHead.nll`).
- **Do**: implement NB and ZINB NLLs (reference: scvi-tools `log_nb_positive`,
  `log_zinb_positive`). Add optional per-cell library-size scaling.
- **Done**: unit test on a synthetic counts fixture; NLL finite and lower for the
  matching likelihood than for a mismatched one.

## T6 — GradNorm weighting  (SPEC §5.2)
- **Files**: `mmvae/objectives.py` (`ObjectiveAssembler`, `scheme="gradnorm"`).
- **Do**: implement dynamic gradient-norm balancing across the three heads on the
  shared trunk's last layer; renormalise task weights to sum to M.
- **Done**: toggling `weighting_scheme` changes the logged per-task weights over
  training; run completes.

## T7 — Fusion / gating test matrix  (SPEC §4)
- **Files**: `tests/test_mmvae.py`.
- **Do**: parametrise over `fusion ∈ {poe, moe, mopoe} × gating ∈ {on, off}` and
  over dropping one modality; assert joint `(mu, logvar)` shapes and finiteness.
- **Done**: matrix green.

## T8 — Benchmark hybrid vs joint
- **Files**: new `scripts/benchmark.py`, `config/*.yaml` sweeps.
- **Do**: run both architectures across the sweep grids on the toy cohort (and your
  real cohort); collect `concordance_gap`, `km_iad`, `km_iad_late`, `hr_logcorr`,
  latent fidelity into a results table.
- **Done**: a CSV + short markdown summary; note which architecture wins on
  late-follow-up KM and hazard-ratio preservation.

## T9 (optional) — Tabular embeddings  (SPEC §3.1)
- **Files**: `mmvae/encoders.py` (`TabularEncoder`).
- **Do**: per-column `nn.Embedding` for high-cardinality categoricals; concat with
  continuous block.

## T10 (optional) — Batch correction
- **Files**: `data/preprocessing.py` (`combat_hook`).
- **Do**: wire in pyComBat / scanpy ComBat before HVG selection when batch labels
  are supplied.
