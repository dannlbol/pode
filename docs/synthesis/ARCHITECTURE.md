# Architecture

Two synthesis architectures share this repo. This document explains the design
decisions and how they relate, so you can decide which to run and why.

## 1. The hybrid (recommended, runnable)

```
                 ┌─────────────────────────────────────────────┐
expression (G) ─▶│ HVG select ─▶ Compressor.fit(X, time, event) │─▶ latents (k)
                 └─────────────────────────────────────────────┘
                                                                     │
clinical ───────────────────────────────────────────────┐          │
survival (time,event) ───────────────────────────────┐   │          │
                                                      ▼   ▼          ▼
                                        ┌───────────────────────────────────┐
                                        │  Synthesizer (SDV TVAE) over the   │
                                        │  table [clinical | latents | surv] │
                                        └───────────────────────────────────┘
                                                          │ sample n
                                                          ▼
                              [clinical | latents | surv] ─▶ split
                                                          │
                          latents ─▶ Compressor.inverse_transform ─▶ expression
```

Two swappable interfaces:

* **Compressor** (`compressors/base.py`): `PCACompressor` (baseline, linear, robust
  off-manifold decoder) or `SurvivalVAE` (encoder/decoder + discrete-time survival
  head; the survival term retains prognostic variance a pure-reconstruction latent
  would discard). Both expose `fit/transform/inverse_transform`.
* **Synthesizer** (`synthesizers/base.py`): `SdvTvaeSynthesizer` (your production
  TVAE over the joint table) or `GaussianMixtureFallback` (plumbing stand-in for
  torch/SDV-free runs).

Why this shape:

* **Survival supervision at the compressor** is where it is cheapest to protect
  prognostic signal, because the compressor is the bottleneck that would otherwise
  throw it away. Vanilla VAEs are documented to drift on survival in mid-to-late
  follow-up; the survival head is the fix.
* **TVAE over the latents** reuses your working tabular synthesis and models the
  joint clinical+latent+survival distribution flexibly — better than forcing a
  simple prior to match the aggregate posterior.
* **Linear/decodable latents** (PCA first) generate more reliably than
  reconstruction-optimal ones; always record PCA's numbers as the floor.

### Loss inside `SurvivalVAE`

```
L = w_recon · gaussian_nll(x, x̂) + β(t) · KL_freebits(q(z|x)‖N(0,I)) + w_surv · disc_nll
```

`β` annealed over `kl_warmup_epochs`; **free bits** floor per-dim KL so survival
dims survive; `w_recon`/`w_surv` fixed or learned via homoscedastic uncertainty.
The discrete-time (logistic-hazard) survival head is *generatable* — unlike a Cox
head it lets you sample `(time, event)` — and minibatches cleanly (no risk set).

## 2. Why the hybrid and the joint model are the same thing

A two-stage pipeline (compress → model latents with TVAE) and a joint multimodal
VAE differ only in **where the latent density model lives**:

* Two-stage learns the latent density *explicitly and expressively* — the TVAE is
  the density model over `z`.
* The joint VAE learns it *implicitly* through the prior + KL.

Because a TVAE is a far more flexible density estimator than a unit Gaussian, the
two-stage route can yield better-distributed latents — at the price of not sharing
gradients, so its latent is shaped by survival only if the *compressor* is
survival-supervised (which is exactly why the recommended compressor is the
`SurvivalVAE`). The joint model buys shared gradients and native cross-modal /
missing-modality generation, at the price of a harder, more antagonistic
optimisation (three reconstruction scales + KL to balance).

Practical consequence: start with the hybrid; escalate to the joint model
(`mmvae/`) only when you need cross-modal conditional generation or when the
two-stage coupling measurably loses joint structure.

## 3. The joint multimodal VAE (scaffolding)

```
x_tab  ─▶ TabularEncoder    ─▶ (μ_t, logσ²_t) ┐
                                              ├─▶ Fusion(PoE/MoE/MoPoE, +gating) ─▶ (μ, logσ²)
x_expr ─▶ ExpressionEncoder ─▶ (μ_e, logσ²_e) ┘                                        │
                                                                            reparam ─▶ z
                                          ┌──────────────┬──────────────┬──────────────┘
                                          ▼              ▼              ▼
                                     TabularHead   ExpressionHead   SurvivalHead
                                   (mixed likel.)  (gauss/nb/zinb)  (discrete-time)
```

The options you can explore, all as config switches (`configs/synthesis/mmvae.yaml`):

* **Fusion strategy** — PoE (precision-weighted product; confident modality wins),
  MoE (mixture; better cross-generation), MoPoE (mixture over subset-PoEs; balanced,
  robust to missingness). See `SPECIFICATION.md` §4.
* **Gating** — data-dependent per-modality weights scaling expert precision (PoE)
  or mixture weights (MoE). The lever to stop high-dimensional expression from
  dominating clinical.
* **Expression likelihood** — gaussian (vst/log-CPM) vs NB/ZINB (counts).
* **Weighting scheme** — fixed / uncertainty / gradnorm for balancing the three
  reconstruction scales against the KL.
* **Regularisation magnitude** — `beta_max`, `free_bits`, and `w_survival`, the
  settings that (per the literature) most govern whether survival signal survives.

Status (implemented): encoders, all three fusion strategies, gating, tabular +
expression (gaussian + NB/ZINB) + survival heads, objective assembler
(fixed/uncertainty/GradNorm), model forward, the training loop (`fit`), and
generation (`sample` + `JointVAESynthesizer`). Driven by `scripts/run_mmvae.py` and
compared to the hybrid by `scripts/benchmark.py`; covered by `tests/test_mmvae.py`.
Remaining optional item: per-column tabular embeddings (`SPECIFICATION.md` §3.1 /
`TASKS.md` T9). See `SPECIFICATION.md` §9 and `TASKS.md`.
