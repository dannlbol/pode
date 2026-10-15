# Specification — Joint Multimodal VAE for survival-preserving synthesis

Status: **implemented** (this document was the contract; it is now satisfied). The
training loop, generation, NB/ZINB count likelihoods, and GradNorm weighting are
all in place — see `scripts/run_mmvae.py`, `scripts/benchmark.py`, and
`tests/test_mmvae.py`. The only remaining item is the optional per-column tabular
embedding encoder (§3.1 / `TASKS.md` T9). Section numbers are referenced from
docstrings in `synthvae/mmvae/`.

---

## 1. Purpose and scope

Learn a single shared latent `z` that jointly generates three modalities for a
patient — clinico-pathological table, gene expression, and time-to-event survival —
such that synthetic cohorts preserve (a) each modality's marginals, (b) the joint
/ cross-modal structure, and (c) survival associations (concordance, hazard ratios,
Kaplan–Meier shape, especially in late follow-up).

This is the "Option 2" alternative to the hybrid. Prefer the hybrid
(`synthvae.pipeline.hybrid`) as the baseline; build this only when the two-stage
coupling demonstrably loses cross-modal structure. See `ARCHITECTURE.md` §2 for the
equivalence argument.

Non-goals: images; more than ~3 modalities (MoPoE's 2^M cost is acceptable at M≤3);
federated/private training (out of scope for the MMVAE; the tabular TVAE has a
federated/DP implementation in `pode federated`, see `docs/federated/FEDERATED.md`).

## 2. Data contract

Inputs per patient, aligned by row:

| Modality    | Symbol   | Shape      | Notes |
|-------------|----------|------------|-------|
| Tabular     | `x_tab`  | (n, D_tab) | clinical; categoricals encoded per §3.1 |
| Expression  | `x_expr` | (n, G)     | HVG-selected; transform per likelihood §3.3 |
| Survival    | `(t, e)` | (n,), (n,) | continuous time + 1/0 event |

Survival is discretized once at fit time into `K` intervals via
`synthvae.survival.discretize.TimeGrid` (quantile cuts on event times by default).
The grid is persisted with the model. Targets `(idx, event)` are produced by
`TimeGrid.index(t)`.

Preprocessing order is fixed (see `data/preprocessing.py`): batch-correction hook →
transform → HVG selection. All fitted state (selected genes, grid, scalers) must be
saved for the train→synthesis handoff.

### 3.1 Tabular encoding

Current `TabularEncoder` assumes the caller has numerically encoded categoricals
(one-hot or ordinal) upstream and passes a dense matrix. **Optional upgrade**: a
per-column embedding front end for high-cardinality categoricals — an
`nn.Embedding` per categorical column, concatenated with the continuous block
before the MLP. The decoder side already handles categoricals as softmax heads
(`heads.TabularHead`).

### 3.3 Expression likelihoods

`ExpressionHead.likelihood ∈ {gaussian, nb, zinb}`.

* **gaussian** (implemented): for vst / log-CPM inputs. Shared learned log-scale.
* **nb** (TODO): for raw counts. Decoder emits `mu = softplus(out)` (optionally
  scaled by a per-cell library size); per-gene dispersion `theta = exp(log_theta)`.
  NLL = `-log NB(x; mu, theta)` (reference: scvi-tools `log_nb_positive`).
* **zinb** (TODO): NB plus a zero-inflation gate `pi = sigmoid(dropout_logits(z))`
  (reference: `log_zinb_positive`).

Whichever is chosen, keep the head's `nll(z, target)` returning a per-sample-scale
scalar so the objective assembler stays likelihood-agnostic.

## 4. Fusion strategies (the option matrix)

Per-modality encoders produce Gaussian posteriors `q(z|x_m)=N(mu_m, diag exp lv_m)`.
Fusion combines them into the joint posterior. Implemented in `experts.py`; select
via `MMVAEConfig.fusion`.

| Strategy | Joint posterior | Strengths | Costs |
|----------|-----------------|-----------|-------|
| **PoE**  | precision-weighted product of experts (+ prior expert) | closed form; confident modality dominates; missing-modality = drop expert | one modality can swamp another (mitigate via gating) |
| **MoE**  | (weighted) mixture, moment-matched to one Gaussian | strong one-way cross-generation | weaker joint coherence |
| **MoPoE**| uniform mixture over PoE of every modality subset | balances PoE/MoE; robust to missing modalities | 2^M subsets |

**Gating** (`experts.GatingNetwork`, toggle `MMVAEConfig.gating`): a small network
maps the concatenated posterior means to one positive scalar per modality. For PoE
these scale each expert's *precision*; for MoE they scale *mixture weights*. This is
the mechanism to adaptively down-weight the high-dimensional expression modality
relative to clinical — the POEMS-style per-omic contribution control.

Requirements for the fusion implementation (met by the current code, keep them):
* PoE must include an optional unit-Gaussian prior expert for stability and to
  define the posterior under missing modalities.
* All math operates on `logvar` for numerical stability; clamp logvars in encoders.
* `encode(..., present=...)` must accept a subset of modalities so a modality can be
  dropped at inference (cross-modal / missing-modality generation).

## 5. Objective and weighting

```
L = Σ_m  w_m · nll_m(z, x_m)            # tabular, expression, survival
  + β(t) · KL( q(z|X) || N(0, I) )       # annealed, free-bits floored
```

`nll_m` come from the heads. KL uses `losses.vae.kl_standard_normal` with **free
bits** (`free_bits` nats/dim floor) — this specifically protects low-variance but
survival-relevant latent dimensions from collapse, which is the reported failure
mode of survival-supervised VAEs. `β` is annealed 0→`beta_max` over
`kl_warmup_epochs` (`objectives.beta_schedule`).

**The weighting `w_m` is the crux.** Empirically, survival preservation is governed
primarily by the *magnitude and structure of latent regularisation and survival
weighting*, not the exact divergence — so sweep `beta_max`, `w_survival`, and
`free_bits` jointly. Three schemes (`objectives.ObjectiveAssembler`):

* **fixed** — constant weights from config. Start here; bias `w_survival` up
  (e.g. 2×) to protect prognostic signal.
* **uncertainty** (default) — Kendall & Gal homoscedastic learned weights.
* ### 5.2 gradnorm (TODO)
  Dynamic balancing from per-task gradient norms on the shared trunk. Implement:
  keep learnable task weights `w_m`; each step compute the gradient norm of each
  `w_m · nll_m` w.r.t. the last shared layer; set a target norm (mean norm × the
  task's relative training rate `r_m^α`); add an L1 loss between actual and target
  norms; renormalise `w_m` to sum to M. Gate behind `weighting_scheme="gradnorm"`.

## 6. Generation

`JointMultimodalVAE.sample(n)`:

1. **Latent source.** Sampling `z ~ N(0, I)` hits aggregate-posterior/prior
   mismatch ("holes"). Provide two modes: (a) prior sampling; (b) *ex-post density*
   — fit a GMM (or normalizing flow) on encoded training `z` and sample from it.
   Mode (b) is the same trick the hybrid uses when TVAE models the latents, and is
   recommended by default.
2. **Decode heads.** Tabular → continuous values + argmax/sampled categories from
   softmax logits; Gaussian molecular heads → mean, historical independent draw,
   or a correlated low-rank residual-factor draw; expression counts → NB/ZINB;
   binary matrices → Bernoulli; survival → sample
   `(time, event)` from the predicted discrete hazards (draw per-interval Bernoulli
   with the head's hazards; first "1" is the event interval, else censor at the
   horizon; map interval → a time via the grid, e.g. interval midpoint).
3. Return aligned modalities as a dataclass mirroring
   `pipeline.hybrid.SynthesisResult`.

## 7. Evaluation

Reuse `synthvae.evaluation`:
* TSTR vs TRTR concordance (train synthetic / test real, Cox on clinical + latents
  or clinical + decoded expression).
* Kaplan–Meier integrated absolute difference — report overall **and** late
  follow-up (`km_iad_late`).
* Hazard-ratio preservation (log-HR correlation + max abs difference) between Cox
  fit on real and on synthetic.
* Latent fidelity (marginal + covariance gaps) as a joint-distribution sanity check.

`pode synthesis mmvae evaluate` adds real-fitted feature-space checks for marginal
variance, shrinkage covariance, PCA spectra, neighbour mixing/coverage, subgroup
dispersion and differential effects, cross-modal covariance/correlation/canonical
spectra, subgroup-conditional source mixing, radial-distribution agreement,
manifold precision/recall, and separate duplicate/DCR/NNDR privacy diagnostics.
Source overlap is checked at 512 and up to 5000 real-variable features. UMAP is a diagnostic
plot and is never a selection metric.

## 8. Config reference

`configs/synthesis/mmvae.yaml` ↔ `MMVAEConfig`. Every option above is a field; the `sweeps`
block lists the recommended experiment grid.

`latent_structure: shared_private` adds a separate private Gaussian posterior
for each molecular modality. Molecular decoders receive `[z_shared, z_private]`;
clinical and survival receive only `z_shared`. `private_latent_dims` resolves an
explicit head name before `default`, and `private_kl_weight` scales the mean KL
over active private heads. The default `shared` architecture is checkpoint- and
state-dictionary-compatible with earlier releases.

## 9. Definition of done

* [ ] `fit()` implemented mirroring `SurvivalVAE.fit` (annealing, free bits,
      chosen weighting, grad clip, early stopping on val objective at `beta_max`).
* [ ] `sample()` implemented with both latent-source modes and survival sampling.
* [ ] NB and ZINB expression likelihoods implemented and unit-tested against a
      counts fixture.
* [ ] GradNorm weighting implemented behind its flag.
* [ ] All three fusion strategies pass a shape/finiteness test with and without
      gating and with a modality dropped.
* [ ] End-to-end run on the toy cohort reproduces or beats the hybrid's
      `concordance_gap` and `km_iad_late` (record both in a results table).
* [ ] Optional: per-column embedding tabular encoder (§3.1).
