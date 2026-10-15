# Synthesis theme: end-to-end workflows

These are the copy-paste pipelines that bridge the **synthvae** omics front end
(compressors / joint multimodal VAE) with the **PODE** downstream TVAE. Everything
runs from the repo root with a single virtual environment, through the unified
`pode` command — the synthvae capabilities are the `pode synthesis …` subcommands
(they used to be `python scripts/*.py`; those script paths still work as thin
deprecated shims).

Assumptions:

- You run from the repo root (`~/pode`) with the project installed
  (`pip install -e .[all]`).
- Your real cohort CSVs live under `data/` (gitignored):
  `data/clinical.csv`, `data/expression.csv`, `data/methylation.csv`,
  `data/survival.csv`, one row per sample in the same order (an optional leading
  `sample_id` column is stripped on load).
- Pipeline outputs go under `runs/` and the various `trained_*/`, `tuned_*/`,
  `generated_*/` directories (all gitignored).
- The example clinical schema used below is
  `--continuous-cols Age` and
  `--categorical-cols Sex M._versus_M. ConsensusMYC WCA MNP_Subgroup`.
  Adjust these to your own columns.

The commands referenced:

| Command | Role |
| --- | --- |
| `pode synthesis hybrid` | two-stage hybrid: compressor + synthesizer; `run` / `tune` / `export-latents` / `sample-decode` |
| `pode synthesis mmvae` | joint multimodal VAE: `run` / `tune` / `train` / `generate` |
| `pode synthesis bundle` | concatenate one or more handoff bundles into a PODE training table + starter config |
| `pode synthesis promote` | turn a PODE tuning result into a final trainable config |
| `pode synthesis decode` | split a PODE-generated table and decode latent blocks back to full matrices |
| `pode synthesis impute` | generative multiple imputation with TVAE + masking benchmark (see [`IMPUTATION.md`](IMPUTATION.md)) |
| `Rscript R/scripts/plot_real_vs_synthetic_umap.R` | UMAP / heatmap validation for expression |
| `Rscript R/scripts/plot_real_vs_synthetic_methylation_umap.R` | UMAP / heatmap validation for methylation |

---

## A. Expression hybrid → PODE TVAE

```bash
# 1. Tune the hybrid compressor+synthesizer on expression.
pode synthesis hybrid tune \
    --sweep configs/synthesis/hybrid_sweep.yaml \
    --clinical data/clinical.csv \
    --expression data/expression.csv \
    --survival data/survival.csv \
    --out-dir runs/tun

# 2. Export the selected expression latent handoff bundle.
#    --include-input copies the original expression matrix into the bundle.
pode synthesis hybrid export-latents \
    --config runs/tun/selected_config.json \
    --clinical data/clinical.csv \
    --expression data/expression.csv \
    --survival data/survival.csv \
    --out-dir runs/final_handoff \
    --include-input

# 3. (Optional) PCA-16 baseline handoff + evaluation for comparison.
pode synthesis hybrid export-latents \
    --compressor pca --n-latent 16 --n-top-genes 0 \
    --clinical data/clinical.csv --expression data/expression.csv --survival data/survival.csv \
    --out-dir runs/final_handoff_pca16
pode synthesis hybrid run \
    --compressor pca --synthesizer gmm --n-latent 16 --n-top-genes 0 \
    --clinical data/clinical.csv --expression data/expression.csv --survival data/survival.csv \
    --out-dir runs/pca16_eval

# 4. (Optional) decode a sampled latent table back to expression, purely inside synthvae.
pode synthesis hybrid sample-decode \
    --bundle-dir runs/final_handoff \
    --latents sampled_latents.csv \
    --out-dir runs/synthetic_from_external_tvae

# 5. Prepare the bundle as one wide PODE training table + starter config.
pode synthesis bundle \
    --bundle-dir runs/final_handoff \
    --out-dir runs/final_handoff/pode_prep \
    --continuous-cols Age \
    --categorical-cols Sex M._versus_M. ConsensusMYC WCA MNP_Subgroup

# 6. Train an initial PODE model.
pode synthesis train \
    --data runs/final_handoff/pode_prep/training_table.csv \
    --config runs/final_handoff/pode_prep/pode_config.yaml \
    --output-dir ./trained_hybrid --seed 42

# 7. Tune the downstream PODE TVAE, then promote + retrain the winner.
pode synthesis tune \
    --data runs/final_handoff/pode_prep/training_table.csv \
    --config runs/final_handoff/pode_prep/pode_config.yaml \
    --output-dir ./tuned_hybrid
pode synthesis promote \
    --base-config runs/final_handoff/pode_prep/pode_config.yaml \
    --best-config ./tuned_hybrid/best_configuration.json \
    --out-config ./tuned_hybrid/pode_config_final.yaml \
    --train-epochs 1500
pode synthesis train \
    --data runs/final_handoff/pode_prep/training_table.csv \
    --config ./tuned_hybrid/pode_config_final.yaml \
    --output-dir ./trained_hybrid_tuned --seed 42

# 8. Generate synthetic rows.
pode synthesis generate \
    --model ./trained_hybrid_tuned/model.pkl \
    --schema ./trained_hybrid_tuned/schema.yaml \
    --num-samples 301 --output-dir ./generated_hybrid_tuned --seed 42

# 9. Split the generated table and decode the latent block back to expression.
pode synthesis decode \
    --bundle-dir runs/final_handoff \
    --pode-table ./generated_hybrid_tuned/synthetic_data.csv \
    --out-dir ./generated_hybrid_tuned/decoded

# 10. Validate real vs synthetic expression.
Rscript R/scripts/plot_real_vs_synthetic_umap.R \
    --real data/expression.csv \
    --synthetic ./generated_hybrid_tuned/decoded/synthetic_expression.csv \
    --real-clinical data/clinical.csv \
    --synthetic-clinical ./generated_hybrid_tuned/decoded/synthetic_clinical.csv
```

---

## B. Methylation hybrid → PODE TVAE

Identical to workflow A, swapping `--expression data/expression.csv` for
`--methylation data/methylation.csv` and using `_methylation` output dirs.

```bash
pode synthesis hybrid tune \
    --sweep configs/synthesis/hybrid_sweep.yaml \
    --clinical data/clinical.csv --methylation data/methylation.csv --survival data/survival.csv \
    --out-dir runs/tun_methylation

pode synthesis hybrid export-latents \
    --config runs/tun_methylation/selected_config.json \
    --clinical data/clinical.csv --methylation data/methylation.csv --survival data/survival.csv \
    --out-dir runs/final_handoff_methylation --include-input

pode synthesis bundle \
    --bundle-dir runs/final_handoff_methylation \
    --out-dir runs/final_handoff_methylation/pode_prep \
    --continuous-cols Age \
    --categorical-cols Sex M._versus_M. ConsensusMYC WCA MNP_Subgroup

pode synthesis train  --data runs/final_handoff_methylation/pode_prep/training_table.csv \
    --config runs/final_handoff_methylation/pode_prep/pode_config.yaml \
    --output-dir ./trained_hybrid_methylation --seed 42
pode synthesis tune   --data runs/final_handoff_methylation/pode_prep/training_table.csv \
    --config runs/final_handoff_methylation/pode_prep/pode_config.yaml \
    --output-dir ./tuned_hybrid_methylation
pode synthesis promote \
    --base-config runs/final_handoff_methylation/pode_prep/pode_config.yaml \
    --best-config ./tuned_hybrid_methylation/best_configuration.json \
    --out-config ./tuned_hybrid_methylation/pode_config_final.yaml --train-epochs 1500
pode synthesis train  --data runs/final_handoff_methylation/pode_prep/training_table.csv \
    --config ./tuned_hybrid_methylation/pode_config_final.yaml \
    --output-dir ./trained_hybrid_methylation_tuned --seed 42

pode synthesis generate \
    --model ./trained_hybrid_methylation_tuned/model.pkl \
    --schema ./trained_hybrid_methylation_tuned/schema.yaml \
    --num-samples 301 --output-dir ./generated_hybrid_methylation_tuned --seed 42

pode synthesis decode \
    --bundle-dir runs/final_handoff_methylation \
    --pode-table ./generated_hybrid_methylation_tuned/synthetic_data.csv \
    --out-dir ./generated_hybrid_methylation_tuned/decoded

Rscript R/scripts/plot_real_vs_synthetic_methylation_umap.R \
    --real data/methylation.csv \
    --synthetic ./generated_hybrid_methylation_tuned/decoded/synthetic_methylation.csv \
    --real-clinical data/clinical.csv \
    --synthetic-clinical ./generated_hybrid_methylation_tuned/decoded/synthetic_clinical.csv
```

---

## C. Joint expression + methylation → a single PODE TVAE

Reuses the two bundles from A and B (`runs/final_handoff` and
`runs/final_handoff_methylation`) and hands **both** latent blocks to one PODE
model, so the TVAE learns the joint expr↔meth (and clinical/survival) structure
rather than two independent models. `prepare_hybrid_bundle.py` and
`decode_hybrid_samples.py` accept multiple `--bundle-dir` arguments.

```bash
# Concatenate both latent blocks into one PODE-ready training table.
pode synthesis bundle \
    --bundle-dir runs/final_handoff runs/final_handoff_methylation \
    --out-dir runs/final_handoff_joint/pode_prep \
    --continuous-cols Age \
    --categorical-cols Sex M._versus_M. ConsensusMYC WCA MNP_Subgroup

# Train / tune / promote / retrain ONE PODE model on the combined table.
pode synthesis train  --data runs/final_handoff_joint/pode_prep/training_table.csv \
    --config runs/final_handoff_joint/pode_prep/pode_config.yaml \
    --output-dir ./trained_hybrid_joint --seed 42
pode synthesis tune   --data runs/final_handoff_joint/pode_prep/training_table.csv \
    --config runs/final_handoff_joint/pode_prep/pode_config.yaml \
    --output-dir ./tuned_hybrid_joint
pode synthesis promote \
    --base-config runs/final_handoff_joint/pode_prep/pode_config.yaml \
    --best-config ./tuned_hybrid_joint/best_configuration.json \
    --out-config ./tuned_hybrid_joint/pode_config_final.yaml --train-epochs 1500
pode synthesis train  --data runs/final_handoff_joint/pode_prep/training_table.csv \
    --config ./tuned_hybrid_joint/pode_config_final.yaml \
    --output-dir ./trained_hybrid_joint_tuned --seed 42

pode synthesis generate \
    --model ./trained_hybrid_joint_tuned/model.pkl \
    --schema ./trained_hybrid_joint_tuned/schema.yaml \
    --num-samples 301 --output-dir ./generated_hybrid_joint_tuned --seed 42

# Decode BOTH blocks: writes synthetic_expression.csv AND synthetic_methylation.csv.
pode synthesis decode \
    --bundle-dir runs/final_handoff runs/final_handoff_methylation \
    --pode-table ./generated_hybrid_joint_tuned/synthetic_data.csv \
    --out-dir ./generated_hybrid_joint_tuned/decoded
```

---

## D. Joint multimodal VAE (mmvae) — expression only

The mmvae path replaces the compressor + external TVAE with one end-to-end joint
model; no PODE handoff is needed.

```bash
pode synthesis mmvae tune --config configs/synthesis/mmvae.yaml \
    --sweep configs/synthesis/mmvae_sweep.yaml \
    --clinical data/clinical.csv --expression data/expression.csv --survival data/survival.csv \
    --out-dir runs/mmvae_tune

pode synthesis mmvae train --config runs/mmvae_tune/selected_config.json \
    --clinical data/clinical.csv --expression data/expression.csv --survival data/survival.csv \
    --out-dir runs/mmvae_final

pode synthesis mmvae generate --model runs/mmvae_final/model.pt --n 200 --out-dir runs/mmvae_synth

Rscript R/scripts/plot_real_vs_synthetic_umap.R \
    --real data/expression.csv \
    --synthetic runs/mmvae_synth/synthetic_expression.csv \
    --real-clinical data/clinical.csv \
    --synthetic-clinical runs/mmvae_synth/synthetic_clinical.csv
```

---

## E. Joint multimodal VAE (mmvae) — expression + methylation

```bash
# Single end-to-end run (both molecular modalities).
pode synthesis mmvae run \
    --clinical data/clinical.csv --expression data/expression.csv \
    --methylation data/methylation.csv --survival data/survival.csv \
    --out runs/mmvae_joint

# Tune, then train the winner on all data (methylation head included).
pode synthesis mmvae tune --config configs/synthesis/mmvae.yaml \
    --sweep configs/synthesis/mmvae_sweep.yaml \
    --clinical data/clinical.csv --expression data/expression.csv \
    --methylation data/methylation.csv --survival data/survival.csv \
    --out-dir runs/mmvae_meth_tune
pode synthesis mmvae train --config runs/mmvae_meth_tune/selected_config.json \
    --clinical data/clinical.csv --expression data/expression.csv \
    --methylation data/methylation.csv --survival data/survival.csv \
    --out-dir runs/mmvae_meth_final

# Generate — writes synthetic_methylation.csv automatically.
pode synthesis mmvae generate --model runs/mmvae_meth_final/model.pt \
    --n 1000 --out-dir runs/mmvae_meth_synth

# Quantify and compare final joint/hybrid outputs in feature space.
pode synthesis mmvae evaluate \
    --clinical data/clinical.csv --expression data/expression.csv \
    --methylation data/methylation.csv --survival data/survival.csv \
    --synthetic joint=runs/mmvae_meth_synth \
    --synthetic hybrid=generated_hybrid_joint_tuned/decoded \
    --out-dir runs/mmvae_quality

Rscript R/scripts/plot_real_vs_synthetic_umap.R \
    --real data/expression.csv \
    --synthetic runs/mmvae_meth_synth/synthetic_expression.csv \
    --real-clinical data/clinical.csv \
    --synthetic-clinical runs/mmvae_meth_synth/synthetic_clinical.csv

Rscript R/scripts/plot_real_vs_synthetic_methylation_umap.R \
    --real data/methylation.csv \
    --synthetic runs/mmvae_meth_synth/synthetic_methylation.csv \
    --real-clinical data/clinical.csv \
    --synthetic-clinical runs/mmvae_meth_synth/synthetic_clinical.csv
```

---

## F. Shared/private MMVAE experiment with rollback

This opt-in workflow writes only new directories. The v2 and hybrid cohorts
remain available if the candidate fails quantitative or visual review.

**Status:** discontinued after failed broad and low-noise calibration reviews.
These commands are retained for provenance only. Use the hybrid workflow for
production; do not launch another residual-factor calibration sweep.

```bash
# Broad four-stage search. Add --resume to continue an interrupted run.
pode synthesis mmvae tune --config configs/synthesis/mmvae.yaml \
    --sweep configs/synthesis/mmvae_shared_private_sweep.yaml \
    --clinical data/clinical.csv --expression data/expression.csv \
    --methylation data/methylation.csv --survival data/survival.csv \
    --out-dir runs/mmvae_shared_private_tune

pode synthesis mmvae train \
    --config runs/mmvae_shared_private_tune/selected_config.json \
    --clinical data/clinical.csv --expression data/expression.csv \
    --methylation data/methylation.csv --survival data/survival.csv \
    --out-dir runs/mmvae_shared_private_final

pode synthesis mmvae generate \
    --model runs/mmvae_shared_private_final/model.pt \
    --n 1000 --out-dir runs/mmvae_shared_private_synth

# Quantitative promotion gate. `joint_v2` supplies the survival baseline.
pode synthesis mmvae evaluate \
    --clinical data/clinical.csv --expression data/expression.csv \
    --methylation data/methylation.csv --survival data/survival.csv \
    --synthetic joint=runs/mmvae_shared_private_synth \
    --synthetic joint_v2=runs/mmvae_joint_v2_synth \
    --synthetic hybrid=generated_hybrid_joint_tuned/decoded \
    --synthetic expression_only=runs/mmvae_expression_synth \
    --joint-baseline joint_v2 --out-dir runs/mmvae_shared_private_quality

# Both fixed-real and combined embeddings are required for visual sign-off.
Rscript R/scripts/plot_real_vs_synthetic_umap.R \
    --real data/expression.csv \
    --synthetic runs/mmvae_shared_private_synth/synthetic_expression.csv \
    --real-clinical data/clinical.csv \
    --synthetic-clinical runs/mmvae_shared_private_synth/synthetic_clinical.csv \
    --embedding-mode both --output-dir runs/mmvae_shared_private_plots

Rscript R/scripts/plot_real_vs_synthetic_methylation_umap.R \
    --real data/methylation.csv \
    --synthetic runs/mmvae_shared_private_synth/synthetic_methylation.csv \
    --real-clinical data/clinical.csv \
    --synthetic-clinical runs/mmvae_shared_private_synth/synthetic_clinical.csv \
    --embedding-mode both --output-dir runs/mmvae_shared_private_plots
```

Promote only after `PASS -- visual review required` and after confirming that
neither embedding shows source-specific bands, rings, or shells. Otherwise keep
the hybrid output. The code-level rollback is `dims.latent_structure: shared`.

---

## G. Low-noise calibration after a failed shared/private review

Use this only after the broad shared/private sweep has completed and its
candidate failed because of boundary bands or shells. It consumes the broad
winner and holds its neural architecture fixed. Outputs use new directories.

**Historical failed experiment:** an earlier run worsened boundary penalties,
manifold recall, and within-subgroup dispersion and has been discontinued. Do
not repeat this workflow unless a new observation-sampling design explicitly
supersedes the residual-factor approach.

```bash
pode synthesis mmvae tune \
    --config runs/mmvae_shared_private_tune/selected_config.json \
    --sweep configs/synthesis/mmvae_shared_private_calibration_sweep.yaml \
    --clinical data/clinical.csv --expression data/expression.csv \
    --methylation data/methylation.csv --survival data/survival.csv \
    --out-dir runs/mmvae_shared_private_calibration_tune

pode synthesis mmvae train \
    --config runs/mmvae_shared_private_calibration_tune/selected_config.json \
    --clinical data/clinical.csv --expression data/expression.csv \
    --methylation data/methylation.csv --survival data/survival.csv \
    --out-dir runs/mmvae_shared_private_calibration_final

pode synthesis mmvae generate \
    --model runs/mmvae_shared_private_calibration_final/model.pt \
    --n 1000 --out-dir runs/mmvae_shared_private_calibration_synth

pode synthesis mmvae evaluate \
    --clinical data/clinical.csv --expression data/expression.csv \
    --methylation data/methylation.csv --survival data/survival.csv \
    --synthetic joint=runs/mmvae_shared_private_calibration_synth \
    --synthetic shared_private_uncalibrated=runs/mmvae_shared_private_synth \
    --synthetic joint_v2=runs/mmvae_joint_v2_synth \
    --synthetic hybrid=generated_hybrid_joint_tuned/decoded \
    --synthetic expression_only=runs/mmvae_expression_synth \
    --joint-baseline joint_v2 \
    --out-dir runs/mmvae_shared_private_calibration_quality
```

Repeat both R plotting commands from workflow F, changing the synthetic input to
`runs/mmvae_shared_private_calibration_synth` and the output directory to
`runs/mmvae_shared_private_calibration_plots`. If the new quantitative or visual
review fails, keep the hybrid model; do not restore rank-32/full-scale residual
noise simply to force the marginal variance ratio toward one.

---

## H. Generative imputation with a PODE TVAE

Benchmark imputation engines by hiding known values in a complete training
table, then impute a real incomplete cohort. Engines, metrics and leakage rules
are described in [`IMPUTATION.md`](IMPUTATION.md).

```bash
# 1. Benchmark (add --quick for a smoke run, --engines to subset).
pode synthesis impute benchmark \
    --data runs/final_handoff/pode_prep/training_table.csv \
    --config runs/final_handoff/pode_prep/pode_config.yaml \
    --impute-config configs/imputation/benchmark.example.yaml \
    --output-dir runs/impute_bench_joint

# 2a. Impute with an existing PODE model (decoder-only conditioning).
pode synthesis impute apply \
    --model trained_hybrid_tuned/model.pkl \
    --data incomplete_training_table.csv --m 20 \
    --output-dir runs/imputed_decoder --seed 42

# 2b. Or train a mask-aware TVAE on the incomplete cohort itself, then apply it.
pode synthesis impute train --data incomplete_training_table.csv \
    --config pode_config.yaml --impute-config configs/imputation/benchmark.example.yaml \
    --output-dir runs/masked_tvae --seed 42
pode synthesis impute apply --model runs/masked_tvae \
    --data incomplete_training_table.csv --m 20 --output-dir runs/imputed_masked --seed 42

# 2c. Or fit an engine directly: hybrid = MICE for gaps + mask-aware TVAE for absent blocks.
#     For prediction work, fit on the training cohort and keep test outcomes hidden.
pode synthesis impute apply --engine hybrid --reference train_table.csv \
    --data test_table.csv --config pode_config.yaml \
    --impute-config configs/imputation/benchmark.example.yaml \
    --outcome-policy hide --fit-scope reference --m 20 --output-dir runs/imputed_hybrid

# 3. (Optional) decode imputed latent blocks back to expression / methylation.
pode synthesis decode --bundle-dir final_handoff_joint \
    --pode-table runs/imputed_masked/imputed_001.csv --out-dir runs/imputed_masked/decoded_001
```
