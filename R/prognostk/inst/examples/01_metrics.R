# Example 1: prognostic metrics end-to-end.
# Fit a Cox model, predict 5-year survival with bootstrap CIs, assign risk groups,
# optimise cutoffs, and externally validate a frozen model.

local({
  for (p in c("_setup.R", "r/inst/examples/_setup.R", "inst/examples/_setup.R"))
    if (file.exists(p)) { source(p, local = FALSE); return(invisible()) }
  stop("Cannot find _setup.R; run from the repo root or r/inst/examples/.")
})

# Split into a discovery and validation cohort.
set.seed(1)
idx  <- sample(nrow(cohort), 200)
disc <- cohort[idx, ]
val  <- cohort[-idx, ]

# --- Fit + predict ----------------------------------------------------------
fit <- fit_cox(disc, cfg)                       # formula from cfg$columns$covariates
cat("Cox coefficients:\n"); print(round(coef(fit$fit), 3))

pred5 <- predict_survival_at_time(fit, disc)    # horizon from cfg$survival$horizon
cat(sprintf("\nPredicted 5-yr survival: mean %.3f\n", mean(pred5)))

# --- Bootstrap CIs + risk groups -------------------------------------------
boot <- bootstrap_survival_ci(disc, cfg, n = 200)
disc$risk <- assign_risk_groups(boot$pred_bias_corrected, cfg)
cat("\nRisk-group sizes:\n"); print(table(disc$risk))

# --- Data-driven cutoffs ----------------------------------------------------
oc <- optimise_cutoffs(disc, boot$pred_bias_corrected, cfg,
                       labels = c("Low", "Medium", "High"),
                       fav_min_surv = 0.65, grid = seq(0.3, 0.85, 0.05))
cat(sprintf("\nOptimised cutoffs: %s (score %.1f)\n",
            paste(round(oc$best, 3), collapse = ", "), oc$best_score))

# --- External validation of the frozen model --------------------------------
ev <- external_validate(val, fit, cfg)
cat(sprintf("\nExternal validation: C-index %.3f [%.3f, %.3f], E/O %.3f\n",
            ev$c_index, ev$c_index_lower, ev$c_index_upper, ev$eo_ratio))

cal <- calibration_table(val, predict_survival_at_time(fit, val), cfg, n_bins = 5)
cat(sprintf("Calibration E/O ratio: %.3f\n", cal$eo_ratio))
