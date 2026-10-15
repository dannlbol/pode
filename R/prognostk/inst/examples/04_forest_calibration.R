# Example 4: forest plot of hazard ratios and a calibration curve.
local({
  for (p in c("_setup.R", "r/inst/examples/_setup.R", "inst/examples/_setup.R"))
    if (file.exists(p)) { source(p, local = FALSE); return(invisible()) }
  stop("Cannot find _setup.R; run from the repo root or r/inst/examples/.")
})

set.seed(1)
idx  <- sample(nrow(cohort), 200)
disc <- cohort[idx, ]
val  <- cohort[-idx, ]

# Forest plot comparing discovery vs validation hazard ratios.
plot_forest(list(discovery = disc, validation = val), cfg,
            term_labels = c(metaCpG = "metaCpG score", metastasis = "M+ stage"),
            title = "Hazard ratios: discovery vs validation",
            filename = "example_forest.pdf")

# Calibration of the frozen model on the validation cohort.
fit  <- fit_cox(disc, cfg)
pred <- predict_survival_at_time(fit, val)
plot_calibration(val, cfg, pred = pred, n_bins = 6,
                 title = "Validation calibration",
                 filename = "example_calibration.pdf")

cat("Wrote forest and calibration figures to", OUT, "\n")
