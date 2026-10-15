# Example 2: survival visualisations -- KM curves and per-patient curves.
local({
  for (p in c("_setup.R", "r/inst/examples/_setup.R", "inst/examples/_setup.R"))
    if (file.exists(p)) { source(p, local = FALSE); return(invisible()) }
  stop("Cannot find _setup.R; run from the repo root or r/inst/examples/.")
})

# Assign risk groups (the KM plot groups by cfg$columns$group = "riskGrpBoot").
fit <- fit_cox(cohort, cfg)
cohort$riskGrpBoot <- assign_risk_groups(predict_survival_at_time(fit, cohort), cfg)

plot_km(cohort, cfg, title = "Overall survival by risk group",
        filename = "example_km.pdf")

plot_individual_curves(fit, cohort, cfg,
                       title = "Per-patient predicted survival",
                       filename = "example_individual_curves.pdf")

cat("Wrote KM and individual-curve figures to", OUT, "\n")
