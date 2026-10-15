# Example 6: the SAME pipeline on a different dataset via a different config.
# Uses config.template.yaml (neutral column names: time/status/score/group_a) on
# toy_cohort_neutral.csv -- no function code changes, only the config differs.
# This is the generalisability demonstration.
local({
  for (p in c("_setup.R", "r/inst/examples/_setup.R", "inst/examples/_setup.R"))
    if (file.exists(p)) { source(p, local = FALSE); return(invisible()) }
  stop("Cannot find _setup.R; run from the repo root or r/inst/examples/.")
})

ncfg   <- load_config(file.path(CONFIG, "config.template.yaml"),
                      figure = list(outdir = OUT))
ndata  <- read.csv(file.path(DATA, "toy_cohort_neutral.csv"))

fit <- fit_cox(ndata, ncfg, covariates = c("score"))
ndata$risk_group <- assign_risk_groups(predict_survival_at_time(fit, ndata, cfg = ncfg), ncfg)

cat("Neutral-config risk groups:\n"); print(table(ndata$risk_group))

plot_km(ndata, ncfg, title = "Neutral config: survival by risk group",
        filename = "example_neutral_km.pdf")
plot_distribution(ndata, ncfg, group_col = "group_a", type = "violin",
                  title = "Neutral config: score by group_a",
                  filename = "example_neutral_violin.pdf")

cat("Same code, different config -> wrote neutral-config figures to", OUT, "\n")
