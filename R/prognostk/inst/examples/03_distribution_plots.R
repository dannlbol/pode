# Example 3: distribution plots of the score across groups.
local({
  for (p in c("_setup.R", "r/inst/examples/_setup.R", "inst/examples/_setup.R"))
    if (file.exists(p)) { source(p, local = FALSE); return(invisible()) }
  stop("Cannot find _setup.R; run from the repo root or r/inst/examples/.")
})

# Violin of metaCpG by methylation group, with pairwise brackets.
plot_distribution(cohort, cfg, group_col = "MethGroup", type = "violin",
                  pairwise = TRUE, title = "metaCpG by methylation group",
                  filename = "example_violin.pdf")

# Ridge of metaCpG by molecular subgroup.
plot_distribution(cohort, cfg, group_col = "MNP_Subgroup", type = "ridge",
                  title = "metaCpG by subgroup", filename = "example_ridge.pdf")

# Boxplot by histology.
plot_distribution(cohort, cfg, group_col = "Histology", type = "box",
                  title = "metaCpG by histology", filename = "example_box.pdf")

cat("Wrote distribution figures to", OUT, "\n")
