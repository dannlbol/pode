# Example 5: oncoplot -- annotated methylation heatmap.
# The annotation tracks, colours and ordering all come from cfg$oncoplot / palettes.
local({
  for (p in c("_setup.R", "r/inst/examples/_setup.R", "inst/examples/_setup.R"))
    if (file.exists(p)) { source(p, local = FALSE); return(invisible()) }
  stop("Cannot find _setup.R; run from the repo root or r/inst/examples/.")
})

plot_oncoplot(cohort, cfg, matrix = meth,
              title = "Toy medulloblastoma oncoplot",
              filename = "example_oncoplot.pdf")

cat("Wrote oncoplot to", OUT, "\n")
