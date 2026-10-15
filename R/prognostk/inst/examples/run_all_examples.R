# Run every prognostk example script in order.
# Usage (from repo root):  Rscript r/inst/examples/run_all_examples.R
here <- tryCatch(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE))), error = function(e) ".")
if (length(here) == 0 || !nzchar(here)) here <- "r/inst/examples"

scripts <- sort(list.files(here, pattern = "^[0-9].*\\.R$", full.names = TRUE))
for (s in scripts) {
  cat("\n==== ", basename(s), " ====\n", sep = "")
  source(s, local = new.env())
}
cat("\nAll examples completed.\n")
