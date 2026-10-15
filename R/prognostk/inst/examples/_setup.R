# Shared setup for the prognostk examples.
# Locates the repo, loads the package (installed, or from source via devtools),
# reads the synthetic toy data, and builds a config pointed at example_output/.

# Locate the repo root (contains config/, data/, r/).
.find_repo <- function() {
  for (p in c(".", "..", "../..", "../../..")) {
    if (file.exists(file.path(p, "config", "config.template.yaml")))
      return(normalizePath(p))
  }
  stop("Could not locate repo root (config/config.template.yaml).")
}
REPO   <- .find_repo()
DATA   <- file.path(REPO, "data")
CONFIG <- file.path(REPO, "config")
OUT    <- file.path(REPO, "example_output")
dir.create(OUT, showWarnings = FALSE)

suppressMessages({
  if (requireNamespace("prognostk", quietly = TRUE)) {
    library(prognostk)
  } else if (requireNamespace("devtools", quietly = TRUE)) {
    devtools::load_all(file.path(REPO, "r"), quiet = TRUE)
  } else {
    for (f in list.files(file.path(REPO, "r", "R"), full.names = TRUE)) source(f)
  }
})

cohort <- read.csv(file.path(DATA, "toy_cohort.csv"))
meth   <- read.csv(file.path(DATA, "toy_methylation.csv"), check.names = FALSE)
cfg    <- load_config(file.path(CONFIG, "config.example.yaml"),
                      figure = list(outdir = OUT))
