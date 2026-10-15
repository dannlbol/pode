# =============================================================================
# generate_toy_data.R
# Creates a fully synthetic, shareable "medulloblastoma-like" cohort used by the
# prognostk examples and tests. No real patient data is involved: survival times
# are simulated from a Cox-like model where a continuous score drives risk.
#
# Writes (next to this script):
#   toy_cohort.csv       one row per sample: id, time, status, score, markers, risk group
#   toy_methylation.csv  CpG (rows) x sample (cols) beta-value matrix, samples in cohort order
#
# Usage:  Rscript data/generate_toy_data.R
# =============================================================================

set.seed(2026)

out_dir <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  flag <- grep("^--file=", args, value = TRUE)
  if (length(flag)) dirname(normalizePath(sub("^--file=", "", flag))) else getwd()
}, error = function(e) getwd())

n <- 300L

# --- Continuous prognostic score (bimodal-ish, 0..1) -------------------------
score <- pmin(pmax(rbeta(n, 2, 2) * 0.9 + rnorm(n, 0, 0.03), 0.01), 0.99)

# --- Categorical clinical markers --------------------------------------------
sex        <- sample(c("M", "F"), n, replace = TRUE, prob = c(0.55, 0.45))
infant     <- rbinom(n, 1, 0.20)
metastasis <- rbinom(n, 1, 0.35)                         # M+ disease
histology  <- sample(c("CLA", "LCA", "DN", "MBEN"), n, replace = TRUE,
                     prob = c(0.55, 0.20, 0.15, 0.10))
myc        <- rbinom(n, 1, 0.12)                         # MYC amplification
mycn       <- rbinom(n, 1, 0.10)                         # MYCN amplification
methgrp    <- sample(c("Grp3", "Grp4"), n, replace = TRUE, prob = c(0.4, 0.6))
subgroup   <- sample(c("I","II","III","IV","V","VI","VII","VIII"), n, replace = TRUE)
age        <- round(pmin(pmax(rgamma(n, shape = 3, scale = 2.2), 0.2), 16), 1)

# --- Simulate survival from a Cox-like linear predictor ----------------------
# Higher score = better outcome, so it enters with a negative coefficient.
lp <- -2.4 * (score - 0.5) + 0.55 * metastasis + 0.6 * myc + 0.4 * mycn +
      0.25 * (histology == "LCA")
baseline_rate <- 0.06
u <- runif(n)
event_time <- -log(u) / (baseline_rate * exp(lp))         # exponential survival
cens_time  <- runif(n, 2, 14)                             # administrative censoring
OS_Time    <- pmin(event_time, cens_time)
OS_Status  <- as.integer(event_time <= cens_time)
OS_Time    <- round(pmin(OS_Time, 15), 2)

cohort <- data.frame(
  sample_id     = sprintf("TOY%03d", seq_len(n)),
  OS_Time       = OS_Time,
  OS_Status     = OS_Status,
  metaCpG       = round(score, 4),
  Age           = age,
  Sex           = sex,
  Infant        = infant,
  metastasis    = metastasis,
  Histology     = histology,
  ConsensusMYC  = myc,
  ConsensusMYCN = mycn,
  MethGroup   = methgrp,
  MNP_Subgroup  = subgroup,
  stringsAsFactors = FALSE
)

# --- A neutral-column copy of the same data (for the config.template example) -
# Demonstrates that the identical pipeline runs on differently-named columns.
neutral <- data.frame(
  sample_id = cohort$sample_id,
  time      = cohort$OS_Time,
  status    = cohort$OS_Status,
  score     = cohort$metaCpG,
  group_a   = cohort$MethGroup,
  group_b   = cohort$Histology,
  stringsAsFactors = FALSE
)

# --- Synthetic CpG methylation matrix (correlated with the score) ------------
n_cpg <- 19L
cpg_names <- sprintf("cg%08d", sample(1e7, n_cpg))
# Each CpG's beta value tracks the score with its own weight and noise.
weights <- runif(n_cpg, -1, 1)
meth <- matrix(0, nrow = n_cpg, ncol = n)
for (i in seq_len(n_cpg)) {
  centre <- plogis((score - 0.5) * 6 * weights[i])
  meth[i, ] <- pmin(pmax(centre + rnorm(n, 0, 0.08), 0.001), 0.999)
}
meth <- round(meth, 4)
meth_df <- data.frame(CpG = cpg_names, meth, check.names = FALSE)
colnames(meth_df) <- c("CpG", cohort$sample_id)

write.csv(cohort,  file.path(out_dir, "toy_cohort.csv"),        row.names = FALSE)
write.csv(neutral, file.path(out_dir, "toy_cohort_neutral.csv"), row.names = FALSE)
write.csv(meth_df, file.path(out_dir, "toy_methylation.csv"),   row.names = FALSE)

cat(sprintf("Wrote toy data (%d samples, %d CpGs, %d events) to %s\n",
            n, n_cpg, sum(OS_Status), out_dir))
