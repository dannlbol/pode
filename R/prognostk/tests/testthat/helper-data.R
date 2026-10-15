# Small synthetic cohort generated inline so tests need no external data files.
make_toy <- function(n = 150, seed = 7) {
  set.seed(seed)
  score <- pmin(pmax(rbeta(n, 2, 2), 0.01), 0.99)
  metastasis <- rbinom(n, 1, 0.35)
  strat <- sample(c("A", "B"), n, replace = TRUE)
  lp <- -2.2 * (score - 0.5) + 0.5 * metastasis
  et <- -log(runif(n)) / (0.06 * exp(lp))
  ct <- runif(n, 2, 14)
  data.frame(
    sample_id  = sprintf("S%03d", seq_len(n)),
    OS_Time    = round(pmin(et, ct), 2),
    OS_Status  = as.integer(et <= ct),
    metaCpG    = round(score, 4),
    metastasis = metastasis,
    grp        = strat,
    Sex        = sample(c("M", "F"), n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}

toy_cfg <- function(...) {
  load_config(NULL,
    columns = list(id = "sample_id", time = "OS_Time", status = "OS_Status",
                   score = "metaCpG", group = "risk", covariates = list("metaCpG", "metastasis"),
                   categorical = list("Sex", "grp")),
    ...)
}
