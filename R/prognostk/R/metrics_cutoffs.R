# =============================================================================
# Data-driven cutoff optimisation.
# A penalised grid search over risk thresholds, generalised from earlier
# study-specific cutoff-optimisation scripts.
# =============================================================================

# Kaplan-Meier survival estimate at `time` for a set of subjects.
.km_surv_at <- function(time_vec, status_vec, at) {
  if (length(time_vec) < 2 || sum(status_vec, na.rm = TRUE) == 0) return(NA_real_)
  sf <- survival::survfit(survival::Surv(time_vec, status_vec) ~ 1)
  s <- summary(sf, times = at, extend = TRUE)$surv
  if (length(s) == 0) NA_real_ else s[1]
}

# Score one candidate set of (descending) cutoffs. Returns -Inf for infeasible.
.score_cutoffs <- function(cuts, pred, time_vec, status_vec, at,
                           fav_min_surv, min_group_frac, hi_max_surv, labels) {
  groups <- assign_risk_groups(pred, thresholds = cuts, labels = labels)
  tab <- table(groups)
  fracs <- as.numeric(tab) / length(pred)
  if (any(fracs < min_group_frac)) return(-Inf)

  # KM survival in each group at the horizon
  surv_by_grp <- tapply(seq_along(pred), groups, function(idx)
    .km_surv_at(time_vec[idx], status_vec[idx], at))

  fav_surv <- surv_by_grp[[labels[1]]]
  worst_surv <- surv_by_grp[[labels[length(labels)]]]

  # Hard constraint: favourable group must be safe enough.
  if (is.na(fav_surv) || fav_surv < fav_min_surv) return(-Inf)

  score <- fracs[1] * 100                     # reward: % in favourable tier
  # Soft penalty: worst tier should genuinely be high-risk.
  if (!is.na(hi_max_surv) && !is.na(worst_surv) && worst_surv > hi_max_surv) {
    score <- score - (worst_surv - hi_max_surv) * 300
  }
  score
}

#' Optimise risk-group cutoffs by penalised grid search
#'
#' Searches combinations of \code{n_groups - 1} descending thresholds on predicted
#' survival to maximise the proportion of patients placed in the top ("favourable")
#' tier, subject to that tier's observed Kaplan-Meier survival at the horizon being
#' at least \code{fav_min_surv}, every group meeting a minimum size, and (softly)
#' the worst tier being genuinely high risk.
#'
#' @param data Data frame containing the survival columns.
#' @param pred Numeric vector of predicted survival (length \code{nrow(data)}), or
#'   the name of a column in \code{data}.
#' @param cfg A \code{prognostk_config}.
#' @param n_groups Number of risk tiers (default: from \code{cfg$risk_groups$labels}).
#' @param labels Tier labels (default: from config).
#' @param fav_min_surv Minimum KM survival for the favourable tier (default 0.90).
#' @param hi_max_surv Soft ceiling on the worst tier's KM survival (default 0.60; NA to disable).
#' @param min_group_frac Minimum fraction of patients per group (default 0.02).
#' @param grid Candidate threshold values to search (default \code{seq(0.15, 0.95, 0.02)}).
#' @return A list: \code{best} (named cutoffs), \code{best_score}, \code{groups}
#'   (assigned factor under best cutoffs), and \code{results} (all feasible
#'   candidates, best first).
#' @export
optimise_cutoffs <- function(data, pred, cfg, n_groups = NULL, labels = NULL,
                             fav_min_surv = 0.90, hi_max_surv = 0.60,
                             min_group_frac = 0.02, grid = seq(0.15, 0.95, by = 0.02)) {
  if (is.character(pred) && length(pred) == 1) pred <- data[[pred]]
  labels <- labels %||% cfg$risk_groups$labels
  if (!is.null(n_groups)) {
    if (n_groups != length(labels))
      stop("optimise_cutoffs: n_groups must match length(labels).", call. = FALSE)
  }
  n_cuts <- length(labels) - 1L
  if (n_cuts < 1) stop("optimise_cutoffs: need at least 2 tiers.", call. = FALSE)

  time_col <- col_of(cfg, "time"); status_col <- col_of(cfg, "status")
  time_vec <- as.numeric(data[[time_col]])
  status_vec <- as.numeric(data[[status_col]])
  at <- cfg$survival$horizon

  # Enumerate strictly-descending cutoff tuples from the shared grid.
  combos <- utils::combn(sort(unique(grid), decreasing = TRUE), n_cuts,
                         simplify = FALSE)

  rows <- lapply(combos, function(cuts) {
    sc <- .score_cutoffs(cuts, pred, time_vec, status_vec, at,
                         fav_min_surv, min_group_frac, hi_max_surv, labels)
    if (!is.finite(sc)) return(NULL)
    setNames(as.list(c(cuts, sc)), c(paste0("cut", seq_len(n_cuts)), "score"))
  })
  rows <- Filter(Negate(is.null), rows)
  if (length(rows) == 0) {
    stop("optimise_cutoffs: no feasible cutoff set found. Relax constraints ",
         "(fav_min_surv / min_group_frac) or widen the grid.", call. = FALSE)
  }
  results <- do.call(rbind, lapply(rows, as.data.frame))
  results <- results[order(-results$score), , drop = FALSE]
  rownames(results) <- NULL

  best_cuts <- as.numeric(results[1, paste0("cut", seq_len(n_cuts))])
  names(best_cuts) <- paste0("cut", seq_len(n_cuts))
  list(
    best = best_cuts,
    best_score = results$score[1],
    groups = assign_risk_groups(pred, thresholds = best_cuts, labels = labels),
    results = results
  )
}
