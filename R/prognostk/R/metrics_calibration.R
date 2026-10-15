# =============================================================================
# Calibration & external validation metrics.
# Generalised from earlier study-specific calibration / validation scripts.
# =============================================================================

#' Calibration table: predicted vs observed survival by risk bin
#'
#' Bins patients by predicted survival into \code{n_bins} quantile groups and
#' compares the mean predicted survival to the Kaplan-Meier observed survival at
#' the horizon in each bin. Also returns the overall expected/observed (E/O) ratio.
#'
#' @param data Data frame with the survival columns.
#' @param pred Numeric vector of predicted survival, or a column name in \code{data}.
#' @param cfg A \code{prognostk_config}.
#' @param n_bins Number of quantile bins (default 10; auto-reduced for small n).
#' @return A list: \code{table} (bin, n, mean_pred, obs_km, obs_lower, obs_upper)
#'   and \code{eo_ratio}.
#' @export
calibration_table <- function(data, pred, cfg, n_bins = 10) {
  if (is.character(pred) && length(pred) == 1) pred <- data[[pred]]
  time_col <- col_of(cfg, "time"); status_col <- col_of(cfg, "status")
  at <- cfg$survival$horizon

  ok <- is.finite(pred) & !is.na(data[[time_col]]) & !is.na(data[[status_col]])
  pred <- pred[ok]
  tv <- as.numeric(data[[time_col]][ok]); sv <- as.numeric(data[[status_col]][ok])

  n_bins <- max(2, min(n_bins, floor(length(pred) / 15)))
  brks <- unique(quantile(pred, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE))
  bin <- cut(pred, breaks = brks, include.lowest = TRUE, labels = FALSE)

  rows <- lapply(sort(unique(bin)), function(b) {
    idx <- which(bin == b)
    sf <- survival::survfit(survival::Surv(tv[idx], sv[idx]) ~ 1)
    s <- summary(sf, times = at, extend = TRUE)
    data.frame(bin = b, n = length(idx),
               mean_pred = mean(pred[idx]),
               obs_km = ifelse(length(s$surv), s$surv[1], NA_real_),
               obs_lower = ifelse(length(s$lower), s$lower[1], NA_real_),
               obs_upper = ifelse(length(s$upper), s$upper[1], NA_real_))
  })
  tab <- do.call(rbind, rows)

  overall_obs <- .km_surv_at(tv, sv, at)
  eo_ratio <- mean(pred) / overall_obs
  list(table = tab, eo_ratio = eo_ratio)
}

#' External validation of a frozen prognostic model
#'
#' Applies a fixed Cox model to a validation cohort and reports discrimination
#' (Harrell's C-index with CI), calibration slope, calibration-in-the-large, and
#' the expected/observed ratio at the horizon. Generalised from
#' \code{external_validate()}.
#'
#' @param valid_data Validation cohort data frame.
#' @param frozen_fit A \code{coxph} fit (or \code{fit_cox} result) to apply as-is.
#' @param cfg A \code{prognostk_config}.
#' @return A named list of metrics.
#' @export
external_validate <- function(valid_data, frozen_fit, cfg) {
  if (is.list(frozen_fit) && !is.null(frozen_fit$fit)) frozen_fit <- frozen_fit$fit
  time_col <- col_of(cfg, "time"); status_col <- col_of(cfg, "status")
  valid_data <- .apply_landmark(valid_data, cfg, time_col, status_col)

  surv_obj <- survival::Surv(valid_data[[time_col]], as.numeric(valid_data[[status_col]]))
  lp <- .centered_lp(frozen_fit, valid_data)

  ok <- is.finite(lp) & !is.na(valid_data[[time_col]]) & !is.na(valid_data[[status_col]])
  surv_obj <- surv_obj[ok]; lp <- lp[ok]

  # Discrimination: Harrell's C
  cc <- survival::concordance(surv_obj ~ lp, reverse = TRUE)
  cidx <- cc$concordance
  cse  <- sqrt(cc$var)

  # Calibration slope (ideal = 1) and calibration-in-the-large.
  slope_fit <- tryCatch(survival::coxph(surv_obj ~ lp), error = function(e) NULL)
  cal_slope <- if (!is.null(slope_fit)) unname(coef(slope_fit)[1]) else NA_real_
  citl_fit <- tryCatch(survival::coxph(surv_obj ~ offset(lp)), error = function(e) NULL)
  citl <- if (!is.null(citl_fit)) unname(survival::survfit(citl_fit)$surv[1]) else NA_real_

  # E/O at horizon
  pred <- predict_survival_at_time(frozen_fit, valid_data[ok, , drop = FALSE],
                                   time = cfg$survival$horizon)
  eo <- mean(pred, na.rm = TRUE) /
    .km_surv_at(valid_data[[time_col]][ok], as.numeric(valid_data[[status_col]])[ok],
                cfg$survival$horizon)

  z <- qnorm(1 - (1 - cfg$survival$bootstrap$conf) / 2)
  list(
    n = sum(ok),
    events = sum(as.numeric(valid_data[[status_col]])[ok], na.rm = TRUE),
    c_index = cidx,
    c_index_lower = cidx - z * cse,
    c_index_upper = cidx + z * cse,
    calibration_slope = cal_slope,
    eo_ratio = eo
  )
}
