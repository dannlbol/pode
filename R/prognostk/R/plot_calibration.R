# =============================================================================
# Calibration plot (predicted vs observed survival) and per-patient survival
# "spaghetti" curves. Generalised from Figure4C_calibration.R::plot_calibration()
# and survival_functions.R::plot_individual_curves_by_risk().
# =============================================================================

#' Calibration plot: predicted vs observed survival
#'
#' Plots binned mean predicted survival against Kaplan-Meier observed survival at
#' the horizon (with observed CIs) against the identity line. Accepts either a
#' precomputed \code{calibration_table()} result or raw data + predictions.
#'
#' @param x Either the list returned by \code{calibration_table()}, or a data frame.
#' @param cfg A \code{prognostk_config}.
#' @param pred If \code{x} is a data frame: predicted survival vector or column name.
#' @param n_bins If computing from raw data: number of bins.
#' @param title Plot title.
#' @param filename Optional output filename.
#' @return A ggplot, or the written path if \code{filename} is set.
#' @export
plot_calibration <- function(x, cfg, pred = NULL, n_bins = 10, title = NULL,
                             filename = NULL) {
  cal <- if (is.list(x) && !is.null(x$table)) x
         else calibration_table(x, pred, cfg, n_bins = n_bins)
  tab <- cal$table
  base_size <- cfg$figure$page$base_size

  p <- ggplot2::ggplot(tab, ggplot2::aes(mean_pred, obs_km)) +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                         colour = "grey50") +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = obs_lower, ymax = obs_upper),
                           width = 0.01, colour = "grey40") +
    ggplot2::geom_point(size = 2, colour = prognostk_palette(cfg, "cohort")[1]) +
    ggplot2::geom_line(colour = prognostk_palette(cfg, "cohort")[1], alpha = 0.6) +
    ggplot2::coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    ggplot2::scale_x_continuous(labels = scales::percent) +
    ggplot2::scale_y_continuous(labels = scales::percent) +
    ggplot2::labs(
      x = sprintf("Predicted %g-%s survival", cfg$survival$horizon, cfg$survival$time_unit),
      y = "Observed (KM) survival", title = title,
      subtitle = sprintf("E/O = %.2f", cal$eo_ratio)) +
    theme_pub(base_size, cfg)

  if (!is.null(filename)) return(.save_figure(p, cfg, filename,
                                              width = cfg$figure$page$width_in * 0.7,
                                              height = cfg$figure$page$width_in * 0.7))
  p
}

#' Per-patient predicted survival curves coloured by risk group
#'
#' Draws each patient's model-predicted survival curve over time, coloured by
#' their risk group, with a dashed line at the horizon. Generalised from
#' \code{plot_individual_curves_by_risk()}.
#'
#' @param fit A \code{coxph} fit (or \code{fit_cox} result).
#' @param data Data to predict for.
#' @param cfg A \code{prognostk_config}.
#' @param risk_groups Optional factor of risk groups per row; if \code{NULL},
#'   computed from predicted survival at the horizon.
#' @param max_time Maximum time on the x-axis (default \code{2 * horizon}).
#' @param title Plot title.
#' @param filename Optional output filename.
#' @return A ggplot, or the written path if \code{filename} is set.
#' @export
plot_individual_curves <- function(fit, data, cfg, risk_groups = NULL,
                                   max_time = NULL, title = NULL, filename = NULL) {
  coxfit <- if (is.list(fit) && !is.null(fit$fit)) fit$fit else fit
  horizon <- cfg$survival$horizon
  max_time <- max_time %||% (2 * horizon)
  grid <- seq(0, max_time, length.out = 60)

  if (is.null(risk_groups)) {
    ps <- predict_survival_at_time(coxfit, data, time = horizon)
    risk_groups <- assign_risk_groups(ps, cfg)
  }
  # Predicted survival for each row across the time grid (reuse the robust predictor).
  curves <- do.call(rbind, lapply(grid, function(t) {
    data.frame(id = seq_len(nrow(data)), time = t,
               surv = predict_survival_at_time(coxfit, data, time = t),
               risk = risk_groups, stringsAsFactors = FALSE)
  }))
  palette <- prognostk_palette(cfg, "risk", levels(risk_groups))
  base_size <- cfg$figure$page$base_size

  p <- ggplot2::ggplot(curves, ggplot2::aes(time, surv, group = id, colour = risk)) +
    ggplot2::geom_line(linewidth = 0.3, alpha = 0.4) +
    ggplot2::geom_vline(xintercept = horizon, linetype = "dashed", colour = "grey50") +
    ggplot2::scale_colour_manual(values = palette, name = NULL, drop = FALSE) +
    ggplot2::scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
    ggplot2::labs(x = sprintf("Time (%s)", cfg$survival$time_unit),
                  y = "Predicted survival", title = title) +
    theme_pub(base_size, cfg) +
    ggplot2::theme(legend.position = "top")

  if (!is.null(filename)) return(.save_figure(p, cfg, filename,
                                              height = cfg$figure$page$width_in * 0.8))
  p
}
