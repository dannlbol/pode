# =============================================================================
# Forest plot of Cox hazard ratios. Generalised from
# survival_functions.R::create_comparison_forest_plot() and ForestPlotComparison_*.
# =============================================================================

# Tidy a coxph fit into a HR / CI / p data frame.
.tidy_cox <- function(fit, label = NULL) {
  s <- summary(fit)
  data.frame(
    term  = rownames(s$coefficients),
    HR    = s$coefficients[, "exp(coef)"],
    lower = s$conf.int[, "lower .95"],
    upper = s$conf.int[, "upper .95"],
    p     = s$coefficients[, "Pr(>|z|)"],
    model = label %||% "model",
    row.names = NULL, stringsAsFactors = FALSE
  )
}

#' Forest plot of hazard ratios from one or more Cox models
#'
#' Fits a Cox model per named cohort in \code{data_list} (using the config's
#' covariates unless overridden) and draws their hazard ratios on a shared
#' log-scale forest plot, dodged by cohort. A single data frame may be passed
#' instead of a list.
#'
#' @param data_list A named list of data frames (one per cohort), or a single data frame.
#' @param cfg A \code{prognostk_config}.
#' @param covariates Optional model-term override (see \code{fit_cox}).
#' @param term_labels Optional named vector mapping model terms to display labels.
#' @param palette Named colour vector for cohorts (default \code{cfg$palettes$cohort}).
#' @param title Plot title.
#' @param filename Optional output filename.
#' @return A ggplot, or the written path if \code{filename} is set.
#' @export
plot_forest <- function(data_list, cfg, covariates = NULL, term_labels = NULL,
                        palette = NULL, title = NULL, filename = NULL) {
  if (is.data.frame(data_list)) data_list <- list(cohort = data_list)
  cohorts <- names(data_list) %||% paste0("cohort", seq_along(data_list))

  tidy <- do.call(rbind, lapply(seq_along(data_list), function(i) {
    fit <- fit_cox(data_list[[i]], cfg, covariates = covariates)
    .tidy_cox(fit$fit, cohorts[i])
  }))
  if (!is.null(term_labels)) {
    mapped <- term_labels[tidy$term]
    tidy$term <- ifelse(is.na(mapped), tidy$term, mapped)
  }
  tidy$term  <- factor(tidy$term, levels = rev(unique(tidy$term)))
  tidy$model <- factor(tidy$model, levels = cohorts)
  if (is.null(palette)) palette <- prognostk_palette(cfg, "cohort", cohorts)
  base_size <- cfg$figure$page$base_size

  p <- ggplot2::ggplot(tidy, ggplot2::aes(x = HR, y = term, colour = model)) +
    ggplot2::geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50") +
    ggplot2::geom_errorbar(ggplot2::aes(xmin = lower, xmax = upper),
                           orientation = "y", width = 0.2,
                           position = ggplot2::position_dodge(width = 0.5)) +
    ggplot2::geom_point(size = 2, position = ggplot2::position_dodge(width = 0.5)) +
    ggplot2::scale_x_log10() +
    ggplot2::scale_colour_manual(values = palette, name = NULL,
                                 drop = FALSE, guide = if (length(cohorts) > 1)
                                   "legend" else "none") +
    ggplot2::labs(x = "Hazard ratio (95% CI)", y = NULL, title = title) +
    theme_pub(base_size, cfg) +
    ggplot2::theme(legend.position = "top")

  if (!is.null(filename)) return(.save_figure(p, cfg, filename,
                                              height = cfg$figure$page$width_in * 0.6))
  p
}
