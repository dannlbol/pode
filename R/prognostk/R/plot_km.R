# =============================================================================
# Kaplan-Meier survival curves by group, with an optional number-at-risk table.
# Generalised from survival_functions.R::km_single_cohort() and the from-scratch
# geom_step KM in Plot_bootPred_distributions.R. Built directly on survfit so it
# needs no survminer.
# =============================================================================

# Convert a survfit into a step-ready long data frame with a starting (t=0, s=1) row.
.survfit_to_df <- function(sf, group_levels) {
  strata <- if (is.null(sf$strata)) rep(group_levels[1], length(sf$time))
            else rep(sub("^[^=]*=", "", names(sf$strata)), sf$strata)
  df <- data.frame(time = sf$time, surv = sf$surv, group = strata,
                   stringsAsFactors = FALSE)
  # Prepend the (0, 1) origin for each group so curves start at 100%.
  starts <- do.call(rbind, lapply(unique(df$group), function(g)
    data.frame(time = 0, surv = 1, group = g, stringsAsFactors = FALSE)))
  rbind(starts, df)
}

#' Kaplan-Meier survival curves by group
#'
#' Draws stratified KM curves coloured by a grouping column, with survival on a
#' percent axis and an optional number-at-risk table beneath. All columns,
#' palette, horizon and page size come from the config unless overridden.
#'
#' @param data A data frame.
#' @param cfg A \code{prognostk_config}.
#' @param group_col Grouping column (default \code{cfg$columns$group}).
#' @param time_col,status_col Optional survival column overrides.
#' @param palette Named colour vector (default resolved from \code{cfg$palettes$risk}).
#' @param time_points Numeric breaks / at-risk table times (default \code{seq(0, 2*horizon, ...)}).
#' @param title Plot title.
#' @param risk_table Logical; append a number-at-risk table (requires patchwork).
#' @param filename Optional output filename (written to \code{cfg$figure$outdir}).
#' @return A ggplot (or patchwork) object, invisibly if written to file.
#' @export
plot_km <- function(data, cfg, group_col = NULL, time_col = NULL, status_col = NULL,
                    palette = NULL, time_points = NULL, title = NULL,
                    risk_table = TRUE, filename = NULL) {
  group_col  <- col_of(cfg, "group", group_col)
  time_col   <- col_of(cfg, "time", time_col)
  status_col <- col_of(cfg, "status", status_col)

  d <- data[!is.na(data[[group_col]]) & !is.na(data[[time_col]]) &
              !is.na(data[[status_col]]), , drop = FALSE]
  d[[status_col]] <- as.numeric(d[[status_col]])
  # Order group levels by config risk labels if they match, else natural order.
  glevels <- if (all(unique(d[[group_col]]) %in% cfg$risk_groups$labels))
    cfg$risk_groups$labels[cfg$risk_groups$labels %in% unique(d[[group_col]])]
    else sort(unique(as.character(d[[group_col]])))
  d[[group_col]] <- factor(as.character(d[[group_col]]), levels = glevels)

  horizon <- cfg$survival$horizon
  if (is.null(time_points)) time_points <- seq(0, ceiling(2 * horizon), length.out = 6)
  if (is.null(palette)) palette <- prognostk_palette(cfg, "risk", glevels)

  sf <- survival::survfit(
    survival::Surv(d[[time_col]], d[[status_col]]) ~ d[[group_col]])
  curve_df <- .survfit_to_df(sf, glevels)
  curve_df$group <- factor(curve_df$group, levels = glevels)

  base_size <- cfg$figure$page$base_size
  p <- ggplot2::ggplot(curve_df, ggplot2::aes(time, surv, colour = group)) +
    ggplot2::geom_step(linewidth = 0.7) +
    ggplot2::geom_vline(xintercept = horizon, linetype = "dashed",
                        colour = "grey50", alpha = 0.6) +
    ggplot2::scale_colour_manual(values = palette, name = NULL, drop = FALSE) +
    ggplot2::scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
    ggplot2::scale_x_continuous(breaks = round(time_points, 1)) +
    ggplot2::labs(x = sprintf("Time (%s)", cfg$survival$time_unit),
                  y = "Overall survival", title = title) +
    theme_pub(base_size, cfg) +
    ggplot2::theme(legend.position = "top")

  if (risk_table && requireNamespace("patchwork", quietly = TRUE)) {
    at <- summary(sf, times = time_points, extend = TRUE)
    strata_lab <- if (is.null(at$strata)) glevels[1] else sub("^[^=]*=", "", at$strata)
    rt <- data.frame(time = at$time, n.risk = at$n.risk,
                     group = factor(strata_lab, levels = glevels))
    tbl <- ggplot2::ggplot(rt, ggplot2::aes(time, group, label = n.risk)) +
      ggplot2::geom_text(size = base_size / 3) +
      ggplot2::scale_x_continuous(breaks = round(time_points, 1),
                                  limits = range(time_points)) +
      ggplot2::scale_y_discrete(limits = rev(glevels)) +
      ggplot2::labs(x = NULL, y = NULL, title = "Number at risk") +
      theme_pub(base_size, cfg) +
      ggplot2::theme(panel.grid = ggplot2::element_blank(),
                     axis.text.x = ggplot2::element_blank(),
                     plot.title = ggplot2::element_text(size = base_size, hjust = 0))
    p <- patchwork::wrap_plots(p, tbl, ncol = 1, heights = c(4, 1))
  }

  if (!is.null(filename)) return(.save_figure(p, cfg, filename,
                                              height = cfg$figure$page$width_in * 0.9))
  p
}
