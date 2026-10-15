# =============================================================================
# Distribution of a continuous score across groups: violin / box / ridge, with
# an omnibus statistic and optional pairwise comparison brackets.
# Generalised from Figure2_combined_violin.R and the COG oncoplot ridge panels.
# =============================================================================

#' Distribution plot of a score across groups
#'
#' Plots a continuous score (default \code{cfg$columns$score}) grouped by a
#' categorical variable, as violins, boxplots or ridges. Annotates an omnibus
#' test (t-test / ANOVA) and, optionally, pairwise significance brackets.
#'
#' @param data A data frame.
#' @param cfg A \code{prognostk_config}.
#' @param group_col Categorical grouping column. Defaults to the first entry of
#'   \code{cfg$columns$categorical}.
#' @param score_col Continuous column (default \code{cfg$columns$score}).
#' @param type One of \code{"violin"}, \code{"box"}, \code{"ridge"}.
#' @param palette Named colour vector; default auto-generated for the group levels.
#' @param add_points Overlay jittered points (violin/box only).
#' @param pairwise Add pairwise comparison brackets (requires ggpubr).
#' @param title Plot title.
#' @param filename Optional output filename.
#' @return A ggplot, or the written path if \code{filename} is set.
#' @export
plot_distribution <- function(data, cfg, group_col = NULL, score_col = NULL,
                              type = c("violin", "box", "ridge"),
                              palette = NULL, add_points = TRUE, pairwise = FALSE,
                              title = NULL, filename = NULL) {
  type <- match.arg(type)
  score_col <- col_of(cfg, "score", score_col)
  if (is.null(group_col)) {
    group_col <- cfg$columns$categorical[[1]]
    if (is.null(group_col)) stop("plot_distribution: no group_col and no columns.categorical.",
                                 call. = FALSE)
  }
  d <- data[!is.na(data[[group_col]]) & !is.na(data[[score_col]]), , drop = FALSE]
  d[[group_col]] <- factor(as.character(d[[group_col]]))
  glevels <- levels(d[[group_col]])
  if (is.null(palette)) palette <- prognostk_palette(cfg, group_col, glevels)
  base_size <- cfg$figure$page$base_size

  # Omnibus statistic
  ng <- length(glevels)
  stat_lbl <- tryCatch({
    if (ng == 2) {
      tt <- t.test(d[[score_col]] ~ d[[group_col]])
      sprintf("t = %.2f, %s", tt$statistic, format_pvalue(tt$p.value))
    } else if (ng > 2) {
      av <- summary(aov(d[[score_col]] ~ d[[group_col]]))[[1]]
      sprintf("F = %.2f, %s", av[["F value"]][1], format_pvalue(av[["Pr(>F)"]][1]))
    } else ""
  }, error = function(e) "")

  if (type == "ridge" && requireNamespace("ggridges", quietly = TRUE)) {
    p <- ggplot2::ggplot(d, ggplot2::aes(x = .data[[score_col]], y = .data[[group_col]],
                                         fill = .data[[group_col]])) +
      ggridges::geom_density_ridges(alpha = 0.7, scale = 1.1, linewidth = 0.3) +
      ggplot2::scale_fill_manual(values = palette, guide = "none") +
      ggplot2::labs(x = score_col, y = NULL, title = title, subtitle = stat_lbl)
  } else {
    p <- ggplot2::ggplot(d, ggplot2::aes(x = .data[[group_col]], y = .data[[score_col]],
                                         colour = .data[[group_col]]))
    p <- p + if (type == "box") ggplot2::geom_boxplot(outlier.shape = NA, linewidth = 0.4)
             else ggplot2::geom_violin(trim = FALSE, linewidth = 0.4)
    if (add_points)
      p <- p + ggplot2::geom_jitter(width = 0.15, size = 0.5, alpha = 0.4)
    p <- p +
      ggplot2::scale_colour_manual(values = palette, guide = "none") +
      ggplot2::labs(x = NULL, y = score_col, title = title, subtitle = stat_lbl)
    if (pairwise && ng > 1 && requireNamespace("ggpubr", quietly = TRUE)) {
      combs <- utils::combn(glevels, 2, simplify = FALSE)
      p <- p + ggpubr::stat_compare_means(comparisons = combs, label = "p.signif",
                                          size = base_size / 4)
    }
  }
  p <- p + theme_pub(base_size, cfg)

  if (!is.null(filename)) return(.save_figure(p, cfg, filename,
                                              height = cfg$figure$page$width_in * 0.8))
  p
}
