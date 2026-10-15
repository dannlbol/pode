# =============================================================================
# Shared publication theme, palette resolution and small helpers.
# Consolidates the theme_pub() / format_pvalue() / palette definitions that were
# copy-pasted (with drift) across many earlier analysis scripts.
# =============================================================================

#' Publication ggplot theme
#'
#' A clean theme keyed to a base font size (from \code{cfg$figure$page$base_size}).
#'
#' @param base_size Base font size in points. If \code{cfg} is given and
#'   \code{base_size} is \code{NULL}, it is taken from the config.
#' @param cfg Optional \code{prognostk_config}.
#' @return A ggplot2 theme object.
#' @export
theme_pub <- function(base_size = NULL, cfg = NULL) {
  if (is.null(base_size)) base_size <- if (!is.null(cfg)) cfg$figure$page$base_size else 11
  ggplot2::theme_bw(base_size = base_size) +
    ggplot2::theme(
      plot.title    = ggplot2::element_text(face = "bold", size = base_size + 1, hjust = 0),
      plot.subtitle = ggplot2::element_text(size = base_size, hjust = 0.5),
      axis.title    = ggplot2::element_text(size = base_size),
      axis.text     = ggplot2::element_text(size = base_size - 1),
      legend.title  = ggplot2::element_text(size = base_size),
      legend.text   = ggplot2::element_text(size = base_size - 1),
      legend.key.size  = ggplot2::unit(0.3, "cm"),
      panel.grid.minor = ggplot2::element_blank(),
      strip.text    = ggplot2::element_text(size = base_size),
      plot.margin   = ggplot2::margin(t = 2, r = 2, b = 2, l = 2)
    )
}

#' Format a p-value for display
#'
#' @param p A numeric p-value.
#' @return A character label such as \code{"p = 0.032"} or \code{"p < 0.001"}.
#' @export
format_pvalue <- function(p) {
  if (is.na(p)) return("p = NA")
  if (p < 2.2e-16) "p < 2.2e-16"
  else if (p < 0.001) paste0("p = ", formatC(p, format = "e", digits = 2))
  else paste0("p = ", round(p, 3))
}

#' Resolve a named colour palette from the config
#'
#' Returns a named character vector of colours for a palette family. If a level
#' present in \code{levels} has no configured colour, a colour is auto-generated
#' so the function never fails on an unmapped category.
#'
#' @param cfg A \code{prognostk_config}.
#' @param which Palette family: one of \code{"risk"}, \code{"cohort"},
#'   \code{"heatmap"}, or a categorical variable name under \code{palettes.categorical}.
#' @param levels Optional character vector of levels that must be covered.
#' @return A named character vector of hex colours (or an unnamed ramp for "heatmap").
#' @export
prognostk_palette <- function(cfg, which = "risk", levels = NULL) {
  if (which == "heatmap") return(unlist(cfg$palettes$heatmap, use.names = FALSE))
  pal <- cfg$palettes[[which]]
  if (is.null(pal)) pal <- cfg$palettes$categorical[[which]]
  pal <- if (is.null(pal)) character(0) else unlist(pal)
  if (!is.null(levels)) {
    missing <- setdiff(as.character(levels), names(pal))
    if (length(missing) > 0) {
      auto <- grDevices::hcl.colors(length(missing), palette = "Dark 3")
      names(auto) <- missing
      pal <- c(pal, auto)
    }
    pal <- pal[as.character(levels)]
  }
  pal
}

# Ensure an output directory exists and return a full path for a filename.
.out_path <- function(cfg, filename) {
  outdir <- cfg$figure$outdir %||% "."
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  file.path(outdir, filename)
}

# Save a ggplot/grob to the configured device, honouring page size.
.save_figure <- function(plot, cfg, filename, width = NULL, height = NULL) {
  if (is.null(filename)) return(invisible(plot))
  width  <- width  %||% cfg$figure$page$width_in
  height <- height %||% cfg$figure$page$height_in
  path <- .out_path(cfg, filename)
  ggplot2::ggsave(path, plot = plot, width = width, height = height,
                  device = if (identical(cfg$figure$device, "png")) "png" else cfg$figure$device,
                  dpi = cfg$figure$dpi %||% 300)
  message("Wrote ", path)
  invisible(path)
}
