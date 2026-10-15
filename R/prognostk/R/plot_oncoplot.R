# =============================================================================
# Oncoplot: a ComplexHeatmap of an (optional) feature matrix with a stack of
# spec-driven clinical annotation tracks. Generalised from lateMetaCpG_oncoplot.R,
# COG_G34_oncoplot.R and Figure4_Feb2026_addCpGs.R -- every track (points /
# categorical / line), colour map, ordering and baseline comes from the config,
# with NO hard-coded column names or indices.
# =============================================================================

# Build a single HeatmapAnnotation track from a spec entry.
.build_track <- function(track, data, cfg, base_size) {
  col <- track$column
  vals <- data[[col]]
  if (is.null(vals)) stop(sprintf("oncoplot: track column '%s' not in data.", col),
                          call. = FALSE)

  if (identical(track$type, "points")) {
    ylim <- if (!is.null(track$ylim)) unlist(track$ylim) else range(vals, na.rm = TRUE)
    pch <- 20; col_pt <- "grey30"
    if (!is.null(track$status_col) && track$status_col %in% names(data)) {
      st <- as.numeric(data[[track$status_col]])
      pch <- ifelse(st == 1, 4, 20)
      col_pt <- ifelse(st == 1, "firebrick3", "grey60")
    }
    anno <- ComplexHeatmap::anno_points(
      pmin(pmax(vals, ylim[1]), ylim[2]), ylim = ylim, pch = pch, size = grid::unit(1, "mm"),
      gp = grid::gpar(col = col_pt),
      axis_param = list(gp = grid::gpar(fontsize = base_size - 2)))
    # Baseline is drawn afterwards via decorate_annotation() in plot_oncoplot().
    return(list(name = track$name, anno = anno, baseline = track$baseline))
  }

  if (identical(track$type, "line")) {
    anno <- ComplexHeatmap::anno_lines(
      vals, gp = grid::gpar(col = "darkred", lwd = 1.2),
      axis_param = list(gp = grid::gpar(fontsize = base_size - 2)))
    return(list(name = track$name, anno = anno, baseline = NULL))
  }

  # categorical
  lev <- unique(as.character(vals[!is.na(vals)]))
  colmap <- prognostk_palette(cfg, col, lev)
  anno <- ComplexHeatmap::anno_simple(as.character(vals), col = colmap, border = TRUE,
                                      height = grid::unit(3, "mm"))
  list(name = track$name, anno = anno, baseline = NULL, legend = colmap)
}

#' Oncoplot: annotated heatmap of samples
#'
#' Draws an oncoplot as a \code{ComplexHeatmap}: an optional feature heatmap body
#' (e.g. a CpG methylation matrix) with a stack of clinical annotation tracks
#' defined by \code{cfg$oncoplot$tracks} (or the \code{tracks} argument). Samples
#' are ordered by \code{order_by} (default the score column). Track types are
#' \code{points}, \code{categorical} and \code{line}; all colours come from the config.
#'
#' @param data Sample metadata, one row per sample.
#' @param cfg A \code{prognostk_config}.
#' @param matrix Optional feature matrix (features x samples) as a matrix, or a
#'   data frame whose first column is the feature id and remaining columns are
#'   samples. Columns are matched to \code{data} by the id column.
#' @param tracks Optional track spec list (default \code{cfg$oncoplot$tracks}).
#' @param order_by Column to order samples by (default \code{cfg$columns$score}).
#' @param id_col Sample id column (default \code{cfg$columns$id}).
#' @param title Heatmap title.
#' @param filename Optional output filename.
#' @return The drawn \code{HeatmapList} invisibly (and writes a file if \code{filename} set).
#' @export
plot_oncoplot <- function(data, cfg, matrix = NULL, tracks = NULL, order_by = NULL,
                          id_col = NULL, title = "Oncoplot", filename = NULL) {
  if (!requireNamespace("ComplexHeatmap", quietly = TRUE) ||
      !requireNamespace("circlize", quietly = TRUE)) {
    stop("plot_oncoplot requires the 'ComplexHeatmap' and 'circlize' packages.",
         call. = FALSE)
  }
  tracks <- tracks %||% cfg$oncoplot$tracks
  if (length(tracks) == 0) stop("plot_oncoplot: no tracks configured.", call. = FALSE)
  id_col   <- id_col %||% cfg$columns$id
  order_by <- order_by %||% cfg$columns$score
  base_size <- cfg$figure$page$base_size

  # Order samples
  if (!is.null(order_by) && order_by %in% names(data))
    data <- data[order(data[[order_by]]), , drop = FALSE]

  # Align matrix columns to sample order (if a matrix was supplied)
  mat <- NULL
  if (!is.null(matrix)) {
    if (is.data.frame(matrix)) {
      rn <- as.character(matrix[[1]]); matrix <- as.matrix(matrix[, -1, drop = FALSE])
      rownames(matrix) <- rn
    }
    if (is.null(id_col)) stop("plot_oncoplot: matrix supplied but no id column configured.",
                              call. = FALSE)
    ids <- as.character(data[[id_col]])
    keep <- ids[ids %in% colnames(matrix)]
    data <- data[as.character(data[[id_col]]) %in% keep, , drop = FALSE]
    mat  <- matrix[, as.character(data[[id_col]]), drop = FALSE]
  }

  built <- lapply(tracks, .build_track, data = data, cfg = cfg, base_size = base_size)
  anno_args <- stats::setNames(lapply(built, function(b) b$anno),
                               vapply(built, function(b) b$name, character(1)))
  anno_args$annotation_name_side <- "left"
  anno_args$annotation_name_gp <- grid::gpar(fontsize = base_size)
  anno_args$gap <- grid::unit(0.5, "mm")
  top_anno <- do.call(ComplexHeatmap::HeatmapAnnotation, anno_args)

  ramp <- prognostk_palette(cfg, "heatmap")
  col_fun <- if (!is.null(mat)) {
    rng <- range(mat, na.rm = TRUE)
    mids <- if (length(ramp) >= 3) c(rng[1], mean(rng), rng[2]) else rng
    circlize::colorRamp2(mids, ramp[seq_along(mids)])
  } else NULL

  if (is.null(mat)) {
    # No feature matrix: draw a 1-row placeholder so annotations still render.
    mat <- matrix(NA_real_, nrow = 1, ncol = nrow(data))
    ht <- ComplexHeatmap::Heatmap(
      mat, top_annotation = top_anno, show_heatmap_legend = FALSE,
      cluster_columns = FALSE, cluster_rows = FALSE,
      show_row_names = FALSE, show_column_names = FALSE, height = grid::unit(2, "mm"),
      column_title = title, column_title_gp = grid::gpar(fontsize = base_size + 1))
  } else {
    ht <- ComplexHeatmap::Heatmap(
      mat, name = "value", col = col_fun, top_annotation = top_anno,
      cluster_columns = FALSE, cluster_rows = TRUE,
      show_row_names = FALSE, show_column_names = FALSE, border = TRUE,
      column_title = title, column_title_gp = grid::gpar(fontsize = base_size + 1),
      heatmap_legend_param = list(labels_gp = grid::gpar(fontsize = base_size - 2),
                                  title_gp = grid::gpar(fontsize = base_size - 1)))
  }

  draw_it <- function() {
    ComplexHeatmap::draw(ht, merge_legend = TRUE)
    # Draw baselines on point tracks.
    for (b in built) if (!is.null(b$baseline)) {
      ComplexHeatmap::decorate_annotation(b$name, {
        grid::grid.lines(grid::unit(c(0, 1), "npc"), grid::unit(rep(b$baseline, 2), "native"),
                         gp = grid::gpar(col = "grey50", lty = 2, lwd = 0.6))
      })
    }
  }

  if (!is.null(filename)) {
    path <- .out_path(cfg, filename)
    w <- cfg$figure$page$width_in; h <- cfg$figure$page$width_in * 0.85
    dev <- cfg$figure$device
    if (identical(dev, "png")) grDevices::png(path, width = w, height = h, units = "in",
                                              res = cfg$figure$dpi %||% 300)
    else if (identical(dev, "cairo_pdf")) grDevices::cairo_pdf(path, width = w, height = h)
    else grDevices::pdf(path, width = w, height = h)
    draw_it(); grDevices::dev.off()
    message("Wrote ", path)
    return(invisible(path))
  }
  draw_it()
  invisible(ht)
}
