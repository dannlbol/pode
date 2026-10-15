#!/usr/bin/env Rscript
#
# Real vs synthetic copy-number heatmap, bins ordered by genomic position
# (chromosome then start), samples grouped by subgroup with real/synthetic
# blocks side by side within each subgroup -- the CNV analogue of
# plot_real_vs_synthetic_methylation_umap.R's differential-feature heatmap.
#
# Unlike the methylation/expression heatmaps, rows are NOT selected by a
# differential test: copy-number bins are already comparable across samples on
# a shared log2-ratio (or ordinal gain/loss) scale, so every bin is plotted in
# its natural genomic order and no per-feature standardization is applied --
# doing so would destroy the "0 = neutral" reference point that makes a CNV
# heatmap readable.

suppressPackageStartupMessages({
  library(ggplot2)
})

ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      sprintf(
        "Package '%s' is required but not installed. Install it with install.packages('%s').",
        pkg, pkg
      ),
      call. = FALSE
    )
  }
}

ensure_package("pheatmap")

defaults <- list(
  real = "data/copy_number.csv",
  synthetic = "decoded/synthetic_copy_number.csv",
  real_clinical = "data/clinical.csv",
  synthetic_clinical = "decoded/synthetic_clinical.csv",
  color_col = "MNP_Subgroup",
  value_type = "continuous",       # continuous (log2 ratio) | discrete ({-1,0,1} calls)
  color_limit = NA,                # continuous only; NA -> auto from real-data quantile
  color_limit_quantile = 0.99,
  max_rows = 4000L,                # evenly-spaced downsample cap for the plot only
  output_dir = "data",
  out_heatmap = "data/real_vs_synthetic_cnv_heatmap.png"
)

parse_args <- function(defaults) {
  args <- commandArgs(trailingOnly = TRUE)
  out <- defaults
  provided <- character()

  if ("--help" %in% args || "-h" %in% args) {
    cat(
      paste0(
        "Usage: Rscript plot_real_vs_synthetic_cnv_heatmap.R [options]\n\n",
        "Options:\n",
        "  --real PATH               real copy-number CSV (bins as columns, e.g. copy_number.csv)\n",
        "  --synthetic PATH          synthetic copy-number CSV (e.g. synthetic_copy_number.csv)\n",
        "  --real-clinical PATH      real/original clinical CSV\n",
        "  --synthetic-clinical PATH synthetic clinical CSV\n",
        "  --color-col NAME          clinical column defining the subgroup blocks\n",
        "  --value-type TYPE         'continuous' (log2 ratio) or 'discrete' ({-1,0,1} calls)\n",
        "  --color-limit FLOAT       symmetric colour-scale limit (continuous only); default auto\n",
        "  --color-limit-quantile F  quantile of |real value| used for the auto colour limit\n",
        "  --max-rows INT            evenly-spaced cap on rendered bins (rendering aid only)\n",
        "  --output-dir PATH         directory for generated outputs\n",
        "  --out-heatmap PATH        output PNG heatmap path\n"
      )
    )
    quit(save = "no", status = 0)
  }

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument: %s", key), call. = FALSE)
    }
    if (i == length(args)) {
      stop(sprintf("Missing value for argument: %s", key), call. = FALSE)
    }
    value <- args[[i + 1L]]
    name <- gsub("-", "_", substring(key, 3L))
    if (!name %in% names(out)) {
      stop(sprintf("Unknown argument: %s", key), call. = FALSE)
    }
    out[[name]] <- value
    provided <- c(provided, name)
    i <- i + 2L
  }

  out$color_limit <- suppressWarnings(as.numeric(out$color_limit))
  out$color_limit_quantile <- as.numeric(out$color_limit_quantile)
  out$max_rows <- as.integer(out$max_rows)
  if (!out$value_type %in% c("continuous", "discrete")) {
    stop("--value-type must be 'continuous' or 'discrete'", call. = FALSE)
  }
  output_names <- c("out_heatmap")
  for (name in output_names) {
    if (!name %in% provided) {
      out[[name]] <- file.path(out$output_dir, basename(defaults[[name]]))
    }
  }
  out
}

opt <- parse_args(defaults)
dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)

read_matrix <- function(path, label) {
  df <- read.csv(path, check.names = FALSE)
  # Input CSVs may carry a unifying sample_id column; it is not a
  # copy-number bin, so drop it before the numeric-column check and plotting.
  if ("sample_id" %in% colnames(df)) {
    df <- df[, setdiff(colnames(df), "sample_id"), drop = FALSE]
  }
  if (!all(vapply(df, is.numeric, logical(1)))) {
    stop(sprintf("All columns in %s must be numeric %s values.", path, label), call. = FALSE)
  }
  df
}

read_clinical <- function(path, expected_rows, color_col) {
  if (is.null(path) || is.na(path) || !nzchar(path) || !file.exists(path)) {
    return(rep("Unknown", expected_rows))
  }
  df <- read.csv(path, check.names = FALSE)
  if (nrow(df) != expected_rows) {
    stop(
      sprintf("Clinical file %s has %d rows but expected %d.", path, nrow(df), expected_rows),
      call. = FALSE
    )
  }
  if (!color_col %in% colnames(df)) {
    return(rep("Unknown", expected_rows))
  }
  vals <- as.character(df[[color_col]])
  vals[is.na(vals) | vals == ""] <- "Unknown"
  vals
}

# Bin columns are named "chrN:start-end" by synthvae.data.copy_number. Parse
# chromosome + start so rows can be ordered genomically rather than selected.
genomic_bin_order <- function(bin_names) {
  m <- regmatches(bin_names, regexec("^(chr[0-9XYM]+):([0-9]+)-([0-9]+)$", bin_names))
  chrom <- vapply(m, function(x) if (length(x) == 4) x[[2]] else NA_character_, character(1))
  start <- vapply(m, function(x) if (length(x) == 4) as.numeric(x[[3]]) else NA_real_, numeric(1))
  unresolved <- is.na(chrom)
  if (any(unresolved)) {
    warning(
      sprintf(
        "%d bin column(s) did not match the 'chrN:start-end' pattern; placing them last, in file order.",
        sum(unresolved)
      ),
      call. = FALSE
    )
  }
  numeric_chrom <- suppressWarnings(as.integer(sub("^chr", "", chrom)))
  rank <- ifelse(
    !is.na(numeric_chrom), numeric_chrom,
    ifelse(chrom == "chrX", 90, ifelse(chrom == "chrY", 91, ifelse(chrom == "chrM", 92, 99)))
  )
  rank[unresolved] <- 100
  order(rank, start, seq_along(bin_names))
}

plot_cnv_heatmap <- function(real_mat, synthetic_mat, real_groups, synthetic_groups,
                             bin_order_idx, chrom_labels, value_type,
                             color_limit, color_limit_quantile, max_rows, out_path) {
  n_bins <- length(bin_order_idx)
  if (n_bins > max_rows) {
    keep <- unique(round(seq(1, n_bins, length.out = max_rows)))
    bin_order_idx <- bin_order_idx[keep]
    chrom_labels <- chrom_labels[keep]
  }
  bin_order <- colnames(real_mat)[bin_order_idx]

  real_groups <- as.character(real_groups)
  synthetic_groups <- as.character(synthetic_groups)
  real_groups[is.na(real_groups) | real_groups == ""] <- "Unknown"
  synthetic_groups[is.na(synthetic_groups) | synthetic_groups == ""] <- "Unknown"
  subgroup_levels <- sort(unique(c(real_groups, synthetic_groups)))

  ordered_blocks <- lapply(subgroup_levels, function(subgroup) {
    real_idx <- which(real_groups == subgroup)
    synthetic_idx <- which(synthetic_groups == subgroup)
    list(
      mat = cbind(
        t(as.matrix(real_mat[real_idx, bin_order, drop = FALSE])),
        t(as.matrix(synthetic_mat[synthetic_idx, bin_order, drop = FALSE]))
      ),
      sample_ids = c(sprintf("Real_%03d", real_idx), sprintf("Synthetic_%03d", synthetic_idx)),
      source = c(rep("Real", length(real_idx)), rep("Synthetic", length(synthetic_idx))),
      subgroup = rep(subgroup, length(real_idx) + length(synthetic_idx))
    )
  })

  combined_mat <- do.call(cbind, lapply(ordered_blocks, `[[`, "mat"))
  combined_mat[is.na(combined_mat)] <- 0
  colnames(combined_mat) <- unlist(lapply(ordered_blocks, `[[`, "sample_ids"), use.names = FALSE)
  rownames(combined_mat) <- bin_order

  annotation_col <- data.frame(
    Source = factor(unlist(lapply(ordered_blocks, `[[`, "source"), use.names = FALSE),
                    levels = c("Real", "Synthetic")),
    Subgroup = factor(unlist(lapply(ordered_blocks, `[[`, "subgroup"), use.names = FALSE),
                      levels = subgroup_levels),
    check.names = FALSE
  )
  rownames(annotation_col) <- colnames(combined_mat)

  chrom_levels <- unique(chrom_labels)
  annotation_row <- data.frame(
    Chromosome = factor(chrom_labels, levels = chrom_levels),
    check.names = FALSE
  )
  rownames(annotation_row) <- bin_order
  chrom_colors <- setNames(grDevices::rainbow(length(chrom_levels)), chrom_levels)

  subgroup_change_idx <- which(annotation_col$Subgroup[-1] != annotation_col$Subgroup[-nrow(annotation_col)])
  source_change_idx <- which(annotation_col$Source[-1] != annotation_col$Source[-nrow(annotation_col)])
  gaps_col <- sort(unique(c(subgroup_change_idx, source_change_idx)))
  gaps_row <- which(annotation_row$Chromosome[-1] != annotation_row$Chromosome[-nrow(annotation_row)])

  if (value_type == "discrete") {
    colors <- c("#2166AC", "#F7F7F7", "#B2182B")   # loss / balanced / gain
    breaks <- c(-1.5, -0.5, 0.5, 1.5)
    legend_note <- "Values: -1 loss, 0 balanced, 1 gain"
  } else {
    lim <- color_limit
    if (is.na(lim)) {
      lim <- as.numeric(quantile(abs(as.matrix(real_mat[, bin_order, drop = FALSE])),
                                 probs = color_limit_quantile, na.rm = TRUE))
    }
    if (!is.finite(lim) || lim <= 0) lim <- 1
    colors <- grDevices::colorRampPalette(c("#2166AC", "#F7F7F7", "#B2182B"))(100)
    breaks <- seq(-lim, lim, length.out = 101)
    legend_note <- sprintf("Log2 ratio, clipped at +/-%.2f (%.0fth pct of |real|)",
                           lim, 100 * color_limit_quantile)
  }

  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  pheatmap::pheatmap(
    mat = combined_mat,
    filename = out_path,
    color = colors,
    breaks = breaks,
    cluster_cols = FALSE,
    cluster_rows = FALSE,
    show_colnames = FALSE,
    show_rownames = FALSE,
    border_color = NA,
    annotation_col = annotation_col,
    annotation_row = annotation_row,
    annotation_colors = list(Chromosome = chrom_colors),
    gaps_col = gaps_col,
    gaps_row = gaps_row,
    main = sprintf(
      "Real vs synthetic copy number (%s)\n%d genomic bins, chr1..chr22 order | %s",
      value_type, length(bin_order), legend_note
    ),
    fontsize = 9,
    width = max(10, min(28, ncol(combined_mat) * 0.085 + 4)),
    height = max(8, min(26, length(bin_order) * 0.012 + 4))
  )
  invisible(TRUE)
}

real_mat <- read_matrix(opt$real, "copy-number")
synthetic_mat <- read_matrix(opt$synthetic, "copy-number")

shared_bins <- intersect(colnames(real_mat), colnames(synthetic_mat))
if (length(shared_bins) == 0) {
  stop("The real and synthetic matrices do not share any copy-number bin columns.", call. = FALSE)
}
real_mat <- real_mat[, shared_bins, drop = FALSE]
synthetic_mat <- synthetic_mat[, shared_bins, drop = FALSE]

if (opt$value_type == "discrete") {
  vals <- c(as.matrix(real_mat), as.matrix(synthetic_mat))
  bad <- !(vals %in% c(-1, 0, 1))
  if (any(bad, na.rm = TRUE)) {
    warning(
      sprintf("%d value(s) outside {-1,0,1} found for --value-type discrete; clamping.", sum(bad, na.rm = TRUE)),
      call. = FALSE
    )
  }
}

order_idx <- genomic_bin_order(shared_bins)
chrom_labels <- sub(":.*$", "", shared_bins[order_idx])

n_real <- nrow(real_mat)
real_color <- read_clinical(opt$real_clinical, n_real, opt$color_col)
synthetic_color <- read_clinical(opt$synthetic_clinical, nrow(synthetic_mat), opt$color_col)

plot_cnv_heatmap(
  real_mat, synthetic_mat, real_color, synthetic_color,
  order_idx, chrom_labels, opt$value_type,
  opt$color_limit, opt$color_limit_quantile, opt$max_rows, opt$out_heatmap
)

cat(sprintf("Copy-number heatmap written to %s\n", opt$out_heatmap))
cat(sprintf("Shared bins considered: %d (rendered: %d)\n",
            length(shared_bins), min(length(shared_bins), opt$max_rows)))
cat(sprintf("Real samples: %d | Synthetic samples: %d | Subgroup column: %s\n",
            n_real, nrow(synthetic_mat), opt$color_col))
