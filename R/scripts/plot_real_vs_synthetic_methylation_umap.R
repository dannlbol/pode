#!/usr/bin/env Rscript

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

ensure_package("uwot")
ensure_package("pheatmap")

defaults <- list(
  real = "data/methylation.csv",
  synthetic = "../pode/generated_hybrid_methylation_tuned/decoded/synthetic_methylation.csv",
  real_clinical = "data/clinical.csv",
  synthetic_clinical = "../pode/generated_hybrid_methylation_tuned/decoded/synthetic_clinical.csv",
  shape_col = "WCA",
  color_col = "MNP_Subgroup",
  n_features = 5000L,
  dm_top_n = 10L,
  seed = 42L,
  neighbors = 15L,
  min_dist = 0.3,
  embedding_mode = "real_reference",
  output_dir = "data",
  out_plot = "data/real_vs_synthetic_methylation_umap.png",
  out_coords = "data/real_vs_synthetic_methylation_umap_coords.csv",
  out_dm_features = "data/real_subgroup_top_dm_features.csv",
  out_heatmap = "data/real_vs_synthetic_methylation_dm_heatmap.png"
)

parse_args <- function(defaults) {
  args <- commandArgs(trailingOnly = TRUE)
  out <- defaults
  provided <- character()

  if ("--help" %in% args || "-h" %in% args) {
    cat(
      paste0(
        "Usage: Rscript plot_real_vs_synthetic_methylation_umap.R [options]\n\n",
        "Options:\n",
        "  --real PATH              real/original methylation CSV\n",
        "  --synthetic PATH         synthetic methylation CSV\n",
        "  --real-clinical PATH     real/original clinical CSV\n",
        "  --synthetic-clinical PATH synthetic clinical CSV\n",
        "  --shape-col NAME         clinical column to map to point shape\n",
        "  --color-col NAME         clinical column to map to point colour\n",
        "  --n-features INT         number of most variable methylation features to keep\n",
        "  --dm-top-n INT           number of top real-data differential methylation features per subgroup\n",
        "  --seed INT               random seed for UMAP\n",
        "  --neighbors INT          UMAP n_neighbors\n",
        "  --min-dist FLOAT         UMAP min_dist\n",
        "  --embedding-mode MODE    real_reference (default), combined, or both\n",
        "  --output-dir PATH        directory for generated outputs\n",
        "  --out-plot PATH          output PNG path\n",
        "  --out-coords PATH        output CSV of UMAP coordinates\n",
        "  --out-dm-features PATH   output CSV of top subgroup differential methylation features\n",
        "  --out-heatmap PATH       output PNG heatmap of selected methylation features\n"
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

  out$n_features <- as.integer(out$n_features)
  out$dm_top_n <- as.integer(out$dm_top_n)
  out$seed <- as.integer(out$seed)
  out$neighbors <- as.integer(out$neighbors)
  out$min_dist <- as.numeric(out$min_dist)
  if (!out$embedding_mode %in% c("combined", "real_reference", "both")) {
    stop("--embedding-mode must be 'combined', 'real_reference', or 'both'", call. = FALSE)
  }
  output_names <- c("out_plot", "out_coords", "out_dm_features", "out_heatmap")
  for (name in output_names) {
    if (!name %in% provided) {
      out[[name]] <- file.path(out$output_dir, basename(defaults[[name]]))
    }
  }
  out
}

opt <- parse_args(defaults)

if (opt$embedding_mode == "both") {
  suffix_path <- function(path, mode) {
    ext <- tools::file_ext(path)
    stem <- if (nzchar(ext)) substr(path, 1, nchar(path) - nchar(ext) - 1L) else path
    paste0(stem, "_", mode, if (nzchar(ext)) paste0(".", ext) else "")
  }
  set_arg <- function(args, key, value) {
    where <- which(args == key)
    if (length(where)) {
      args[where[[1]] + 1L] <- value
      args
    } else {
      c(args, key, value)
    }
  }
  script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  script <- sub("^--file=", "", script_arg[[1]])
  base_args <- commandArgs(trailingOnly = TRUE)
  for (mode in c("real_reference", "combined")) {
    child <- set_arg(base_args, "--embedding-mode", mode)
    child <- set_arg(child, "--out-plot", suffix_path(opt$out_plot, mode))
    child <- set_arg(child, "--out-coords", suffix_path(opt$out_coords, mode))
    status <- system2(file.path(R.home("bin"), "Rscript"),
                      c(shQuote(script), vapply(child, shQuote, character(1))))
    if (!identical(status, 0L)) stop(sprintf("%s embedding failed", mode), call. = FALSE)
  }
  quit(save = "no", status = 0L)
}
dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)

read_matrix <- function(path, label) {
  df <- read.csv(path, check.names = FALSE)
  if (!all(vapply(df, is.numeric, logical(1)))) {
    stop(sprintf("All columns in %s must be numeric %s values.", path, label), call. = FALSE)
  }
  df
}

read_clinical <- function(path, expected_rows, shape_col) {
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
  if (!shape_col %in% colnames(df)) {
    return(rep("Unknown", expected_rows))
  }
  vals <- as.character(df[[shape_col]])
  vals[is.na(vals) | vals == ""] <- "Unknown"
  vals
}

differential_methylation_by_subgroup <- function(mat, groups, top_n) {
  groups <- as.character(groups)
  groups[is.na(groups) | groups == ""] <- "Unknown"
  subgroups <- sort(unique(groups[groups != "Unknown"]))

  results <- lapply(subgroups, function(subgroup) {
    in_group <- groups == subgroup
    out_group <- groups != subgroup & groups != "Unknown"

    if (sum(in_group) < 2 || sum(out_group) < 2) {
      warning(
        sprintf(
          "Skipping subgroup '%s': need at least two samples inside and outside the subgroup.",
          subgroup
        ),
        call. = FALSE
      )
      return(NULL)
    }

    x <- as.matrix(mat[in_group, , drop = FALSE])
    y <- as.matrix(mat[out_group, , drop = FALSE])

    n_x <- colSums(!is.na(x))
    n_y <- colSums(!is.na(y))
    mean_x <- colMeans(x, na.rm = TRUE)
    mean_y <- colMeans(y, na.rm = TRUE)
    var_x <- apply(x, 2, var, na.rm = TRUE)
    var_y <- apply(y, 2, var, na.rm = TRUE)

    se <- sqrt((var_x / n_x) + (var_y / n_y))
    t_stat <- (mean_x - mean_y) / se
    df <- ((var_x / n_x) + (var_y / n_y))^2 /
      (((var_x / n_x)^2 / (n_x - 1)) + ((var_y / n_y)^2 / (n_y - 1)))
    p_val <- 2 * pt(abs(t_stat), df = df, lower.tail = FALSE)

    bad <- !is.finite(t_stat) | !is.finite(df) | is.na(p_val)
    t_stat[bad] <- 0
    p_val[bad] <- 1

    out <- data.frame(
      subgroup = subgroup,
      feature = colnames(mat),
      n_subgroup = n_x,
      n_other = n_y,
      mean_subgroup = mean_x,
      mean_other = mean_y,
      avg_diff = mean_x - mean_y,
      t_stat = t_stat,
      p_val = p_val,
      p_adj = p.adjust(p_val, method = "BH"),
      stringsAsFactors = FALSE,
      check.names = FALSE
    )

    out <- out[order(out$p_adj, -abs(out$avg_diff), out$feature), , drop = FALSE]
    head(out, top_n)
  })

  results <- Filter(Negate(is.null), results)
  if (length(results) == 0) {
    return(data.frame())
  }
  combined_results <- do.call(rbind, results)
  rownames(combined_results) <- NULL
  combined_results
}

plot_dm_heatmap <- function(real_mat, synthetic_mat, real_groups, synthetic_groups,
                            dm_results, out_path, dm_top_n) {
  if (nrow(dm_results) == 0) {
    warning("No differential methylation features were available for heatmap generation.", call. = FALSE)
    return(invisible(FALSE))
  }

  selected_features <- unique(dm_results$feature)
  selected_features <- selected_features[selected_features %in% colnames(real_mat) &
                                           selected_features %in% colnames(synthetic_mat)]
  if (length(selected_features) == 0) {
    warning("No selected methylation features were shared by real and synthetic matrices.", call. = FALSE)
    return(invisible(FALSE))
  }

  feature_meta <- dm_results[
    match(selected_features, dm_results$feature),
    c("feature", "subgroup", "avg_diff", "p_adj")
  ]
  colnames(feature_meta) <- c("feature", "assigned_subgroup", "avg_diff", "p_adj")
  feature_meta <- feature_meta[
    order(feature_meta$assigned_subgroup, -feature_meta$avg_diff,
          feature_meta$p_adj, feature_meta$feature),
    ,
    drop = FALSE
  ]
  feature_order <- feature_meta$feature

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
        t(as.matrix(real_mat[real_idx, feature_order, drop = FALSE])),
        t(as.matrix(synthetic_mat[synthetic_idx, feature_order, drop = FALSE]))
      ),
      sample_ids = c(
        sprintf("Real_%03d", real_idx),
        sprintf("Synthetic_%03d", synthetic_idx)
      ),
      source = c(
        rep("Real", length(real_idx)),
        rep("Synthetic", length(synthetic_idx))
      ),
      subgroup = c(
        rep(subgroup, length(real_idx)),
        rep(subgroup, length(synthetic_idx))
      )
    )
  })

  combined_mat <- do.call(cbind, lapply(ordered_blocks, `[[`, "mat"))
  combined_mat[is.na(combined_mat)] <- 0
  colnames(combined_mat) <- unlist(lapply(ordered_blocks, `[[`, "sample_ids"), use.names = FALSE)
  rownames(combined_mat) <- feature_order
  real_reference <- t(as.matrix(real_mat[, feature_order, drop = FALSE]))
  real_center <- rowMeans(real_reference, na.rm = TRUE)
  real_scale <- apply(real_reference, 1, sd, na.rm = TRUE)
  real_scale[!is.finite(real_scale) | real_scale < 1e-8] <- 1
  combined_mat <- sweep(sweep(combined_mat, 1, real_center, "-"), 1, real_scale, "/")

  annotation_col <- data.frame(
    Source = factor(unlist(lapply(ordered_blocks, `[[`, "source"), use.names = FALSE),
                    levels = c("Real", "Synthetic")),
    Subgroup = factor(unlist(lapply(ordered_blocks, `[[`, "subgroup"), use.names = FALSE),
                      levels = subgroup_levels),
    check.names = FALSE
  )
  rownames(annotation_col) <- colnames(combined_mat)

  annotation_row <- data.frame(
    AssignedSubgroup = factor(feature_meta$assigned_subgroup, levels = subgroup_levels),
    check.names = FALSE
  )
  rownames(annotation_row) <- feature_meta$feature

  subgroup_change_idx <- which(annotation_col$Subgroup[-1] != annotation_col$Subgroup[-nrow(annotation_col)])
  source_change_idx <- which(annotation_col$Source[-1] != annotation_col$Source[-nrow(annotation_col)])
  gaps_col <- sort(unique(c(subgroup_change_idx, source_change_idx)))
  row_change_idx <- which(annotation_row$AssignedSubgroup[-1] != annotation_row$AssignedSubgroup[-nrow(annotation_row)])

  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  pheatmap::pheatmap(
    mat = combined_mat,
    filename = out_path,
    scale = "none",
    cluster_cols = FALSE,
    cluster_rows = FALSE,
    show_colnames = FALSE,
    border_color = NA,
    annotation_col = annotation_col,
    annotation_row = annotation_row,
    gaps_col = gaps_col,
    gaps_row = row_change_idx,
    main = sprintf(
      "Real vs synthetic individual-patient methylation heatmap\n%d unique features from top %d real-data differential features per subgroup",
      length(feature_order),
      dm_top_n
    ),
    fontsize_row = 8,
    fontsize_col = 6,
    fontsize = 9,
    width = max(10, min(28, ncol(combined_mat) * 0.085 + 4)),
    height = max(8, min(22, nrow(combined_mat) * 0.22 + 3))
  )

  invisible(TRUE)
}

real_mat <- read_matrix(opt$real, "methylation")
synthetic_mat <- read_matrix(opt$synthetic, "methylation")

shared_features <- intersect(colnames(real_mat), colnames(synthetic_mat))
if (length(shared_features) == 0) {
  stop("The real and synthetic matrices do not share any methylation feature columns.", call. = FALSE)
}

real_mat <- real_mat[, shared_features, drop = FALSE]
synthetic_mat <- synthetic_mat[, shared_features, drop = FALSE]

combined <- rbind(real_mat, synthetic_mat)
# Preserve the historical joint display by default; real_reference is available
# for strict cross-run comparison but can concentrate transformed query points.
feature_var <- apply(if (opt$embedding_mode == "combined") combined else real_mat, 2, var, na.rm = TRUE)
feature_var[is.na(feature_var)] <- 0

n_keep <- min(opt$n_features, length(feature_var))
top_features <- names(sort(feature_var, decreasing = TRUE))[seq_len(n_keep)]

n_real <- nrow(real_mat)
real_shape <- read_clinical(opt$real_clinical, n_real, opt$shape_col)
synthetic_shape <- read_clinical(opt$synthetic_clinical, nrow(synthetic_mat), opt$shape_col)
real_color <- read_clinical(opt$real_clinical, n_real, opt$color_col)
synthetic_color <- read_clinical(opt$synthetic_clinical, nrow(synthetic_mat), opt$color_col)

dm_results <- differential_methylation_by_subgroup(real_mat, real_color, opt$dm_top_n)
dir.create(dirname(opt$out_dm_features), recursive = TRUE, showWarnings = FALSE)
write.csv(dm_results, opt$out_dm_features, row.names = FALSE)

heatmap_written <- plot_dm_heatmap(
  real_mat,
  synthetic_mat,
  real_color,
  synthetic_color,
  dm_results,
  opt$out_heatmap,
  opt$dm_top_n
)

set.seed(opt$seed)
if (opt$embedding_mode == "combined") {
  scaled <- scale(combined[, top_features, drop = FALSE])
  scaled[is.na(scaled)] <- 0
  umap_coords <- uwot::umap(scaled, n_neighbors = opt$neighbors,
                            min_dist = opt$min_dist, metric = "euclidean", verbose = TRUE)
} else {
  real_top <- real_mat[, top_features, drop = FALSE]
  synthetic_top <- synthetic_mat[, top_features, drop = FALSE]
  real_center <- colMeans(real_top, na.rm = TRUE)
  real_scale <- apply(real_top, 2, sd, na.rm = TRUE)
  real_scale[!is.finite(real_scale) | real_scale == 0] <- 1
  scaled_real <- sweep(sweep(real_top, 2, real_center, "-"), 2, real_scale, "/")
  scaled_synthetic <- sweep(sweep(synthetic_top, 2, real_center, "-"), 2, real_scale, "/")
  scaled_real[is.na(scaled_real)] <- 0; scaled_synthetic[is.na(scaled_synthetic)] <- 0
  umap_model <- uwot::umap(scaled_real, n_neighbors = opt$neighbors,
                            min_dist = opt$min_dist, metric = "euclidean", verbose = TRUE,
                            ret_model = TRUE)
  umap_coords <- rbind(umap_model$embedding, uwot::umap_transform(scaled_synthetic, umap_model))
}

plot_df <- data.frame(
  UMAP1 = umap_coords[, 1],
  UMAP2 = umap_coords[, 2],
  source = c(rep("Real", n_real), rep("Synthetic", nrow(synthetic_mat))),
  color_group = c(real_color, synthetic_color),
  shape_group = c(real_shape, synthetic_shape)
)
plot_df$color_group <- factor(plot_df$color_group)
plot_df$shape_group <- factor(plot_df$shape_group)
plot_df$source <- factor(plot_df$source, levels = c("Real", "Synthetic"))

dir.create(dirname(opt$out_plot), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(opt$out_coords), recursive = TRUE, showWarnings = FALSE)

write.csv(plot_df, opt$out_coords, row.names = FALSE)

p <- ggplot(plot_df, aes(x = UMAP1, y = UMAP2, color = color_group, shape = source)) +
  geom_point(alpha = 0.8, size = 1.8) +
  scale_shape_manual(values = c("Real" = 16, "Synthetic" = 17)) +
  labs(
    title = sprintf("UMAP of real and synthetic methylation (%d most variable features)", n_keep),
    subtitle = sprintf(
      "Shared features: %d | Real: %d | Synthetic: %d | Colour: %s | Shape: Source",
      length(shared_features), nrow(real_mat), nrow(synthetic_mat), opt$color_col
    ),
    x = "UMAP1",
    y = "UMAP2",
    color = opt$color_col,
    shape = "Source"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  )

ggsave(opt$out_plot, plot = p, width = 8, height = 6, dpi = 300)

cat(sprintf("UMAP plot written to %s\n", opt$out_plot))
cat(sprintf("UMAP coordinates written to %s\n", opt$out_coords))
cat(sprintf("Outputs written under %s\n", opt$output_dir))
cat(sprintf("Shared methylation features considered: %d\n", length(shared_features)))
cat(sprintf("Top variable methylation features used: %d\n", n_keep))
cat(sprintf("Top real-data differential methylation features written to %s\n", opt$out_dm_features))
if (isTRUE(heatmap_written)) {
  cat(sprintf("Differential methylation heatmap written to %s\n", opt$out_heatmap))
}
