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
  real = "data/expression.csv",
  synthetic = "runs/final_handoff/decoded_from_pode_2/synthetic_expression.csv",
  real_clinical = "data/clinical.csv",
  synthetic_clinical = "runs/final_handoff/decoded_from_pode_2/synthetic_clinical.csv",
  shape_col = "WCA",
  color_col = "MNP_Subgroup",
  n_genes = 5000L,
  de_top_n = 10L,
  seed = 42L,
  neighbors = 15L,
  min_dist = 0.3,
  embedding_mode = "real_reference",
  output_dir = "data",
  out_plot = "data/real_vs_synthetic_umap.png",
  out_coords = "data/real_vs_synthetic_umap_coords.csv",
  out_de_genes = "data/real_subgroup_top_de_genes.csv",
  out_heatmap = "data/real_vs_synthetic_de_heatmap.png"
)

parse_args <- function(defaults) {
  args <- commandArgs(trailingOnly = TRUE)
  out <- defaults
  provided <- character()

  if ("--help" %in% args || "-h" %in% args) {
    cat(
      paste0(
        "Usage: Rscript data/plot_real_vs_synthetic_umap.R [options]\n\n",
        "Options:\n",
        "  --real PATH          real/original expression CSV\n",
        "  --synthetic PATH     synthetic expression CSV\n",
        "  --real-clinical PATH real/original clinical CSV\n",
        "  --synthetic-clinical PATH synthetic clinical CSV\n",
        "  --shape-col NAME     clinical column to map to point shape\n",
        "  --color-col NAME     clinical column to map to point colour\n",
        "  --n-genes INT        number of most variable genes to keep\n",
        "  --de-top-n INT       number of top real-data DE genes to keep per subgroup\n",
        "  --seed INT           random seed for UMAP\n",
        "  --neighbors INT      UMAP n_neighbors\n",
        "  --min-dist FLOAT     UMAP min_dist\n",
        "  --embedding-mode MODE real_reference (default), combined, or both\n",
        "  --output-dir PATH    directory for generated outputs\n",
        "  --out-plot PATH      output PNG path\n",
        "  --out-coords PATH    output CSV of UMAP coordinates\n",
        "  --out-de-genes PATH  output CSV of top subgroup DE genes from real data\n",
        "  --out-heatmap PATH   output PNG heatmap of selected DE genes\n"
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

  out$n_genes <- as.integer(out$n_genes)
  out$de_top_n <- as.integer(out$de_top_n)
  out$seed <- as.integer(out$seed)
  out$neighbors <- as.integer(out$neighbors)
  out$min_dist <- as.numeric(out$min_dist)
  if (!out$embedding_mode %in% c("combined", "real_reference", "both")) {
    stop("--embedding-mode must be 'combined', 'real_reference', or 'both'", call. = FALSE)
  }
  output_names <- c("out_plot", "out_coords", "out_de_genes", "out_heatmap")
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

read_expression <- function(path) {
  df <- read.csv(path, check.names = FALSE)
  if (!all(vapply(df, is.numeric, logical(1)))) {
    stop(sprintf("All columns in %s must be numeric expression values.", path), call. = FALSE)
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

differential_expression_by_subgroup <- function(expr, groups, top_n) {
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

    x <- as.matrix(expr[in_group, , drop = FALSE])
    y <- as.matrix(expr[out_group, , drop = FALSE])

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
      gene = colnames(expr),
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

    out <- out[order(out$p_adj, -abs(out$avg_diff), out$gene), , drop = FALSE]
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

plot_de_heatmap <- function(real_expr, synthetic_expr, real_groups, synthetic_groups,
                            de_results, out_path, color_col, de_top_n) {
  if (nrow(de_results) == 0) {
    warning("No DE genes were available for heatmap generation.", call. = FALSE)
    return(invisible(FALSE))
  }

  selected_genes <- unique(de_results$gene)
  selected_genes <- selected_genes[selected_genes %in% colnames(real_expr) &
                                     selected_genes %in% colnames(synthetic_expr)]
  if (length(selected_genes) == 0) {
    warning("No selected DE genes were shared by real and synthetic matrices.", call. = FALSE)
    return(invisible(FALSE))
  }

  gene_meta <- de_results[match(selected_genes, de_results$gene), c("gene", "subgroup", "avg_diff", "p_adj")]
  colnames(gene_meta) <- c("gene", "assigned_subgroup", "avg_diff", "p_adj")
  gene_meta <- gene_meta[order(gene_meta$assigned_subgroup, -gene_meta$avg_diff, gene_meta$p_adj, gene_meta$gene), , drop = FALSE]
  gene_order <- gene_meta$gene

  real_groups <- as.character(real_groups)
  synthetic_groups <- as.character(synthetic_groups)
  real_groups[is.na(real_groups) | real_groups == ""] <- "Unknown"
  synthetic_groups[is.na(synthetic_groups) | synthetic_groups == ""] <- "Unknown"
  subgroup_levels <- sort(unique(c(real_groups, synthetic_groups)))

  ordered_blocks <- lapply(subgroup_levels, function(subgroup) {
    real_idx <- which(real_groups == subgroup)
    synthetic_idx <- which(synthetic_groups == subgroup)

    list(
      expr = cbind(
        t(as.matrix(real_expr[real_idx, gene_order, drop = FALSE])),
        t(as.matrix(synthetic_expr[synthetic_idx, gene_order, drop = FALSE]))
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

  combined_expr <- do.call(cbind, lapply(ordered_blocks, `[[`, "expr"))
  combined_expr[is.na(combined_expr)] <- 0
  colnames(combined_expr) <- unlist(lapply(ordered_blocks, `[[`, "sample_ids"), use.names = FALSE)
  rownames(combined_expr) <- gene_order
  # Use real-data row statistics only.  Scaling on the combined matrix lets a
  # noisy synthetic cohort redefine the colour range and visually compress the
  # real signal.
  real_reference <- t(as.matrix(real_expr[, gene_order, drop = FALSE]))
  real_center <- rowMeans(real_reference, na.rm = TRUE)
  real_scale <- apply(real_reference, 1, sd, na.rm = TRUE)
  real_scale[!is.finite(real_scale) | real_scale < 1e-8] <- 1
  combined_expr <- sweep(sweep(combined_expr, 1, real_center, "-"), 1, real_scale, "/")

  annotation_col <- data.frame(
    Source = factor(unlist(lapply(ordered_blocks, `[[`, "source"), use.names = FALSE),
                    levels = c("Real", "Synthetic")),
    Subgroup = factor(unlist(lapply(ordered_blocks, `[[`, "subgroup"), use.names = FALSE),
                      levels = subgroup_levels),
    check.names = FALSE
  )
  rownames(annotation_col) <- colnames(combined_expr)

  annotation_row <- data.frame(
    AssignedSubgroup = factor(gene_meta$assigned_subgroup, levels = subgroup_levels),
    check.names = FALSE
  )
  rownames(annotation_row) <- gene_meta$gene

  subgroup_change_idx <- which(annotation_col$Subgroup[-1] != annotation_col$Subgroup[-nrow(annotation_col)])
  source_change_idx <- which(annotation_col$Source[-1] != annotation_col$Source[-nrow(annotation_col)])
  gaps_col <- sort(unique(c(subgroup_change_idx, source_change_idx)))
  row_change_idx <- which(annotation_row$AssignedSubgroup[-1] != annotation_row$AssignedSubgroup[-nrow(annotation_row)])

  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  pheatmap::pheatmap(
    mat = combined_expr,
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
      "Real vs synthetic individual-patient heatmap\n%d unique genes from top %d real-data DE genes per subgroup",
      length(gene_order),
      de_top_n
    ),
    fontsize_row = 8,
    fontsize_col = 6,
    fontsize = 9,
    width = max(10, min(28, ncol(combined_expr) * 0.085 + 4)),
    height = max(8, min(22, nrow(combined_expr) * 0.22 + 3))
  )

  invisible(TRUE)
}

real_expr <- read_expression(opt$real)
synthetic_expr <- read_expression(opt$synthetic)

shared_genes <- intersect(colnames(real_expr), colnames(synthetic_expr))
if (length(shared_genes) == 0) {
  stop("The real and synthetic matrices do not share any gene columns.", call. = FALSE)
}

real_expr <- real_expr[, shared_genes, drop = FALSE]
synthetic_expr <- synthetic_expr[, shared_genes, drop = FALSE]

combined <- rbind(real_expr, synthetic_expr)
# `combined` preserves the historical, intuitive joint display.  The optional
# real_reference mode is stricter for comparing generator runs, but UMAP's query
# transform can visually concentrate out-of-sample synthetic points.
gene_var <- apply(if (opt$embedding_mode == "combined") combined else real_expr, 2, var, na.rm = TRUE)
gene_var[is.na(gene_var)] <- 0

n_keep <- min(opt$n_genes, length(gene_var))
top_genes <- names(sort(gene_var, decreasing = TRUE))[seq_len(n_keep)]

n_real <- nrow(real_expr)
real_shape <- read_clinical(opt$real_clinical, n_real, opt$shape_col)
synthetic_shape <- read_clinical(opt$synthetic_clinical, nrow(synthetic_expr), opt$shape_col)
# colour by MNP_Subgroup (or whichever clinical column contains MNP labels)
real_color <- read_clinical(opt$real_clinical, n_real, opt$color_col)
synthetic_color <- read_clinical(opt$synthetic_clinical, nrow(synthetic_expr), opt$color_col)

de_results <- differential_expression_by_subgroup(real_expr, real_color, opt$de_top_n)
dir.create(dirname(opt$out_de_genes), recursive = TRUE, showWarnings = FALSE)
write.csv(de_results, opt$out_de_genes, row.names = FALSE)

heatmap_written <- plot_de_heatmap(
  real_expr,
  synthetic_expr,
  real_color,
  synthetic_color,
  de_results,
  opt$out_heatmap,
  opt$color_col,
  opt$de_top_n
)

set.seed(opt$seed)
if (opt$embedding_mode == "combined") {
  scaled <- scale(combined[, top_genes, drop = FALSE])
  scaled[is.na(scaled)] <- 0
  umap_coords <- uwot::umap(scaled, n_neighbors = opt$neighbors,
                            min_dist = opt$min_dist, metric = "euclidean", verbose = TRUE)
} else {
  real_top <- real_expr[, top_genes, drop = FALSE]
  synthetic_top <- synthetic_expr[, top_genes, drop = FALSE]
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
  source = c(rep("Real", n_real), rep("Synthetic", nrow(synthetic_expr))),
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
    title = sprintf("UMAP of real and synthetic expression (%d most variable genes)", n_keep),
    subtitle = sprintf(
      "Shared genes: %d | Real: %d | Synthetic: %d | Colour: %s | Shape: Source",
      length(shared_genes), nrow(real_expr), nrow(synthetic_expr), opt$color_col
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
cat(sprintf("Shared genes considered: %d\n", length(shared_genes)))
cat(sprintf("Top variable genes used: %d\n", n_keep))
cat(sprintf("Top real-data DE genes written to %s\n", opt$out_de_genes))
if (isTRUE(heatmap_written)) {
  cat(sprintf("DE heatmap written to %s\n", opt$out_heatmap))
}
