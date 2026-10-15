#!/usr/bin/env Rscript
## Generic CNV feature preparation for MANTIS-MB via conumee2, ported from a
## study-specific adaptation script into a config-driven, cohort-agnostic tool. The algorithm is
## unchanged from that source: compute one common probe set across every
## query Mset AND the control Mset (a documented correctness fix over naive
## per-batch annotation subsetting, which gives each batch different bins and
## non-comparable log2 ratios), build the conumee2 annotation once from that
## common set, aggregate to MANTIS's 16,067-gene order (median CNV ratio per
## gene across its overlapping probes, minus the per-sample bin shift), and
## impute unmapped genes with 0 (the model's neutral prior, "no change from
## reference"), tracked in two distinct categories (no coordinate match at
## all, vs. mapped but all overlapping probes excluded by the common-probe
## restriction).
##
## --control-mset and --noise-threshold have NO default and are required.
## A normal-tissue reference appropriate for medulloblastoma (cerebellum) is
## not necessarily appropriate for a different tumour type/location -- this
## is a judgement call for you to make deliberately, not something a generic
## tool should silently assume. The noise threshold is similarly cohort- and
## fit-specific: an a priori "typical conumee" guess of 0.3 can exclude almost
## every sample in a cohort --
## derive yours from your OWN cohort's observed noise distribution (this
## script writes conumee_noise per sample to <out-dir>/cnv_qc.csv; a common
## starting point is the Tukey outlier fence, Q3 + 1.5*IQR, over the pooled
## noise column), don't reuse a value tuned for a different dataset.
##
## Usage:
##   Rscript R/scripts/prepare_mantis_cnv.R \
##     --msets "cohort1=path1.rds,cohort2=path2.rds" \
##     --control-mset control.rds \
##     --gene-order annot/cnv_gene_order.txt \
##     --noise-threshold 0.77 \
##     --out-dir outputs/mantis/cnv \
##     [--array-type 450k] [--n-test N] [--phenotype path.csv]

ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Package '%s' is required but not installed. Install it with install.packages('%s').",
                 pkg, pkg), call. = FALSE)
  }
}

ensure_bioc_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf(
      "Package '%s' is required but not installed. It is a Bioconductor package, not CRAN -- install it with:\n  install.packages('BiocManager')\n  BiocManager::install('%s')",
      pkg, pkg
    ), call. = FALSE)
  }
}

defaults <- list(
  msets = "",
  control_mset = "",
  gene_order = "",
  noise_threshold = "",
  phenotype = "",
  array_type = "450k",
  out_dir = "outputs/mantis/cnv",
  n_test = ""
)

parse_args <- function(defaults) {
  args <- commandArgs(trailingOnly = TRUE)
  out <- defaults
  if ("--help" %in% args || "-h" %in% args) {
    cat(paste0(
      "Usage: Rscript R/scripts/prepare_mantis_cnv.R [options]\n\n",
      "Options:\n",
      "  --msets \"name=path,name=path\"  one or more query Msets (RDS), REQUIRED\n",
      "  --control-mset PATH             normal-tissue reference Mset (RDS), REQUIRED, no default\n",
      "  --gene-order PATH               MANTIS's 16,067-gene order (annot/cnv_gene_order.txt), REQUIRED\n",
      "  --noise-threshold FLOAT         conumee noise exclusion cutoff, REQUIRED, no default -- derive\n",
      "                                  from your own cohort's noise distribution, see script header\n",
      "  --phenotype PATH                optional sample sheet with ConsensusMYC/ConsensusMYCN columns,\n",
      "                                  for the MYC/MYCN positive-control QC check (skipped if absent)\n",
      "  --array-type TYPE               450k (default) or EPIC -- do not mix array types in one run\n",
      "  --out-dir PATH                  output directory (default: outputs/mantis/cnv)\n",
      "  --n-test N                      subset every Mset to its first N samples, for a cheap dry run\n"
    ))
    quit(save = "no", status = 0)
  }
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop(sprintf("Unexpected argument: %s", key), call. = FALSE)
    if (i == length(args)) stop(sprintf("Missing value for argument: %s", key), call. = FALSE)
    value <- args[[i + 1L]]
    name <- gsub("-", "_", substring(key, 3L))
    if (!name %in% names(out)) stop(sprintf("Unknown argument: %s", key), call. = FALSE)
    out[[name]] <- value
    i <- i + 2L
  }
  if (!nzchar(out$msets)) stop("--msets is required (e.g. \"cohort1=path1.rds,cohort2=path2.rds\").", call. = FALSE)
  if (!nzchar(out$control_mset)) stop("--control-mset is required, no default -- see script header.", call. = FALSE)
  if (!nzchar(out$gene_order)) stop("--gene-order is required.", call. = FALSE)
  if (!nzchar(out$noise_threshold)) stop("--noise-threshold is required, no default -- see script header.", call. = FALSE)
  out$noise_threshold <- as.numeric(out$noise_threshold)
  out$n_test <- if (nzchar(out$n_test)) as.integer(out$n_test) else NA_integer_
  out
}

opt <- parse_args(defaults)

MSETS <- (function(spec) {
  pairs <- strsplit(spec, ",")[[1]]
  kv <- strsplit(pairs, "=")
  setNames(vapply(kv, `[`, character(1), 2), vapply(kv, `[`, character(1), 1))
})(opt$msets)

suppressMessages({
  ensure_bioc_package("minfi")
  ensure_bioc_package("conumee2")
  ensure_bioc_package("GenomicRanges")
  ensure_package("digest")
  library(minfi)
  library(conumee2)
  library(GenomicRanges)
  library(digest)
})

dir.create(opt$out_dir, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("=== MANTIS CNV preparation %s ===\n",
            if (!is.na(opt$n_test)) sprintf("(TEST MODE: first %d samples/cohort)", opt$n_test) else "(FULL RUN)"))

## ===== Load all Msets + control, compute the common probe set ONCE =====
cat("\nLoading Msets...\n")
msets <- lapply(MSETS, function(p) { cat(" ", p, "\n"); readRDS(p) })
control.mSet <- readRDS(opt$control_mset)

if (!is.na(opt$n_test)) {
  msets <- lapply(msets, function(m) m[, seq_len(min(opt$n_test, ncol(m)))])
  cat(sprintf("TEST MODE: subset each Mset to %d samples\n", opt$n_test))
}

common_probes <- Reduce(intersect, c(lapply(msets, rownames), list(rownames(control.mSet))))
cat("Common probes across all Msets + control:", length(common_probes), "\n")

writeLines(common_probes, file.path(opt$out_dir, "common_probes.txt"))
common_probes_md5 <- digest(paste(sort(common_probes), collapse = "\n"), algo = "md5")
writeLines(common_probes_md5, file.path(opt$out_dir, "common_probes_md5.txt"))
cat("common_probes.txt + md5 written:", common_probes_md5, "\n")

## ===== Build the annotation ONCE, subset to common_probes =====
## (the correctness fix this script exists to make: NOT subsetting per-batch,
## which would give each cohort a different probe set -> different bins ->
## non-comparable log2 ratios across cohorts)
data(exclude_regions)
data(detail_regions)
anno <- CNV.create_anno(chrXY = FALSE, exclude_regions = exclude_regions,
                        detail_regions = detail_regions, array_type = opt$array_type)
anno@probes <- anno@probes[names(anno@probes) %in% common_probes]
cat("Annotation probes after common-probe restriction:", length(anno@probes), "\n")

## ===== target_genes from --gene-order (MANTIS's 16,067), not conumee's own set =====
## conumee2 ships an internal (unexported) `genes` GRanges - used here only as a
## coordinate lookup, subset down to MANTIS's own gene list. Matched on SYMBOL;
## unmapped symbols are reported, not silently dropped.
gene_order <- readLines(opt$gene_order)
cat("\nMANTIS CNV gene order:", length(gene_order), "genes\n")

all_genes <- conumee2:::genes
matched <- gene_order %in% all_genes$SYMBOL
unmapped_genes <- gene_order[!matched]
cat("Genes matched to annotation coordinates:", sum(matched), "/", length(gene_order), "\n")
cat("Unmapped genes (no coordinate match - will be imputed 0):", length(unmapped_genes), "\n")
writeLines(unmapped_genes, file.path(opt$out_dir, "cnv_unmapped_genes.txt"))

target_genes <- all_genes[all_genes$SYMBOL %in% gene_order]
if (any(duplicated(target_genes$SYMBOL))) {
  n_dup <- sum(duplicated(target_genes$SYMBOL))
  cat("NOTE:", n_dup, "duplicate SYMBOL entries in the annotation - keeping first occurrence only\n")
  target_genes <- target_genes[!duplicated(target_genes$SYMBOL)]
}
cat("target_genes GRanges built:", length(target_genes), "unique genes with coordinates\n")

## Precompute gene->probe overlaps ONCE (shared across all cohorts - same anno@probes)
d1 <- as.matrix(findOverlaps(query = target_genes, subject = anno@probes))
d2 <- data.frame(gene  = values(target_genes)$SYMBOL[d1[, "queryHits"]],
                 probe = names(anno@probes)[d1[, "subjectHits"]],
                 stringsAsFactors = FALSE)
genes_with_probes <- unique(d2$gene)
cat("Genes with >=1 overlapping probe in the common-probe annotation:", length(genes_with_probes),
    "/", length(gene_order), "\n")

## Optional MYC/MYCN positive-control ground truth (only if a phenotype sheet is supplied
## with these exact columns -- this is a diagnostic aid, not a required pipeline step).
pheno <- NULL
if (nzchar(opt$phenotype)) {
  pheno <- tryCatch(read.csv(opt$phenotype, stringsAsFactors = FALSE), error = function(e) NULL)
  if (is.null(pheno) || !all(c("ConsensusMYC", "ConsensusMYCN") %in% names(pheno))) {
    cat("[warn] --phenotype given but missing ConsensusMYC/ConsensusMYCN columns -- skipping MYC/MYCN QC.\n")
    pheno <- NULL
  }
}

## ===== Per-cohort CNV pipeline =====
control.cnv <- CNV.load(control.mSet[common_probes, ])
qc_rows <- list()

for (cohort in names(msets)) {
  cat(sprintf("\n=== Cohort: %s ===\n", cohort))
  mset <- msets[[cohort]]
  sample_ids <- colnames(mset)  ## native Mset order - MUST match this same cohort's
                                 ## beta_sample_order.txt from pode_mantis prepare-beta
  n_samples <- length(sample_ids)
  cat("Samples:", n_samples, "\n")

  t0 <- Sys.time()
  test.cnv <- CNV.load(mset[common_probes, ])
  names(test.cnv) <- sample_ids

  x <- CNV.fit(test.cnv, control.cnv, anno)
  x <- CNV.bin(x)
  x <- CNV.detail(x)
  x <- CNV.segment(x)
  cat("CNV.fit/bin/detail/segment done in", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s\n")
  ## "intensities are abnormally low (< 5000)" warning here is expected/benign.

  gene_mat <- matrix(NA_real_, nrow = length(gene_order), ncol = n_samples,
                      dimnames = list(gene_order, sample_ids))
  noise_vals <- setNames(numeric(n_samples), sample_ids)

  for (i in seq_len(n_samples)) {
    gene_ratio <- sapply(split(x@fit$ratio[d2$probe, i], d2$gene), median, na.rm = TRUE)
    gene_ratio <- gene_ratio - x@bin$shift[i]
    gene_mat[names(gene_ratio), i] <- gene_ratio
    noise_vals[i] <- x@fit$noise[i]
    if (i %% 50 == 0 || i == n_samples) cat("  gene-level aggregation:", i, "/", n_samples, "\n")
  }

  ## Unmapped genes (no coordinate match at all) -> impute 0, the model's neutral prior.
  gene_mat[unmapped_genes, ] <- 0
  ## Mapped genes whose overlapping probes were all excluded by the common-probe
  ## restriction are a distinct, smaller category - also impute 0, reported separately.
  mapped_but_still_na <- setdiff(rownames(gene_mat)[rowSums(is.na(gene_mat)) > 0], unmapped_genes)
  if (length(mapped_but_still_na) > 0) {
    cat("Mapped genes with no surviving probes post common-probe restriction (imputed 0):",
        length(mapped_but_still_na), "\n")
  }
  gene_mat[is.na(gene_mat)] <- 0

  ## --- Outputs: bare numeric CSV, genes as rows (canonical order), samples as columns --
  ## matching the orientation load_cnv_features actually requires on disk, NOT the
  ## "samples x 16,067" phrasing that describes the in-memory shape after the reference
  ## implementation's own internal transpose.
  out_path <- file.path(opt$out_dir, sprintf("cnv_%s_numeric.csv", cohort))
  write.table(gene_mat, out_path, sep = ",", row.names = FALSE, col.names = FALSE)
  cat("Wrote", out_path, ":", nrow(gene_mat), "x", ncol(gene_mat), "\n")

  order_path <- file.path(opt$out_dir, sprintf("cnv_%s_sample_order.txt", cohort))
  writeLines(sample_ids, order_path)

  myc_gt <- mycn_gt <- rep(NA, n_samples)
  if (!is.null(pheno)) {
    m <- match(sample_ids, pheno[[1]])
    myc_gt <- pheno$ConsensusMYC[m]
    mycn_gt <- pheno$ConsensusMYCN[m]
  }

  for (i in seq_len(n_samples)) {
    qc_rows[[length(qc_rows) + 1]] <- data.frame(
      cohort = cohort,
      sample_id_raw = sample_ids[i],
      conumee_noise = noise_vals[i],
      n_genes_imputed_unmapped = length(unmapped_genes),
      n_genes_imputed_no_probes = length(mapped_but_still_na),
      pct_genes_imputed = 100 * (length(unmapped_genes) + length(mapped_but_still_na)) / length(gene_order),
      noise_exclude_flag = noise_vals[i] > opt$noise_threshold,
      myc_ratio = gene_mat["MYC", i],
      mycn_ratio = gene_mat["MYCN", i],
      myc_ground_truth = myc_gt[i],
      mycn_ground_truth = mycn_gt[i]
    )
  }
}

qc_df <- do.call(rbind, qc_rows)
write.csv(qc_df, file.path(opt$out_dir, "cnv_qc.csv"), row.names = FALSE)
cat("\nWrote", file.path(opt$out_dir, "cnv_qc.csv"), ":", nrow(qc_df), "rows\n")
cat("Noise threshold:", opt$noise_threshold, "- samples flagged for exclusion:",
    sum(qc_df$noise_exclude_flag), "/", nrow(qc_df), "\n")

if (!is.null(pheno)) {
  has_gt <- !is.na(qc_df$myc_ground_truth)
  if (any(has_gt)) {
    pred_myc <- qc_df$myc_ratio[has_gt] > 0.3
    true_myc <- as.logical(qc_df$myc_ground_truth[has_gt])
    sens <- sum(pred_myc & true_myc) / max(1, sum(true_myc))
    spec <- sum(!pred_myc & !true_myc) / max(1, sum(!true_myc))
    cat(sprintf("MYC QC (ratio > 0.3 vs ConsensusMYC, n=%d): sensitivity=%.3f specificity=%.3f\n",
                sum(has_gt), sens, spec))
  }
}

cat("\n=== MANTIS CNV preparation done ===\n")
