.schema_version <- 1L
.weight_refit_modes <- c("covariate_aware", "legacy_unadjusted")

weight_refit_mode <- function(cfg) cfg$final_model$weight_refit %||% "covariate_aware"

weight_refit_explicit <- function(cfg) {
  isTRUE(cfg$`_weight_refit_explicit`) || (!is.null(cfg$final_model) && "weight_refit" %in% names(cfg$final_model))
}

validate_weight_refit <- function(cfg) {
  mode <- weight_refit_mode(cfg)
  if (!mode %in% .weight_refit_modes)
    stop("final_model.weight_refit must be covariate_aware or legacy_unadjusted")
  mode
}

resolve_path <- function(path, base) {
  path <- path.expand(path)
  if (!grepl("^/", path)) path <- file.path(base, path)
  normalizePath(path, mustWork = FALSE)
}

read_config <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  cfg <- yaml::read_yaml(path)
  if (!identical(as.integer(cfg$schema_version), .schema_version))
    stop("schema_version must be ", .schema_version)
  cfg$`_weight_refit_explicit` <- !is.null(cfg$final_model$weight_refit)
  validate_weight_refit(cfg)
  cfg$`_config_path` <- path
  cfg$`_config_dir` <- dirname(path)
  cfg$matrix$path <- resolve_path(cfg$matrix$path, dirname(path))
  cfg$phenotype$path <- resolve_path(cfg$phenotype$path, dirname(path))
  if (!is.null(cfg$annotation$path)) cfg$annotation$path <- resolve_path(cfg$annotation$path, dirname(path))
  cfg$output_dir <- resolve_path(cfg$output_dir %||% "results", dirname(path))
  cfg
}

`%||%` <- function(x, y) if (is.null(x)) y else x

read_table_auto <- function(path, sep = NULL, check.names = FALSE) {
  if (is.null(sep)) sep <- if (grepl("\\.(tsv|txt)$", path, ignore.case = TRUE)) "\t" else ","
  read.table(path, header = TRUE, sep = sep, quote = "\"", comment.char = "",
             check.names = check.names, stringsAsFactors = FALSE)
}

beta_to_m <- function(x) {
  x <- as.matrix(x); storage.mode(x) <- "double"
  if (any(x <= 0 | x >= 1, na.rm = TRUE)) stop("beta values must be strictly between 0 and 1")
  log2(x / (1 - x))
}

validate_inputs <- function(config) {
  cfg <- if (is.character(config)) read_config(config) else config
  required <- c("analysis_id", "matrix", "phenotype", "endpoints")
  missing <- setdiff(required, names(cfg))
  if (length(missing)) stop("missing required keys: ", paste(missing, collapse = ", "))
  if ((cfg$methodology %||% "legacy_exact") != "legacy_exact") stop("methodology must be legacy_exact")
  if ((cfg$search$cross_validation %||% "loo") != "loo") stop("legacy_exact requires search.cross_validation: loo")
  if (!(cfg$selection$mode %||% "manual") %in% c("manual","max_auc")) stop("selection.mode must be manual or max_auc")
  validate_weight_refit(cfg)
  if (length(cfg$endpoints) < 1 || length(cfg$endpoints) > 2) stop("endpoints must contain one or two entries")
  if (!file.exists(cfg$matrix$path) || !file.exists(cfg$phenotype$path)) stop("input file does not exist")
  raw <- read_table_auto(cfg$matrix$path, cfg$matrix$separator)
  feature_col <- cfg$matrix$feature_id_column %||% "first"
  if (identical(feature_col, "first")) feature_col <- 1L
  if (is.character(feature_col)) feature_col <- match(feature_col, names(raw))
  feature_ids <- as.character(raw[[feature_col]])
  if (anyDuplicated(feature_ids)) stop("matrix feature identifiers are not unique")
  matrix <- as.matrix(raw[-feature_col]); rownames(matrix) <- feature_ids; storage.mode(matrix) <- "double"
  if (identical(cfg$matrix$orientation %||% "features_by_samples", "samples_by_features")) matrix <- t(matrix)
  if (anyDuplicated(colnames(matrix))) stop("matrix sample identifiers are not unique")
  if (any(!is.finite(matrix))) stop("matrix contains missing or non-finite values")
  pheno <- read_table_auto(cfg$phenotype$path, cfg$phenotype$separator)
  sample_col <- cfg$phenotype$sample_id_column
  if (!sample_col %in% names(pheno)) stop("phenotype sample_id_column is missing: ", sample_col)
  ids <- as.character(pheno[[sample_col]])
  if (anyDuplicated(ids)) stop("phenotype sample identifiers are not unique")
  rownames(pheno) <- ids
  needed <- unlist(lapply(cfg$endpoints, function(x) c(x$time, x$status)))
  needed <- c(needed, vapply(cfg$covariates %||% list(), `[[`, "", "column"))
  absent <- setdiff(needed, names(pheno))
  if (length(absent)) stop("phenotype columns are missing: ", paste(absent, collapse = ", "))
  common <- intersect(colnames(matrix), rownames(pheno))
  allow <- isTRUE(cfg$validation$allow_unmatched_samples)
  if (!allow && (length(common) != ncol(matrix) || length(common) != nrow(pheno)))
    stop("sample IDs do not match between matrix and phenotype")
  if (!length(common)) stop("matrix and phenotype have no sample IDs in common")
  matrix <- matrix[, common, drop = FALSE]; pheno <- pheno[common, , drop = FALSE]
  scale <- cfg$matrix$scale %||% "beta"
  if (scale == "beta") matrix <- beta_to_m(matrix) else if (scale != "m_value") stop("matrix.scale must be beta or m_value")
  for (ep in cfg$endpoints) {
    time <- suppressWarnings(as.numeric(pheno[[ep$time]])); status <- suppressWarnings(as.numeric(pheno[[ep$status]]))
    if (any(time <= 0, na.rm = TRUE)) stop("endpoint contains non-positive times: ", ep$name)
    if (!all(na.omit(status) %in% c(0, 1))) stop("endpoint status must be 0/1: ", ep$name)
  }
  for (cov in cfg$covariates %||% list()) {
    if (!cov$type %in% c("categorical", "numeric")) stop("covariate.type must be categorical or numeric")
    if (!(cov$mode %||% "adjust") %in% c("strata", "adjust")) stop("covariate.mode must be strata or adjust")
    if (cov$type == "numeric" && anyNA(suppressWarnings(as.numeric(pheno[[cov$column]]))))
      stop("numeric covariate cannot be converted: ", cov$column)
  }
  structure(list(matrix = matrix, phenotype = pheno, endpoints = cfg$endpoints,
                 covariates = cfg$covariates %||% list()), class = "metacpg_inputs")
}

endpoint_rows <- function(data, endpoint) {
  ok <- !is.na(data$phenotype[[endpoint$time]]) & !is.na(data$phenotype[[endpoint$status]])
  for (cov in data$covariates) ok <- ok & !is.na(data$phenotype[[cov$column]])
  which(ok)
}

cox_formula <- function(covariates) {
  adjust <- vapply(Filter(function(x) (x$mode %||% "adjust") == "adjust", covariates), `[[`, "", "column")
  strata <- vapply(Filter(function(x) (x$mode %||% "adjust") == "strata", covariates), `[[`, "", "column")
  rhs <- c("feature_value", adjust, if (length(strata)) paste0("strata(`", strata, "`)") else character())
  as.formula(paste("survival::Surv(.time, .status) ~", paste(rhs, collapse = " + ")))
}

endpoint_stats <- function(matrix, pheno, endpoint, covariates, cores = 1L) {
  work <- function(i) {
    d <- pheno
    d$.time <- as.numeric(d[[endpoint$time]]); d$.status <- as.numeric(d[[endpoint$status]])
    d$feature_value <- as.numeric(matrix[i, ])
    fit <- try(suppressWarnings(survival::coxph(cox_formula(covariates), data = d, ties = "efron")), silent = TRUE)
    if (inherits(fit, "try-error")) return(rep(NA_real_, 5))
    sm <- summary(fit)$coefficients[1, ]
    sdv <- stats::sd(d$feature_value)
    c(coef = sm[["coef"]], se = sm[["se(coef)"]], ordinary_z = sm[["z"]],
      standardized_coef = sm[["coef"]] / sdv,
      standardized_z = (sm[["coef"]] / sdv) / sm[["se(coef)"]], p_value = sm[["Pr(>|z|)"]])
  }
  values <- parallel::mclapply(seq_len(nrow(matrix)), work, mc.cores = max(1L, as.integer(cores)))
  out <- data.frame(feature = rownames(matrix), do.call(rbind, values), check.names = FALSE)
  out$endpoint <- endpoint$name
  out
}

meta_stats <- function(stats, matrix) {
  features <- stats[[1]]$feature
  rows <- lapply(seq_along(features), function(i) {
    te <- vapply(stats, function(x) x$coef[i], numeric(1)); se <- vapply(stats, function(x) x$se[i], numeric(1))
    if (any(!is.finite(c(te, se)))) return(rep(NA_real_, 5))
    fit <- suppressWarnings(meta::metagen(TE = te, seTE = se, common = FALSE, random = TRUE, method.tau = "EB"))
    coef <- fit$TE.random; out_se <- fit$seTE.random
    c(coef = coef, se = out_se, ordinary_z = coef / out_se,
      standardized_coef = coef / stats::sd(as.numeric(matrix[i, ])),
      standardized_z = (coef / stats::sd(as.numeric(matrix[i, ]))) / out_se)
  })
  out <- data.frame(feature = features, do.call(rbind, rows), check.names = FALSE)
  out$p_value <- 2 * stats::pnorm(-abs(out$ordinary_z)); out$endpoint <- "meta"; out
}

rank_stats <- function(stats) {
  stats$abs_z <- abs(stats$ordinary_z)
  stats <- stats[order(-stats$abs_z, na.last = TRUE, method = "radix"), , drop = FALSE]
  stats$rank <- seq_len(nrow(stats)); rownames(stats) <- NULL; stats
}

legacy_scale <- function(raw) {
  center <- stats::median(raw); scale <- stats::IQR(raw)
  list(score = if (scale == 0) rep(NA_real_, length(raw)) else (raw - center) / scale,
       center = center, scale = scale)
}

legacy_fold_score <- function(ranking, n, training_matrix, heldout) {
  top <- head(ranking[is.finite(ranking$standardized_coef), , drop=FALSE], n)
  features <- top$feature; weights <- top$standardized_coef
  train_raw <- apply(training_matrix[features,,drop=FALSE], 2, function(x) mean(x * weights, na.rm=TRUE))
  scaling <- legacy_scale(train_raw)
  heldout_raw <- mean(heldout[features] * weights, na.rm=TRUE)
  list(score=if (scaling$scale == 0) NA_real_ else (heldout_raw-scaling$center)/scaling$scale,
       features=features, center=scaling$center, scale=scaling$scale)
}

legacy_full_scores <- function(matrix, stats, features, scaling_matrix = matrix) {
  chosen <- stats[match(features,stats$feature),,drop=FALSE]
  weights <- chosen$standardized_coef
  raw <- apply(matrix[features,,drop=FALSE],2,function(x) mean(x*weights,na.rm=TRUE))
  scaling_raw <- apply(scaling_matrix[features,,drop=FALSE],2,function(x) mean(x*weights,na.rm=TRUE))
  scaling <- legacy_scale(scaling_raw)
  scores <- if (scaling$scale == 0) rep(NA_real_,length(raw)) else (raw-scaling$center)/scaling$scale
  list(scores=data.frame(sample_id=colnames(matrix),risk_score=scores,raw_score=raw),
       selected=chosen, scaling=list(center=scaling$center,scale=scaling$scale))
}

harrell_c <- function(time, status, score) {
  fit <- survival::coxph(survival::Surv(time, status) ~ score)
  unname(summary(fit)$concordance[[1]])
}

auc_at_time <- function(time, status, score, horizon) {
  ok <- is.finite(time) & is.finite(status) & is.finite(score)
  time <- as.numeric(time[ok]); status <- as.numeric(status[ok]); score <- as.numeric(score[ok])
  if (requireNamespace("survivalROC", quietly = TRUE)) {
    roc <- try(survivalROC::survivalROC(Stime = time, status = status, marker = score,
      predict.time = horizon, span = .25 * length(time)^(-.20)), silent = TRUE)
    if (!inherits(roc, "try-error") && is.finite(roc$AUC)) return(roc$AUC)
  }
  cases <- time <= horizon & status == 1; controls <- time > horizon
  if (!any(cases) || !any(controls)) return(NA_real_)
  mean(outer(score[cases], score[controls], `>`) + .5 * outer(score[cases], score[controls], `==`))
}

legacy_task <- function(data,n,e,heldout) {
  all_stats <- list(); endpoint_sets <- list(); fit_count <- 0L
  for (k in seq_along(data$endpoints)) {
    rows <- endpoint_rows(data,data$endpoints[[k]])
    if (k == e) rows <- setdiff(rows,heldout)
    endpoint_sets[[k]] <- data$matrix[,rows,drop=FALSE]
    all_stats[[k]] <- endpoint_stats(endpoint_sets[[k]],data$phenotype[rows,,drop=FALSE],
      data$endpoints[[k]],data$covariates,1L)
    fit_count <- fit_count+nrow(data$matrix)
  }
  if (length(all_stats)==1) {
    combined <- all_stats[[1]]; normalization_matrix <- endpoint_sets[[1]]
  } else {
    combined <- meta_stats(all_stats,do.call(cbind,endpoint_sets))
    if (e==1) {
      secondary_rows <- endpoint_rows(data,data$endpoints[[2]])
      normalization_matrix <- cbind(endpoint_sets[[1]],data$matrix[,secondary_rows,drop=FALSE])
    } else {
      secondary_rows <- endpoint_rows(data,data$endpoints[[2]])
      normalization_matrix <- cbind(data$matrix[,secondary_rows,drop=FALSE],endpoint_sets[[2]])
    }
  }
  ranking <- rank_stats(combined)
  fold <- legacy_fold_score(ranking,n,normalization_matrix,data$matrix[,heldout])
  ep <- data$endpoints[[e]]
  list(score=data.frame(sample_id=colnames(data$matrix)[heldout],endpoint=ep$name,
    endpoint_index=e-1L,heldout_index=heldout-1L,feature_count=n,risk_score=fold$score,
    fold_center=fold$center,fold_scale=fold$scale,time=as.numeric(data$phenotype[heldout,ep$time]),
    status=as.numeric(data$phenotype[heldout,ep$status])),
    selected=data.frame(feature_count=n,feature=fold$features,endpoint=ep$name),fit_count=fit_count)
}

exact_loo <- function(data,counts,phase,cores,cfg,allow_resume=FALSE) {
  tasks <- do.call(rbind,lapply(counts,function(n) do.call(rbind,lapply(seq_along(data$endpoints),function(e)
    cbind(n=n,e=e,heldout=endpoint_rows(data,data$endpoints[[e]]))))))
  resume_id <- if (allow_resume) cfg$execution$resume_run_id else NULL
  cache_dir <- if (!is.null(resume_id)) file.path(cfg$output_dir,"cache",resume_id,config_fingerprint(cfg),phase) else NULL
  if (!is.null(cache_dir)) dir.create(cache_dir,recursive=TRUE,showWarnings=FALSE)
  worker <- function(task) {
    n <- task[[1]]; e <- task[[2]]; heldout <- task[[3]]
    cache <- if (!is.null(cache_dir)) file.path(cache_dir,paste0("n",n,"_endpoint",e-1L,"_fold",heldout-1L,".rds")) else NULL
    if (!is.null(cache) && file.exists(cache)) { result <- readRDS(cache); result$fit_count <- 0; resumed <- TRUE } else {
      result <- legacy_task(data,n,e,heldout); resumed <- FALSE
      if (!is.null(cache)) { temporary <- paste0(cache,".tmp"); saveRDS(result,temporary); file.rename(temporary,cache) }
    }
    result$provenance <- data.frame(phase=phase,feature_count=n,endpoint_index=e-1L,
      heldout_index=heldout-1L,resumed=resumed,run_id=cfg$`_run_id`,
      ranking_statistic="abs_ordinary_cox_z",scaling="fold_median_iqr")
    result
  }
  out <- parallel::mclapply(split(tasks,row(tasks)),worker,mc.cores=max(1L,as.integer(cores)))
  scores <- do.call(rbind,lapply(out,`[[`,"score")); selected <- do.call(rbind,lapply(out,`[[`,"selected"))
  freq <- aggregate(list(fold_count=rep(1L,nrow(selected))),selected[c("feature_count","feature")],sum)
  totals <- aggregate(sample_id~feature_count,scores,length); names(totals)[2] <- "total_folds"
  freq <- merge(freq,totals,by="feature_count"); freq$stability_denominator <- length(endpoint_rows(data,data$endpoints[[1]]))
  freq$inclusion_percent <- 100*freq$fold_count/freq$stability_denominator
  freq <- freq[order(freq$feature_count,-freq$inclusion_percent,freq$feature),]
  list(scores=scores,inclusion_frequency=freq,task_provenance=do.call(rbind,lapply(out,`[[`,"provenance")),
       fit_count=sum(vapply(out,`[[`,numeric(1),"fit_count")))
}

legacy_performance <- function(time,status,score,horizon) {
  ok <- is.finite(time)&is.finite(status)&is.finite(score)
  fit <- suppressWarnings(survival::coxph(survival::Surv(time[ok],status[ok])~score[ok]))
  lp <- stats::predict(fit,type="lp")
  list(auc=auc_at_time(time[ok],status[ok],lp,horizon),c_index=harrell_c(time[ok],status[ok],lp))
}

search_features <- function(config, data = NULL) {
  cfg <- if (is.character(config)) read_config(config) else config
  if (is.null(data)) data <- validate_inputs(cfg)
  if ((cfg$methodology %||% "legacy_exact") != "legacy_exact") stop("only methodology: legacy_exact is currently supported")
  cores <- as.integer(cfg$execution$workers %||% 1L); full_fit_count <- 0L; endpoint_matrices <- list()
  stats <- lapply(seq_along(data$endpoints), function(k) {
    rows <- endpoint_rows(data, data$endpoints[[k]])
    endpoint_matrices[[k]] <<- data$matrix[,rows,drop=FALSE]; full_fit_count <<- full_fit_count+nrow(data$matrix)
    endpoint_stats(data$matrix[, rows, drop = FALSE], data$phenotype[rows, , drop = FALSE], data$endpoints[[k]], data$covariates, cores)
  })
  ranking <- rank_stats(if (length(stats) == 1) stats[[1]] else meta_stats(stats, do.call(cbind,endpoint_matrices)))
  s <- cfg$search %||% list(); counts <- seq(s$min_features %||% 5L, s$max_features %||% 60L, s$step %||% 1L)
  if ((s$cross_validation %||% "loo") != "loo") stop("legacy_exact requires leave-one-out cross-validation")
  cv <- exact_loo(data,counts,"search",cores,cfg,TRUE); scores <- cv$scores; frequency <- cv$inclusion_frequency
  horizon <- min(vapply(data$endpoints, function(x) as.numeric(x$prediction_time %||% 1095), numeric(1)))
  metrics <- do.call(rbind,lapply(split(scores,scores$feature_count),function(x) {
    perf <- legacy_performance(x$time,x$status,x$risk_score,horizon)
    data.frame(feature_count=x$feature_count[1],auc=perf$auc,c_index=perf$c_index)
  }))
  logical_fits <- length(data$endpoints)*nrow(data$matrix)*(1+nrow(scores))
  list(ranking=ranking,scores=scores,metrics=metrics,endpoint_stats=do.call(rbind,stats),
    inclusion_frequency=frequency,task_provenance=cv$task_provenance,search_cox_fit_count=logical_fits,
    search_cox_fits_executed=full_fit_count+cv$fit_count)
}

select_feature_count <- function(metrics, override = NULL, mode = "manual") {
  if (!is.null(override)) {
    if (!override %in% metrics$feature_count) stop("feature-count override is outside the search range")
    return(as.integer(override))
  }
  if (mode=="manual") stop("manual feature selection requires an explicit feature-count override")
  if (mode!="max_auc") stop("selection.mode must be manual or max_auc")
  valid <- metrics[is.finite(metrics$auc), ]
  if (!nrow(valid)) stop("no finite AUC values are available")
  min(valid$feature_count[valid$auc == max(valid$auc)])
}

fit_metacpg <- function(config, search_result = NULL, feature_count = NULL, data = NULL, .resume_extraction = FALSE) {
  cfg <- if (is.character(config)) read_config(config) else config
  if (is.null(data)) data <- validate_inputs(cfg)
  if (is.null(search_result)) search_result <- search_features(cfg, data)
  configured <- cfg$selection$feature_count %||% cfg$search$feature_count_override
  selected <- select_feature_count(search_result$metrics,feature_count %||% configured,cfg$selection$mode %||% "manual")
  weight_refit <- validate_weight_refit(cfg)
  refit_covariates <- if (weight_refit == "covariate_aware") data$covariates else list()
  if (length(data$covariates) && !weight_refit_explicit(cfg))
    message("WARNING: final_model.weight_refit now defaults to covariate_aware; LOO selection is unchanged, but exported weights/fixed scores differ from legacy runs. Rerun downstream validation or set legacy_unadjusted explicitly.")
  extraction <- exact_loo(data,selected,"extraction",cfg$execution$workers %||% 1L,cfg,.resume_extraction)
  freq <- extraction$inclusion_frequency; features <- head(freq$feature[freq$feature_count==selected],selected)
  full_stats <- list(); endpoint_matrices <- list(); full_fit_count <- 0L
  for (k in seq_along(data$endpoints)) {
    rows <- endpoint_rows(data,data$endpoints[[k]]); endpoint_matrices[[k]] <- data$matrix[,rows,drop=FALSE]
    full_stats[[k]] <- endpoint_stats(endpoint_matrices[[k]],data$phenotype[rows,,drop=FALSE],data$endpoints[[k]],refit_covariates,cfg$execution$workers %||% 1L)
    full_fit_count <- full_fit_count+nrow(data$matrix)
  }
  final_stats <- if (length(full_stats)==1) full_stats[[1]] else meta_stats(full_stats,do.call(cbind,endpoint_matrices))
  combined_matrix <- do.call(cbind,endpoint_matrices)
  final <- legacy_full_scores(data$matrix,final_stats,features,combined_matrix); scores <- final$scores; selected_features <- final$selected
  selected_features$rank <- seq_len(nrow(selected_features))
  if (!is.null(cfg$annotation$path)) {
    ann <- read_table_auto(cfg$annotation$path, cfg$annotation$separator)
    selected_features <- merge(selected_features, ann, by.x = "feature", by.y = cfg$annotation$feature_id_column %||% "feature", all.x = TRUE, sort = FALSE)
  }
  horizon <- min(vapply(data$endpoints,function(x) as.numeric(x$prediction_time %||% 1095),numeric(1)))
  perf <- legacy_performance(extraction$scores$time,extraction$scores$status,extraction$scores$risk_score,horizon)
  c(list(selected_features=selected_features,risk_scores=scores,scaling=final$scaling,weight_refit=weight_refit,
    performance=list(feature_count=selected,c_index=perf$c_index,auc=perf$auc),
    extraction_scores=extraction$scores,extraction_inclusion_frequency=freq,
    extraction_task_provenance=extraction$task_provenance,
    extraction_cox_fit_count=length(data$endpoints)*nrow(data$matrix)*(1+nrow(extraction$scores)),
    extraction_cox_fits_executed=extraction$fit_count+full_fit_count),search_result)
}

config_fingerprint <- function(cfg) {
  clean <- cfg[!grepl("^_", names(cfg))]
  clean$final_model$weight_refit <- validate_weight_refit(cfg)
  config_sum <- sum(as.integer(serialize(clean, NULL))) %% 2147483647
  paste(c(as.character(tools::md5sum(c(cfg$matrix$path, cfg$phenotype$path))), config_sum), collapse = "-")
}

write_results <- function(cfg, result, elapsed) {
  cfg$output_dir <- file.path(cfg$output_dir, "r")
  dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)
  for (name in c("ranking","scores","metrics","endpoint_stats","inclusion_frequency","task_provenance",
    "selected_features","risk_scores","extraction_scores","extraction_inclusion_frequency","extraction_task_provenance"))
    if (is.data.frame(result[[name]])) write.csv(result[[name]], file.path(cfg$output_dir, paste0(name, ".csv")), row.names = FALSE)
  jsonlite::write_json(list(analysis_id = cfg$analysis_id, engine = "r", schema_version = .schema_version,
    methodology=cfg$methodology %||% "legacy_exact",run_id=cfg$`_run_id`,
    fingerprint = config_fingerprint(cfg), elapsed_seconds = elapsed, R = R.version.string,
    performance=result$performance,scaling=result$scaling,weight_refit=result$weight_refit %||% weight_refit_mode(cfg),
    cox_fit_counts=list(search=result$search_cox_fit_count,extraction=result$extraction_cox_fit_count,
      search_executed_in_process=result$search_cox_fits_executed,
      extraction_executed_in_process=result$extraction_cox_fits_executed)),
    file.path(cfg$output_dir,"result.json"),auto_unbox=TRUE,pretty=TRUE,na="null")
  try({
    grDevices::png(file.path(cfg$output_dir, "search_metrics.png"), width=900, height=600)
    graphics::matplot(result$metrics$feature_count, result$metrics[,c("auc","c_index")], type="l", lty=1,
      xlab="Feature count", ylab="Performance", main=cfg$analysis_id)
    graphics::legend("bottomright",c("AUC","C-index"),col=1:2,lty=1); grDevices::dev.off()
    grDevices::png(file.path(cfg$output_dir, "risk_scores.png"), width=900, height=600)
    graphics::hist(result$risk_scores$risk_score, breaks=20, xlab="Risk score", main="Final risk-score distribution")
    grDevices::dev.off()
  }, silent=TRUE)
  cfg$output_dir
}

run_pipeline <- function(config, backend = "local", dry_run = FALSE, feature_count = NULL) {
  cfg <- if (is.character(config)) read_config(config) else config
  if (backend == "slurm") return(submit_slurm(cfg, dry_run))
  if (backend != "local") stop("backend must be local or slurm")
  data <- validate_inputs(cfg)
  if (dry_run) return(list(valid = TRUE, output_dir = cfg$output_dir))
  cfg$`_run_id` <- cfg$execution$resume_run_id %||% paste0(format(Sys.time(),"%Y%m%dT%H%M%S"),"-",Sys.getpid())
  start <- proc.time()[["elapsed"]]; search <- search_features(cfg, data)
  result <- fit_metacpg(cfg, search, feature_count, data)
  result$output_dir <- write_results(cfg, result, proc.time()[["elapsed"]] - start); result
}

run_worker <- function(config,phase,feature_count,run_id) {
  cfg <- if (is.character(config)) read_config(config) else config
  if (!phase %in% c("search","extraction")) stop("phase must be search or extraction")
  cfg$execution$resume_run_id <- run_id; cfg$`_run_id` <- run_id; data <- validate_inputs(cfg)
  result <- exact_loo(data,as.integer(feature_count),phase,cfg$execution$workers %||% 1L,cfg,TRUE)
  list(phase=phase,feature_count=as.integer(feature_count),run_id=run_id,
       tasks=nrow(result$task_provenance),cox_fits_executed=result$fit_count)
}

aggregate_run <- function(config,run_id,feature_count) {
  cfg <- if (is.character(config)) read_config(config) else config
  cfg$execution$resume_run_id <- run_id; cfg$`_run_id` <- run_id; start <- proc.time()[["elapsed"]]
  data <- validate_inputs(cfg); search <- search_features(cfg,data)
  result <- fit_metacpg(cfg,search,as.integer(feature_count),data,.resume_extraction=TRUE)
  result$output_dir <- write_results(cfg,result,proc.time()[["elapsed"]]-start); result
}

slurm_directives <- function(cfg,suffix,array=NULL) {
  s <- cfg$execution$slurm %||% list(); directives <- c(
    paste0("#SBATCH --job-name=", cfg$analysis_id,"-",suffix), paste0("#SBATCH --cpus-per-task=", s$cpus %||% 4),
    paste0("#SBATCH --mem=", s$memory %||% "16G"), paste0("#SBATCH --time=", s$time %||% "24:00:00"))
  if (!is.null(array)) directives <- c(directives,paste0("#SBATCH --array=",array))
  for (key in c("partition", "account", "mail_user", "mail_type")) if (!is.null(s[[key]]))
    directives <- c(directives, paste0("#SBATCH --", gsub("_", "-", key), "=", s[[key]]))
  directives
}

submit_slurm <- function(config, dry_run = FALSE) {
  cfg <- if (is.character(config)) read_config(config) else config
  dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)
  selected <- cfg$selection$feature_count %||% cfg$search$feature_count_override
  if (is.null(selected)) stop("SLURM legacy_exact submission requires a manually selected feature count")
  counts <- seq(cfg$search$min_features %||% 5L,cfg$search$max_features %||% 60L,cfg$search$step %||% 1L)
  run_id <- cfg$execution$resume_run_id %||% paste0(format(Sys.time(),"%Y%m%dT%H%M%S"),"-",Sys.getpid())
  setup <- unlist(cfg$execution$slurm$setup %||% list()); prefix <- c("#!/usr/bin/env bash","set -euo pipefail")
  search_path <- file.path(cfg$output_dir,"search-array.slurm.sh")
  writeLines(c(prefix,slurm_directives(cfg,"search",paste0("0-",length(counts)-1L)),setup,
    paste0("COUNTS=(",paste(counts,collapse=" "),")"),
    paste("metacpg-r worker --config",shQuote(cfg$`_config_path`),"--phase search --feature-count ${COUNTS[$SLURM_ARRAY_TASK_ID]} --run-id",shQuote(run_id))),search_path)
  extraction_path <- file.path(cfg$output_dir,"extraction.slurm.sh")
  writeLines(c(prefix,slurm_directives(cfg,"extract"),setup,
    paste("metacpg-r worker --config",shQuote(cfg$`_config_path`),"--phase extraction --feature-count",selected,"--run-id",shQuote(run_id))),extraction_path)
  aggregate_path <- file.path(cfg$output_dir,"aggregate.slurm.sh")
  writeLines(c(prefix,slurm_directives(cfg,"aggregate"),setup,
    paste("metacpg-r aggregate --config",shQuote(cfg$`_config_path`),"--feature-count",selected,"--run-id",shQuote(run_id))),aggregate_path)
  paths <- c(search_path,extraction_path,aggregate_path); Sys.chmod(paths,"0750")
  if (dry_run) return(list(scripts=paths,submitted=FALSE,run_id=run_id))
  search_job <- trimws(system2("sbatch",c("--parsable",shQuote(search_path)),stdout=TRUE)[[1]])
  extraction_job <- trimws(system2("sbatch",c("--parsable",paste0("--dependency=afterok:",search_job),shQuote(extraction_path)),stdout=TRUE)[[1]])
  aggregate_job <- trimws(system2("sbatch",c("--parsable",paste0("--dependency=afterok:",extraction_job),shQuote(aggregate_path)),stdout=TRUE)[[1]])
  list(scripts=paths,submitted=TRUE,run_id=run_id,job_ids=list(search=search_job,extraction=extraction_job,aggregate=aggregate_job))
}

compare_results <- function(reference_dir, candidate_dir, engine = "r") {
  rf <- read.csv(file.path(reference_dir, "selected_features.csv")); cf <- read.csv(file.path(candidate_dir, "selected_features.csv"))
  rs <- read.csv(file.path(reference_dir, "risk_scores.csv")); cs <- read.csv(file.path(candidate_dir, "risk_scores.csv"))
  overlap <- length(intersect(rf$feature, cf$feature)) / max(1, length(unique(rf$feature)))
  merged <- merge(rs, cs, by = "sample_id", suffixes = c("_ref", "_candidate"))
  rho <- cor(merged$risk_score_ref, merged$risk_score_candidate, method = "spearman")
  rj <- jsonlite::fromJSON(file.path(reference_dir, "result.json")); cj <- jsonlite::fromJSON(file.path(candidate_dir, "result.json"))
  values <- c(feature_overlap = overlap, risk_spearman = rho,
    auc_delta = abs(rj$performance$auc - cj$performance$auc), c_index_delta = abs(rj$performance$c_index - cj$performance$c_index))
  thresholds <- if (engine == "r") c(feature_overlap=.95, risk_spearman=.99, auc_delta=.01, c_index_delta=.01) else
    c(feature_overlap=.80, risk_spearman=.95, auc_delta=.03, c_index_delta=.03)
  checks <- c(values[1:2] >= thresholds[1:2], values[3:4] <= thresholds[3:4])
  list(engine = engine, metrics = as.list(values), thresholds = as.list(thresholds), checks = as.list(checks), passed = all(checks))
}
