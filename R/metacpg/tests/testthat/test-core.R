test_that("beta conversion and deterministic selection work", {
  expect_equal(as.numeric(beta_to_m(c(.2,.5,.8))), c(-2,0,2))
  expect_equal(select_feature_count(data.frame(feature_count=3:5,auc=c(.7,.8,.8)),mode="max_auc"),4)
})

test_that("validation, run and SLURM rendering work", {
  root <- tempfile(); cfg <- read_config(make_fixture(root)); data <- validate_inputs(cfg)
  expect_equal(dim(data$matrix),c(5,10)); result <- run_pipeline(cfg,feature_count=2)
  expect_true(file.exists(file.path(root,"out","r","selected_features.csv")))
  expect_equal(result$performance$feature_count,2)
  submit <- submit_slurm(cfg,TRUE); search_script <- paste(readLines(submit$scripts[[1]]),collapse="\n")
  expect_match(search_script,"cpus-per-task=2"); expect_match(search_script,"#SBATCH --array=0-1")
  expect_match(search_script,"metacpg-r worker"); expect_match(paste(readLines(submit$scripts[[2]]),collapse="\n"),"phase extraction")
  expect_match(paste(readLines(submit$scripts[[3]]),collapse="\n"),"metacpg-r aggregate")
})

test_that("paired, strata, categorical and numeric adjustment modes run", {
  covariates <- list(list(column="group",type="categorical",mode="strata"),
    list(column="group",type="categorical",mode="adjust"),list(column="age",type="numeric",mode="adjust"))
  for (cov in covariates) {
    root <- tempfile(); result <- run_pipeline(make_fixture(root,TRUE,cov),feature_count=2)
    expect_equal(nrow(result$selected_features),2)
  }
})

test_that("leave-one-out search computes reusable inclusion frequencies", {
  root <- tempfile(); path <- make_fixture(root); cfg <- yaml::read_yaml(path)
  cfg$search$cross_validation <- "loo"; cfg$search$max_features <- 2
  yaml::write_yaml(cfg,path)
  result <- run_pipeline(path,feature_count=2)
  expect_gt(nrow(result$inclusion_frequency),0)
  expect_equal(sort(unique(result$scores$feature_count)),1:2)
  expect_equal(length(list.files(file.path(root,"out","cache"),pattern=".*rds$",recursive=TRUE)),0)
})

test_that("legacy Cox statistics, ranking and fold normalization are exact", {
  root <- tempfile(); cfg <- read_config(make_fixture(root)); data <- validate_inputs(cfg)
  ep <- data$endpoints[[1]]; stats <- metaCpG:::endpoint_stats(data$matrix,data$phenotype,ep,list(),1)
  feature_value <- as.numeric(data$matrix[1,]); fit <- survival::coxph(survival::Surv(data$phenotype$time1,data$phenotype$status1)~feature_value)
  sm <- summary(fit)$coefficients[1,]
  expect_equal(stats$coef[1],unname(sm[["coef"]]),tolerance=1e-12)
  expect_equal(stats$se[1],unname(sm[["se(coef)"]]),tolerance=1e-12)
  expect_equal(stats$ordinary_z[1],unname(sm[["z"]]),tolerance=1e-12)
  expect_equal(stats$standardized_coef[1],unname(sm[["coef"]])/sd(feature_value),tolerance=1e-12)
  heldout <- 1; train <- data$matrix[,-heldout,drop=FALSE]
  train_stats <- metaCpG:::endpoint_stats(train,data$phenotype[-heldout,,drop=FALSE],ep,list(),1)
  ranking <- metaCpG:::rank_stats(train_stats); score <- metaCpG:::legacy_fold_score(ranking,2,train,data$matrix[,heldout])
  features <- ranking$feature[1:2]; weights <- ranking$standardized_coef[1:2]
  raw <- apply(train[features,,drop=FALSE],2,function(x) mean(x*weights))
  expected <- (mean(data$matrix[features,heldout]*weights)-median(raw))/IQR(raw)
  expect_equal(score$score,expected,tolerance=1e-12)
})

test_that("literal refit counts include a fresh extraction", {
  root <- tempfile(); result <- run_pipeline(make_fixture(root),feature_count=2)
  expect_equal(result$search_cox_fit_count,5+5*10*2)
  expect_equal(result$extraction_cox_fit_count,5*10+5)
  expect_true(all(result$task_provenance$resumed==FALSE))
  expect_true(all(result$extraction_task_provenance$phase=="extraction"))
})

test_that("covariate stability selection exports covariate-aware weights by default", {
  root <- tempfile(); cov <- list(column="group",type="categorical",mode="strata")
  path <- make_fixture(root,covariate=cov); cfg <- read_config(path)
  cfg$covariates[[2]] <- list(column="age",type="numeric",mode="adjust")
  data <- validate_inputs(cfg); result <- run_pipeline(cfg,feature_count=2)
  expected <- metaCpG:::endpoint_stats(data$matrix,data$phenotype,data$endpoints[[1]],data$covariates,1)
  expect_equal(result$selected_features$standardized_coef,
    expected$standardized_coef[match(result$selected_features$feature,expected$feature)],tolerance=1e-12)
  expect_equal(result$weight_refit,"covariate_aware")
  saved <- jsonlite::fromJSON(file.path(root,"out","r","result.json"))
  expect_equal(saved$weight_refit,"covariate_aware")
})

test_that("legacy unadjusted exported weights remain available", {
  root <- tempfile(); cov <- list(column="group",type="categorical",mode="strata")
  path <- make_fixture(root,covariate=cov); cfg <- metaCpG::read_config(path)
  cfg$final_model <- list(weight_refit="legacy_unadjusted")
  data <- validate_inputs(cfg); result <- run_pipeline(cfg,feature_count=2)
  expected <- metaCpG:::endpoint_stats(data$matrix,data$phenotype,data$endpoints[[1]],list(),1)
  expect_equal(result$selected_features$standardized_coef,
    expected$standardized_coef[match(result$selected_features$feature,expected$feature)],tolerance=1e-12)
  historical <- metaCpG:::legacy_full_scores(data$matrix,expected,result$selected_features$feature)
  expect_equal(result$risk_scores,historical$scores,tolerance=1e-12)
  expect_equal(result$scaling,historical$scaling,tolerance=1e-12)
  expect_equal(result$weight_refit,"legacy_unadjusted")
})

test_that("weight refit modes validate, fingerprint distinctly, and agree without covariates", {
  root <- tempfile(); path <- make_fixture(root)
  aware_cfg <- read_config(path); aware_cfg$final_model <- list(weight_refit="covariate_aware")
  legacy_cfg <- read_config(path); legacy_cfg$final_model <- list(weight_refit="legacy_unadjusted")
  aware <- run_pipeline(aware_cfg,feature_count=2)
  legacy <- run_pipeline(legacy_cfg,feature_count=2)
  expect_equal(aware$selected_features,legacy$selected_features,tolerance=1e-12)
  expect_equal(aware$risk_scores,legacy$risk_scores,tolerance=1e-12)
  expect_equal(aware$scaling,legacy$scaling,tolerance=1e-12)
  expect_false(identical(metaCpG:::config_fingerprint(aware_cfg),metaCpG:::config_fingerprint(legacy_cfg)))
  invalid <- read_config(path); invalid$final_model <- list(weight_refit="invalid")
  expect_error(validate_inputs(invalid),"final_model.weight_refit")
})

test_that("final scaling uses endpoint-eligible training observations only", {
  root <- tempfile(); path <- make_fixture(root)
  pheno_path <- file.path(root,"pheno.csv"); pheno <- read.csv(pheno_path)
  pheno[9:10,c("time1","status1")] <- NA; write.csv(pheno,pheno_path,row.names=FALSE)
  result <- suppressWarnings(run_pipeline(path,feature_count=2))
  eligible <- result$risk_scores$sample_id %in% pheno$sample_id[complete.cases(pheno[,c("time1","status1")])]
  expect_equal(result$scaling$center,median(result$risk_scores$raw_score[eligible]),tolerance=1e-12)
  expect_equal(result$scaling$scale,IQR(result$risk_scores$raw_score[eligible]),tolerance=1e-12)
  expect_false(isTRUE(all.equal(result$scaling$center,median(result$risk_scores$raw_score),tolerance=1e-12)))
})

test_that("distributed workers aggregate phase-separated legacy tasks", {
  root <- tempfile(); path <- make_fixture(root); run_id <- "distributed-test"
  expect_equal(run_worker(path,"search",1,run_id)$cox_fits_executed,50)
  expect_equal(run_worker(path,"search",2,run_id)$cox_fits_executed,50)
  expect_equal(run_worker(path,"extraction",2,run_id)$cox_fits_executed,50)
  result <- aggregate_run(path,run_id,2)
  expect_true(all(result$task_provenance$resumed))
  expect_true(all(result$extraction_task_provenance$resumed))
  expect_equal(result$search_cox_fit_count,105)
  expect_equal(result$extraction_cox_fit_count,55)
})
