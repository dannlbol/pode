library(metaCpG)

make_fixture <- function(path, paired = FALSE, covariate = NULL) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  set.seed(7); samples <- sprintf("S%02d", 0:9); features <- sprintf("cg%05d", 0:4)
  beta <- matrix(runif(5 * 10, .05, .95), 5, dimnames = list(features, samples))
  write.csv(data.frame(feature = features, beta, check.names = FALSE), file.path(path, "matrix.csv"), row.names = FALSE)
  risk <- beta[1,] - beta[2,]
  pheno <- data.frame(sample_id=samples, time1=pmax(1,300-100*risk+rnorm(10,0,20)), status1=rep(c(1,0),5),
    time2=pmax(1,250-80*risk+rnorm(10,0,20)), status2=rep(c(0,1,1,0,1),2), group=rep(c("A","B"),5), age=2:11)
  write.csv(pheno, file.path(path, "pheno.csv"), row.names = FALSE)
  endpoints <- list(list(name="one",time="time1",status="status1",prediction_time=200))
  if (paired) endpoints[[2]] <- list(name="two",time="time2",status="status2",prediction_time=200)
  cfg <- list(schema_version=1,methodology="legacy_exact",analysis_id="synthetic",
    matrix=list(path="matrix.csv",orientation="features_by_samples",scale="beta",feature_id_column="feature"),
    phenotype=list(path="pheno.csv",sample_id_column="sample_id"), endpoints=endpoints,
    covariates=if (is.null(covariate)) list() else list(covariate),
    search=list(min_features=1,max_features=2,step=1,cross_validation="loo"),
    selection=list(mode="manual",feature_count=2),
    execution=list(workers=1,seed=1,slurm=list(cpus=2,memory="2G",time="00:10:00")), annotation=NULL,
    validation=list(allow_unmatched_samples=FALSE),output_dir="out")
  yaml::write_yaml(cfg,file.path(path,"config.yml")); file.path(path,"config.yml")
}
