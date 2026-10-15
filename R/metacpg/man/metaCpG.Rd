\name{metaCpG}
\alias{metaCpG}
\alias{read_config}
\alias{validate_inputs}
\alias{search_features}
\alias{select_feature_count}
\alias{fit_metacpg}
\alias{run_pipeline}
\alias{submit_slurm}
\alias{compare_results}
\alias{beta_to_m}
\alias{run_worker}
\alias{aggregate_run}
\title{Configurable CpG survival metagene modelling}
\description{
Read and validate a versioned analysis configuration, search CpG feature counts,
fit a final metagene, run locally or through SLURM, and compare portable results.
}
\details{
For schema-v1 configurations, final exported CpG weights are refitted with all
configured adjustment covariates and Cox strata by default. Set
\code{final_model.weight_refit: legacy_unadjusted} in YAML to reproduce the
historical unadjusted exported weights. This setting does not alter the LOO
search curve or selected CpGs, but fixed-model scores and downstream validation
must be regenerated when it changes.
}
\usage{
read_config(path)

validate_inputs(config)

search_features(config, data = NULL)

select_feature_count(metrics, override = NULL, mode = "manual")

fit_metacpg(config, search_result = NULL, feature_count = NULL, data = NULL,
  .resume_extraction = FALSE)

run_pipeline(config, backend = "local", dry_run = FALSE, feature_count = NULL)

submit_slurm(config, dry_run = FALSE)

compare_results(reference_dir, candidate_dir, engine = "r")

beta_to_m(x)

run_worker(config, phase, feature_count, run_id)

aggregate_run(config, run_id, feature_count)
}
\arguments{
  \item{path}{Path to a version 1 metaCpG YAML configuration.}
  \item{config}{A configuration path or object returned by \code{read_config}.}
  \item{data}{Optional validated input object.}
  \item{metrics}{Search metric data frame.}
  \item{override}{Optional explicit feature count.}
  \item{mode}{Selection mode: legacy-default \code{"manual"} or explicit \code{"max_auc"}.}
  \item{search_result}{Optional result from \code{search_features}.}
  \item{feature_count}{Optional final feature-count override.}
  \item{.resume_extraction}{Internal distributed-aggregation flag for reading extraction-task results from its separate phase cache.}
  \item{backend}{Either \code{"local"} or \code{"slurm"}.}
  \item{dry_run}{Validate or render without running/submitting.}
  \item{reference_dir}{Directory containing portable reference results.}
  \item{candidate_dir}{Directory containing portable candidate results.}
  \item{engine}{Threshold profile, \code{"r"} or \code{"python"}.}
  \item{x}{Numeric beta values strictly between zero and one.}
  \item{phase}{Internal distributed phase, \code{"search"} or \code{"extraction"}.}
  \item{run_id}{Explicit distributed/resume run identifier.}
}
\value{
Configuration and validation functions return structured lists. Search and fit
functions return named tables and metrics. Pipeline and submission functions
also return output/job metadata. \code{compare_results} returns metrics,
threshold checks and an overall pass flag. \code{beta_to_m} returns M-values.
}
