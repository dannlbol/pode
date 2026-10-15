#' prognostk: generic prognostic figures, metrics and risk stratification
#'
#' A configuration-driven toolkit for survival / biomarker prognosis studies.
#' All dataset specifics (column names, survival horizon, thresholds, palettes)
#' come from an editable YAML config, so the same functions apply to any cohort.
#'
#' @importFrom survival Surv coxph survfit cox.zph concordance
#' @importFrom rlang .data
#' @importFrom stats as.formula approx aov coef complete.cases delete.response
#'   median model.matrix offset predict qnorm quantile sd setNames t.test terms
#' @importFrom grDevices cairo_pdf dev.off hcl.colors pdf png
#' @importFrom utils head tail combn
#' @keywords internal
"_PACKAGE"

# ggplot2 aes() references columns by bare name via non-standard evaluation;
# declare them so R CMD check does not flag "no visible binding".
utils::globalVariables(c(
  "time", "surv", "group", "n.risk", "risk", "id",
  "HR", "term", "model", "lower", "upper",
  "mean_pred", "obs_km", "obs_lower", "obs_upper"))
