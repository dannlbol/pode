# =============================================================================
# Prognostic metrics: Cox modelling, survival prediction, bootstrap CIs,
# risk-group assignment, generalised from earlier study-specific scripts.
# All column names and settings come from the config (or explicit overrides).
# =============================================================================

# Build a "Surv(time, status) ~ terms" formula from a config + covariate spec.
# Covariate terms prefixed with "strata:" become strata() terms.
.build_surv_formula <- function(cfg, covariates = NULL, time_col = NULL,
                                status_col = NULL) {
  time_col   <- col_of(cfg, "time", time_col)
  status_col <- col_of(cfg, "status", status_col)
  covariates <- covariates %||% cfg$columns$covariates
  if (is.null(covariates) || length(covariates) == 0) {
    stop("fit_cox: no covariates supplied (set columns.covariates in the config).",
         call. = FALSE)
  }
  terms <- vapply(covariates, function(tm) {
    if (grepl("^strata:", tm)) sprintf("strata(%s)", sub("^strata:", "", tm)) else tm
  }, character(1))
  list(
    # Unqualified Surv() (imported into the namespace): survfit.coxph fails to
    # reconstruct the model frame if the formula uses survival::Surv().
    formula = as.formula(sprintf("Surv(%s, %s) ~ %s",
                                 time_col, status_col, paste(terms, collapse = " + "))),
    time_col = time_col, status_col = status_col,
    model_vars = sub("^strata:", "", covariates)
  )
}

# Apply an optional landmark: keep subjects surviving past the offset and shift
# the clock. Returns the (possibly filtered) data with time column reset.
.apply_landmark <- function(data, cfg, time_col, status_col) {
  off <- cfg$survival$landmark_offset
  if (is.null(off) || is.na(off)) return(data)
  data <- data[!is.na(data[[time_col]]) & data[[time_col]] > off, , drop = FALSE]
  data[[time_col]] <- data[[time_col]] - off
  data
}

#' Fit a Cox proportional-hazards model
#'
#' Wraps \code{survival::coxph} with complete-case handling, factor level
#' dropping, an optional landmark reset, and a proportional-hazards test. The
#' model formula is assembled from the config's \code{columns.covariates}
#' (terms prefixed \code{strata:} become strata) unless \code{covariates} is given.
#'
#' @param data A data frame.
#' @param cfg A \code{prognostk_config}.
#' @param covariates Optional character vector of model terms (overrides config).
#' @param time_col,status_col Optional column-name overrides.
#' @return A list with \code{fit} (coxph), \code{zph} (cox.zph), \code{data}
#'   (complete-case data used), and \code{spec}.
#' @export
fit_cox <- function(data, cfg, covariates = NULL, time_col = NULL, status_col = NULL) {
  spec <- .build_surv_formula(cfg, covariates, time_col, status_col)
  data <- .apply_landmark(data, cfg, spec$time_col, spec$status_col)
  data[[spec$status_col]] <- as.numeric(data[[spec$status_col]])

  keep_vars <- c(spec$time_col, spec$status_col, spec$model_vars)
  d <- data[complete.cases(data[, keep_vars, drop = FALSE]), , drop = FALSE]
  for (v in spec$model_vars) if (is.factor(d[[v]])) d[[v]] <- droplevels(d[[v]])
  if (nrow(d) < 10) stop("fit_cox: fewer than 10 complete cases.", call. = FALSE)

  # model = TRUE stores the model frame so survfit()/predict() need not
  # re-evaluate the formula (whose environment is this local frame).
  fit <- survival::coxph(spec$formula, data = d, model = TRUE)
  zph <- tryCatch(survival::cox.zph(fit), error = function(e) NULL)
  list(fit = fit, zph = zph, data = d, spec = spec)
}

# Names of strata() variables in a coxph model (character(0) if none).
.strata_vars <- function(coxfit) {
  tl <- attr(coxfit$terms, "term.labels")
  sl <- grep("^strata\\(", tl, value = TRUE)
  gsub("^strata\\(|\\)$", "", sl)
}

# Centered linear predictor for newdata, computed manually to avoid the
# predict.coxph(newdata=, type="lp") strata bug. Matches the survfit(coxfit)
# baseline (which is centered at the covariate means).
.centered_lp <- function(coxfit, newdata) {
  beta <- coef(coxfit); beta <- beta[!is.na(beta)]
  if (length(beta) == 0) return(rep(0, nrow(newdata)))
  tl <- attr(coxfit$terms, "term.labels")
  tl <- tl[!grepl("^strata\\(", tl)]
  mm <- stats::model.matrix(as.formula(paste("~", paste(tl, collapse = " + "))),
                            data = newdata)
  common <- intersect(names(beta), colnames(mm))
  lp_raw  <- as.vector(mm[, common, drop = FALSE] %*% beta[common])
  lp_raw - sum(coxfit$means[common] * beta[common])
}

# Normalise a stratum label to a value-only key that is independent of how
# survfit() names strata (bare level "A" for character vars, "g=0" for numeric,
# "v1=a, v2=b" for multiple). Strips any "var=" prefixes and joins values with "|".
.norm_stratum <- function(nm) {
  parts <- strsplit(nm, ", ", fixed = TRUE)[[1]]
  vals <- vapply(parts, function(p) sub("^[^=]*=", "", p), character(1))
  paste(vals, collapse = "|")
}

# Per-stratum baseline survival (with CI bands) at a single time point.
# Returns a list of numeric vectors keyed by a normalised value key, or key ""
# for non-stratified models.
.baseline_at <- function(coxfit, time) {
  bh <- survival::survfit(coxfit)
  interp <- function(tt, ss) approx(tt, ss, xout = time, method = "constant", rule = 2)$y
  pull <- function(field, idx) {
    v <- bh[[field]]
    if (is.null(v)) return(NA_real_)
    interp(bh$time[idx], v[idx])
  }
  if (is.null(bh$strata)) {
    idx <- seq_along(bh$time)
    return(lapply(c(surv = "surv", lower = "lower", upper = "upper"),
                  function(fld) setNames(pull(fld, idx), "")))
  }
  keys <- vapply(names(bh$strata), .norm_stratum, character(1))
  ends <- cumsum(bh$strata); starts <- c(1, head(ends, -1) + 1)
  grab <- function(fld) setNames(vapply(seq_along(bh$strata), function(k)
    pull(fld, starts[k]:ends[k]), numeric(1)), keys)
  list(surv = grab("surv"), lower = grab("lower"), upper = grab("upper"))
}

# Normalised value key for each row of newdata (matches .baseline_at keys).
.stratum_key <- function(coxfit, newdata) {
  sv <- .strata_vars(coxfit)
  if (length(sv) == 0) return(rep("", nrow(newdata)))
  apply(newdata[, sv, drop = FALSE], 1, function(r)
    paste(as.character(r), collapse = "|"))
}

#' Per-patient survival probability at a given time
#'
#' Computes S(t) for each row of \code{newdata} from a fitted Cox model as
#' \code{S0_stratum(t) ^ exp(lp)}, using the per-stratum baseline hazard. This is
#' strata-safe and much faster than per-row \code{survfit}. Generalised from
#' \code{get_predicted_survival()} / \code{get_predicted_survival_lp()}.
#'
#' @param fit A \code{coxph} fit (or the list returned by \code{fit_cox}).
#' @param newdata Data to predict on.
#' @param time Time horizon; defaults to \code{cfg$survival$horizon} if \code{cfg} given.
#' @param cfg Optional config supplying the default horizon.
#' @return Numeric vector of predicted survival probabilities.
#' @export
predict_survival_at_time <- function(fit, newdata, time = NULL, cfg = NULL) {
  if (is.list(fit) && !is.null(fit$fit)) fit <- fit$fit
  if (is.null(time)) time <- if (!is.null(cfg)) cfg$survival$horizon else 5
  lp  <- .centered_lp(fit, newdata)
  s0  <- .baseline_at(fit, time)$surv
  key <- .stratum_key(fit, newdata)
  base <- if (length(s0) == 1 && identical(names(s0), "")) rep(s0, length(lp)) else s0[key]
  # Rows whose stratum was unseen in training get NA baseline.
  pmin(pmax(base ^ exp(lp), 0), 1)
}

# Bootstrap statistic: refit on resampled rows, predict on the original data.
.boot_surv_stat <- function(data, indices, formula, orig_data, time) {
  d <- data[indices, , drop = FALSE]
  fit <- tryCatch(survival::coxph(formula, data = d, model = TRUE), error = function(e) NULL)
  if (is.null(fit)) return(rep(NA_real_, nrow(orig_data)))
  tryCatch(predict_survival_at_time(fit, orig_data, time = time),
           error = function(e) rep(NA_real_, nrow(orig_data)))
}

#' Bootstrapped survival prediction with bias-corrected point estimates and CIs
#'
#' Refits the model on \code{n} bootstrap resamples, predicts each patient's
#' survival at the horizon, and returns BCa confidence intervals plus a
#' bias-corrected point estimate. Generalised from \code{bootstrap_survival_ci()}.
#'
#' @param data A data frame.
#' @param cfg A \code{prognostk_config} (supplies covariates, horizon, bootstrap settings).
#' @param covariates,time_col,status_col Optional overrides passed to \code{fit_cox}.
#' @param n,conf,seed Optional overrides for bootstrap settings.
#' @return A data frame: id, pred_original, pred_bias_corrected, ci_lower, ci_upper.
#' @export
bootstrap_survival_ci <- function(data, cfg, covariates = NULL, time_col = NULL,
                                  status_col = NULL, n = NULL, conf = NULL, seed = NULL) {
  n    <- n    %||% cfg$survival$bootstrap$n
  conf <- conf %||% cfg$survival$bootstrap$conf
  seed <- seed %||% cfg$survival$bootstrap$seed
  time <- cfg$survival$horizon

  fitres <- fit_cox(data, cfg, covariates, time_col, status_col)
  d <- fitres$data
  formula <- fitres$spec$formula

  pred_original <- predict_survival_at_time(fitres$fit, d, time = time)

  set.seed(seed)
  bt <- boot::boot(d, statistic = function(dd, idx)
    .boot_surv_stat(dd, idx, formula, d, time), R = n)

  np <- nrow(d)
  ci_lower <- ci_upper <- bias <- numeric(np)
  for (i in seq_len(np)) {
    ti <- bt$t[, i]
    ti <- ti[is.finite(ti)]
    if (length(ti) < 10) { ci_lower[i] <- ci_upper[i] <- NA; bias[i] <- 0; next }
    # BCa can warn about extreme order statistics for small n_boot; the
    # percentile fallback below handles those cases.
    bca <- tryCatch(suppressWarnings(
      boot::boot.ci(bt, index = i, type = "bca", conf = conf)),
      error = function(e) NULL)
    if (!is.null(bca) && !is.null(bca$bca)) {
      ci_lower[i] <- bca$bca[4]; ci_upper[i] <- bca$bca[5]
    } else {
      qs <- quantile(ti, c((1 - conf) / 2, 1 - (1 - conf) / 2), na.rm = TRUE)
      ci_lower[i] <- qs[1]; ci_upper[i] <- qs[2]
    }
    bias[i] <- mean(ti) - pred_original[i]
  }

  id_col <- cfg$columns$id
  ids <- if (!is.null(id_col) && id_col %in% names(d)) d[[id_col]] else seq_len(np)
  data.frame(
    id = ids,
    pred_original = pred_original,
    pred_bias_corrected = pmin(pmax(pred_original - bias, 0), 1),
    ci_lower = pmin(pmax(ci_lower, 0), 1),
    ci_upper = pmin(pmax(ci_upper, 0), 1),
    stringsAsFactors = FALSE
  )
}

#' External-validation survival predictions from a frozen model
#'
#' Applies a fixed (already-fitted) Cox model to a new cohort with NO refitting
#' and NO bias correction; CIs come from the log-log survfit interval. Generalised
#' from \code{bootstrap_validation_ci_frozen()}.
#'
#' @param data New cohort data frame.
#' @param frozen_fit A \code{coxph} fit (or \code{fit_cox} result) to apply as-is.
#' @param cfg A \code{prognostk_config}.
#' @param conf Optional confidence level override.
#' @return Data frame: id, pred, ci_lower, ci_upper.
#' @export
validate_frozen_model <- function(data, frozen_fit, cfg, conf = NULL) {
  if (is.list(frozen_fit) && !is.null(frozen_fit$fit)) frozen_fit <- frozen_fit$fit
  time <- cfg$survival$horizon
  time_col <- col_of(cfg, "time"); status_col <- col_of(cfg, "status")
  data <- .apply_landmark(data, cfg, time_col, status_col)

  lp   <- .centered_lp(frozen_fit, data)
  base <- .baseline_at(frozen_fit, time)
  key  <- .stratum_key(frozen_fit, data)
  onestrat <- length(base$surv) == 1 && identical(names(base$surv), "")
  pick <- function(v) if (onestrat) rep(v, length(lp)) else v[key]

  r <- exp(lp)
  # Frozen model: coefficients fixed, propagate baseline-survival uncertainty.
  pred     <- pick(base$surv)  ^ r
  ci_lower <- pick(base$lower) ^ r
  ci_upper <- pick(base$upper) ^ r

  id_col <- cfg$columns$id
  ids <- if (!is.null(id_col) && id_col %in% names(data)) data[[id_col]] else seq_len(nrow(data))
  data.frame(id = ids,
             pred = pmin(pmax(pred, 0), 1),
             ci_lower = pmin(pmax(ci_lower, 0), 1),
             ci_upper = pmin(pmax(ci_upper, 0), 1),
             stringsAsFactors = FALSE)
}

#' Assign risk groups from predicted survival
#'
#' Cuts predicted survival into ordered, named risk tiers using descending
#' thresholds. Defaults come from \code{cfg$risk_groups}. Generalised from
#' \code{assign_risk_groups()}.
#'
#' @param pred_surv Numeric vector of predicted survival probabilities.
#' @param cfg Optional config supplying thresholds/labels.
#' @param thresholds Optional descending numeric vector (overrides config).
#' @param labels Optional character vector, length = length(thresholds)+1.
#' @return An ordered factor of risk-group labels (highest tier first).
#' @export
assign_risk_groups <- function(pred_surv, cfg = NULL, thresholds = NULL, labels = NULL) {
  if (is.null(thresholds)) thresholds <- cfg$risk_groups$thresholds
  if (is.null(labels))     labels     <- cfg$risk_groups$labels
  if (is.null(thresholds) || is.null(labels)) {
    stop("assign_risk_groups: need thresholds and labels (via cfg or arguments).",
         call. = FALSE)
  }
  if (length(labels) != length(thresholds) + 1) {
    stop("assign_risk_groups: length(labels) must equal length(thresholds)+1.",
         call. = FALSE)
  }
  breaks <- c(Inf, sort(thresholds, decreasing = TRUE), -Inf)
  # cut() with decreasing breaks: reverse to increasing, then map labels.
  g <- cut(pred_surv, breaks = rev(breaks), labels = rev(labels), right = TRUE,
           include.lowest = TRUE)
  factor(as.character(g), levels = labels)
}
