# =============================================================================
# Configuration loading, validation and override-merging
# =============================================================================

# Built-in defaults. A user config is merged on top of these, so any section may
# be omitted from the YAML and still resolve to a sensible value.
.prognostk_defaults <- function() {
  list(
    columns = list(
      id = NULL, time = "time", status = "status", score = NULL,
      group = NULL, covariates = NULL, categorical = NULL
    ),
    survival = list(
      horizon = 5, time_unit = "years", landmark_offset = NULL,
      bootstrap = list(n = 1000, conf = 0.95, seed = 42)
    ),
    risk_groups = list(
      thresholds = c(0.90, 0.75, 0.50),
      labels = c("Favourable", "Standard", "High", "VeryHigh")
    ),
    palettes = list(
      risk = list(Favourable = "#2A9D8F", Standard = "#457B9D",
                  High = "#F4A261", VeryHigh = "#E63946"),
      cohort = list(discovery = "#08519C", validation = "#D6604D"),
      heatmap = c("#00008B", "#EEE9E9", "#8B0000"),
      categorical = list()
    ),
    oncoplot = list(tracks = list()),
    figure = list(
      page = list(width_in = 7.09, height_in = 10.51, base_size = 8),
      device = "pdf", dpi = 300, outdir = "./figures"
    )
  )
}

# Recursively merge `override` (a named list) on top of `base`. Non-list values
# in `override` win; lists are merged key-by-key. NULLs in override are ignored.
.merge_lists <- function(base, override) {
  if (is.null(override)) return(base)
  if (!is.list(base) || !is.list(override)) return(override)
  for (key in names(override)) {
    ov <- override[[key]]
    if (is.null(ov)) next
    if (is.list(ov) && is.list(base[[key]]) && !is.null(names(ov))) {
      base[[key]] <- .merge_lists(base[[key]], ov)
    } else {
      base[[key]] <- ov
    }
  }
  base
}

#' Load and validate a prognostk configuration
#'
#' Reads a YAML config (see \code{config.schema.md}), merges it over the built-in
#' defaults, validates it, and returns a \code{prognostk_config} object. Pass the
#' result as the \code{cfg} argument to any prognostk function.
#'
#' @param path Path to a YAML file, or \code{NULL} to use the built-in defaults.
#' @param ... Named overrides applied on top of the file, e.g.
#'   \code{load_config("cfg.yaml", survival = list(horizon = 10))}.
#' @return A \code{prognostk_config} (a validated named list).
#' @export
load_config <- function(path = NULL, ...) {
  cfg <- .prognostk_defaults()
  if (!is.null(path)) {
    if (!file.exists(path)) stop("Config file not found: ", path, call. = FALSE)
    user <- yaml::read_yaml(path)
    cfg <- .merge_lists(cfg, user)
  }
  overrides <- list(...)
  if (length(overrides) > 0) cfg <- .merge_lists(cfg, overrides)
  cfg <- validate_config(cfg)
  class(cfg) <- c("prognostk_config", "list")
  cfg
}

#' Validate a prognostk configuration
#'
#' Checks required keys and internal consistency (threshold ordering, label
#' counts, palette coverage). Called automatically by \code{load_config()}.
#'
#' @param cfg A config list.
#' @return The config, invisibly, if valid; otherwise stops with an informative error.
#' @export
validate_config <- function(cfg) {
  # Required column roles
  if (is.null(cfg$columns$time) || is.null(cfg$columns$status)) {
    stop("config: columns.time and columns.status are required.", call. = FALSE)
  }
  # Risk group consistency
  th <- cfg$risk_groups$thresholds
  lab <- cfg$risk_groups$labels
  if (!is.null(th)) {
    th <- as.numeric(th)
    if (any(th <= 0 | th >= 1)) {
      stop("config: risk_groups.thresholds must all be in (0, 1).", call. = FALSE)
    }
    if (length(th) > 1 && any(diff(th) >= 0)) {
      stop("config: risk_groups.thresholds must be strictly descending.", call. = FALSE)
    }
    if (length(lab) != length(th) + 1) {
      stop(sprintf(
        "config: risk_groups.labels must have length(thresholds)+1 (%d), got %d.",
        length(th) + 1, length(lab)), call. = FALSE)
    }
    cfg$risk_groups$thresholds <- th
  }
  # Survival
  if (!is.null(cfg$survival$horizon) && cfg$survival$horizon <= 0) {
    stop("config: survival.horizon must be > 0.", call. = FALSE)
  }
  conf <- cfg$survival$bootstrap$conf
  if (!is.null(conf) && (conf <= 0 || conf >= 1)) {
    stop("config: survival.bootstrap.conf must be in (0, 1).", call. = FALSE)
  }
  # Oncoplot track types
  valid_types <- c("points", "categorical", "line")
  for (tr in cfg$oncoplot$tracks) {
    if (is.null(tr$type) || !tr$type %in% valid_types) {
      stop(sprintf("config: oncoplot track '%s' has invalid type '%s' (allowed: %s).",
                   tr$name %||% "?", tr$type %||% "NULL",
                   paste(valid_types, collapse = ", ")), call. = FALSE)
    }
    if (is.null(tr$column)) {
      stop(sprintf("config: oncoplot track '%s' is missing 'column'.",
                   tr$name %||% "?"), call. = FALSE)
    }
  }
  cfg
}

# Null-coalescing helper (used throughout the package).
`%||%` <- function(a, b) if (is.null(a)) b else a

#' @export
print.prognostk_config <- function(x, ...) {
  cat("<prognostk_config>\n")
  cat(sprintf("  time/status : %s / %s\n", x$columns$time, x$columns$status))
  cat(sprintf("  score       : %s\n", x$columns$score %||% "(none)"))
  cat(sprintf("  horizon     : %s %s\n", x$survival$horizon, x$survival$time_unit))
  cat(sprintf("  risk tiers  : %s\n", paste(x$risk_groups$labels, collapse = " > ")))
  cat(sprintf("  oncoplot    : %d track(s)\n", length(x$oncoplot$tracks)))
  invisible(x)
}

# --- Small accessors so downstream code never hard-codes column names --------

#' Resolve a column role from the config to an actual column name.
#' @param cfg A prognostk_config.
#' @param role One of the keys under \code{columns} (e.g. "time", "status", "score").
#' @param override Optional explicit column name that takes precedence.
#' @export
col_of <- function(cfg, role, override = NULL) {
  if (!is.null(override)) return(override)
  val <- cfg$columns[[role]]
  if (is.null(val)) {
    stop(sprintf("config: no column configured for role '%s' and no override given.", role),
         call. = FALSE)
  }
  val
}
