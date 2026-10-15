test_that("defaults load and validate", {
  cfg <- load_config(NULL)
  expect_s3_class(cfg, "prognostk_config")
  expect_equal(cfg$survival$horizon, 5)
  expect_length(cfg$risk_groups$labels, length(cfg$risk_groups$thresholds) + 1)
})

test_that("overrides win over file/defaults", {
  cfg <- load_config(NULL, survival = list(horizon = 10))
  expect_equal(cfg$survival$horizon, 10)
})

test_that("missing required columns error", {
  expect_error(validate_config(list(columns = list(time = NULL, status = "s"))),
               "columns.time")
})

test_that("bad thresholds are rejected", {
  expect_error(load_config(NULL, risk_groups = list(
    thresholds = c(0.5, 0.7), labels = c("a", "b", "c"))),
    "descending")
  expect_error(load_config(NULL, risk_groups = list(
    thresholds = c(0.7, 0.5), labels = c("a", "b"))),
    "length")
})

test_that("col_of resolves roles and overrides", {
  cfg <- toy_cfg()
  expect_equal(col_of(cfg, "time"), "OS_Time")
  expect_equal(col_of(cfg, "time", "custom"), "custom")
})
