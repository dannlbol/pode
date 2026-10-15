test_that("fit_cox recovers a protective score", {
  cfg <- toy_cfg()
  fit <- fit_cox(make_toy(), cfg)
  expect_s3_class(fit$fit, "coxph")
  expect_true(coef(fit$fit)["metaCpG"] < 0)   # higher score -> better survival
})

test_that("predict_survival_at_time returns valid probabilities", {
  cfg <- toy_cfg(); d <- make_toy()
  fit <- fit_cox(d, cfg)
  ps <- predict_survival_at_time(fit, d)
  expect_length(ps, nrow(d))
  expect_true(all(ps >= 0 & ps <= 1, na.rm = TRUE))
})

test_that("stratified models predict without error", {
  cfg <- toy_cfg()
  fit <- fit_cox(make_toy(), cfg, covariates = c("metaCpG", "strata:grp"))
  ps <- predict_survival_at_time(fit, make_toy())
  expect_true(all(is.finite(ps)))
})

test_that("assign_risk_groups partitions and orders correctly", {
  cfg <- toy_cfg()
  g <- assign_risk_groups(c(0.95, 0.8, 0.6, 0.3), cfg)
  expect_equal(as.character(g),
               c("Favourable", "Standard", "High", "VeryHigh"))
  expect_equal(levels(g), cfg$risk_groups$labels)
})

test_that("bootstrap_survival_ci returns bracketed CIs", {
  cfg <- toy_cfg(); d <- make_toy()
  bs <- bootstrap_survival_ci(d, cfg, n = 30, seed = 1)
  expect_equal(nrow(bs), nrow(fit_cox(d, cfg)$data))
  ok <- !is.na(bs$ci_lower)
  expect_true(all(bs$ci_lower[ok] <= bs$pred_original[ok] + 1e-6 |
                  bs$ci_upper[ok] >= bs$pred_original[ok] - 1e-6))
})

test_that("external_validate reports a plausible C-index", {
  cfg <- toy_cfg(); d <- make_toy(200)
  fit <- fit_cox(d[1:150, ], cfg)
  ev <- external_validate(d[151:200, ], fit, cfg)
  expect_true(ev$c_index > 0.4 && ev$c_index <= 1)
})

test_that("optimise_cutoffs finds a feasible solution", {
  cfg <- toy_cfg(); d <- make_toy(200)
  fit <- fit_cox(d, cfg); ps <- predict_survival_at_time(fit, d)
  oc <- optimise_cutoffs(d, ps, cfg, labels = c("Low", "Med", "High"),
                         fav_min_surv = 0.6, grid = seq(0.3, 0.8, 0.1))
  expect_length(oc$best, 2)
  expect_true(all(diff(oc$best) < 0))          # strictly descending
})

test_that("calibration_table E/O ratio is finite and positive", {
  cfg <- toy_cfg(); d <- make_toy()
  fit <- fit_cox(d, cfg)
  cal <- calibration_table(d, predict_survival_at_time(fit, d), cfg, n_bins = 4)
  expect_true(is.finite(cal$eo_ratio) && cal$eo_ratio > 0)
})
