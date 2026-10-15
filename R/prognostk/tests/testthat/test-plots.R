# Plot smoke tests: functions build a ggplot / write a file without error.

test_that("theme_pub and palette helpers work", {
  cfg <- toy_cfg()
  expect_s3_class(theme_pub(cfg = cfg), "theme")
  pal <- prognostk_palette(cfg, "risk", cfg$risk_groups$labels)
  expect_length(pal, length(cfg$risk_groups$labels))
  # unmapped level gets an auto colour rather than erroring
  expect_length(prognostk_palette(cfg, "Sex", c("M", "F", "Unknown")), 3)
})

test_that("plot_km returns an object", {
  cfg <- toy_cfg(); d <- make_toy()
  fit <- fit_cox(d, cfg); d$risk <- assign_risk_groups(predict_survival_at_time(fit, d), cfg)
  p <- plot_km(d, cfg, risk_table = FALSE)
  expect_true(inherits(p, "ggplot") || inherits(p, "patchwork"))
})

test_that("plot_distribution builds violin/box/ridge", {
  cfg <- toy_cfg(); d <- make_toy()
  expect_s3_class(plot_distribution(d, cfg, group_col = "grp", type = "violin"), "ggplot")
  expect_s3_class(plot_distribution(d, cfg, group_col = "grp", type = "box"), "ggplot")
})

test_that("plot_forest and plot_calibration build", {
  cfg <- toy_cfg(); d <- make_toy(200)
  expect_s3_class(
    plot_forest(list(a = d[1:100, ], b = d[101:200, ]), cfg), "ggplot")
  fit <- fit_cox(d, cfg)
  expect_s3_class(
    plot_calibration(d, cfg, pred = predict_survival_at_time(fit, d), n_bins = 4), "ggplot")
})

test_that("plot functions write files to the configured outdir", {
  tmp <- file.path(tempdir(), "prognostk_test_figs")
  cfg <- toy_cfg(figure = list(outdir = tmp, device = "pdf"))
  d <- make_toy(); fit <- fit_cox(d, cfg)
  d$risk <- assign_risk_groups(predict_survival_at_time(fit, d), cfg)
  plot_km(d, cfg, risk_table = FALSE, filename = "km.pdf")
  expect_true(file.exists(file.path(tmp, "km.pdf")))
})

test_that("plot_oncoplot runs when ComplexHeatmap is available", {
  skip_if_not_installed("ComplexHeatmap")
  skip_if_not_installed("circlize")
  cfg <- toy_cfg(oncoplot = list(tracks = list(
    list(name = "Sex", column = "Sex", type = "categorical"),
    list(name = "Score", column = "metaCpG", type = "line"))))
  tmp <- file.path(tempdir(), "prognostk_onco")
  cfg$figure$outdir <- tmp
  d <- make_toy()
  expect_error(plot_oncoplot(d, cfg, filename = "onco.pdf"), NA)
  expect_true(file.exists(file.path(tmp, "onco.pdf")))
})
