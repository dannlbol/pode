"""Tests for the configurable objective and its metric registry."""

from pode.config import ObjectiveConfig
from pode.evaluation.objective import (
    available_metrics,
    compute_composite_score,
    register_metric,
)
from pode.schema import DatasetSchema


def test_builtin_metrics_registered():
    metrics = available_metrics()
    for name in ("hr_mae", "tvd_mean", "surv_prob_diff_5yr", "quality_penalty"):
        assert name in metrics


def test_custom_metric_plugs_in(sample_df):
    """A registered metric can drive the composite via config weights alone."""

    @register_metric("constant_penalty")
    def _constant(real_df, synth_df, schema, **ctx):
        return 0.25

    schema = DatasetSchema.from_dataframe(sample_df)
    config = ObjectiveConfig(weights={"constant_penalty": 2.0})
    result = compute_composite_score(sample_df, sample_df, schema, config)

    assert result["metrics"]["constant_penalty"] == 0.25
    assert result["composite"] == 0.5  # 2.0 * 0.25


def test_zero_weight_metric_skipped(sample_df):
    schema = DatasetSchema.from_dataframe(sample_df)

    @register_metric("should_not_run")
    def _boom(real_df, synth_df, schema, **ctx):
        raise AssertionError("zero-weight metric must not be evaluated")

    config = ObjectiveConfig(weights={"should_not_run": 0.0})
    result = compute_composite_score(sample_df, sample_df, schema, config)
    assert result["composite"] == 0.0
