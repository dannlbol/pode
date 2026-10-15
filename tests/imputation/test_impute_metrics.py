import numpy as np
import pandas as pd
import pytest

from pode.imputation.metrics import (
    apply_delta,
    cell_metrics,
    crps_ensemble,
    interval_coverage,
    pooled_cox,
    rubin_pool,
)


def test_crps_point_mass_is_absolute_error():
    y = np.array([0.0, 1.0])
    draws = np.array([[0.5, 3.0]] * 10)
    assert crps_ensemble(draws, y) == pytest.approx([0.5, 2.0])


def test_crps_gaussian_matches_closed_form():
    rng = np.random.default_rng(0)
    draws = rng.normal(0, 1, (4000, 1))
    # CRPS(N(0,1), 0) = 2*phi(0) - 1/sqrt(pi)
    expected = 2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi)
    assert crps_ensemble(draws, np.array([0.0]))[0] == pytest.approx(expected, rel=0.03)


def test_interval_coverage_is_calibrated_for_correct_model():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 2000)
    draws = rng.normal(0, 1, (200, 2000))
    covered, _ = interval_coverage(draws, y, 0.9)
    assert covered.mean() == pytest.approx(0.9, abs=0.03)


def test_rubin_pool_hand_computed():
    q = np.array([1.0, 2.0, 3.0])
    se = np.array([1.0, 1.0, 1.0])
    out = rubin_pool(q, se, n=1000)
    # qbar = 2, ubar = 1, b = 1, t = 1 + (4/3) * 1
    assert out["coef"] == pytest.approx(2.0)
    assert out["between_var"] == pytest.approx(1.0)
    assert out["se"] == pytest.approx(np.sqrt(1 + 4 / 3))
    assert 0 < out["fmi"] < 1
    single = rubin_pool(np.array([2.0]), np.array([0.5]), n=1000)
    assert single["between_var"] == 0 and single["se"] == pytest.approx(0.5)


def test_cell_metrics_perfect_and_constant_imputer(cohort, cohort_schema):
    mask = np.zeros(cohort.shape, bool)
    cols = list(cohort.columns)
    mask[:50, cols.index("bio")] = True
    mask[:50, cols.index("subgroup")] = True
    perfect = cell_metrics(cohort, [cohort.copy()] * 5, mask, cohort_schema)
    by = {r["column"]: r for r in perfect}
    assert by["bio"]["mae"] == pytest.approx(0) and by["bio"]["crps"] == pytest.approx(0)
    assert by["subgroup"]["accuracy"] == 1.0 and by["subgroup"]["brier"] < 0.1

    const = cohort.copy()
    const.loc[:49, "bio"] = cohort["bio"].mean()
    rows = {r["column"]: r for r in cell_metrics(cohort, [const] * 5, mask, cohort_schema)}
    assert rows["bio"]["coverage90"] < 0.2  # a point mass rarely covers the truth


def test_pooled_cox_and_delta(cohort, cohort_schema):
    mask = np.zeros(cohort.shape, bool)
    mask[:60, list(cohort.columns).index("bio")] = True
    mask[:60, list(cohort.columns).index("treated")] = True
    draws = [cohort.copy() for _ in range(3)]
    pooled = pooled_cox(draws, ["bio", "treated"], cohort_schema, "OS_Time", "OS_Status", cohort)
    assert set(pooled["covariate"]) == {"bio", "treated"}
    shifted = apply_delta(draws, mask, ["bio", "treated"], cohort_schema, cohort, 1.0, np.random.default_rng(0))
    sd = cohort["bio"].std()
    assert np.allclose(shifted[0]["bio"][:60], cohort["bio"][:60] + sd)
    assert np.allclose(shifted[0]["bio"][60:], cohort["bio"][60:])
    assert apply_delta(draws, mask, ["bio"], cohort_schema, cohort, 0.0, np.random.default_rng(0)) is draws
