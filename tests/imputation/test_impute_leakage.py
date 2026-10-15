"""What the imputer may see: outcome policy, excluded columns, fit scope."""

import numpy as np
import pandas as pd
import pytest

from pode.config import PodeConfig
from pode.imputation.baselines import KNNEngine, MeanModeImputer, MICEEngine, MissForestEngine
from pode.imputation.config import EngineSpec, ImputationConfig
from pode.imputation.engines import build_imputer


def _split(cohort):
    train, test = cohort.iloc[:150].copy(), cohort.iloc[150:].copy()
    test.loc[test.index[:25], "bio"] = np.nan
    test.loc[test.index[10:30], "treated"] = np.nan
    return train, test


def _perturb_outcomes(test):
    out = test.copy()
    out["OS_Time"] = out["OS_Time"][::-1].to_numpy() * 3 + 1
    out["OS_Status"] = 1 - out["OS_Status"]
    return out


def _imputed(draws, test):
    return np.stack([pd.to_numeric(d.loc[test["bio"].isna(), "bio"]).to_numpy() for d in draws])


@pytest.mark.parametrize("policy,changes", [("use", True), ("hide", False)])
def test_outcome_hide_blocks_own_outcome(cohort, cohort_schema, policy, changes):
    train, test = _split(cohort)
    imp = MICEEngine(max_iter=3, n_jobs=1).configure(policy).fit(train, cohort_schema)
    a = imp.impute(test, m=3, seed=0)
    b = imp.impute(_perturb_outcomes(test), m=3, seed=0)
    assert (not np.allclose(_imputed(a, test), _imputed(b, test))) == changes
    # Outcomes come back exactly as given.
    assert np.allclose(a[0]["OS_Time"], test["OS_Time"])
    assert (a[0]["OS_Status"].to_numpy() == test["OS_Status"].to_numpy()).all()


def test_outcome_exclude_and_column_exclusion(cohort, cohort_schema):
    train, test = _split(cohort)
    test.loc[test.index[:5], "lat_0"] = np.nan
    imp = MICEEngine(max_iter=2, n_jobs=1).configure("exclude", exclude_columns=["lat_*"])
    imp.fit(train, cohort_schema)
    assert not {"OS_Time", "OS_Status", "lat_0", "lat_1"} & set(imp.columns)
    assert set(imp.excluded_columns) == {"OS_Time", "OS_Status", "lat_0", "lat_1"}
    d = imp.impute(test, m=2, seed=0)[0]
    assert d["lat_0"].isna().sum() == 5  # excluded -> not imputed
    assert not d["bio"].isna().any()
    assert np.allclose(d["OS_Time"], test["OS_Time"])


def test_custom_outcome_columns(cohort, cohort_schema):
    train, _ = _split(cohort)
    imp = MeanModeImputer().configure("exclude", outcome_columns=["stage"]).fit(train, cohort_schema)
    assert "stage" not in imp.columns and "OS_Time" in imp.columns


def test_exclude_on_fitted_model_degrades_to_hide(cohort, cohort_schema):
    imp = MeanModeImputer().fit(cohort, cohort_schema)
    with pytest.warns(UserWarning, match="cannot be"):
        imp.configure("exclude")
    assert imp.outcome_policy == "hide" and set(imp.outcome_columns) == {"OS_Time", "OS_Status"}


@pytest.mark.parametrize("make", [
    lambda s: MICEEngine(max_iter=3, n_jobs=1, fit_scope=s),
    lambda s: KNNEngine(fit_scope=s),
    lambda s: MissForestEngine(n_estimators=5, max_iter=2, fit_scope=s),
])
def test_reference_scope_isolates_target_rows(cohort, cohort_schema, make):
    """Under fit_scope='reference', other target rows cannot affect a row's imputation."""
    train, test = _split(cohort)
    other = test.copy()
    rows = other.index[30:]  # fully observed in bio; shift their observed values
    other.loc[rows, "bio"] = other.loc[rows, "bio"] + 5
    other.loc[rows, "Age"] = other.loc[rows, "Age"] * 0.2

    def first_rows(t, scope):
        imp = make(scope).fit(train, cohort_schema)
        return _imputed(imp.impute(t, m=2, seed=1), test)

    assert np.allclose(first_rows(test, "reference"), first_rows(other, "reference"))
    if isinstance(make("stacked"), (MICEEngine, KNNEngine)):
        assert not np.allclose(first_rows(test, "stacked"), first_rows(other, "stacked"))


def test_transductive_use_does_not_duplicate_rows(cohort, cohort_schema):
    df = cohort.copy()
    df.loc[:30, "bio"] = np.nan
    imp = MICEEngine(max_iter=1, n_jobs=1).fit(df, cohort_schema)
    mat, n_learn, start = imp._stack(df[imp.columns])
    assert mat.shape[0] == len(df) and start == 0
    mat2, _, start2 = imp._stack(df[imp.columns].iloc[:50])
    assert mat2.shape[0] == len(df) + 50 and start2 == len(df)


def test_build_imputer_applies_spec_and_global_leakage():
    icfg = ImputationConfig(outcome_policy="hide", fit_scope="reference", exclude_columns=["x*"])
    imp = build_imputer(EngineSpec("mice", params={"outcome_policy": "use"}), PodeConfig(), icfg)
    assert imp.outcome_policy == "use" and imp.fit_scope == "reference"
    assert imp._exclude_patterns == ["x*"]
    with pytest.raises(ValueError):
        MICEEngine(fit_scope="nope")
    with pytest.raises(ValueError):
        MeanModeImputer().configure("sometimes")
