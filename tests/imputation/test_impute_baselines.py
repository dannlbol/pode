import numpy as np
import pytest

from pode.imputation.base import category_key
from pode.imputation.baselines import KNNEngine, MeanModeImputer, MICEEngine, MissForestEngine
from pode.imputation.gain import GAINEngine
from pode.imputation.config import GainConfig
from pode.imputation.masking import MaskSpec, apply_mask, make_mask

ENGINES = [
    lambda: MeanModeImputer(),
    lambda: KNNEngine(),
    lambda: MICEEngine(max_iter=2, n_jobs=1),
    lambda: MissForestEngine(n_estimators=5, max_iter=2),
    lambda: GAINEngine(GainConfig(epochs=50), fit_seed=0, device="cpu"),
]


@pytest.mark.parametrize("make", ENGINES)
def test_baseline_fills_all_and_keeps_observed(cohort, cohort_schema, make):
    train, test = cohort.iloc[:150], cohort.iloc[150:]
    mask = make_mask(test, MaskSpec("m", "mcar", rate=0.3), cohort_schema, np.random.default_rng(0))
    masked = apply_mask(test, mask)
    imp = make().fit(train, cohort_schema)
    draws = imp.impute(masked, m=3, seed=1)
    assert len(draws) == 3
    for d in draws:
        assert list(d.index) == list(test.index)
        assert not d.isna().any().any()
        obs = masked.notna()
        for c in cohort.columns:
            assert (category_key(d.loc[obs[c], c]) == category_key(masked.loc[obs[c], c])).all()
        assert set(d["subgroup"]) <= set(cohort["subgroup"])
        assert set(d["treated"]) <= {0, 1}


def test_mice_draws_differ_but_single_imputers_do_not(cohort, cohort_schema):
    masked = cohort.copy()
    masked.loc[:40, "bio"] = np.nan
    mice = MICEEngine(max_iter=2, n_jobs=1).fit(cohort.iloc[100:], cohort_schema).impute(masked, m=2, seed=0)
    assert not np.allclose(mice[0]["bio"][:41], mice[1]["bio"][:41])
    mm = MeanModeImputer().fit(cohort, cohort_schema).impute(masked, m=2, seed=0)
    assert np.allclose(mm[0]["bio"], mm[1]["bio"])
