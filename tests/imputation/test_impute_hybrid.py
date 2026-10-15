import numpy as np
import pytest

from pode.imputation.base import category_key
from pode.imputation.baselines import MeanModeImputer, MICEEngine
from pode.imputation.hybrid import HybridImputer
from pode.imputation.masking import MaskSpec, make_mask

GROUPS = {"lat": ["lat_*"]}


def _gappy(cohort):
    df = cohort.iloc[150:].copy()
    df.loc[df.index[:10], ["lat_0", "lat_1"]] = np.nan  # absent block
    df.loc[df.index[10:20], "bio"] = np.nan  # scattered gaps
    df.loc[df.index[5], "Age"] = np.nan  # block row with an extra gap
    return df


def test_routes_block_rows_and_gap_rows(cohort, cohort_schema):
    gap, block = MeanModeImputer(), MICEEngine(max_iter=2, n_jobs=1)
    hyb = HybridImputer(gap, block, GROUPS).fit(cohort.iloc[:150], cohort_schema)
    df = _gappy(cohort)
    route = hyb.route(df)
    assert route[:10].all() and not route[10:].any()
    draws = hyb.impute(df, m=3, seed=0)
    diag = hyb.diagnostics()
    assert diag["rows_to_block_engine"] == 10 and diag["rows_to_gap_engine"] == 10
    for d in draws:
        assert not d.isna().any().any()
        # Gap rows -> mean/mode (constant); block rows -> MICE (varied).
        assert np.allclose(d.loc[df.index[10:20], "bio"], gap.fill["bio"])
        assert d.loc[df.index[:10], "lat_0"].nunique() > 1
        obs = df.notna()
        for c in df.columns:
            assert (category_key(d.loc[obs[c], c]) == category_key(df.loc[obs[c], c])).all()


def test_partial_block_threshold(cohort, cohort_schema):
    hyb = HybridImputer(MeanModeImputer(), MeanModeImputer(), GROUPS, block_min_frac=0.5)
    hyb.fit(cohort.iloc[:150], cohort_schema)
    df = cohort.iloc[150:155].copy()
    df.loc[df.index[0], "lat_0"] = np.nan  # half the group missing
    assert hyb.route(df)[0] and not hyb.route(df)[1:].any()


def test_hybrid_requires_block_groups():
    with pytest.raises(ValueError, match="block_groups"):
        HybridImputer(MeanModeImputer(), MeanModeImputer(), {})


def test_union_mask_is_or_of_parts(cohort, cohort_schema):
    parts = [{"kind": "mcar", "rate": 0.1}, {"kind": "block", "group": "lat", "row_frac": 0.3}]
    union = make_mask(cohort, MaskSpec.from_dict({"name": "u", "kind": "union", "parts": parts}),
                      cohort_schema, np.random.default_rng(5), groups=GROUPS)
    rng = np.random.default_rng(5)
    a = make_mask(cohort, MaskSpec.from_dict({"name": "a", **parts[0]}), cohort_schema, rng, groups=GROUPS)
    b = make_mask(cohort, MaskSpec.from_dict({"name": "b", **parts[1]}), cohort_schema, rng, groups=GROUPS)
    assert (union == (a | b)).all()
