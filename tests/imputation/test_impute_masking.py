import numpy as np
import pytest

from pode.imputation.masking import MaskSpec, apply_mask, eligible_columns, make_mask


def _mask(cohort, schema, seed=0, groups=None, **spec):
    spec.setdefault("name", spec["kind"])
    return make_mask(cohort, MaskSpec.from_dict(spec), schema, np.random.default_rng(seed), groups=groups)


def test_mcar_rate_and_protection(cohort, cohort_schema):
    mask = _mask(cohort, cohort_schema, kind="mcar", rate=0.3)
    cols = list(cohort.columns)
    for surv in ("OS_Time", "OS_Status"):
        assert not mask[:, cols.index(surv)].any()
    eligible = [cols.index(c) for c in eligible_columns(cohort, cohort_schema)]
    assert abs(mask[:, eligible].mean() - 0.3) < 0.04


def test_masks_are_deterministic_by_seed(cohort, cohort_schema):
    a = _mask(cohort, cohort_schema, seed=7, kind="mcar", rate=0.2)
    b = _mask(cohort, cohort_schema, seed=7, kind="mcar", rate=0.2)
    c = _mask(cohort, cohort_schema, seed=8, kind="mcar", rate=0.2)
    assert (a == b).all() and (a != c).any()


def test_block_hides_whole_group(cohort, cohort_schema):
    mask = _mask(cohort, cohort_schema, kind="block", group="lat", row_frac=0.4, groups={"lat": ["lat_*"]})
    cols = list(cohort.columns)
    l0, l1 = mask[:, cols.index("lat_0")], mask[:, cols.index("lat_1")]
    assert (l0 == l1).all() and 0.25 < l0.mean() < 0.55
    others = [i for i, c in enumerate(cols) if not c.startswith("lat_")]
    assert not mask[:, others].any()


def test_mar_keeps_drivers_and_depends_on_them(cohort, cohort_schema):
    # seed != 0: the fixture draws Age from default_rng(0), which would correlate with the mask.
    mask = _mask(cohort, cohort_schema, seed=3, kind="mar", rate=0.3, drivers=["Age"], columns=["bio"])
    cols = list(cohort.columns)
    assert not mask[:, cols.index("Age")].any()
    hidden = mask[:, cols.index("bio")]
    assert abs(hidden.mean() - 0.3) < 0.1
    assert cohort.loc[hidden, "Age"].mean() != pytest.approx(cohort.loc[~hidden, "Age"].mean(), abs=0.5)


def test_tail_and_rare_category(cohort, cohort_schema):
    cols = list(cohort.columns)
    tail = _mask(cohort, cohort_schema, kind="tail", columns=["Age"], quantile=0.8, rate=1.0)
    hidden = tail[:, cols.index("Age")]
    assert cohort.loc[hidden, "Age"].min() >= cohort["Age"].quantile(0.8) - 1e-9
    rare = _mask(cohort, cohort_schema, kind="rare_category", columns=["subgroup"], max_freq=0.2, rate=1.0)
    freq = cohort["subgroup"].value_counts(normalize=True)
    assert set(cohort.loc[rare[:, cols.index("subgroup")], "subgroup"]) <= set(freq[freq < 0.2].index)


def test_never_masks_already_missing(cohort, cohort_schema):
    df = cohort.copy()
    df.loc[:9, "bio"] = np.nan
    mask = _mask(df, cohort_schema, kind="mcar", rate=1.0)
    assert not mask[:10, list(df.columns).index("bio")].any()
    assert apply_mask(df, mask)["bio"].isna().all()


def test_unknown_block_group_raises(cohort, cohort_schema):
    with pytest.raises(ValueError):
        _mask(cohort, cohort_schema, kind="block", group="nope", groups={})
