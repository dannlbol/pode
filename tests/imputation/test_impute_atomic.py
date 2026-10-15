"""Modality blocks are all-or-nothing: masks, training masks and apply checks."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from pode.imputation.masking import MaskSpec, make_mask, partial_block_rows

GROUPS = {"lat": ["lat_*"]}


def _mask(df, schema, atomic=True, seed=0, **spec):
    spec.setdefault("name", spec["kind"])
    return make_mask(df, MaskSpec.from_dict(spec), schema, np.random.default_rng(seed),
                     groups=GROUPS, atomic_blocks=atomic)


def test_cell_level_masks_skip_atomic_blocks(cohort, cohort_schema):
    cols = list(cohort.columns)
    lat = [cols.index("lat_0"), cols.index("lat_1")]
    for kind in ({"kind": "mcar", "rate": 0.5}, {"kind": "mar", "rate": 0.3, "drivers": ["Age"]},
                 {"kind": "mnar_self", "rate": 0.3}, {"kind": "tail", "quantile": 0.8, "rate": 1.0}):
        m = _mask(cohort, cohort_schema, **kind)
        assert not m[:, lat].any(), kind
        assert m.any()
    # Without atomic blocks the same MCAR mask does split the block.
    m = _mask(cohort, cohort_schema, atomic=False, kind="mcar", rate=0.5)
    assert (m[:, lat[0]] != m[:, lat[1]]).any()


def test_explicit_partial_block_target_is_refused(cohort, cohort_schema):
    with pytest.raises(ValueError, match="all-or-nothing"):
        _mask(cohort, cohort_schema, kind="mcar", rate=0.3, columns=["lat_0"])


def test_independent_modality_blocks_stay_whole(cohort, cohort_schema):
    groups = {"a": ["lat_0"], "b": ["lat_1", "cg_0001"]}
    df = cohort.assign(cg_0001=np.random.default_rng(1).uniform(size=len(cohort)))
    spec = {"name": "mods", "kind": "union", "parts": [
        {"kind": "block", "group": "a", "row_frac": 0.3},
        {"kind": "block", "group": "b", "row_frac": 0.3},
        {"kind": "mcar", "rate": 0.2}]}
    from pode.schema.schema import DatasetSchema
    schema = DatasetSchema.from_dataframe(df)
    m = make_mask(df, MaskSpec.from_dict(spec), schema, np.random.default_rng(0), groups=groups, atomic_blocks=True)
    cols = list(df.columns)
    b = m[:, [cols.index("lat_1"), cols.index("cg_0001")]]
    assert (b.all(axis=1) | ~b.any(axis=1)).all()          # whole block or nothing
    both = m[:, cols.index("lat_0")] & b[:, 0]
    assert both.any() and (~both).any()                      # some rows lose both modalities
    assert m[:, cols.index("Age")].any()                     # clinical gaps still present


def test_partial_block_rows_detection(cohort):
    df = cohort.copy()
    df.loc[:4, "lat_0"] = np.nan                              # partial: lat_1 present
    df.loc[10:12, ["lat_0", "lat_1"]] = np.nan                # whole block: fine
    assert partial_block_rows(df, GROUPS) == {"lat": 5}


def test_masked_tvae_training_drop_respects_blocks(cohort, cohort_schema):
    from pode.imputation.config import MaskedTVAEConfig
    from pode.imputation.masked_tvae import MaskedTVAEImputer

    imp = MaskedTVAEImputer(masked=MaskedTVAEConfig(p_max=0.9, block_prob=0.4), device="cpu")
    ob = torch.ones(500, 4, dtype=torch.bool)
    groups = torch.tensor([[False, False, True, True]])
    atomic = imp._training_drop(ob, groups, ~groups.any(dim=0))
    blk = atomic[:, 2:]
    assert (blk.all(dim=1) | ~blk.any(dim=1)).all()
    assert blk.all(dim=1).float().mean().item() == pytest.approx(0.4, abs=0.08)
    assert atomic[:, :2].any()
    loose = imp._training_drop(ob, groups, torch.ones(4, dtype=torch.bool))
    assert (loose[:, 2] != loose[:, 3]).any()


def test_apply_reports_partial_blocks(tmp_path, cohort):
    from pode.cli.main import main
    import yaml

    df = cohort.copy()
    df.loc[:4, "lat_0"] = np.nan
    df.loc[20:25, "bio"] = np.nan
    data = tmp_path / "d.csv"
    df.to_csv(data, index=False)
    icfg = tmp_path / "i.yaml"
    icfg.write_text(yaml.safe_dump({"block_groups": GROUPS}))
    out = tmp_path / "o"
    assert main(["synthesis", "impute", "apply", "--engine", "mean_mode", "--data", str(data),
                 "--impute-config", str(icfg), "--m", "1", "--output-dir", str(out)]) == 0
    assert json.loads((out / "diagnostics.json").read_text())["partial_block_rows"] == {"lat": 5}
