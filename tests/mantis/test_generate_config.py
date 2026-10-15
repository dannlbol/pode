import yaml
import pytest

from pode_mantis.core.generate_config import run_generate_config

from conftest import toy_cfg


def _generate(toy_paths, **cfg_overrides):
    cfg = toy_cfg(toy_paths, **cfg_overrides)
    return cfg, run_generate_config(
        cfg,
        cohort_name="my_cohort",
        beta_path=toy_paths["tmp_path"] / "beta_numeric.csv",
        cnv_path=toy_paths["tmp_path"] / "cnv_numeric.csv",
        surv_path=toy_paths["tmp_path"] / "survival.csv",
    )


def test_fixed_sections_copied_verbatim(toy_paths):
    cfg, result = _generate(toy_paths)
    with open(result.out_path) as f:
        generated = yaml.safe_load(f)
    with open(toy_paths["reference_config_path"]) as f:
        reference = yaml.safe_load(f)

    assert generated["data"]["time_bins"] == reference["data"]["time_bins"]
    assert generated["data"]["conn_mat_path"] == reference["data"]["conn_mat_path"]
    assert generated["model"] == reference["model"]
    assert generated["run_setup"]["feature_types"] == reference["run_setup"]["feature_types"]
    assert generated["run_setup"]["metastasis_recoding"] == reference["run_setup"]["metastasis_recoding"]
    assert generated["run_setup"]["training_cohorts"] == reference["run_setup"]["training_cohorts"]


def test_cohort_specific_fields_substituted(toy_paths):
    cfg, result = _generate(toy_paths)
    with open(result.out_path) as f:
        generated = yaml.safe_load(f)

    assert generated["run_setup"]["test_cohort"] == "my_cohort"
    assert "my_cohort_beta_path" in generated["data"]
    assert "my_cohort_cnv_path" in generated["data"]
    assert "my_cohort_surv_path" in generated["data"]
    # written as absolute paths, not carried through relative
    assert generated["data"]["my_cohort_beta_path"].startswith("/")


def test_os_pfs_footgun_closed_for_os_endpoint(toy_paths):
    cfg, result = _generate(toy_paths, reference_config={"endpoint": "os"})
    with open(result.out_path) as f:
        generated = yaml.safe_load(f)
    assert generated["data"]["time_column"] == generated["data"]["os_time_column"]
    assert generated["data"]["event_column"] == generated["data"]["os_event_column"]
    assert generated["data"]["time_column"] == "t"


def test_os_pfs_footgun_closed_for_pfs_endpoint(toy_paths):
    cfg, result = _generate(toy_paths, reference_config={"endpoint": "pfs"})
    with open(result.out_path) as f:
        generated = yaml.safe_load(f)
    # The documented bug: evaluate.py scores using os_time_column/os_event_column
    # regardless of intent. For a PFS run, BOTH pairs must point at the PFS columns.
    assert generated["data"]["time_column"] == "pfs_time"
    assert generated["data"]["os_time_column"] == "pfs_time"
    assert generated["data"]["event_column"] == "pfs_event"
    assert generated["data"]["os_event_column"] == "pfs_event"


def test_requires_fixed_sections_from(toy_paths):
    cfg = toy_cfg(toy_paths, reference_config={"fixed_sections_from": None})
    with pytest.raises(ValueError, match="fixed_sections_from"):
        run_generate_config(
            cfg,
            cohort_name="my_cohort",
            beta_path="x.csv",
            cnv_path="y.csv",
            surv_path="z.csv",
        )


def test_training_cohort_surv_paths_carried_forward(toy_paths):
    cfg, result = _generate(toy_paths)
    with open(result.out_path) as f:
        generated = yaml.safe_load(f)
    assert "toy_train_surv_path" in generated["data"]
