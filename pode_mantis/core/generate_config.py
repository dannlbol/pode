"""Assemble a MANTIS reference-repo ``config.yaml`` for a new cohort.

The reference implementation's own config mixes together things that must
**never** change (``model:``, ``data.time_bins`` -- tied 1:1 to the released
checkpoint's tensor shapes; ``run_setup.feature_types``/``metastasis_recoding``
-- tied to how the checkpoint was trained to interpret its inputs) with
things that are wholly cohort-specific (file paths, column names, which
cohort is under test). This module copies the FIXED sections verbatim from
the user's own copy of the reference repo's ``config.yaml`` and fills in only
the cohort-specific fields -- never hand-edits or guesses at the fixed
values.

It also closes a real bug documented in the source project's
``NOTES_implementation.md``: ``evaluate.py`` scores the *test cohort's*
outcome using ``os_time_column``/``os_event_column`` regardless of which
logical endpoint (OS vs PFS) you intended, falling back to
``time_column``/``event_column`` only if the ``os_*`` keys are absent. A
hand-written "PFS config" that only changes ``time_column``/``event_column``
therefore silently still evaluates on OS. This module sets both pairs
consistently from a single ``endpoint: os|pfs`` choice, so that footgun
cannot recur by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_FIXED_DATA_KEYS = ("conn_mat_path", "conn_mat_is_one_based", "time_bins")
_FIXED_RUN_SETUP_KEYS = ("training_cohorts", "feature_types", "metastasis_recoding")


@dataclass
class GenerateConfigResult:
    """Outcome of a ``run_generate_config`` call. ``out_path`` is what
    ``run_reference``'s ``infer``/``evaluate``/``plot`` subcommands pass to
    the reference implementation via ``--config``."""

    out_path: Path
    cohort_name: str
    endpoint: str


def _abspath(p) -> str:
    return str(Path(p).expanduser().resolve())


def run_generate_config(
    cfg,
    cohort_name: str,
    beta_path,
    cnv_path,
    surv_path,
    endpoint: Optional[str] = None,
    out_path=None,
) -> GenerateConfigResult:
    rc = cfg["reference_config"]
    surv_cfg = cfg["survival"]
    paths = cfg["paths"]

    fixed_from = rc.get("fixed_sections_from")
    if not fixed_from:
        raise ValueError(
            "generate-config: reference_config.fixed_sections_from is required -- point it at the "
            "reference repo's own config.yaml (the one shipped with sstefan01/methylation-survival) "
            "so the model/time_bins/feature_types/metastasis_recoding sections are copied verbatim, "
            "not re-guessed."
        )
    with open(fixed_from) as f:
        reference_yaml = yaml.safe_load(f)

    missing_data = [k for k in _FIXED_DATA_KEYS if k not in reference_yaml.get("data", {})]
    missing_run = [k for k in _FIXED_RUN_SETUP_KEYS if k not in reference_yaml.get("run_setup", {})]
    if missing_data or missing_run or "model" not in reference_yaml:
        raise ValueError(
            f"generate-config: {fixed_from} is missing expected fixed keys "
            f"(data: {missing_data}, run_setup: {missing_run}, model section present: "
            f"{'model' in reference_yaml}) -- is this really the reference repo's own config.yaml?"
        )

    endpoint = endpoint or rc["endpoint"]
    if endpoint == "os":
        time_col, event_col = surv_cfg["time_column"] or "time", surv_cfg["event_column"] or "event"
    elif endpoint == "pfs":
        if not surv_cfg.get("pfs_time_column") or not surv_cfg.get("pfs_event_column"):
            raise ValueError("generate-config: endpoint='pfs' requires survival.pfs_time_column/pfs_event_column.")
        time_col = "pfs_time"
        event_col = "pfs_event"
    else:
        raise ValueError(f"generate-config: unknown endpoint '{endpoint}' (expected 'os' or 'pfs').")

    model_weights = cfg["inference"].get("model_weights_path")
    if not model_weights:
        if not paths.get("zenodo_assets_dir"):
            raise ValueError(
                "generate-config: no model checkpoint given (config inference.model_weights_path, "
                "or paths.zenodo_assets_dir to default to <dir>/mantismb_model_weights.pt)."
            )
        model_weights = str(Path(paths["zenodo_assets_dir"]) / "mantismb_model_weights.pt")

    out = {
        "data": {
            **{k: reference_yaml["data"][k] for k in _FIXED_DATA_KEYS},
            f"{cohort_name}_beta_path": _abspath(beta_path),
            f"{cohort_name}_cnv_path": _abspath(cnv_path),
            f"{cohort_name}_surv_path": _abspath(surv_path),
            "time_column": time_col,
            "event_column": event_col,
            "clinical_feature_col": "m",
            "id_column": "Patient_ID",
            # Set BOTH pairs consistently -- closes the evaluate.py footgun above.
            "os_time_column": time_col,
            "os_event_column": event_col,
        },
        "run_setup": {
            **{k: reference_yaml["run_setup"][k] for k in _FIXED_RUN_SETUP_KEYS},
            "test_cohort": cohort_name,
        },
        "model": reference_yaml["model"],
        "inference": {
            "input_model_path": _abspath(model_weights),
            "output_predictions_path": _abspath(cfg["inference"]["output_predictions_path"]),
            "batch_size": cfg["inference"].get("batch_size", 64),
        },
    }

    # Also carry forward each fixed training cohort's own survival path (only --
    # evaluate.py's IPCW-AUC censoring reference reads config[f"{cohort}_surv_path"]
    # for every cohort in run_setup.training_cohorts; it never reads their beta/cnv,
    # and neither does inference.py, which only loads the test cohort's features).
    # Resolved relative to the reference repo dir, since that's where the reference
    # config's own relative paths are anchored -- everything else in this generated
    # config is written as an absolute path to avoid depending on subprocess cwd.
    reference_repo = paths.get("reference_repo")
    for training_cohort in reference_yaml["run_setup"]["training_cohorts"]:
        key = f"{training_cohort}_surv_path"
        if key in reference_yaml.get("data", {}):
            raw = reference_yaml["data"][key]
            out["data"][key] = raw if Path(raw).is_absolute() or not reference_repo else _abspath(
                Path(reference_repo) / raw
            )

    out_path = Path(out_path or rc["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)

    print(f"[saved] {out_path}  (cohort='{cohort_name}', endpoint='{endpoint}', "
          f"time_column='{time_col}', event_column='{event_col}')")
    return GenerateConfigResult(out_path=out_path, cohort_name=cohort_name, endpoint=endpoint)
