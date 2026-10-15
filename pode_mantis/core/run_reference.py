"""Subprocess dispatch into the reference implementation's isolated environment.

Per the source project's own governing principle -- never modify the
reference implementation -- this module calls `inference.py`/`evaluate.py`/
`plot_results.py` from the user's own checkout of
``sstefan01/methylation-survival`` exactly as published, using the
interpreter from a SEPARATE, externally-managed Python environment
(``paths.mantis_python`` in config; see ``docs/mantis/README.md`` for setup).
That environment's pinned dependencies (``sparselinear``, ``torch-scatter``,
``torch-sparse``, ``scikit-survival==0.23.1``) cannot coexist with PODE's own
environment, so this is a real subprocess boundary, not an in-process import.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _resolve(cfg, key: str, value, what: str) -> str:
    value = value or cfg["paths"].get(key)
    if not value:
        raise ValueError(f"{what}: config paths.{key} is required (see docs/mantis/README.md setup).")
    return value


def _run(mantis_python, reference_repo, script: str, args: list[str], what: str) -> subprocess.CompletedProcess:
    script_path = Path(reference_repo) / script
    if not script_path.exists():
        raise FileNotFoundError(
            f"{what}: {script_path} not found -- is paths.reference_repo pointed at a checkout of "
            f"sstefan01/methylation-survival (v1.0.0)?"
        )
    cmd = [mantis_python, str(script_path), *args]
    print(f"[subprocess] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=reference_repo, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"{what} failed (exit {result.returncode}). stderr:\n{result.stderr}\n\n"
            f"Common causes: paths.mantis_python doesn't point at an environment with MANTIS's "
            f"pinned deps installed (see docs/mantis/README.md), or the generated config has a "
            f"stale/incorrect path."
        )
    if result.stderr:
        print(f"[stderr]\n{result.stderr}")
    return result


def run_inference(cfg, mantis_config_path, mantis_python=None, reference_repo=None) -> str:
    """Run the reference ``inference.py`` against a generated MANTIS config.

    ``mantis_python``/``reference_repo`` override ``cfg["paths"]`` if given
    (e.g. from a CLI flag); otherwise both are required in config. Returns
    ``cfg["inference"]["output_predictions_path"]`` -- the reference
    implementation writes there itself, this just reports where.
    """
    mantis_python = _resolve(cfg, "mantis_python", mantis_python, "infer")
    reference_repo = _resolve(cfg, "reference_repo", reference_repo, "infer")
    _run(mantis_python, reference_repo, "inference.py", ["--config", str(mantis_config_path)], "infer")
    return cfg["inference"]["output_predictions_path"]


def run_evaluate(cfg, mantis_config_path, pred_path, output_metrics=None, mantis_python=None, reference_repo=None) -> str:
    """Run the reference ``evaluate.py`` (C-index / time-dependent AUC / Brier
    score) against a predictions CSV from :func:`run_inference`. Needs the
    same generated config's ``run_setup.training_cohorts`` survival files for
    the AUC censoring reference -- see ``generate_config``'s docstring for
    why those are carried forward automatically."""
    mantis_python = _resolve(cfg, "mantis_python", mantis_python, "evaluate")
    reference_repo = _resolve(cfg, "reference_repo", reference_repo, "evaluate")
    output_metrics = output_metrics or str(Path(pred_path).with_suffix("")) + "_metrics.json"
    _run(
        mantis_python,
        reference_repo,
        "evaluate.py",
        ["--config", str(mantis_config_path), "--pred_path", str(pred_path), "--output_metrics", output_metrics],
        "evaluate",
    )
    return output_metrics


def run_plot(cfg, mantis_config_path, metrics_file, pred_path, out_dir, mantis_python=None, reference_repo=None) -> str:
    """Run the reference ``plot_results.py`` (survival curves + AUC/Brier
    plots) against the outputs of :func:`run_inference`/:func:`run_evaluate`.
    Optional -- nothing else in this theme depends on its output."""
    mantis_python = _resolve(cfg, "mantis_python", mantis_python, "plot")
    reference_repo = _resolve(cfg, "reference_repo", reference_repo, "plot")
    _run(
        mantis_python,
        reference_repo,
        "plot_results.py",
        [
            "--config", str(mantis_config_path),
            "--metrics_file", str(metrics_file),
            "--plot_survival",
            "--predictions_file", str(pred_path),
            "--output_dir", str(out_dir),
        ],
        "plot",
    )
    return str(out_dir)
