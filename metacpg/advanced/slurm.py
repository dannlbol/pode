from __future__ import annotations

import csv
import json
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable

from .data import read_advanced_config


ProgressCallback = Callable[[str], None]


def _announce(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def submit_advanced_slurm(config: dict[str, Any] | str | Path, dry_run: bool = False,
                          *, progress: ProgressCallback | None = None) -> dict[str, Any]:
    cfg = read_advanced_config(config) if not isinstance(config, dict) else config
    _announce(progress, "Preparing schema-v2 SLURM task manifest")
    out = Path(cfg["output_dir"]) / "slurm"
    out.mkdir(parents=True, exist_ok=True)
    run_id = cfg.get("execution", {}).get("resume_run_id") or uuid.uuid4().hex
    tasks = out / f"tasks-{run_id}.csv"
    from .data import load_advanced_data
    from .pipeline import _endpoint_rows, _fingerprint, _folds
    data = load_advanced_data(cfg)
    fingerprint = _fingerprint(cfg)
    outer_folds = int(cfg.get("evaluation", {}).get("outer_folds", 5))
    cohort_validation = str(cfg.get("evaluation", {}).get("cohort_validation", "stratified_kfold"))
    seed = int(cfg.get("execution", {}).get("seed", 1))
    rows = []
    for method in cfg["methods"]["enabled"]:
        for endpoint_index, endpoint in enumerate(data.endpoints):
            endpoint_rows = _endpoint_rows(data, endpoint)
            for fold_index, _ in enumerate(_folds(
                data, endpoint, endpoint_rows, outer_folds, seed + endpoint_index,
                cohort_validation,
            )):
                rows.append({"task_index": len(rows), "run_id": run_id, "fingerprint": fingerprint,
                             "method": method,
                             "endpoint": endpoint["name"], "fold": fold_index, "stage": "outer_cv"})
    with tasks.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    slurm = cfg.get("execution", {}).get("slurm", {})
    script = out / f"advanced-{run_id}.sh"
    aggregate_script = out / f"aggregate-{run_id}.sh"
    setup = slurm.get("setup", [])
    config_path = cfg.get("_config_path")
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={cfg['analysis_id']}-signatures",
        f"#SBATCH --cpus-per-task={slurm.get('cpus', 4)}",
        f"#SBATCH --mem={slurm.get('memory', '16G')}",
        f"#SBATCH --time={slurm.get('time', '24:00:00')}",
        f"#SBATCH --array=0-{len(rows)-1}",
    ]
    if slurm.get("partition"):
        lines.append(f"#SBATCH --partition={slurm['partition']}")
    if slurm.get("gres"):
        lines.append(f"#SBATCH --gres={slurm['gres']}")
    lines.extend(str(line) for line in setup)
    lines.append(
        f"pode signatures worker --config {shlex.quote(str(config_path))} "
        f"--task-manifest {shlex.quote(str(tasks))} "
        "--task-index ${SLURM_ARRAY_TASK_ID}"
    )
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    aggregate_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={cfg['analysis_id']}-aggregate",
        f"#SBATCH --cpus-per-task={slurm.get('cpus', 4)}",
        f"#SBATCH --mem={slurm.get('memory', '16G')}",
        f"#SBATCH --time={slurm.get('time', '24:00:00')}",
    ]
    aggregate_lines.extend(str(line) for line in setup)
    aggregate_lines.append(
        f"pode signatures aggregate --config {shlex.quote(str(config_path))} "
        f"--run-id {shlex.quote(str(run_id))} --task-manifest {shlex.quote(str(tasks))}"
    )
    aggregate_script.write_text("\n".join(aggregate_lines) + "\n", encoding="utf-8")
    result = {"submitted": False, "run_id": run_id, "task_manifest": str(tasks),
              "scripts": [str(script), str(aggregate_script)], "tasks": rows}
    if not dry_run:
        job = subprocess.check_output(["sbatch", str(script)], text=True).strip().split()[-1]
        aggregate_job = subprocess.check_output(
            ["sbatch", f"--dependency=afterok:{job}", str(aggregate_script)], text=True
        ).strip().split()[-1]
        result.update({"submitted": True, "job_id": job, "aggregate_job_id": aggregate_job})
    (out / f"submission-{run_id}.json").write_text(json.dumps(result, indent=2) + "\n")
    _announce(progress, f"SLURM submission prepared with {len(rows)} task(s): {out}")
    return result


def run_advanced_worker(config: str | Path, task_manifest: str | Path, task_index: int,
                        *, progress: ProgressCallback | None = None) -> dict[str, Any]:
    import pandas as pd
    from .pipeline import run_advanced_pipeline

    tasks = pd.read_csv(task_manifest)
    match = tasks[tasks["task_index"] == int(task_index)]
    if len(match) != 1:
        raise ValueError(f"task index {task_index} is not present exactly once")
    row = match.iloc[0]
    _announce(progress, f"Running task {task_index}: {row['method']} / {row['endpoint']} fold {int(row['fold']) + 1}")
    cfg = read_advanced_config(config)
    cache = Path(task_manifest).parent / "cache" / str(row["run_id"]) / str(row["fingerprint"])
    cfg["_advanced_task_cache"] = str(cache)
    result = run_advanced_pipeline(cfg, str(row["method"]), write_results=False,
                                   task_filter=(str(row["endpoint"]), int(row["fold"])),
                                   finalize=False, progress=progress)
    _announce(progress, f"Task {task_index} complete")
    return {"task_index": int(task_index), "method": row["method"],
            "endpoint": row["endpoint"], "fold": int(row["fold"]),
            "stage": row["stage"], "cache_dir": str(cache)}


def aggregate_advanced_tasks(config: str | Path, task_manifest: str | Path, run_id: str,
                             *, progress: ProgressCallback | None = None) -> dict[str, Any]:
    import pandas as pd
    from .pipeline import run_advanced_pipeline

    cfg = read_advanced_config(config)
    tasks = pd.read_csv(task_manifest)
    tasks = tasks[tasks["run_id"].astype(str) == str(run_id)]
    if tasks.empty:
        raise ValueError(f"task manifest has no rows for run ID {run_id}")
    fingerprints = tasks["fingerprint"].astype(str).unique()
    if len(fingerprints) != 1:
        raise ValueError("task manifest contains multiple configuration fingerprints")
    cache = Path(task_manifest).parent / "cache" / str(run_id) / fingerprints[0]
    missing = []
    for row in tasks.itertuples(index=False):
        path = cache / str(row.method) / str(row.endpoint) / f"fold-{int(row.fold)}.json"
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise ValueError(f"cannot aggregate: {len(missing)} fold task results are missing")
    cfg["_advanced_task_cache"] = str(cache)
    outputs = {}
    for method in tasks["method"].drop_duplicates():
        _announce(progress, f"Aggregating cached tasks for {method}")
        result = run_advanced_pipeline(cfg, str(method), progress=progress)
        outputs[str(method)] = result.get("output_dir")
    _announce(progress, "Aggregation complete")
    return {"run_id": str(run_id), "methods": outputs, "cache_dir": str(cache)}
