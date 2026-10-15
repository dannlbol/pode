"""Thin wrapper dispatching CNV feature preparation to the R/Bioconductor script.

CNV calling (conumee2) has no Python equivalent, so this just translates
``pode_mantis`` config into ``Rscript R/scripts/prepare_mantis_cnv.R --flag
value ...`` and surfaces its output paths. See that script's own header for
the full algorithm and, critically, why ``cnv.control_mset`` and
``cnv.noise_threshold`` have no default and must be set deliberately.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PrepareCnvResult:
    """Outcome of a ``run_prepare_cnv`` call, one entry per cohort in
    ``cnv.msets``. ``qc_path`` is worth checking before trusting anything
    else here -- it has the per-sample ``conumee_noise`` column you need to
    derive ``cnv.noise_threshold`` from (see docs/mantis/README.md), plus the
    unmapped-gene and noise-exclusion counts."""

    out_dir: Path
    numeric_paths: dict[str, Path]
    sample_order_paths: dict[str, Path]
    qc_path: Path


def _repo_root() -> Path:
    # pode_mantis/core/prepare_cnv.py -> pode_mantis/core -> pode_mantis -> repo root
    return Path(__file__).resolve().parents[2]


def run_prepare_cnv(cfg, out_dir=None, n_test=None) -> PrepareCnvResult:
    """Validate config, then dispatch to R/scripts/prepare_mantis_cnv.R.

    ``n_test``, if given, subsets every Mset to its first N samples for a
    cheap dry run before committing to a full cohort (passed straight
    through to the R script's own ``--n-test``).
    """
    cnv = cfg["cnv"]
    if not cnv.get("msets"):
        raise ValueError("prepare-cnv: cnv.msets is required (a name -> RDS-path mapping).")
    if not cnv.get("control_mset"):
        raise ValueError(
            "prepare-cnv: cnv.control_mset is required, no default -- pick a normal-tissue "
            "reference appropriate for your tumour type/location (see docs/mantis/README.md)."
        )
    if not cnv.get("gene_order_path"):
        raise ValueError("prepare-cnv: cnv.gene_order_path is required (from the Zenodo release).")
    if cnv.get("noise_threshold") is None:
        raise ValueError(
            "prepare-cnv: cnv.noise_threshold is required, no default -- derive it from your own "
            "cohort's noise distribution (see docs/mantis/README.md and the R script's own header)."
        )

    rscript = shutil.which("Rscript")
    if not rscript:
        raise RuntimeError(
            "prepare-cnv: 'Rscript' not found on PATH. Install R plus Bioconductor's minfi/conumee2 "
            "(see docs/mantis/README.md) -- CNV calling has no Python equivalent."
        )

    out = Path(out_dir or cnv["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    msets_spec = ",".join(f"{name}={path}" for name, path in cnv["msets"].items())
    r_script = _repo_root() / "R" / "scripts" / "prepare_mantis_cnv.R"

    cmd = [
        rscript, str(r_script),
        "--msets", msets_spec,
        "--control-mset", str(cnv["control_mset"]),
        "--gene-order", str(cnv["gene_order_path"]),
        "--noise-threshold", str(cnv["noise_threshold"]),
        "--array-type", cnv.get("array_type", "450k"),
        "--out-dir", str(out),
    ]
    if cnv.get("phenotype_path"):
        cmd += ["--phenotype", str(cnv["phenotype_path"])]
    if n_test:
        cmd += ["--n-test", str(n_test)]

    print(f"[subprocess] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"prepare-cnv: R script failed (exit {result.returncode}). stderr:\n{result.stderr}")
    if result.stderr:
        print(f"[stderr]\n{result.stderr}")

    numeric_paths = {name: out / f"cnv_{name}_numeric.csv" for name in cnv["msets"]}
    sample_order_paths = {name: out / f"cnv_{name}_sample_order.txt" for name in cnv["msets"]}
    return PrepareCnvResult(
        out_dir=out, numeric_paths=numeric_paths, sample_order_paths=sample_order_paths, qc_path=out / "cnv_qc.csv"
    )
