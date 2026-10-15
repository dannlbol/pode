"""R-side smoke test, guarded like tests/signatures/test_core.py's own
Rscript-based cross-validation: skip entirely if R isn't on PATH, since CI
here has no Bioconductor/conumee2 installed. A full real run against
Msets + a control reference is a manual verification step (see
docs/mantis/README.md) -- this only checks the script parses its own
arguments and fails clearly on missing required options, which needs R but
not conumee2/minfi.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

RSCRIPT = shutil.which("Rscript")
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "R" / "scripts" / "prepare_mantis_cnv.R"

pytestmark = pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")


def test_help_exits_cleanly():
    result = subprocess.run([RSCRIPT, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--noise-threshold" in result.stdout
    assert "--control-mset" in result.stdout


def test_missing_control_mset_errors_clearly():
    result = subprocess.run(
        [RSCRIPT, str(SCRIPT), "--msets", "a=x.rds", "--gene-order", "g.txt", "--noise-threshold", "0.5"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--control-mset is required" in result.stderr


def test_missing_noise_threshold_errors_clearly():
    result = subprocess.run(
        [RSCRIPT, str(SCRIPT), "--msets", "a=x.rds", "--control-mset", "c.rds", "--gene-order", "g.txt"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--noise-threshold is required" in result.stderr


def test_missing_msets_errors_clearly():
    result = subprocess.run(
        [RSCRIPT, str(SCRIPT), "--control-mset", "c.rds", "--gene-order", "g.txt", "--noise-threshold", "0.5"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--msets is required" in result.stderr
