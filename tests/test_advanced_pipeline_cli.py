import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "scripts/extract_molformer_tokens.py",
        "scripts/train_molecular_mpnn.py",
        "scripts/export_molecular_mpnn_tokens.py",
        "scripts/train_primekg_hgt.py",
        "build_advanced_features.py",
        "configure_advanced_experiment.py",
        "build_teacher_cache.py",
        "run_advanced_pipeline.py",
    ],
)
def test_advanced_pipeline_entry_points_have_runnable_help(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
