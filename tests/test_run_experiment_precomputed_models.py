import json

import pytest

import run_experiment
from src.models.factory import UnsupportedModelConfiguration


def test_precomputed_model_fails_before_creating_run_artifacts(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "benchmark_id": "fixture",
                "seed": 1,
                "scenario": "warm_pair",
                "manifest_hash": "fixture_hash",
            }
        )
    )
    config_path = tmp_path / "student.yaml"
    config_path.write_text("model_type: distilled_pair_student\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_experiment, "verify_manifest", lambda path: json.loads(manifest_path.read_text()))
    with pytest.raises(UnsupportedModelConfiguration, match="precomputed|ManifestPolypharmacyDataset|cached"):
        run_experiment.run_single_experiment(str(config_path), str(manifest_path))
    assert not (tmp_path / "artifacts" / "runs").exists()
