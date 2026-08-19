import json

import pytest
import torch

import run_experiment
from src.models.factory import UnsupportedModelConfiguration


def test_runner_rejects_unavailable_model_before_training(monkeypatch, tmp_path):
    manifest = {
        "benchmark_id": "fixture",
        "seed": 7,
        "scenario": "warm_pair",
        "manifest_hash": "fixture_hash",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config_path = tmp_path / "unified.yaml"
    config_path.write_text("model_type: unified\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_experiment, "verify_manifest", lambda path: manifest)

    class TinyDataset:
        labels = [{"index": 0, "cui": "C001"}]
        records = [{"labels": [1.0]}]

        @classmethod
        def from_manifest(cls, *_args, **_kwargs):
            return cls()

    monkeypatch.setattr(run_experiment, "ManifestPolypharmacyDataset", TinyDataset)
    started = []

    class RecordingTrainer:
        def __init__(self, _run_dir):
            pass

        def begin_training(self):
            started.append(True)

    monkeypatch.setattr(run_experiment, "StateGuardedTrainer", RecordingTrainer)

    with pytest.raises(UnsupportedModelConfiguration, match="unavailable"):
        run_experiment.run_single_experiment(str(config_path), str(manifest_path))

    assert started == []
    assert not list((tmp_path / "artifacts" / "runs").glob("*/completion.json"))


def test_runner_executes_prevalence_without_optimizer(monkeypatch, tmp_path):
    manifest = {
        "benchmark_id": "fixture",
        "seed": 7,
        "scenario": "warm_pair",
        "manifest_hash": "fixture_hash",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config_path = tmp_path / "prevalence.yaml"
    config_path.write_text("model_type: prevalence\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_experiment, "verify_manifest", lambda path: manifest)

    class TinyDataset:
        labels = [{"index": 0, "cui": "C001"}, {"index": 1, "cui": "C002"}]

        def __init__(self, split):
            self.records = [{"labels": [1.0, 0.0]}] if split == "train" else [{"labels": [0.0, 0.0]}]

        @classmethod
        def from_manifest(cls, _manifest_path, _manifest, *, split, drug_features_path):
            assert drug_features_path == "artifacts/morgan_fingerprints.parquet"
            return cls(split)

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return (
                torch.zeros(20, 768),
                torch.zeros(20, 768),
                torch.zeros(20, dtype=torch.bool),
                torch.zeros(20, dtype=torch.bool),
                torch.tensor(self.records[index]["labels"]),
                bool(self.records[index]["labels"][0]),
            )

    monkeypatch.setattr(run_experiment, "ManifestPolypharmacyDataset", TinyDataset)
    monkeypatch.setattr(
        run_experiment,
        "compute_all_metrics",
        lambda *_args, **_kwargs: {"macro_ap": 0.5},
    )
    monkeypatch.setattr(
        run_experiment,
        "calibrate_validation_logits",
        lambda predictions, targets: (predictions, {"specific": 1.0}),
    )
    monkeypatch.setattr(run_experiment, "select_thresholds", lambda *_args, **_kwargs: {})

    run_experiment.run_single_experiment(str(config_path), str(manifest_path))

    run_dir = tmp_path / "artifacts" / "runs" / "prevalence__warm_pair__seed_7__fixture_hash"
    assert json.loads((run_dir / "completion.json").read_text())["status"] == "complete"
