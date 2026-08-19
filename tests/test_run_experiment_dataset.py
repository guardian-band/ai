import json

import torch
from torch import nn

import run_experiment


def test_run_loads_test_dataset_only_after_validation_freeze(monkeypatch, tmp_path):
    events = []
    manifest = {
        "benchmark_id": "fixture",
        "seed": 7,
        "scenario": "warm_pair",
        "manifest_hash": "fixture_hash",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text("model_type: symmetric_mlp\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_experiment, "verify_manifest", lambda path: manifest)

    class TinyDataset:
        labels = [{"index": index, "cui": f"C{index:03d}"} for index in range(100)]

        @classmethod
        def from_manifest(cls, _manifest_path, _manifest, *, split, drug_features_path):
            assert drug_features_path == "artifacts/morgan_fingerprints.parquet"
            events.append(f"load:{split}")
            if split == "test":
                assert events[-2] == "validation_frozen"
            return cls()

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return (
                torch.zeros(20, 768),
                torch.zeros(20, 768),
                torch.zeros(20, dtype=torch.bool),
                torch.zeros(20, dtype=torch.bool),
                torch.zeros(100),
                True,
            )

    monkeypatch.setattr(run_experiment, "ManifestPolypharmacyDataset", TinyDataset)

    class TinyModel(nn.Module):
        def __init__(self, num_labels=100):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(num_labels))

        def forward(self, drug_a, drug_b, mask_a, mask_b):
            return self.bias.unsqueeze(0).expand(drug_a.shape[0], -1)

    monkeypatch.setattr(run_experiment, "create_model", lambda *_args, **_kwargs: TinyModel())

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

    class RecordingTrainer:
        def __init__(self, _run_dir):
            self.state = "initialized"

        def begin_training(self):
            self.state = "training"

        def model_selected(self):
            self.state = "model_selected"

        def validation_frozen(self):
            self.state = "validation_frozen"
            events.append(self.state)

        def evaluate_test(self):
            assert self.state == "validation_frozen"
            self.state = "test_evaluated"
            events.append(self.state)

        def complete(self, _metrics):
            self.state = "complete"

    monkeypatch.setattr(run_experiment, "StateGuardedTrainer", RecordingTrainer)

    run_experiment.run_single_experiment(str(experiment_path), str(manifest_path))

    assert events == [
        "load:train",
        "load:validation",
        "validation_frozen",
        "load:test",
        "test_evaluated",
    ]
    run_dir = tmp_path / "artifacts" / "runs" / "symmetric_mlp__warm_pair__seed_7__fixture_hash"
    resolved = json.loads((run_dir / "config.resolved.json").read_text())
    assert resolved["model_type"] == "symmetric_mlp"
