import json

import numpy as np
import pandas as pd
import torch

import run_experiment


def test_runner_calibrates_raw_validation_logits_and_bounds_test_probabilities(
    monkeypatch, tmp_path
):
    manifest = {
        "benchmark_id": "fixture",
        "seed": 7,
        "scenario": "warm_pair",
        "manifest_hash": "fixture_hash",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config_path = tmp_path / "advanced.yaml"
    config_path.write_text("model_type: prevalence\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_experiment, "verify_manifest", lambda path: manifest)

    class TinyDataset:
        labels = [{"index": 0, "cui": "C001"}]

        def __init__(self, split):
            labels = [1.0] if split == "train" else [0.0]
            self.records = [{"labels": labels}]

        @classmethod
        def from_manifest(cls, _manifest_path, _manifest, *, split):
            return cls(split)

        def __len__(self):
            return 1

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
    captured = {}

    def calibrate(validation_logits, validation_labels):
        captured["logits"] = validation_logits["specific"].copy()
        captured["labels"] = validation_labels["specific"].copy()
        raw = validation_logits["specific"]
        probabilities = 1.0 / (1.0 + np.exp(-raw.to_numpy() / 2.0))
        return {"specific": pd.DataFrame(probabilities, columns=raw.columns)}, {"specific": 2.0}

    monkeypatch.setattr(run_experiment, "calibrate_validation_logits", calibrate)
    monkeypatch.setattr(run_experiment, "select_thresholds", lambda *_args, **_kwargs: {"0": {"threshold": 0.5}})
    metric_inputs = {}

    def metrics(y_true, y_prob, thresholds, train_prevalences):
        metric_inputs["probabilities"] = y_prob.copy()
        return {"macro_ap": 0.5}

    monkeypatch.setattr(run_experiment, "compute_all_metrics", metrics)

    run_experiment.run_single_experiment(str(config_path), str(manifest_path))

    raw_val = captured["logits"].to_numpy()
    assert np.any(np.abs(raw_val) > 0.0)
    assert np.array_equal(captured["labels"].to_numpy(), np.array([[0.0]]))
    test_probabilities = metric_inputs["probabilities"].to_numpy()
    assert np.all((test_probabilities >= 0.0) & (test_probabilities <= 1.0))

    run_dir = tmp_path / "artifacts" / "runs" / "prevalence__warm_pair__seed_7__fixture_hash"
    assert json.loads((run_dir / "calibration.json").read_text()) == {
        "temperatures": {"specific": 2.0},
        "input": "validation_raw_logits",
    }
    assert json.loads((run_dir / "thresholds.json").read_text()) == {
        "0": {"threshold": 0.5}
    }
