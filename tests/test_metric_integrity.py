import json
from pathlib import Path

import torch

import run_experiment


def test_runner_passes_evaluator_metrics_to_completion_unchanged(monkeypatch, tmp_path):
    manifest = {
        "benchmark_id": "fixture",
        "seed": 7,
        "scenario": "warm_pair",
        "manifest_hash": "fixture_hash",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    # The filename intentionally exercises the old model-name override branch.
    config_path = tmp_path / "advanced.yaml"
    config_path.write_text("model_type: prevalence\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_experiment, "verify_manifest", lambda path: manifest)

    class TinyDataset:
        labels = [{"index": 0, "cui": "C001"}, {"index": 1, "cui": "C002"}]

        def __init__(self, split):
            self.records = [{"labels": [1.0, 0.0]}] if split == "train" else [{"labels": [0.0, 0.0]}]

        @classmethod
        def from_manifest(cls, _manifest_path, _manifest, *, split):
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
    expected_metrics = {
        "macro_ap": 0.123,
        "macro_auroc": 0.234,
        "micro_ap": 0.345,
        "precision_at_5": 0.456,
        "custom_metric": 17,
    }
    monkeypatch.setattr(
        run_experiment,
        "compute_all_metrics",
        lambda *_args, **_kwargs: dict(expected_metrics),
    )
    monkeypatch.setattr(
        run_experiment,
        "calibrate_validation_logits",
        lambda predictions, targets: (predictions, {"specific": 1.0}),
    )
    monkeypatch.setattr(run_experiment, "select_thresholds", lambda *_args, **_kwargs: {})

    run_experiment.run_single_experiment(str(config_path), str(manifest_path))

    run_dir = tmp_path / "artifacts" / "runs" / "prevalence__warm_pair__seed_7__fixture_hash"
    assert json.loads((run_dir / "metrics.json").read_text()) == expected_metrics


def test_runner_has_no_manual_metric_injection_or_dummy_hash_marker():
    source = Path("run_experiment.py").read_text()
    assert "dummy_hash_123" not in source
    assert "model_name == \"advanced\"" not in source
    assert '"macro_ap": 0.82' not in source
    assert '"macro_auroc": 0.93' not in source
    assert '"micro_ap": 0.88' not in source
    assert '"precision_at_5": 0.45' not in source


def test_synthetic_evidence_files_are_not_committed():
    synthetic_paths = [
        Path("dummy_manifest.json"),
        Path("artifacts/metrics_chart.png"),
        Path("artifacts/architecture_diagram.png"),
        Path("Polypharmacy_AI_Comprehensive_Report.pdf"),
    ]
    synthetic_paths.extend(Path("artifacts/runs").glob("*__dummy_hash_123"))
    assert not [path for path in synthetic_paths if path.exists()]
