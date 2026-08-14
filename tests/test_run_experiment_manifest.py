import json
import sys
from pathlib import Path

import pytest

import run_experiment


def test_run_single_experiment_rejects_missing_manifest_before_training(
    monkeypatch, tmp_path
):
    """A missing manifest must fail before a run directory or trainer exists."""

    monkeypatch.chdir(tmp_path)

    def fail_if_training_starts(*_args, **_kwargs):
        raise AssertionError("training started before manifest validation")

    monkeypatch.setattr(run_experiment, "StateGuardedTrainer", fail_if_training_starts)

    with pytest.raises((TypeError, ValueError), match="manifest"):
        run_experiment.run_single_experiment("configs/experiment.yaml", None)

    assert not (tmp_path / "artifacts").exists()


def test_run_single_experiment_verifies_manifest_before_creating_run(
    monkeypatch, tmp_path
):
    """Manifest verification must happen before filesystem writes or training."""

    monkeypatch.chdir(tmp_path)
    calls = []

    def reject_manifest(path):
        calls.append(path)
        raise ValueError("invalid manifest")

    monkeypatch.setattr(run_experiment, "verify_manifest", reject_manifest, raising=False)

    def fail_if_training_starts(*_args, **_kwargs):
        raise AssertionError("training started before manifest validation")

    monkeypatch.setattr(run_experiment, "StateGuardedTrainer", fail_if_training_starts)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")

    with pytest.raises(ValueError, match="invalid manifest"):
        run_experiment.run_single_experiment("configs/experiment.yaml", str(manifest_path))

    assert calls == [str(manifest_path)]
    assert not (tmp_path / "artifacts").exists()


def test_run_single_experiment_uses_verified_manifest_identity(monkeypatch, tmp_path):
    """Run IDs and resolved config must use the verified manifest fields."""

    monkeypatch.chdir(tmp_path)
    manifest = {
        "benchmark_id": "benchmark_from_manifest",
        "seed": 17,
        "scenario": "cold_1",
        "manifest_hash": "abc123def456",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    experiment_path = tmp_path / "advanced.yaml"
    experiment_path.write_text("model_type: symmetric_mlp\n")
    monkeypatch.setattr(run_experiment, "verify_manifest", lambda path: manifest)

    class StopBeforeTraining:
        def __init__(self, run_dir):
            raise RuntimeError(f"stop before training: {run_dir}")

    monkeypatch.setattr(run_experiment, "StateGuardedTrainer", StopBeforeTraining)

    with pytest.raises(RuntimeError, match="stop before training") as exc_info:
        run_experiment.run_single_experiment(str(experiment_path), str(manifest_path))

    expected_run_id = "symmetric_mlp__cold_1__seed_17__abc123def456"
    assert expected_run_id in str(exc_info.value)
    resolved = tmp_path / "artifacts" / "runs" / expected_run_id / "config.resolved.json"
    assert json.loads(resolved.read_text())["benchmark_id"] == "benchmark_from_manifest"


def test_cli_requires_benchmark_when_not_running_all(monkeypatch):
    """Single-run CLI mode must reject an omitted --benchmark argument."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiment.py", "--experiment", "configs/advanced.yaml"],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_experiment.main()

    assert exc_info.value.code == 2


def test_all_benchmarks_preflights_and_runs_sorted_once(monkeypatch, tmp_path):
    import run_experiment

    root = tmp_path / "benchmarks"
    manifests = []
    for name, scenario, seed in [("b", "cold_1", 2), ("a", "warm_pair", 1)]:
        path = root / name / "manifest.json"
        path.parent.mkdir(parents=True)
        payload = {"benchmark_id": "fixture", "scenario": scenario, "seed": seed, "manifest_hash": "hash" + str(seed), "pairs_path": "pairs.parquet", "triples_path": "triples.parquet", "labels_path": "labels.json"}
        for artifact in ("pairs.parquet", "triples.parquet", "labels.json"):
            (path.parent / artifact).write_bytes(b"fixture")
        path.write_text(json.dumps(payload))
        manifests.append(path)
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("model_type: prevalence\n")
    monkeypatch.setattr(run_experiment, "verify_manifest", lambda path: json.loads(Path(path).read_text()))
    called = []
    monkeypatch.setattr(run_experiment, "run_single_experiment", lambda exp, manifest: called.append((exp, manifest)))
    monkeypatch.setattr(sys, "argv", [
        "run_experiment.py", "--experiment", str(experiment), "--all-benchmarks",
        "--benchmarks-root", str(root),
    ])
    run_experiment.main()
    assert [Path(item[1]).name for item in called] == ["manifest.json", "manifest.json"]
    assert [Path(item[1]).parent.name for item in called] == ["a", "b"]


def test_all_benchmarks_invalid_manifest_fails_before_any_run(monkeypatch, tmp_path):
    import run_experiment

    root = tmp_path / "benchmarks"
    valid = root / "a" / "manifest.json"
    valid.parent.mkdir(parents=True)
    valid.write_text(json.dumps({"benchmark_id": "fixture", "scenario": "warm_pair", "seed": 1, "manifest_hash": "h1", "pairs_path": "pairs.parquet", "triples_path": "triples.parquet", "labels_path": "labels.json"}))
    for artifact in ("pairs.parquet", "triples.parquet", "labels.json"):
        (valid.parent / artifact).write_bytes(b"fixture")
    invalid = root / "b" / "manifest.json"
    invalid.parent.mkdir()
    invalid.write_text("not-json")
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("model_type: prevalence\n")
    monkeypatch.setattr(run_experiment, "verify_manifest", lambda path: json.loads(Path(path).read_text()))
    called = []
    monkeypatch.setattr(run_experiment, "run_single_experiment", lambda *_args: called.append(True))
    monkeypatch.setattr(sys, "argv", [
        "run_experiment.py", "--experiment", str(experiment), "--all-benchmarks",
        "--benchmarks-root", str(root),
    ])
    with pytest.raises(ValueError, match="invalid benchmark manifest"):
        run_experiment.main()
    assert called == []
