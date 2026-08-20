import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.features.cached_token_artifact import CachedTokenArtifact
from src.models.multimodal_teacher_student import DistilledPairStudent
from scripts.benchmark_pair_inference import (
    MIN_ITERATIONS,
    MIN_WARMUPS,
    benchmark_completed_pair,
    build_parser,
    load_benchmark_run,
    verify_prediction_parity,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "distilled_pair_student__warm_pair__seed_42__fixture"
    run_dir.mkdir()
    cache_path = tmp_path / "cache.npz"
    CachedTokenArtifact.write(
        cache_path,
        ["drug_a", "drug_b"],
        np.random.default_rng(4).normal(size=(2, 3, 4)).astype(np.float32),
        np.ones((2, 3), dtype=bool),
        teacher_provenance_hash=HASH_A,
        modality_provenance_hash=HASH_B,
        teacher_selection_hash=HASH_C,
        teacher_selected_mode="baseline",
    )
    torch.manual_seed(3)
    model = DistilledPairStudent(4, 3, 8, 1, 3, num_heads=2).eval()
    checkpoint_path = run_dir / "checkpoint_best.pt"
    torch.save(model.state_dict(), checkpoint_path)
    calibration_path = run_dir / "calibration.json"
    calibration_path.write_text(json.dumps({"temperatures": {"specific": 1.0}}))
    thresholds_path = run_dir / "thresholds.json"
    thresholds_path.write_text(
        json.dumps({label: {"threshold": 0.5} for label in ("s0", "s1", "s2")})
    )
    labels_path = run_dir / "per_label_metrics.csv"
    labels_path.write_text("label_cui\ns0\ns1\ns2\n")
    config = {
        "model_type": "distilled_pair_student",
        "run_model_id": "distilled_pair_student",
        "run_id": run_dir.name,
        "token_dim": 4,
        "token_count": 3,
        "hidden_dim": 8,
        "num_heads": 2,
        "num_organ": 1,
        "num_specific": 3,
        "dropout": 0.0,
        "label_names": ["s0", "s1", "s2"],
        "organ_label_names": ["o0"],
        "cached_token_artifact_path": str(cache_path),
        "cached_token_artifact_sha256": _sha256(cache_path),
    }
    config_path = run_dir / "config.resolved.json"
    config_path.write_text(json.dumps(config, indent=2))
    artifacts = {}
    for path in (config_path, checkpoint_path, calibration_path, thresholds_path, labels_path):
        artifacts[path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    (run_dir / "completion.json").write_text(
        json.dumps({"status": "complete", "required_artifacts": artifacts}, indent=2)
    )
    return run_dir


def test_benchmark_helper_supports_small_counts_and_json_native_finite_stats(tmp_path):
    run_dir = _write_run(tmp_path)
    result = benchmark_completed_pair(
        run_dir,
        "drug_a",
        "drug_b",
        warmups=1,
        iterations=3,
        threads=1,
    )
    json.dumps(result, allow_nan=False)
    timing = result["timing_ms"]
    assert result["warmups"] == 1
    assert result["iterations"] == 3
    assert all(isinstance(timing[key], float) and math.isfinite(timing[key]) for key in ("median", "p95", "p99"))
    assert result["run_model_id"] == "distilled_pair_student"
    assert result["pair"] == {"drug_a": "drug_a", "drug_b": "drug_b"}


def test_benchmark_output_and_sha_sidecar_are_json_verifiable(tmp_path):
    output = tmp_path / "benchmark.json"
    benchmark_completed_pair(
        _write_run(tmp_path),
        "drug_a",
        "drug_b",
        warmups=1,
        iterations=1,
        threads=1,
        output_path=output,
    )
    assert json.loads(output.read_text())["iterations"] == 1
    assert output.with_name("benchmark.json.sha256").read_text().strip() == _sha256(output)


def test_production_parser_enforces_safe_minimums():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "run", "--drug-a", "a", "--drug-b", "b", "--output", "out", "--warmups", str(MIN_WARMUPS - 1)])
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "run", "--drug-a", "a", "--drug-b", "b", "--output", "out", "--iterations", str(MIN_ITERATIONS - 1)])


@pytest.mark.parametrize("tamper", ["completion", "checkpoint", "cache"])
def test_benchmark_rejects_tampered_provenance(tmp_path, tamper):
    run_dir = _write_run(tmp_path)
    if tamper == "completion":
        (run_dir / "completion.json").write_text("{}")
    elif tamper == "checkpoint":
        with (run_dir / "checkpoint_best.pt").open("ab") as stream:
            stream.write(b"tampered")
    else:
        cache_path = Path(json.loads((run_dir / "config.resolved.json").read_text())["cached_token_artifact_path"])
        with cache_path.open("ab") as stream:
            stream.write(b"tampered")
    with pytest.raises(ValueError):
        load_benchmark_run(run_dir)


def test_benchmark_rejects_unknown_pair(tmp_path):
    with pytest.raises(KeyError, match="unknown"):
        benchmark_completed_pair(_write_run(tmp_path), "drug_a", "missing", warmups=1, iterations=1, threads=1)


def test_benchmark_rejects_non_student_run(tmp_path):
    run_dir = _write_run(tmp_path)
    config_path = run_dir / "config.resolved.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "symmetric_mlp"
    config_path.write_text(json.dumps(config))
    marker = json.loads((run_dir / "completion.json").read_text())
    marker["required_artifacts"]["config.resolved.json"] = {
        "sha256": _sha256(config_path),
        "bytes": config_path.stat().st_size,
    }
    (run_dir / "completion.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="distilled_pair_student"):
        load_benchmark_run(run_dir)


def test_benchmark_rejects_parseable_tampered_label_csv(tmp_path):
    run_dir = _write_run(tmp_path)
    config_path = run_dir / "config.resolved.json"
    config = json.loads(config_path.read_text())
    config.pop("label_names")
    config_path.write_text(json.dumps(config))
    marker = json.loads((run_dir / "completion.json").read_text())
    marker["required_artifacts"]["config.resolved.json"] = {
        "sha256": _sha256(config_path),
        "bytes": config_path.stat().st_size,
    }
    (run_dir / "completion.json").write_text(json.dumps(marker))
    (run_dir / "per_label_metrics.csv").write_text("label_cui\ns0\nchanged\ns2\n")
    with pytest.raises(ValueError, match="per_label_metrics.csv"):
        load_benchmark_run(run_dir)


def test_specific_only_calibration_uses_default_organ_labels(tmp_path):
    run_dir = _write_run(tmp_path)
    config_path = run_dir / "config.resolved.json"
    config = json.loads(config_path.read_text())
    config.pop("organ_label_names")
    config_path.write_text(json.dumps(config))
    marker = json.loads((run_dir / "completion.json").read_text())
    marker["required_artifacts"]["config.resolved.json"] = {
        "sha256": _sha256(config_path),
        "bytes": config_path.stat().st_size,
    }
    (run_dir / "completion.json").write_text(json.dumps(marker))
    calibration = json.loads((run_dir / "calibration.json").read_text())
    thresholds = json.loads((run_dir / "thresholds.json").read_text())
    assert set(calibration["temperatures"]) == {"specific"}
    assert set(thresholds) == {"s0", "s1", "s2"}
    result = benchmark_completed_pair(run_dir, "drug_a", "drug_b", warmups=1, iterations=1, threads=1)
    assert result["pair"] == {"drug_a": "drug_a", "drug_b": "drug_b"}


def test_prediction_parity_failure_is_explicit():
    with pytest.raises(ValueError, match="parity"):
        verify_prediction_parity({"specific_logits": [0.0]}, {"specific_logits": [1.0]})


def test_benchmark_source_has_no_training_only_graph_or_transformer_runtime():
    source = Path("scripts/benchmark_pair_inference.py").read_text()
    assert "transformers" not in source
    assert "torch_geometric" not in source
