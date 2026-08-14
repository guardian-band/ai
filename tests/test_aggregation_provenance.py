import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from aggregate_results import _verify_run, aggregate_runs
from src.evaluation.metrics import compute_all_metrics


METRIC_KEYS = ["micro_ap", "micro_auroc", "brier_score", "ece_15", "precision_at_k", "recall_at_k", "ndcg_at_k", "macro_ap", "macro_auroc"]


def _sha256(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _write_fixture_run(
    runs_dir: Path,
    *,
    model_type: str,
    scenario: str,
    seed: int,
    benchmark_id: str = "fixture_benchmark",
    completion_status: str = "complete",
    tamper_prediction: bool = False,
    metric_mismatch: bool = False,
    run_suffix: str = "",
    all_negative: bool = False,
) -> Path:
    run_id = f"{model_type}__{scenario}__seed_{seed}__manifest123{run_suffix}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pair_ids = [f"{scenario}_{seed}_p0", f"{scenario}_{seed}_p1", f"{scenario}_{seed}_p2", f"{scenario}_{seed}_p3"]
    labels = ["C001", "C002"]
    truth = pd.DataFrame({"pair_id": pair_ids, "truth__C001": [1, 0, 1, 0], "truth__C002": [0, 1, 0, 1]})
    if all_negative:
        truth[["truth__C001", "truth__C002"]] = 0
    probs = pd.DataFrame({"pair_id": pair_ids, "prob__C001": [0.9, 0.2, 0.8, 0.1], "prob__C002": [0.1, 0.8, 0.2, 0.9]})
    logits = pd.DataFrame({"pair_id": pair_ids, "logit__C001": [2.0, -1.0, 1.4, -2.0], "logit__C002": [-2.0, 1.4, -1.4, 2.0]})
    if tamper_prediction:
        probs.loc[0, "prob__C001"] = 0.05
    truth.to_parquet(run_dir / "test_predictions.parquet", index=False)
    probs.to_parquet(run_dir / "test_predictions.parquet", index=False)
    # Combine truth and probabilities in the contract's prediction table.
    test_predictions = truth.merge(probs, on="pair_id")
    test_predictions.to_parquet(run_dir / "test_predictions.parquet", index=False)
    validation_predictions = test_predictions.copy()
    validation_predictions.to_parquet(run_dir / "validation_predictions.parquet", index=False)
    logits.to_parquet(run_dir / "test_logits.parquet", index=False)
    logits.to_parquet(run_dir / "validation_logits.parquet", index=False)
    thresholds = {label: {"threshold": 0.5} for label in labels}
    (run_dir / "thresholds.json").write_text(json.dumps(thresholds))
    (run_dir / "calibration.json").write_text(json.dumps({"temperatures": {"specific": 1.0}}))
    (run_dir / "environment.json").write_text(json.dumps({"python": "fixture"}))
    (run_dir / "checkpoint_best.pt").write_bytes(b"fixture checkpoint")
    config = {
        "run_id": run_id,
        "model_type": model_type,
        "benchmark_id": benchmark_id,
        "scenario": scenario,
        "seed": seed,
        "manifest_hash": "manifest123",
        "train_prevalences": {"C001": 0.5, "C002": 0.5},
    }
    (run_dir / "config.resolved.json").write_text(json.dumps(config))
    metrics = compute_all_metrics(
        truth.drop(columns="pair_id").rename(columns=lambda c: c.removeprefix("truth__")),
        probs.drop(columns="pair_id").rename(columns=lambda c: c.removeprefix("prob__")),
        thresholds,
        config["train_prevalences"],
    )
    if metric_mismatch:
        metrics["macro_ap"] = float(metrics["macro_ap"]) + 0.1
    (run_dir / "metrics.json").write_text(json.dumps(metrics, allow_nan=True))
    per_label = pd.DataFrame([{"label_cui": label} for label in labels])
    per_label.to_csv(run_dir / "per_label_metrics.csv", index=False)

    required = [
        "config.resolved.json", "environment.json", "checkpoint_best.pt",
        "validation_logits.parquet", "validation_predictions.parquet",
        "test_logits.parquet", "test_predictions.parquet", "metrics.json",
        "per_label_metrics.csv", "thresholds.json", "calibration.json",
    ]
    marker = {"status": completion_status, "required_artifacts": {name: _sha256(run_dir / name) for name in required}}
    (run_dir / "completion.json").write_text(json.dumps(marker))
    return run_dir


@pytest.fixture
def benchmark_fixture(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump({"benchmark_name": "fixture_benchmark", "seeds": [1, 2], "partitions": {"warm_pair": {}, "cold_1": {}}, "bootstrap": {"resamples": 100, "seed": 42, "confidence_level": 0.95}}))
    runs_dir = tmp_path / "runs"
    for model_type in ["logistic", "symmetric_mlp"]:
        for scenario in ["warm_pair", "cold_1"]:
            for seed in [1, 2]:
                _write_fixture_run(runs_dir, model_type=model_type, scenario=scenario, seed=seed)
    return runs_dir, config_path


def test_genuine_cohorts_aggregate_and_champion_is_deterministic(benchmark_fixture, tmp_path):
    runs_dir, config_path = benchmark_fixture
    output = tmp_path / "summary.json"
    result = aggregate_runs(str(runs_dir), str(output), str(config_path))
    assert output.exists()
    assert output.with_suffix(".csv").exists()
    assert len(result["cohorts"]) == 4
    assert result["champion"]["model_type"] == "logistic"
    assert result["champion_rule"].startswith("highest mean macro_ap")


def test_missing_seed_is_rejected(benchmark_fixture, tmp_path):
    runs_dir, config_path = benchmark_fixture
    removed = next(runs_dir.glob("logistic__cold_1__seed_2__*" ))
    removed.rename(tmp_path / "removed_run")
    with pytest.raises(ValueError, match=r"missing seeds \[2\]"):
        aggregate_runs(str(runs_dir), str(tmp_path / "summary.json"), str(config_path))


def test_duplicate_run_is_rejected(benchmark_fixture, tmp_path):
    runs_dir, config_path = benchmark_fixture
    source = next(runs_dir.glob("logistic__cold_1__seed_2__*"))
    duplicate = runs_dir / (source.name + "_duplicate")
    duplicate.mkdir()
    for path in source.iterdir():
        if path.is_file():
            duplicate.joinpath(path.name).write_bytes(path.read_bytes())
    config = json.loads((duplicate / "config.resolved.json").read_text())
    config["run_id"] = duplicate.name
    (duplicate / "config.resolved.json").write_text(json.dumps(config))
    marker = json.loads((duplicate / "completion.json").read_text())
    marker["required_artifacts"]["config.resolved.json"] = _sha256(duplicate / "config.resolved.json")
    (duplicate / "completion.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match=r"duplicate run .* seed 2"):
        aggregate_runs(str(runs_dir), str(tmp_path / "summary.json"), str(config_path))


def test_run_identity_requires_model_type_prefix(benchmark_fixture, tmp_path):
    runs_dir, config_path = benchmark_fixture
    run = next(runs_dir.glob("logistic__warm_pair__seed_1__*"))
    config = json.loads((run / "config.resolved.json").read_text())
    config["model_type"] = "wrong_model"
    (run / "config.resolved.json").write_text(json.dumps(config))
    marker = json.loads((run / "completion.json").read_text())
    marker["required_artifacts"]["config.resolved.json"] = _sha256(run / "config.resolved.json")
    (run / "completion.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="model_type prefix"):
        _verify_run(run)


def test_status_tamper_and_metric_mismatch_are_rejected(benchmark_fixture, tmp_path):
    runs_dir, config_path = benchmark_fixture
    run = next(runs_dir.glob("symmetric_mlp__warm_pair__seed_1__*"))
    marker = json.loads((run / "completion.json").read_text())
    marker["status"] = "failed"
    (run / "completion.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="status"):
        aggregate_runs(str(runs_dir), str(tmp_path / "status.json"), str(config_path))

    # Restore the marker before exercising the independent tamper check.
    marker["status"] = "complete"
    (run / "completion.json").write_text(json.dumps(marker))
    run = next(runs_dir.glob("symmetric_mlp__warm_pair__seed_1__*"))
    prediction_path = run / "test_predictions.parquet"
    original_prediction = prediction_path.read_bytes()
    prediction_path.write_bytes(original_prediction + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        aggregate_runs(str(runs_dir), str(tmp_path / "tamper.json"), str(config_path))
    prediction_path.write_bytes(original_prediction)

    run = next(runs_dir.glob("symmetric_mlp__warm_pair__seed_1__*"))
    metrics = json.loads((run / "metrics.json").read_text())
    metrics["macro_ap"] += 0.1
    (run / "metrics.json").write_text(json.dumps(metrics))
    marker = json.loads((run / "completion.json").read_text())
    marker["required_artifacts"]["metrics.json"] = _sha256(run / "metrics.json")
    (run / "completion.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="mismatch"):
        aggregate_runs(str(runs_dir), str(tmp_path / "metrics.json"), str(config_path))


def test_nonfinite_prediction_values_are_rejected(benchmark_fixture, tmp_path):
    runs_dir, config_path = benchmark_fixture
    run = next(runs_dir.glob("logistic__warm_pair__seed_1__*"))
    predictions = pd.read_parquet(run / "test_predictions.parquet")
    predictions.loc[0, "prob__C001"] = np.inf
    predictions.to_parquet(run / "test_predictions.parquet", index=False)
    marker = json.loads((run / "completion.json").read_text())
    marker["required_artifacts"]["test_predictions.parquet"] = _sha256(run / "test_predictions.parquet")
    (run / "completion.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="finite"):
        aggregate_runs(str(runs_dir), str(tmp_path / "summary.json"), str(config_path))


def test_json_null_is_accepted_for_nonfinite_recomputed_metrics(tmp_path):
    runs_dir = tmp_path / "runs"
    run = _write_fixture_run(
        runs_dir,
        model_type="logistic",
        scenario="warm_pair",
        seed=1,
        all_negative=True,
    )
    metrics = json.loads((run / "metrics.json").read_text())
    metrics = {key: (None if isinstance(value, float) and np.isnan(value) else value) for key, value in metrics.items()}
    (run / "metrics.json").write_text(json.dumps(metrics))
    marker = json.loads((run / "completion.json").read_text())
    marker["required_artifacts"]["metrics.json"] = _sha256(run / "metrics.json")
    (run / "completion.json").write_text(json.dumps(marker))
    assert _verify_run(run)["run_id"] == run.name


def test_cli_output_paths(benchmark_fixture, tmp_path, monkeypatch):
    runs_dir, config_path = benchmark_fixture
    output = tmp_path / "nested" / "result.json"
    import subprocess
    subprocess.run([".venv/bin/python", "aggregate_results.py", "--runs", str(runs_dir), "--output", str(output), "--benchmark-config", str(config_path)], check=True)
    assert output.exists()
    assert output.with_suffix(".csv").exists()
