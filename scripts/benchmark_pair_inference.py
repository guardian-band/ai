"""Benchmark one cached-token Student pair through the CPU deployment path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.features.cached_token_artifact import CachedTokenArtifact
from src.inference.cpu_pair_predictor import CPUPairPredictor, verify_prediction_parity
from src.models.factory import create_model, derive_run_model_id


MIN_WARMUPS = 20
MIN_ITERATIONS = 1000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {name} artifact {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} artifact must contain a JSON object")
    return value


def _artifact_path(run_dir: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} path must be a non-empty string")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    return candidate


def _verify_declared_artifact(run_dir: Path, marker: Mapping[str, Any], name: str) -> Path:
    declared = marker.get("required_artifacts")
    if not isinstance(declared, Mapping) or name not in declared:
        raise ValueError(f"completion.json does not declare required artifact {name}")
    record = declared[name]
    if not isinstance(record, Mapping) or not isinstance(record.get("sha256"), str):
        raise ValueError(f"completion.json has invalid provenance for {name}")
    path = run_dir / name
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {name}")
    actual_hash = _sha256(path)
    actual_bytes = path.stat().st_size
    if record["sha256"] != actual_hash or record.get("bytes") != actual_bytes:
        raise ValueError(f"completion provenance mismatch for {name}")
    return path


def _label_names(
    run_dir: Path,
    config: Mapping[str, Any],
    labels_path: Path | None = None,
) -> list[str]:
    configured = config.get("label_names")
    if isinstance(configured, (list, tuple)) and configured:
        labels = [str(label).strip() for label in configured]
    else:
        path = labels_path or (run_dir / "per_label_metrics.csv")
        if not path.is_file():
            raise ValueError("Student run must provide label_names or per_label_metrics.csv")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        labels = [str(row.get("label_cui", "")).strip() for row in rows]
    if not labels or any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Student label names must be unique non-empty values")
    return labels


def load_benchmark_run(
    run_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a completed Student run and its cached-token artifact."""

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Student run directory does not exist: {run_dir}")
    completion_path = run_dir / "completion.json"
    marker = _read_json(completion_path, "completion")
    if marker.get("status") != "complete":
        raise ValueError("Student run completion status must be complete")
    resolved_config_path = (
        Path(config_path).resolve() if config_path is not None else run_dir / "config.resolved.json"
    )
    if resolved_config_path != run_dir / "config.resolved.json":
        raise ValueError("config source must be the completed run's config.resolved.json")
    config_artifact = _verify_declared_artifact(run_dir, marker, "config.resolved.json")
    checkpoint_artifact = _verify_declared_artifact(run_dir, marker, "checkpoint_best.pt")
    config = _read_json(config_artifact, "resolved config")
    try:
        derived_model_id = derive_run_model_id(config)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("unable to derive Student run_model_id") from exc
    if derived_model_id != "distilled_pair_student":
        raise ValueError("CPU pair benchmark requires run_model_id distilled_pair_student")
    if config.get("run_model_id") not in (None, derived_model_id):
        raise ValueError("config run_model_id does not match the declared Student model")

    configured_cache = config.get("cached_token_artifact_path")
    cache = Path(cache_path).resolve() if cache_path is not None else _artifact_path(run_dir, configured_cache, "cached token artifact")
    if not cache.is_file():
        raise ValueError(f"cached-token artifact is missing: {cache}")
    expected_cache_hash = config.get("cached_token_artifact_sha256")
    actual_cache_hash = _sha256(cache)
    if not isinstance(expected_cache_hash, str) or expected_cache_hash != actual_cache_hash:
        raise ValueError("cached-token artifact SHA-256 does not match resolved config")
    try:
        cached = CachedTokenArtifact.load(cache)
    except (OSError, ValueError) as exc:
        raise ValueError(f"unable to load validated cached-token artifact {cache}") from exc

    calibration_value = config.get("calibration_path", "calibration.json")
    thresholds_value = config.get("thresholds_path", "thresholds.json")
    calibration = _artifact_path(run_dir, calibration_value, "calibration")
    thresholds = _artifact_path(run_dir, thresholds_value, "thresholds")
    calibration_used = calibration.is_file() or thresholds.is_file()
    if calibration_used:
        if not calibration.is_file() or not thresholds.is_file():
            raise ValueError("calibration and thresholds artifacts must be supplied together")
        calibration_name = calibration.relative_to(run_dir).as_posix()
        thresholds_name = thresholds.relative_to(run_dir).as_posix()
        _verify_declared_artifact(run_dir, marker, calibration_name)
        _verify_declared_artifact(run_dir, marker, thresholds_name)

    labels_path = run_dir / "per_label_metrics.csv"
    labels_sha256: str | None = None
    if labels_path.is_file():
        _verify_declared_artifact(run_dir, marker, "per_label_metrics.csv")
        labels_sha256 = _sha256(labels_path)
    elif not isinstance(config.get("label_names"), (list, tuple)) or not config.get("label_names"):
        raise ValueError("Student run must provide label_names or per_label_metrics.csv")

    return {
        "run_dir": run_dir,
        "run_id": config.get("run_id", run_dir.name),
        "run_model_id": derived_model_id,
        "config": config,
        "config_path": config_artifact,
        "config_sha256": _sha256(config_artifact),
        "checkpoint_path": checkpoint_artifact,
        "checkpoint_sha256": _sha256(checkpoint_artifact),
        "cache": cached,
        "cache_path": cache,
        "cache_sha256": actual_cache_hash,
        "calibration_path": calibration if calibration_used else None,
        "thresholds_path": thresholds if calibration_used else None,
        "calibration_sha256": _sha256(calibration) if calibration_used else None,
        "thresholds_sha256": _sha256(thresholds) if calibration_used else None,
        "labels": _label_names(run_dir, config, labels_path if labels_path.is_file() else None),
        "labels_sha256": labels_sha256,
        "organ_labels": config.get("organ_label_names"),
        "completion_sha256": _sha256(completion_path),
    }


def _load_predictor(context: Mapping[str, Any]) -> CPUPairPredictor:
    config = context["config"]
    model = create_model(config, num_labels=len(context["labels"]))
    try:
        state = torch.load(context["checkpoint_path"], map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("unable to load CPU Student checkpoint") from exc
    if not isinstance(state, Mapping):
        raise ValueError("Student checkpoint must contain a state-dict mapping")
    try:
        model.load_state_dict(state)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Student checkpoint does not match resolved config") from exc
    model = model.to(device="cpu").eval()
    common = {
        "label_names": context["labels"],
        "organ_label_names": context["organ_labels"],
        "top_k": int(config.get("top_k", 5)),
    }
    if context["calibration_path"] is not None:
        return CPUPairPredictor.from_artifacts(
            model,
            context["cache"],
            calibration_path=context["calibration_path"],
            thresholds_path=context["thresholds_path"],
            **common,
        )
    return CPUPairPredictor(model, context["cache"], **common)


def _rss() -> dict[str, Any]:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        unit = "bytes"
        normalized = raw
    else:
        unit = "KiB"
        normalized = raw * 1024
    return {
        "value": raw,
        "unit": unit,
        "platform": sys.platform,
        "normalized_bytes": int(normalized),
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def benchmark_completed_pair(
    run_dir: str | Path,
    drug_a: str,
    drug_b: str,
    *,
    config_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    warmups: int = MIN_WARMUPS,
    iterations: int = MIN_ITERATIONS,
    threads: int = 1,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate, parity-check, and time one fixed pair."""

    context = load_benchmark_run(run_dir, config_path=config_path, cache_path=cache_path)
    first = _load_predictor(context)
    second = _load_predictor(context)
    first_prediction = first.predict_pair_json(drug_a, drug_b)
    second_prediction = second.predict_pair_json(drug_a, drug_b)
    verify_prediction_parity(first_prediction, second_prediction)
    measured = first.benchmark_pair(
        drug_a,
        drug_b,
        warmups=warmups,
        iterations=iterations,
        threads=threads,
    )
    cache_path_obj = context["cache_path"]
    cache_bytes = cache_path_obj.stat().st_size
    drug_count = len(context["cache"].drug_ids)
    parameter = next(first.student.parameters(), None)
    dtype = str(parameter.dtype).replace("torch.", "") if parameter is not None else "unknown"
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": context["run_id"],
        "run_model_id": context["run_model_id"],
        "pair": {"drug_a": drug_a, "drug_b": drug_b},
        "prediction": first_prediction,
        **measured,
        "dtype": dtype,
        "checkpoint_bytes": context["checkpoint_path"].stat().st_size,
        "cache_file_bytes": cache_bytes,
        "cache_bytes_per_drug": float(cache_bytes / drug_count),
        "cache_drug_count": drug_count,
        "peak_rss": _rss(),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
        },
        "hashes": {
            "completion_sha256": context["completion_sha256"],
            "config_sha256": context["config_sha256"],
            "checkpoint_sha256": context["checkpoint_sha256"],
            "cache_sha256": context["cache_sha256"],
            "cache_checksum_sha256": _sha256(cache_path_obj.with_name(f"{cache_path_obj.name}.sha256")),
            "calibration_sha256": context["calibration_sha256"],
            "thresholds_sha256": context["thresholds_sha256"],
            "labels_sha256": context["labels_sha256"],
            "cached_token_artifact_sha256": context["config"].get("cached_token_artifact_sha256"),
            "labels_artifact_sha256": context["config"].get("labels_artifact_sha256"),
            "labels_order_sha256": context["config"].get("labels_order_sha256"),
            "manifest_hash": context["config"].get("manifest_hash"),
        },
        "git_commit": _git_commit(),
    }
    if output_path is not None:
        _atomic_write_json_with_sha256(Path(output_path), result)
    return result


def _atomic_write_json_with_sha256(path: Path, value: Mapping[str, Any]) -> None:
    """Write JSON and its checksum sidecar as recoverable atomic replacements."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    checksum_path = path.with_name(f"{path.name}.sha256")
    json_fd, json_temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    checksum_fd, checksum_temporary = tempfile.mkstemp(
        prefix=f".{checksum_path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(json_fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(json_temporary, path)
        with os.fdopen(checksum_fd, "w", encoding="ascii") as stream:
            stream.write(hashlib.sha256(payload).hexdigest())
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(checksum_temporary, checksum_path)
    finally:
        if os.path.exists(json_temporary):
            os.unlink(json_temporary)
        if os.path.exists(checksum_temporary):
            os.unlink(checksum_temporary)


def _minimum_int(minimum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return parsed

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", dest="config_path", type=Path, default=None)
    parser.add_argument("--cache", dest="cache_path", type=Path, default=None)
    parser.add_argument("--drug-a", required=True)
    parser.add_argument("--drug-b", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmups", type=_minimum_int(MIN_WARMUPS), default=MIN_WARMUPS)
    parser.add_argument("--iterations", type=_minimum_int(MIN_ITERATIONS), default=MIN_ITERATIONS)
    parser.add_argument("--threads", type=_minimum_int(1), default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = benchmark_completed_pair(
        args.run_dir,
        args.drug_a,
        args.drug_b,
        config_path=args.config_path,
        cache_path=args.cache_path,
        warmups=args.warmups,
        iterations=args.iterations,
        threads=args.threads,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
