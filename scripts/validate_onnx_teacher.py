#!/usr/bin/env python3
"""Validate frozen Teacher ONNX parity and benchmark its CPU runtime."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_precomputed_experiment import preflight_experiment  # noqa: E402
from src.onnx_teacher import (  # noqa: E402
    BASELINE_INPUT_NAMES,
    FUSED_INPUT_NAMES,
    FrozenTeacherBaselineOnnx,
    FrozenTeacherFusedOnnx,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    checksum = path.with_name(f"{path.name}.sha256")
    json_fd, json_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    sha_fd, sha_tmp = tempfile.mkstemp(prefix=f".{checksum.name}.", dir=path.parent)
    try:
        with os.fdopen(json_fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(json_tmp, path)
        with os.fdopen(sha_fd, "w", encoding="ascii") as stream:
            stream.write(hashlib.sha256(payload).hexdigest() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(sha_tmp, checksum)
    finally:
        for temporary in (json_tmp, sha_tmp):
            if os.path.exists(temporary):
                os.unlink(temporary)


def _sample_inputs(dataset: Any, count: int) -> tuple[dict[str, np.ndarray], list[str]]:
    rows: dict[str, list[np.ndarray]] = {}
    pair_ids: list[str] = []
    for index in range(min(count, len(dataset))):
        drug_a, drug_b, *_rest = dataset[index]
        pair_id = str(_rest[-2])
        values = {
            **{f"{name}_a": value for name, value in drug_a.items()},
            **{f"{name}_b": value for name, value in drug_b.items()},
        }
        for name, value in values.items():
            rows.setdefault(name, []).append(np.asarray(value))
        pair_ids.append(pair_id)
    if not pair_ids:
        raise ValueError("validation dataset contains no pairs")
    return {name: np.stack(values) for name, values in rows.items()}, pair_ids


def _torch_inputs(values: Mapping[str, np.ndarray], names: Sequence[str]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(np.asarray(values[name])) for name in names)


def _original_inputs(values: Mapping[str, np.ndarray]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    first = {
        name[:-2]: torch.from_numpy(value)
        for name, value in values.items()
        if name.endswith("_a")
    }
    second = {
        name[:-2]: torch.from_numpy(value)
        for name, value in values.items()
        if name.endswith("_b")
    }
    return first, second


def _comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    difference = np.abs(reference - candidate)
    return {
        "shape": list(reference.shape),
        "max_absolute_difference": float(difference.max(initial=0.0)),
        "mean_absolute_difference": float(difference.mean()),
        "allclose_atol_1e_5_rtol_1e_4": bool(
            np.allclose(reference, candidate, atol=1e-5, rtol=1e-4)
        ),
    }


def _top5(values: np.ndarray) -> np.ndarray:
    return np.argsort(-values, axis=1, kind="stable")[:, :5]


def _parity(
    reference: tuple[np.ndarray, np.ndarray], candidate: tuple[np.ndarray, np.ndarray]
) -> dict[str, Any]:
    organ_ref, specific_ref = reference
    organ_got, specific_got = candidate
    organ_prob_ref = 1.0 / (1.0 + np.exp(-organ_ref))
    specific_prob_ref = 1.0 / (1.0 + np.exp(-specific_ref))
    organ_prob_got = 1.0 / (1.0 + np.exp(-organ_got))
    specific_prob_got = 1.0 / (1.0 + np.exp(-specific_got))
    return {
        "organ_logits": _comparison(organ_ref, organ_got),
        "specific_logits": _comparison(specific_ref, specific_got),
        "organ_probabilities": _comparison(organ_prob_ref, organ_prob_got),
        "specific_probabilities": _comparison(specific_prob_ref, specific_prob_got),
        "organ_top5_exact_match": bool(
            np.array_equal(_top5(organ_prob_ref), _top5(organ_prob_got))
        ),
        "specific_top5_exact_match": bool(
            np.array_equal(_top5(specific_prob_ref), _top5(specific_prob_got))
        ),
    }


def _summary(samples_ns: Sequence[int], batch_size: int) -> dict[str, float | int]:
    values = np.asarray(samples_ns, dtype=np.float64) / 1_000_000.0
    return {
        "iterations": len(samples_ns),
        "batch_size": batch_size,
        "mean_ms_per_batch": float(values.mean()),
        "mean_ms_per_pair": float(values.mean() / batch_size),
        "p50_ms_per_batch": float(np.percentile(values, 50)),
        "p95_ms_per_batch": float(np.percentile(values, 95)),
        "p99_ms_per_batch": float(np.percentile(values, 99)),
        "pairs_per_second": float(batch_size * 1000.0 / values.mean()),
    }


def _benchmark_session(
    session: ort.InferenceSession,
    values: Mapping[str, np.ndarray],
    names: Sequence[str],
    *,
    warmups: int,
    iterations: int,
) -> dict[str, float | int]:
    feed = {name: np.ascontiguousarray(values[name]) for name in names}
    for _ in range(warmups):
        session.run(None, feed)
    samples: list[int] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        session.run(None, feed)
        samples.append(time.perf_counter_ns() - started)
    return _summary(samples, next(iter(feed.values())).shape[0])


def _make_session(path: Path, threads: int) -> tuple[ort.InferenceSession, float]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return session, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    preflight_started = time.perf_counter()
    plan = preflight_experiment(args.experiment, args.manifest)
    preflight_seconds = time.perf_counter() - preflight_started
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("Teacher checkpoint must contain a state-dict mapping")
    plan.model.load_state_dict(state)
    teacher = plan.model.eval()
    teacher.set_training_stage("fused")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    baseline_teacher = copy.deepcopy(teacher).eval()
    baseline_teacher.set_training_stage("baseline")

    values, pair_ids = _sample_inputs(plan.validation_dataset, args.pairs)
    original_inputs = _original_inputs(values)
    fused_wrapper = FrozenTeacherFusedOnnx(teacher).eval()
    baseline_wrapper = FrozenTeacherBaselineOnnx(teacher).eval()
    with torch.inference_mode():
        original_fused = teacher(*original_inputs)
        original_baseline = baseline_teacher(*original_inputs)
        wrapper_fused = fused_wrapper(*_torch_inputs(values, FUSED_INPUT_NAMES))
        wrapper_baseline = baseline_wrapper(*_torch_inputs(values, BASELINE_INPUT_NAMES))

    def numpy_original(output: Any) -> tuple[np.ndarray, np.ndarray]:
        return output.organ_logits.numpy(), output.specific_logits.numpy()

    def numpy_tuple(output: tuple[torch.Tensor, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        return output[0].numpy(), output[1].numpy()

    fused_path = args.onnx_dir / "teacher_fused.onnx"
    baseline_path = args.onnx_dir / "teacher_baseline.onnx"
    onnx.checker.check_model(onnx.load(str(fused_path)))
    onnx.checker.check_model(onnx.load(str(baseline_path)))
    fused_session, fused_init = _make_session(fused_path, args.cpu_threads)
    baseline_session, baseline_init = _make_session(baseline_path, args.cpu_threads)
    fused_ort = tuple(
        fused_session.run(None, {name: values[name] for name in FUSED_INPUT_NAMES})
    )
    baseline_ort = tuple(
        baseline_session.run(None, {name: values[name] for name in BASELINE_INPUT_NAMES})
    )

    parity = {
        "wrapper_vs_original": {
            "fused": _parity(numpy_original(original_fused), numpy_tuple(wrapper_fused)),
            "baseline": _parity(
                numpy_original(original_baseline), numpy_tuple(wrapper_baseline)
            ),
        },
        "onnxruntime_vs_original": {
            "fused": _parity(numpy_original(original_fused), fused_ort),
            "baseline": _parity(numpy_original(original_baseline), baseline_ort),
        },
    }
    checks = []
    for layer in parity.values():
        for mode in layer.values():
            checks.extend(
                [
                    mode["organ_logits"]["allclose_atol_1e_5_rtol_1e_4"],
                    mode["specific_logits"]["allclose_atol_1e_5_rtol_1e_4"],
                    mode["organ_top5_exact_match"],
                    mode["specific_top5_exact_match"],
                ]
            )
    parity_passed = all(checks)

    benchmarks: dict[str, Any] = {}
    for batch_size in (1, min(args.pairs, len(pair_ids))):
        subset = {name: value[:batch_size] for name, value in values.items()}
        benchmarks[f"batch_{batch_size}"] = {
            "fused": _benchmark_session(
                fused_session,
                subset,
                FUSED_INPUT_NAMES,
                warmups=args.warmups,
                iterations=args.iterations,
            ),
            "baseline": _benchmark_session(
                baseline_session,
                subset,
                BASELINE_INPUT_NAMES,
                warmups=args.warmups,
                iterations=args.iterations,
            ),
        }

    report = {
        "schema_version": 1,
        "status": "PASS" if parity_passed else "FAIL",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "providers": ort.get_available_providers(),
            "cpu_threads": args.cpu_threads,
        },
        "source": {
            "checkpoint_sha256": _sha256(args.checkpoint),
            "fused_onnx_sha256": _sha256(fused_path),
            "baseline_onnx_sha256": _sha256(baseline_path),
            "pair_ids": pair_ids,
        },
        "startup": {
            "full_pytorch_preflight_seconds": preflight_seconds,
            "onnx_fused_session_seconds": fused_init,
            "onnx_baseline_session_seconds": baseline_init,
        },
        "size_bytes": {
            "checkpoint": args.checkpoint.stat().st_size,
            "fused_onnx": fused_path.stat().st_size,
            "baseline_onnx": baseline_path.stat().st_size,
        },
        "parity": parity,
        "latency": benchmarks,
    }
    _atomic_json(args.output, report)
    print(f"ONNX_VALIDATION_{report['status']}: {args.output}")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not parity_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
