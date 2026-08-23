#!/usr/bin/env python3
"""Benchmark the frozen warm Teacher for single-pair inference on CPU and CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_precomputed_experiment import preflight_experiment  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {name}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


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


def _to_device(values: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, non_blocking=False) for key, value in values.items()}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(samples_ns: Sequence[int]) -> dict[str, float | int]:
    values = np.asarray(samples_ns, dtype=np.float64) / 1_000_000.0
    if not len(values):
        raise ValueError("latency summary requires at least one sample")
    return {
        "iterations": int(len(values)),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "minimum_ms": float(values.min()),
        "maximum_ms": float(values.max()),
        "std_ms": float(values.std()),
        "pairs_per_second_from_mean": float(1000.0 / values.mean()),
    }


def _measure(
    model: torch.nn.Module,
    pairs: Sequence[tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]],
    device: torch.device,
    *,
    warmups: int,
    iterations: int,
    postprocess: bool,
) -> dict[str, float | int]:
    def invoke(index: int) -> None:
        drug_a, drug_b = pairs[index % len(pairs)]
        output = model(drug_a, drug_b)
        if postprocess:
            specific = torch.sigmoid(output.specific_logits)
            organ = torch.sigmoid(output.organ_logits)
            torch.topk(specific, k=min(5, specific.shape[-1]), dim=-1)
            torch.topk(organ, k=min(5, organ.shape[-1]), dim=-1)

    with torch.inference_mode():
        for index in range(warmups):
            invoke(index)
        _synchronize(device)
        samples: list[int] = []
        for index in range(iterations):
            _synchronize(device)
            started = time.perf_counter_ns()
            invoke(index)
            _synchronize(device)
            samples.append(time.perf_counter_ns() - started)
    return _summary(samples)


def _load_pairs(dataset: Any, count: int, device: torch.device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    pairs = []
    pair_ids = []
    for batch in loader:
        pairs.append((_to_device(batch[0], device), _to_device(batch[1], device)))
        pair_ids.append(str(batch[4][0]))
        if len(pairs) >= count:
            break
    if not pairs:
        raise ValueError("validation dataset contains no benchmark pairs")
    return pairs, pair_ids


def _load_frozen_model(plan: Any, checkpoint: Path, mode: str, device: torch.device):
    model = plan.model
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("Teacher checkpoint must contain a state-dict mapping")
    model.load_state_dict(state)
    model.set_training_stage(mode)
    return model.to(device=device).eval()


def benchmark_device(
    plan: Any,
    checkpoint: Path,
    mode: str,
    device_name: str,
    *,
    pair_count: int,
    warmups: int,
    iterations: int,
    cpu_threads: int,
) -> tuple[dict[str, Any], np.ndarray]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    if device.type == "cpu":
        torch.set_num_threads(cpu_threads)
    else:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = _load_frozen_model(plan, checkpoint, mode, device)
    pairs, pair_ids = _load_pairs(plan.validation_dataset, pair_count, device)
    with torch.inference_mode():
        parity_output = model(*pairs[0])
        parity_logits = parity_output.specific_logits.detach().cpu().numpy()
    result: dict[str, Any] = {
        "device": str(device),
        "pair_count": len(pairs),
        "pair_ids": pair_ids,
        "warmups": warmups,
        "cpu_threads": cpu_threads if device.type == "cpu" else None,
        "model_forward": _measure(
            model, pairs, device, warmups=warmups, iterations=iterations, postprocess=False
        ),
        "model_plus_sigmoid_top5": _measure(
            model, pairs, device, warmups=warmups, iterations=iterations, postprocess=True
        ),
    }
    if device.type == "cuda":
        result["gpu"] = {
            "name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    model.to("cpu")
    del model, pairs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, parity_logits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=("cpu", "cuda"))
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for name, value in (
        ("pairs", args.pairs),
        ("warmups", args.warmups),
        ("iterations", args.iterations),
        ("cpu_threads", args.cpu_threads),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")

    started = time.perf_counter()
    plan = preflight_experiment(args.experiment, args.manifest)
    preflight_seconds = time.perf_counter() - started
    if plan.config.get("model_type") != "multimodal_teacher":
        raise ValueError("benchmark requires a multimodal_teacher experiment")
    selection = _read_json(args.selection, "Teacher selection")
    selected = selection.get("selected")
    if not isinstance(selected, Mapping) or selected.get("mode") not in {"baseline", "fused"}:
        raise ValueError("Teacher selection must declare selected.mode")
    mode = str(selected["mode"])

    device_results: dict[str, Any] = {}
    parity: dict[str, np.ndarray] = {}
    for device_name in args.devices:
        print(f"[benchmark] device={device_name}", flush=True)
        device_results[device_name], parity[device_name] = benchmark_device(
            plan,
            args.checkpoint,
            mode,
            device_name,
            pair_count=args.pairs,
            warmups=args.warmups,
            iterations=args.iterations,
            cpu_threads=args.cpu_threads,
        )

    parity_report = None
    if "cpu" in parity and "cuda" in parity:
        absolute = np.abs(parity["cpu"] - parity["cuda"])
        parity_report = {
            "specific_logits_max_absolute_difference": float(absolute.max()),
            "specific_logits_mean_absolute_difference": float(absolute.mean()),
            "within_atol_1e_5_rtol_1e_4": bool(
                np.allclose(parity["cpu"], parity["cuda"], atol=1e-5, rtol=1e-4)
            ),
        }

    parameter_count = sum(parameter.numel() for parameter in plan.model.parameters())
    result = {
        "schema_version": 1,
        "measurement_scope": {
            "model_forward": "preloaded single-pair tensors; excludes feature lookup, API, network and JSON serialization",
            "model_plus_sigmoid_top5": "model forward plus sigmoid probabilities and Top-5 tensor selection; excludes API, network and JSON serialization",
        },
        "selected_mode": mode,
        "selected_validation_macro_auprc": float(selected.get("macro_ap")),
        "one_time_preflight_seconds": float(preflight_seconds),
        "parameter_count": int(parameter_count),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "devices": device_results,
        "cpu_cuda_parity": parity_report,
        "hashes": {
            "experiment_sha256": _sha256(args.experiment),
            "manifest_sha256": _sha256(args.manifest),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "selection_sha256": _sha256(args.selection),
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    print(f"TEACHER_LATENCY_COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
