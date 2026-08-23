#!/usr/bin/env python3
"""Export the frozen warm Morgan+MPNN Teacher and Morgan fallback to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_precomputed_experiment import preflight_experiment  # noqa: E402
from src.onnx_teacher import (  # noqa: E402
    BASELINE_INPUT_NAMES,
    FUSED_INPUT_NAMES,
    OUTPUT_NAMES,
    FrozenTeacherBaselineOnnx,
    FrozenTeacherFusedOnnx,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(plan: Any) -> tuple[torch.Tensor, ...]:
    drug_a, drug_b, *_ = plan.validation_dataset[0]
    return (
        drug_a["morgan"].unsqueeze(0),
        drug_a["mpnn_tokens"].unsqueeze(0),
        drug_a["mpnn_padding_mask"].unsqueeze(0),
        drug_b["morgan"].unsqueeze(0),
        drug_b["mpnn_tokens"].unsqueeze(0),
        drug_b["mpnn_padding_mask"].unsqueeze(0),
    )


def _export(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: tuple[str, ...],
    opset: int,
) -> None:
    dynamic_axes = {name: {0: "batch"} for name in (*input_names, *OUTPUT_NAMES)}
    with torch.inference_mode():
        torch.onnx.export(
            model,
            inputs,
            path,
            input_names=list(input_names),
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    plan = preflight_experiment(args.experiment, args.manifest)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected = selection.get("selected")
    if not isinstance(selected, Mapping) or selected.get("mode") != "fused":
        raise ValueError("ONNX export requires validation-selected fused Teacher")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    plan.model.load_state_dict(state)
    teacher = plan.model.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    args.output.mkdir(parents=True, exist_ok=True)
    fused_path = args.output / "teacher_fused.onnx"
    baseline_path = args.output / "teacher_baseline.onnx"
    sample = _inputs(plan)
    _export(
        FrozenTeacherFusedOnnx(teacher).eval(),
        sample,
        fused_path,
        FUSED_INPUT_NAMES,
        args.opset,
    )
    _export(
        FrozenTeacherBaselineOnnx(teacher).eval(),
        (sample[0], sample[3]),
        baseline_path,
        BASELINE_INPUT_NAMES,
        args.opset,
    )

    metadata = {
        "schema_version": 1,
        "runtime": "onnxruntime",
        "opset": args.opset,
        "selected_mode": "fused",
        "input_contract": {
            "fused": list(FUSED_INPUT_NAMES),
            "baseline": list(BASELINE_INPUT_NAMES),
            "outputs": list(OUTPUT_NAMES),
            "dynamic_axes": ["batch"],
        },
        "source_sha256": {
            "checkpoint": _sha256(args.checkpoint),
            "selection": _sha256(args.selection),
        },
        "onnx_sha256": {
            "fused": _sha256(fused_path),
            "baseline": _sha256(baseline_path),
        },
        "size_bytes": {
            "fused": fused_path.stat().st_size,
            "baseline": baseline_path.stat().st_size,
        },
    }
    metadata_path = args.output / "onnx_release.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "onnx_release.json.sha256").write_text(
        _sha256(metadata_path) + "\n", encoding="ascii"
    )
    print(f"ONNX_EXPORT_COMPLETE: {args.output}")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
