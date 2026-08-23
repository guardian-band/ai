from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _required_path(name: str) -> Path:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"missing required environment variable: {name}")
    return Path(value).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    experiment_path: Path
    manifest_path: Path
    checkpoint_path: Path
    selection_path: Path
    runtime: str = "pytorch"
    onnx_fused_path: Path | None = None
    onnx_baseline_path: Path | None = None
    onnx_release_path: Path | None = None
    device: str = "cpu"
    cpu_threads: int = 1
    max_top_k: int = 20
    checkpoint_sha256: str | None = None
    selection_sha256: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        runtime = os.getenv("POLYPHARMACY_RUNTIME", "pytorch").strip().lower()
        if runtime not in {"pytorch", "onnx"}:
            raise ValueError("POLYPHARMACY_RUNTIME must be pytorch or onnx")

        def optional_path(name: str) -> Path | None:
            value = os.getenv(name)
            return Path(value).expanduser().resolve() if value else None

        return cls(
            experiment_path=_required_path("POLYPHARMACY_EXPERIMENT_PATH"),
            manifest_path=_required_path("POLYPHARMACY_MANIFEST_PATH"),
            checkpoint_path=_required_path("POLYPHARMACY_CHECKPOINT_PATH"),
            selection_path=_required_path("POLYPHARMACY_SELECTION_PATH"),
            runtime=runtime,
            onnx_fused_path=optional_path("POLYPHARMACY_ONNX_FUSED_PATH"),
            onnx_baseline_path=optional_path("POLYPHARMACY_ONNX_BASELINE_PATH"),
            onnx_release_path=optional_path("POLYPHARMACY_ONNX_RELEASE_PATH"),
            device=os.getenv("POLYPHARMACY_DEVICE", "cpu"),
            cpu_threads=int(os.getenv("POLYPHARMACY_CPU_THREADS", "1")),
            max_top_k=int(os.getenv("POLYPHARMACY_MAX_TOP_K", "20")),
            checkpoint_sha256=os.getenv("POLYPHARMACY_CHECKPOINT_SHA256"),
            selection_sha256=os.getenv("POLYPHARMACY_SELECTION_SHA256"),
        )
