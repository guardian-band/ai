from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import PredictionRequest
from src.api.settings import Settings


def test_request_normalizes_drug_ids() -> None:
    request = PredictionRequest(drug_a=" db00313 ", drug_b="db01041", top_k=5)
    assert request.drug_a == "DB00313"
    assert request.drug_b == "DB01041"


def test_request_rejects_invalid_top_k() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(drug_a="DB00313", drug_b="DB01041", top_k=0)


def test_settings_accept_onnx_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    required = {
        "POLYPHARMACY_EXPERIMENT_PATH": tmp_path / "teacher.yaml",
        "POLYPHARMACY_MANIFEST_PATH": tmp_path / "manifest.json",
        "POLYPHARMACY_CHECKPOINT_PATH": tmp_path / "checkpoint.pt",
        "POLYPHARMACY_SELECTION_PATH": tmp_path / "selection.json",
        "POLYPHARMACY_ONNX_FUSED_PATH": tmp_path / "teacher_fused.onnx",
        "POLYPHARMACY_ONNX_BASELINE_PATH": tmp_path / "teacher_baseline.onnx",
        "POLYPHARMACY_ONNX_RELEASE_PATH": tmp_path / "onnx_release.json",
    }
    for name, path in required.items():
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("POLYPHARMACY_RUNTIME", "onnx")

    settings = Settings.from_env()

    assert settings.runtime == "onnx"
    assert settings.onnx_fused_path == required["POLYPHARMACY_ONNX_FUSED_PATH"]


def test_settings_reject_unknown_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYPHARMACY_RUNTIME", "unknown")
    with pytest.raises(ValueError, match="POLYPHARMACY_RUNTIME"):
        Settings.from_env()
