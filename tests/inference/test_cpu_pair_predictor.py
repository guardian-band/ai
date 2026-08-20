import sys

import json
import numpy as np
import pytest
import torch

from src.features.cached_token_artifact import CachedTokenArtifact
from src.inference.cpu_pair_predictor import CPUPairPredictor
from src.models.multimodal_teacher_student import DistilledPairStudent

TEACHER_HASH = "a" * 64
MODALITY_HASH = "b" * 64
SELECTION_HASH = "c" * 64


def test_cpu_predictor_applies_temperatures_once_and_orders_ties(tmp_path):
    torch.manual_seed(8)
    student = DistilledPairStudent(4, 3, 8, 1, 3, num_heads=2).eval()
    path = tmp_path / "tokens.npz"
    CachedTokenArtifact.write(
        path,
        ["a", "b"],
        np.random.default_rng(1).normal(size=(2, 3, 4)).astype(np.float32),
        np.ones((2, 3), dtype=bool),
        teacher_provenance_hash=TEACHER_HASH,
        modality_provenance_hash=MODALITY_HASH,
        teacher_selection_hash=SELECTION_HASH,
        teacher_selected_mode="baseline",
    )
    predictor = CPUPairPredictor(
        student,
        path,
        label_names=["s0", "s1", "s2"],
        temperatures={"organ": [2.0], "specific": [2.0, 1.0, 0.5]},
        thresholds={"specific": [0.5, 0.5, 0.5]},
        top_k=3,
    )
    result = predictor.predict_pair("a", "b")
    with torch.no_grad():
        a, av_a = predictor.cache.lookup("a")
        b, av_b = predictor.cache.lookup("b")
        raw = student(
            torch.from_numpy(a).unsqueeze(0),
            torch.from_numpy(b).unsqueeze(0),
            torch.from_numpy(av_a).unsqueeze(0),
            torch.from_numpy(av_b).unsqueeze(0),
        )
    expected = torch.sigmoid(raw.specific_logits / torch.tensor([2.0, 1.0, 0.5]))
    assert np.allclose(result["specific_probabilities"], expected.numpy()[0])
    assert result["top_k"][0]["index"] <= result["top_k"][1]["index"]
    assert result["top_k"][1]["index"] <= result["top_k"][2]["index"]
    with pytest.raises(KeyError, match="unknown"):
        predictor.predict_pair("a", "missing")


def test_cpu_predictor_requires_eval_cpu_and_has_no_graph_runtime_imports(tmp_path):
    student = DistilledPairStudent(4, 3, 8, 1, 3, num_heads=2)
    path = tmp_path / "tokens.npz"
    CachedTokenArtifact.write(
        path,
        ["a", "b"],
        np.zeros((2, 3, 4), dtype=np.float32),
        np.ones((2, 3), dtype=bool),
        teacher_provenance_hash=TEACHER_HASH,
        modality_provenance_hash=MODALITY_HASH,
        teacher_selection_hash=SELECTION_HASH,
        teacher_selected_mode="baseline",
    )
    with pytest.raises(ValueError, match="eval"):
        CPUPairPredictor(student, path, label_names=["s0", "s1", "s2"])
    assert "torch_geometric" not in sys.modules
    assert "transformers" not in sys.modules


def test_cpu_predictor_breaks_probability_ties_by_label_index(tmp_path):
    student = DistilledPairStudent(4, 3, 8, 1, 3, num_heads=2).eval()
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.zero_()
    path = tmp_path / "tokens.npz"
    CachedTokenArtifact.write(
        path,
        ["a", "b"],
        np.zeros((2, 3, 4), dtype=np.float32),
        np.ones((2, 3), dtype=bool),
        teacher_provenance_hash=TEACHER_HASH,
        modality_provenance_hash=MODALITY_HASH,
        teacher_selection_hash=SELECTION_HASH,
        teacher_selected_mode="baseline",
    )
    predictor = CPUPairPredictor(student, path, label_names=["s0", "s1", "s2"], top_k=2)
    result = predictor.predict_pair("a", "b")
    assert [item["index"] for item in result["top_k"]] == [0, 1]


def test_cpu_predictor_loads_repository_calibration_threshold_artifacts(tmp_path):
    student = DistilledPairStudent(4, 3, 8, 1, 3, num_heads=2).eval()
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.zero_()
    cache_path = tmp_path / "tokens.npz"
    CachedTokenArtifact.write(
        cache_path,
        ["a", "b"],
        np.zeros((2, 3, 4), dtype=np.float32),
        np.ones((2, 3), dtype=bool),
        teacher_provenance_hash=TEACHER_HASH,
        modality_provenance_hash=MODALITY_HASH,
        teacher_selection_hash=SELECTION_HASH,
        teacher_selected_mode="baseline",
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({"temperatures": {"specific": 2.0}}))
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps({label: {"threshold": 0.25} for label in ("s0", "s1", "s2")})
    )
    predictor = CPUPairPredictor.from_artifacts(
        student,
        cache_path,
        label_names=["s0", "s1", "s2"],
        calibration_path=calibration_path,
        thresholds_path=thresholds_path,
    )
    result = predictor.predict_pair_json("a", "b")
    json.dumps(result)
    assert result["specific_probabilities"] == [0.5, 0.5, 0.5]


def test_cpu_predictor_rejects_misaligned_threshold_artifact(tmp_path):
    student = DistilledPairStudent(4, 3, 8, 1, 3, num_heads=2).eval()
    cache_path = tmp_path / "tokens.npz"
    CachedTokenArtifact.write(
        cache_path,
        ["a", "b"],
        np.zeros((2, 3, 4), dtype=np.float32),
        np.ones((2, 3), dtype=bool),
        teacher_provenance_hash=TEACHER_HASH,
        modality_provenance_hash=MODALITY_HASH,
        teacher_selection_hash=SELECTION_HASH,
        teacher_selected_mode="baseline",
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({"temperatures": {"specific": 1.0}}))
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(json.dumps({"s0": {"threshold": 0.5}}))
    with pytest.raises(ValueError, match="exactly match"):
        CPUPairPredictor.from_artifacts(
            student,
            cache_path,
            label_names=["s0", "s1", "s2"],
            calibration_path=calibration_path,
            thresholds_path=thresholds_path,
        )
