import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from aggregate_results import _verify_completion, _verify_run
from src.evaluation.metrics import compute_all_metrics
from src.features.multimodal_feature_artifact import MultimodalFeatureArtifact
from src.features.cached_token_artifact import CachedTokenArtifact
from src.models.factory import create_model
from src.training.engine import StateGuardedTrainer
from run_precomputed_experiment import (
    build_optimizer,
    choose_device,
    _construct_test_dataset_after_freeze,
    validate_training_config,
    preflight_experiment,
    run_precomputed_experiment,
    _select_teacher_promotion,
    _build_teacher_selection,
    _atomic_write_json,
    _required_run_artifacts,
    _candidate_grid,
    _select_distillation_candidate,
    _validate_teacher_selection,
    _assert_batch_alignment,
    _validate_student_teacher_ap,
    _derive_run_model_id,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _feature_arrays(count):
    return {
        "morgan": np.ones((count, 4), dtype=np.float32),
        "molformer_tokens": np.ones((count, 2, 3), dtype=np.float32),
        "mpnn_tokens": np.ones((count, 3, 5), dtype=np.float32),
        "kg_tokens": np.ones((count, 1, 2), dtype=np.float32),
        "morgan_available": np.ones(count, dtype=bool),
        "molformer_available": np.ones(count, dtype=bool),
        "mpnn_available": np.ones(count, dtype=bool),
        "kg_available": np.ones(count, dtype=bool),
        "molformer_padding_mask": np.zeros((count, 2), dtype=bool),
        "mpnn_padding_mask": np.zeros((count, 3), dtype=bool),
        "kg_padding_mask": np.zeros((count, 1), dtype=bool),
    }


def _feature_provenance():
    return {
        "morgan_provenance_hash": HASH_A,
        "molformer_provenance_hash": HASH_B,
        "mpnn_provenance_hash": HASH_C,
        "kg_provenance_hash": HASH_D,
    }


def _fixture(tmp_path):
    pairs = pd.DataFrame([
        {"pair_id": "p1", "drug_a": "D1", "drug_b": "D2", "split": "train", "scenario": "warm_pair", "observation_status": "observed_positive", "source_positive_pair_id": None},
        {"pair_id": "p2", "drug_a": "D2", "drug_b": "D3", "split": "validation", "scenario": "warm_pair", "observation_status": "unlabeled", "source_positive_pair_id": None},
        {"pair_id": "p4", "drug_a": "D1", "drug_b": "D2", "split": "validation", "scenario": "warm_pair", "observation_status": "observed_positive", "source_positive_pair_id": None},
        {"pair_id": "p3", "drug_a": "D1", "drug_b": "D3", "split": "test", "scenario": "warm_pair", "observation_status": "unlabeled", "source_positive_pair_id": None},
    ])
    triples = pd.DataFrame([
        {"pair_id": "p1", "drug_a": "D1", "drug_b": "D2", "label_cui": "C1", "observation_status": "observed_positive"},
        {"pair_id": "p4", "drug_a": "D1", "drug_b": "D2", "label_cui": "C1", "observation_status": "observed_positive"},
    ])
    pairs.to_parquet(tmp_path / "pairs.parquet", index=False)
    triples.to_parquet(tmp_path / "triples.parquet", index=False)
    (tmp_path / "labels.json").write_text(json.dumps({"labels": [{"index": 0, "cui": "C1"}]}))
    manifest_body = {"benchmark_id": "fixture", "seed": 1, "scenario": "warm_pair", "pairs_path": "pairs.parquet", "triples_path": "triples.parquet", "labels_path": "labels.json"}
    manifest_hash = hashlib.sha256(json.dumps(manifest_body, sort_keys=True).encode()).hexdigest()[:12]
    manifest = dict(manifest_body, manifest_hash=manifest_hash)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    feature = MultimodalFeatureArtifact.write(
        tmp_path / "features.npz", drug_ids=["D1", "D2", "D3"],
        **_feature_arrays(3), **_feature_provenance(),
        manifest_compatibility={
            "benchmark_id": "fixture", "scenario": "warm_pair", "seed": 1,
            "manifest_hash": manifest_hash,
        },
    )
    hierarchy_path = tmp_path / "hierarchy.json"
    hierarchy_path.write_text(json.dumps({"schema_version": 1, "organ_order": ["O1"], "mappings": [{"specific_cui": "C1", "organ_index": 0}]}))
    config = tmp_path / "teacher.yaml"
    config.write_text(
        "\n".join([
            "model_type: multimodal_teacher", "morgan_dim: 4", "molformer_dim: 3", "mpnn_dim: 5", "kg_dim: 2",
            "molformer_token_count: 2", "mpnn_token_count: 3", "kg_token_count: 1", "hidden_dim: 8", "num_heads: 2",
            "num_organ: 1", "num_specific: 1", "cache_token_count: 3", "cache_token_dim: 6",
            "batch_size: 2", "epochs: 1", "patience: 1", "learning_rate: 0.001", "weight_decay: 0.0", "hierarchy_weight: 0.1",
            f"feature_artifact_path: {feature.metadata.get('path', 'features.npz')}",
            "hierarchy_path: hierarchy.json", f"feature_artifact_sha256: {hashlib.sha256((tmp_path / 'features.npz').read_bytes()).hexdigest()}",
            f"hierarchy_sha256: {hashlib.sha256(hierarchy_path.read_bytes()).hexdigest()}",
        ])
    )
    return config, manifest_path


def test_precomputed_dry_run_performs_preflight_without_writing(tmp_path, monkeypatch):
    config, manifest = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    run_precomputed_experiment(config, manifest, dry_run=True)
    assert not (tmp_path / "artifacts" / "runs").exists()


def test_preflight_seeds_model_initialization_from_manifest(tmp_path):
    config, manifest = _fixture(tmp_path)
    first = preflight_experiment(config, manifest)
    torch.manual_seed(987654321)
    second = preflight_experiment(config, manifest)

    first_state = first.model.state_dict()
    second_state = second.model.state_dict()
    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)


def test_preflight_validates_test_join_without_constructing_test_dataset(tmp_path):
    config, manifest = _fixture(tmp_path)
    manifest_payload = json.loads(manifest.read_text())
    pairs = pd.read_parquet(tmp_path / "pairs.parquet")
    pairs.loc[pairs["pair_id"] == "p3", "drug_b"] = "D4"
    pairs.to_parquet(tmp_path / "pairs.parquet", index=False)
    feature_path = tmp_path / "features_without_test_drug.npz"
    MultimodalFeatureArtifact.write(
        feature_path,
        drug_ids=["D1", "D2", "D3"],
        **_feature_arrays(3), **_feature_provenance(),
        manifest_compatibility={
            "benchmark_id": "fixture", "scenario": "warm_pair", "seed": 1,
            "manifest_hash": manifest_payload["manifest_hash"],
        },
    )
    config_text = config.read_text()
    config_text = config_text.replace("feature_artifact_path: features.npz", "feature_artifact_path: features_without_test_drug.npz")
    config_text = config_text.replace(
        f"feature_artifact_sha256: {hashlib.sha256((tmp_path / 'features.npz').read_bytes()).hexdigest()}",
        f"feature_artifact_sha256: {hashlib.sha256(feature_path.read_bytes()).hexdigest()}",
    )
    config.write_text(config_text)
    with pytest.raises(ValueError, match=r"missing drug IDs in test \(1\): D4"):
        preflight_experiment(config, manifest)


def test_preflight_requires_exact_feature_manifest_compatibility(tmp_path):
    config, manifest = _fixture(tmp_path)
    manifest_payload = json.loads(manifest.read_text())
    original = MultimodalFeatureArtifact.load(tmp_path / "features.npz")
    wrong_path = tmp_path / "features_wrong_seed.npz"
    MultimodalFeatureArtifact.write(
        wrong_path,
        drug_ids=original.drug_ids,
        morgan=original.morgan,
        molformer_tokens=original.molformer_tokens,
        mpnn_tokens=original.mpnn_tokens,
        kg_tokens=original.kg_tokens,
        morgan_available=original.morgan_available,
        molformer_available=original.molformer_available,
        mpnn_available=original.mpnn_available,
        kg_available=original.kg_available,
        molformer_padding_mask=original.molformer_padding_mask,
        mpnn_padding_mask=original.mpnn_padding_mask,
        kg_padding_mask=original.kg_padding_mask,
        **_feature_provenance(),
        manifest_compatibility={
            "benchmark_id": "fixture", "scenario": "warm_pair", "seed": 2,
            "manifest_hash": manifest_payload["manifest_hash"],
        },
    )
    config_text = config.read_text().replace("feature_artifact_path: features.npz", "feature_artifact_path: features_wrong_seed.npz")
    config_text = config_text.replace(
        hashlib.sha256((tmp_path / "features.npz").read_bytes()).hexdigest(),
        hashlib.sha256(wrong_path.read_bytes()).hexdigest(),
    )
    config.write_text(config_text)
    with pytest.raises(ValueError, match="compatibility mismatch for seed"):
        preflight_experiment(config, manifest)


def test_device_priority_and_optimizer_configuration(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert choose_device().type == "cuda"
    model = torch.nn.Linear(2, 1)
    optimizer = build_optimizer(model, {"learning_rate": 0.003, "weight_decay": 0.007})
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.003)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.007)


@pytest.mark.parametrize(
    "key,value",
    [
        ("batch_size", 0),
        ("epochs", True),
        ("patience", -1),
        ("learning_rate", float("nan")),
        ("weight_decay", float("inf")),
        ("hierarchy_weight", -0.1),
    ],
)
def test_invalid_training_controls_fail_preflight_without_run_directory(tmp_path, monkeypatch, key, value):
    config, manifest = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    config_text = config.read_text()
    config_text += f"\n{key}: {value!r}\n"
    config.write_text(config_text)
    with pytest.raises(ValueError, match=key):
        preflight_experiment(config, manifest)
    assert not (tmp_path / "artifacts" / "runs").exists()


def test_optimizer_rejects_bool_and_nonfinite_controls():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="learning_rate"):
        build_optimizer(model, {"learning_rate": True, "weight_decay": 0.0})
    with pytest.raises(ValueError, match="weight_decay"):
        build_optimizer(model, {"learning_rate": 0.001, "weight_decay": float("inf")})


def test_teacher_promotion_requires_validation_delta_and_supports_fallback():
    assert _select_teacher_promotion(0.500, 0.501, 0.002) == "baseline"
    assert _select_teacher_promotion(0.500, 0.502, 0.002) == "baseline"
    assert _select_teacher_promotion(0.500, 0.503, 0.002) == "fused"


def test_morgan_only_teacher_selection_skips_fused_variant():
    selection, mode = _build_teacher_selection(
        {"baseline": {"macro_ap": 0.5}}, {"gates": {}, "coverage": {}}, 0.002
    )
    assert mode == "baseline"
    assert selection["selected"]["mode"] == "baseline"
    assert selection["combined"]["skipped"] is True


def test_run_model_id_rejects_spoofed_teacher_id_and_accepts_ordered_subset():
    assert _derive_run_model_id({"model_type": "multimodal_teacher", "enabled_modalities": []}) == "multimodal_teacher_morgan_only"
    with pytest.raises(ValueError, match="enabled_modalities"):
        _derive_run_model_id({"model_type": "multimodal_teacher", "enabled_modalities": ["kg", "molformer"]})


@pytest.mark.parametrize(
    ("combined_ap", "expected_mode"),
    [(0.501, "baseline"), (0.503, "fused")],
)
def test_teacher_selection_records_harmful_fallback_and_useful_promotion(combined_ap, expected_mode):
    selection, selected_mode = _build_teacher_selection(
        {
            "baseline": {"macro_ap": 0.500},
            "molformer": {"macro_ap": 0.499},
            "mpnn": {"macro_ap": 0.500},
            "kg": {"macro_ap": combined_ap},
            "combined": {"macro_ap": combined_ap},
        },
        {"gates": {}, "coverage": {}},
        0.002,
    )
    assert selected_mode == expected_mode
    assert selection["selected"]["mode"] == expected_mode
    assert selection["baseline"]["macro_ap"] == 0.500
    assert selection["combined"]["macro_ap"] == combined_ap


def test_distillation_candidate_grid_includes_supervised_only_once_and_tie_breaks():
    candidates = _candidate_grid(
        {
            "distillation_weight_candidates": [1.0, 0.0, 0.0],
            "distillation_temperature_candidates": [3.0, 2.0],
        }
    )
    assert candidates.count((0.0, 2.0)) == 1
    assert sum(weight == 0.0 for weight, _ in candidates) == 1
    selected = _select_distillation_candidate(
        [
            {"distillation_weight": 1.0, "temperature": 2.0, "validation_macro_ap": 0.8},
            {"distillation_weight": 0.0, "temperature": 3.0, "validation_macro_ap": 0.8},
            {"distillation_weight": 0.0, "temperature": 2.0, "validation_macro_ap": 0.8},
        ]
    )
    assert selected["distillation_weight"] == 0.0
    assert selected["temperature"] == 2.0


def test_student_template_defaults_cover_weight_and_temperature_candidates():
    config = yaml.safe_load(
        Path("configs/model_distilled_pair_student.yaml").read_text()
    )

    assert config["distillation_weight_candidates"] == [0.0, 0.25, 0.5, 1.0]
    assert config["distillation_temperature_candidates"] == [1.0, 2.0, 4.0]
    candidates = _candidate_grid(config)
    assert sum(weight == 0.0 for weight, _ in candidates) == 1
    assert {
        (weight, temperature)
        for weight, temperature in candidates
        if weight != 0.0
    } == {
        (weight, temperature)
        for weight in (0.25, 0.5, 1.0)
        for temperature in (1.0, 2.0, 4.0)
    }


def test_student_teacher_batch_alignment_rejects_shuffled_or_unequal_ids():
    _assert_batch_alignment(("p1", "p2"), ("p1", "p2"))
    with pytest.raises(ValueError, match="pair_id"):
        _assert_batch_alignment(("p1", "p2"), ("p2", "p1"))
    with pytest.raises(ValueError, match="exhaustion|length"):
        _assert_batch_alignment(("p1",), ("p1", "p2"))


def test_student_validation_drop_gate_is_strict_and_validation_only():
    _validate_student_teacher_ap(0.496, 0.500, 0.005)
    with pytest.raises(ValueError, match="student_max_validation_ap_drop"):
        _validate_student_teacher_ap(0.494, 0.500, 0.005)


def test_teacher_completion_requires_selection_artifact_and_records_tampering(tmp_path):
    required = _required_run_artifacts(student_mode=False)
    assert "teacher_validation_selection.json" in required
    assert "teacher_validation_selection.json" not in _required_run_artifacts(student_mode=True)

    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    for name in required:
        if name != "teacher_validation_selection.json" and name != "metrics.json":
            (missing_dir / name).write_bytes(name.encode())
    trainer = StateGuardedTrainer(str(missing_dir))
    trainer.begin_training()
    trainer.model_selected()
    trainer.validation_frozen()
    trainer.evaluate_test()
    with pytest.raises(FileNotFoundError, match="teacher_validation_selection.json"):
        trainer.complete({}, required_artifacts=required)

    complete_dir = tmp_path / "complete"
    complete_dir.mkdir()
    for name in required:
        if name != "metrics.json":
            (complete_dir / name).write_bytes(name.encode())
    _atomic_write_json(
        complete_dir / "teacher_validation_selection.json",
        {"selected": {"mode": "baseline"}},
    )
    trainer = StateGuardedTrainer(str(complete_dir))
    trainer.begin_training()
    trainer.model_selected()
    trainer.validation_frozen()
    trainer.evaluate_test()
    trainer.complete({}, required_artifacts=required)
    _verify_completion(complete_dir)
    (complete_dir / "teacher_validation_selection.json").write_text("tampered")
    with pytest.raises(ValueError, match="teacher_validation_selection.json"):
        _verify_completion(complete_dir)


def test_teacher_selection_file_requires_hash_mode_and_finite_ap(tmp_path):
    path = tmp_path / "teacher_validation_selection.json"
    path.write_text(json.dumps({"selected": {"mode": "fused", "macro_ap": 0.5}}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    selection = _validate_teacher_selection(path, expected_hash=digest)
    assert selection["selected"]["mode"] == "fused"
    path.write_text(json.dumps({"selected": {"mode": "invalid", "macro_ap": 0.5}}))
    with pytest.raises(ValueError, match="mode|hash"):
        _validate_teacher_selection(path, expected_hash=digest)


@pytest.mark.parametrize("key,value", [("supervised_weight", -1.0), ("distillation_weight", float("nan"))])
def test_student_distillation_weights_are_validated(key, value):
    config = {
        "model_type": "distilled_pair_student",
        "batch_size": 1,
        "epochs": 1,
        "patience": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "hierarchy_weight": 0.0,
        "supervised_weight": 1.0,
        "distillation_weight": 1.0,
        "distillation_temperature": 2.0,
    }
    config[key] = value
    with pytest.raises(ValueError, match=key):
        validate_training_config(config)


def test_distillation_candidates_and_validation_drop_are_validated():
    config = {
        "model_type": "distilled_pair_student",
        "batch_size": 1,
        "epochs": 1,
        "patience": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "hierarchy_weight": 0.0,
        "supervised_weight": 1.0,
        "distillation_weight": 1.0,
        "distillation_temperature": 2.0,
        "distillation_weight_candidates": [0.0, 1.0],
        "distillation_temperature_candidates": [2.0, 3.0],
        "student_max_validation_ap_drop": 0.005,
    }
    validate_training_config(config)
    config["student_max_validation_ap_drop"] = -0.1
    with pytest.raises(ValueError, match="student_max_validation_ap_drop"):
        validate_training_config(config)


def test_validation_macro_ap_requires_defined_label_support():
    from run_precomputed_experiment import _require_defined_macro_ap

    with pytest.raises(ValueError, match="validation macro AP"):
        _require_defined_macro_ap(float("nan"))


def test_resolved_config_contract_is_accepted_by_strict_aggregator(tmp_path):
    run_id = "multimodal_teacher__warm_pair__seed_1__manifest123"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    pair_ids = ["p1", "p2", "p3", "p4"]
    labels = ["C1", "C2"]
    truth = pd.DataFrame({"C1": [1, 0, 1, 0], "C2": [0, 1, 0, 1]})
    probabilities = pd.DataFrame({"C1": [0.9, 0.2, 0.8, 0.1], "C2": [0.1, 0.8, 0.2, 0.9]})
    predictions = pd.concat(
        [truth.add_prefix("truth__"), probabilities.add_prefix("prob__")], axis=1
    )
    predictions.insert(0, "pair_id", pair_ids)
    logits = pd.DataFrame({"pair_id": pair_ids, "logit__C1": [2.0, -1.0, 1.4, -2.0], "logit__C2": [-2.0, 1.4, -1.4, 2.0]})
    predictions.to_parquet(run_dir / "test_predictions.parquet", index=False)
    predictions.to_parquet(run_dir / "validation_predictions.parquet", index=False)
    logits.to_parquet(run_dir / "test_logits.parquet", index=False)
    logits.to_parquet(run_dir / "validation_logits.parquet", index=False)
    thresholds = {label: {"threshold": 0.5} for label in labels}
    (run_dir / "thresholds.json").write_text(json.dumps(thresholds))
    (run_dir / "calibration.json").write_text(json.dumps({"temperatures": {"specific": 1.0}}))
    (run_dir / "environment.json").write_text(json.dumps({"python": "test"}))
    (run_dir / "checkpoint_best.pt").write_bytes(b"checkpoint")
    config = {
        "run_id": run_id,
        "model_type": "multimodal_teacher",
        "benchmark_id": "fixture",
        "scenario": "warm_pair",
        "seed": 1,
        "manifest_hash": "manifest123",
        "train_prevalences": {"C1": 0.5, "C2": 0.5},
    }
    (run_dir / "config.resolved.json").write_text(json.dumps(config))
    metrics = compute_all_metrics(truth, probabilities, thresholds, config["train_prevalences"])
    (run_dir / "metrics.json").write_text(json.dumps(metrics, allow_nan=True))
    pd.DataFrame([{"label_cui": label} for label in labels]).to_csv(run_dir / "per_label_metrics.csv", index=False)
    required = (
        "config.resolved.json", "environment.json", "checkpoint_best.pt",
        "validation_logits.parquet", "validation_predictions.parquet",
        "test_logits.parquet", "test_predictions.parquet", "metrics.json",
        "per_label_metrics.csv", "thresholds.json", "calibration.json",
    )
    def digest(path):
        payload = path.read_bytes()
        return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    (run_dir / "completion.json").write_text(
        json.dumps({"status": "complete", "required_artifacts": {name: digest(run_dir / name) for name in required}})
    )
    assert _verify_run(run_dir)["config"]["train_prevalences"] == {"C1": 0.5, "C2": 0.5}


def test_test_dataset_constructor_requires_validation_freeze(tmp_path):
    config, manifest = _fixture(tmp_path)
    plan = preflight_experiment(config, manifest)
    trainer = StateGuardedTrainer(str(tmp_path / "run"))
    trainer.begin_training()
    trainer.model_selected()
    with pytest.raises(RuntimeError, match="validation_frozen"):
        _construct_test_dataset_after_freeze(plan, False, trainer)
    trainer.validation_frozen()
    dataset = _construct_test_dataset_after_freeze(plan, False, trainer)
    assert len(dataset) == 1


def test_student_preflight_loads_and_freezes_teacher_checkpoint(tmp_path):
    teacher_config, manifest = _fixture(tmp_path)
    teacher = create_model(
        {
            "model_type": "multimodal_teacher", "morgan_dim": 4, "molformer_dim": 3, "mpnn_dim": 5, "kg_dim": 2,
            "molformer_token_count": 2, "mpnn_token_count": 3, "kg_token_count": 1, "hidden_dim": 8, "num_heads": 2,
            "num_organ": 1, "num_specific": 1, "cache_token_count": 3, "cache_token_dim": 6,
        },
        num_labels=1,
    )
    teacher_checkpoint = tmp_path / "teacher.pt"
    torch.save(teacher.state_dict(), teacher_checkpoint)
    cache_path = tmp_path / "cached.npz"
    teacher_checkpoint_hash = hashlib.sha256(teacher_checkpoint.read_bytes()).hexdigest()
    teacher_config_hash = hashlib.sha256(teacher_config.read_bytes()).hexdigest()
    selection_path = tmp_path / "teacher_validation_selection.json"
    selection_path.write_text(json.dumps({"selected": {"mode": "baseline", "macro_ap": 0.5}}))
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    CachedTokenArtifact.write(
        cache_path, ["D1", "D2", "D3"], np.ones((3, 3, 6), dtype=np.float32), np.ones((3, 3), dtype=bool),
        teacher_provenance_hash=teacher_checkpoint_hash, teacher_checkpoint_hash=teacher_checkpoint_hash, teacher_config_hash=teacher_config_hash,
        teacher_selection_hash=selection_hash, teacher_selected_mode="baseline",
        modality_provenance_hash=hashlib.sha256((tmp_path / "features.npz").read_bytes()).hexdigest(),
    )
    teacher_config_text = teacher_config.read_text()
    teacher_config.write_text(teacher_config_text.replace("feature_artifact_path: features.npz", "feature_artifact_path: features.npz"))
    student_config = tmp_path / "student.yaml"
    student_config.write_text("\n".join([
        "model_type: distilled_pair_student", "token_dim: 6", "token_count: 3", "hidden_dim: 8", "num_heads: 2",
        "num_organ: 1", "num_specific: 1", "batch_size: 2", "epochs: 1", "patience: 1", "learning_rate: 0.001", "weight_decay: 0.0",
        "hierarchy_weight: 0.1", "supervised_weight: 1.0", "distillation_weight: 1.0", "distillation_temperature: 2.0",
        "cached_token_artifact_path: cached.npz", f"cached_token_artifact_sha256: {hashlib.sha256(cache_path.read_bytes()).hexdigest()}",
        "teacher_config_path: teacher.yaml", f"teacher_config_sha256: {teacher_config_hash}",
        "teacher_checkpoint_path: teacher.pt", f"teacher_checkpoint_sha256: {teacher_checkpoint_hash}",
        "teacher_selection_path: teacher_validation_selection.json", f"teacher_selection_sha256: {selection_hash}", "teacher_selected_mode: baseline",
        "hierarchy_path: hierarchy.json", f"hierarchy_sha256: {hashlib.sha256((tmp_path / 'hierarchy.json').read_bytes()).hexdigest()}",
        "feature_artifact_path: features.npz", f"feature_artifact_sha256: {hashlib.sha256((tmp_path / 'features.npz').read_bytes()).hexdigest()}",
    ]))
    plan = preflight_experiment(student_config, manifest)
    assert plan.teacher_model is not None
    assert plan.teacher_model.training is False
    assert plan.teacher_model.training_stage == "baseline"
    assert all(parameter.requires_grad is False for parameter in plan.teacher_model.parameters())

    selection_path.write_text(json.dumps({"selected": {"mode": "baseline", "macro_ap": 0.6}}))
    with pytest.raises(ValueError, match="selection hash"):
        preflight_experiment(student_config, manifest)
    selection_path.write_text(json.dumps({"selected": {"mode": "baseline", "macro_ap": 0.5}}))

    wrong_cache = tmp_path / "cached_wrong_modality.npz"
    CachedTokenArtifact.write(
        wrong_cache,
        ["D1", "D2", "D3"],
        np.ones((3, 3, 6), dtype=np.float32),
        np.ones((3, 3), dtype=bool),
        teacher_provenance_hash=teacher_checkpoint_hash,
        teacher_checkpoint_hash=teacher_checkpoint_hash,
        teacher_config_hash=teacher_config_hash,
        teacher_selection_hash=selection_hash,
        teacher_selected_mode="baseline",
        multimodal_feature_artifact_hash=HASH_C,
    )
    wrong_config = student_config.read_text()
    wrong_config = wrong_config.replace("cached.npz", "cached_wrong_modality.npz")
    wrong_config = wrong_config.replace(
        hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        hashlib.sha256(wrong_cache.read_bytes()).hexdigest(),
    )
    student_config.write_text(wrong_config)
    with pytest.raises(ValueError, match="modality provenance"):
        preflight_experiment(student_config, manifest)
