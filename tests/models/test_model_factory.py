import torch
from pathlib import Path
import pytest
import yaml

from src.models.factory import (
    LogisticPairModel,
    PrevalenceModel,
    SymmetricMLPModel,
    UnsupportedModelConfiguration,
    create_model,
    load_experiment_config,
    derive_run_model_id,
)


@pytest.mark.parametrize(
    "config_name, expected_type",
    [
        ("prevalence", PrevalenceModel),
        ("logistic", LogisticPairModel),
        ("symmetric_mlp", SymmetricMLPModel),
    ],
)
def test_supported_configs_create_distinct_models(config_name, expected_type):
    config = load_experiment_config(f"configs/models/{config_name}.yaml")
    model = create_model(config, num_labels=3, input_dim=8)
    assert isinstance(model, expected_type)


@pytest.mark.parametrize("config_name", ["unified"])
def test_unavailable_configs_fail_instead_of_substituting(config_name):
    config = load_experiment_config(f"configs/models/{config_name}.yaml")
    with pytest.raises(UnsupportedModelConfiguration, match="unavailable"):
        create_model(config, num_labels=3, input_dim=8)


def test_advanced_config_is_the_executable_multimodal_teacher_template():
    from src.models.multimodal_teacher_student import MultiModalTeacher

    config = load_experiment_config("configs/models/advanced.yaml")
    assert config["model_type"] == "multimodal_teacher"
    assert isinstance(create_model(config, num_labels=100), MultiModalTeacher)


def test_multimodal_teacher_factory_passes_balanced_modality_controls():
    from src.models.multimodal_teacher_student import MultiModalTeacher

    config = {
        "model_type": "multimodal_teacher",
        "morgan_dim": 8,
        "molformer_dim": 6,
        "mpnn_dim": 7,
        "kg_dim": 5,
        "molformer_token_count": 3,
        "mpnn_token_count": 4,
        "kg_token_count": 2,
        "hidden_dim": 16,
        "num_organ": 2,
        "num_specific": 4,
        "num_heads": 4,
        "modality_summary_token_count": 2,
        "modality_dropout": 0.2,
        "enabled_modalities": ["molformer", "kg"],
    }

    model = create_model(config, num_labels=4)

    assert isinstance(model, MultiModalTeacher)
    assert model.modality_summary_token_count == 2
    assert model.modality_dropout == 0.2
    assert model.enabled_modalities == ("molformer", "kg")


@pytest.mark.parametrize(
    ("enabled_modalities", "expected"),
    [
        ([], "multimodal_teacher_morgan_only"),
        (["molformer"], "multimodal_teacher_morgan_molformer"),
        (["mpnn"], "multimodal_teacher_morgan_mpnn"),
        (["kg"], "multimodal_teacher_morgan_hgt"),
        (["molformer", "mpnn", "kg"], "multimodal_teacher_full"),
    ],
)
def test_teacher_run_model_id_is_derived_from_enabled_modalities(enabled_modalities, expected):
    assert derive_run_model_id(
        {"model_type": "multimodal_teacher", "enabled_modalities": enabled_modalities}
    ) == expected
    assert derive_run_model_id({"model_type": "distilled_pair_student", "run_model_id": "spoof"}) == "distilled_pair_student"


@pytest.mark.parametrize(
    "filename",
    [
        "multimodal_teacher_morgan_only.yaml",
        "multimodal_teacher_morgan_molformer.yaml",
        "multimodal_teacher_morgan_mpnn.yaml",
        "multimodal_teacher_morgan_hgt.yaml",
        "multimodal_teacher_full.yaml",
        "distilled_pair_student.yaml",
    ],
)
def test_ablation_templates_have_null_artifact_contract(filename):
    payload = yaml.safe_load(Path("configs/ablations", filename).read_text())
    assert payload["feature_artifact_path"] is None if payload["model_type"] == "multimodal_teacher" else True
    assert payload["hierarchy_path"] is None
    if payload["model_type"] == "multimodal_teacher":
        assert "enabled_modalities" in payload


def test_invalid_model_type_config_fails_clearly(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("model_typ: logistic\n")
    with pytest.raises(ValueError, match="model_type"):
        load_experiment_config(path)


def test_prevalence_model_emits_train_label_prevalence():
    model = PrevalenceModel(num_labels=2)
    train_labels = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    model.fit(train_labels)
    probabilities = model.predict_proba(batch_size=4)
    expected = torch.tensor([[2.0 / 3.0, 1.0 / 3.0]]).expand(4, -1)
    assert torch.allclose(probabilities, expected)


def test_prevalence_fit_state_survives_checkpoint_round_trip():
    model = PrevalenceModel(num_labels=1)
    model.fit(torch.tensor([[1.0], [0.0]]))
    restored = PrevalenceModel(num_labels=1)
    restored.load_state_dict(model.state_dict())
    assert torch.allclose(restored.predict_proba(batch_size=1), torch.tensor([[0.5]]))


@pytest.mark.parametrize("model_type", ["logistic", "symmetric_mlp"])
def test_pair_models_are_swap_invariant(model_type):
    model = create_model({"model_type": model_type}, num_labels=3, input_dim=8)
    model.eval()
    drug_a = torch.randn(2, 4, 8)
    drug_b = torch.randn(2, 4, 8)
    mask_a = torch.zeros(2, 4, dtype=torch.bool)
    mask_b = torch.zeros(2, 4, dtype=torch.bool)
    forward = model(drug_a, drug_b, mask_a, mask_b)
    reverse = model(drug_b, drug_a, mask_b, mask_a)
    assert torch.allclose(forward, reverse, atol=1e-6, rtol=0.0)
    assert torch.allclose(torch.sigmoid(forward), torch.sigmoid(reverse), atol=1e-6, rtol=0.0)


def test_dual_head_model_is_swap_invariant_and_named():
    from src.models.factory import SymmetricDualHeadModel

    model = SymmetricDualHeadModel(input_dim=8, num_organ=2, num_specific=3)
    model.eval()
    drug_a = torch.randn(2, 4, 8)
    drug_b = torch.randn(2, 4, 8)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    output = model(drug_a, drug_b, mask, mask)
    reverse = model(drug_b, drug_a, mask, mask)
    assert output.organ_logits.shape == (2, 2)
    assert output.specific_logits.shape == (2, 3)
    assert torch.allclose(output.organ_logits, reverse.organ_logits, atol=1e-6, rtol=0.0)
    assert torch.allclose(output.specific_logits, reverse.specific_logits, atol=1e-6, rtol=0.0)


def test_runner_has_no_dead_asymmetric_advanced_architecture():
    source = Path(__file__).resolve().parents[2] / "run_experiment.py"
    text = source.read_text()
    assert "class DummyEncoder" not in text
    assert "class AdvancedPolypharmacyModel" not in text
    assert "torch.cat([pooled_a, pooled_b]" not in text


def test_multimodal_teacher_and_student_factory_dispatch():
    from src.models.multimodal_teacher_student import MultiModalTeacher, DistilledPairStudent

    teacher_config = {
        "model_type": "multimodal_teacher",
        "morgan_dim": 8,
        "molformer_dim": 6,
        "mpnn_dim": 7,
        "kg_dim": 5,
        "molformer_token_count": 3,
        "mpnn_token_count": 4,
        "kg_token_count": 2,
        "hidden_dim": 16,
        "num_organ": 2,
        "num_specific": 4,
        "num_heads": 4,
    }
    student_config = {
        "model_type": "distilled_pair_student",
        "token_dim": 12,
        "token_count": 5,
        "hidden_dim": 16,
        "num_organ": 2,
        "num_specific": 4,
        "num_heads": 4,
    }
    assert isinstance(create_model(teacher_config, num_labels=4), MultiModalTeacher)
    assert isinstance(create_model(student_config, num_labels=4), DistilledPairStudent)
