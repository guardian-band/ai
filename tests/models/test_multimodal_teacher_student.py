import torch

from src.models.factory import DualHeadOutput
from src.models.multimodal_teacher_student import (
    DistillationLoss,
    DistilledPairStudent,
    MultiModalTeacher,
)


def _drug_batch(batch_size=2):
    return {
        "morgan": torch.randn(batch_size, 8),
        "molecular_tokens": torch.randn(batch_size, 3, 6),
        "kg_tokens": torch.randn(batch_size, 2, 5),
        "morgan_available": torch.ones(batch_size, dtype=torch.bool),
        "molecular_available": torch.ones(batch_size, dtype=torch.bool),
        "kg_available": torch.ones(batch_size, dtype=torch.bool),
    }


def test_teacher_uses_multiple_tokens_and_is_swap_invariant():
    torch.manual_seed(3)
    model = MultiModalTeacher(
        morgan_dim=8,
        molecular_dim=6,
        kg_dim=5,
        molecular_token_count=3,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
    ).eval()
    drug_a = _drug_batch()
    drug_b = _drug_batch()
    output = model(drug_a, drug_b)
    reverse = model(drug_b, drug_a)
    assert isinstance(output, DualHeadOutput)
    assert output.organ_logits.shape == (2, 2)
    assert output.specific_logits.shape == (2, 4)
    assert torch.allclose(output.organ_logits, reverse.organ_logits, atol=1e-6, rtol=0.0)
    assert torch.allclose(output.specific_logits, reverse.specific_logits, atol=1e-6, rtol=0.0)

    changed = {key: value.clone() for key, value in drug_a.items()}
    changed["molecular_tokens"][:, 1, :] += 10.0
    changed_output = model(changed, drug_b)
    assert not torch.allclose(output.specific_logits, changed_output.specific_logits)


def test_teacher_handles_all_missing_modalities_with_finite_logits():
    model = MultiModalTeacher(
        morgan_dim=8,
        molecular_dim=6,
        kg_dim=5,
        molecular_token_count=3,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
    ).eval()
    drug = _drug_batch()
    for key in ("morgan_available", "molecular_available", "kg_available"):
        drug[key].fill_(False)
    output = model(drug, drug)
    assert torch.isfinite(output.organ_logits).all()
    assert torch.isfinite(output.specific_logits).all()


def test_teacher_cache_encoding_matches_student_schema_and_uses_modalities():
    torch.manual_seed(31)
    teacher = MultiModalTeacher(
        morgan_dim=8,
        molecular_dim=6,
        kg_dim=5,
        molecular_token_count=3,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
        cache_token_count=5,
        cache_token_dim=12,
    ).eval()
    student = DistilledPairStudent(12, 5, 16, 2, 4, num_heads=4).eval()
    drug = _drug_batch()
    first = teacher.encode_drug_for_cache(drug)
    assert first.shape == (2, 5, 12)
    assert torch.isfinite(first).all()
    assert torch.allclose(first, teacher.encode_drug_for_cache(drug), atol=0.0, rtol=0.0)
    assert student(
        first,
        first,
    ).specific_logits.shape == (2, 4)
    teacher.train()
    teacher(_drug_batch(), _drug_batch()).specific_logits.sum().backward()
    assert teacher.cache_queries.grad is not None
    assert teacher.cache_output_projection[1].weight.grad is not None
    teacher.eval()
    for modality, field in (
        ("morgan", "morgan_available"),
        ("molecular_tokens", "molecular_available"),
        ("kg_tokens", "kg_available"),
    ):
        changed = {key: value.clone() for key, value in drug.items()}
        changed[modality] = changed[modality] + 2.0
        changed_output = teacher.encode_drug_for_cache(changed)
        assert not torch.allclose(first, changed_output)
        missing = {key: value.clone() for key, value in drug.items()}
        missing[field].fill_(False)
        missing_output = teacher.encode_drug_for_cache(missing)
        assert torch.isfinite(missing_output).all()


def test_student_is_swap_invariant_and_has_fixed_token_contract():
    torch.manual_seed(4)
    model = DistilledPairStudent(
        token_dim=12,
        token_count=5,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
    ).eval()
    a = torch.randn(2, 5, 12)
    b = torch.randn(2, 5, 12)
    mask_a = torch.ones(2, 5, dtype=torch.bool)
    mask_b = torch.ones(2, 5, dtype=torch.bool)
    output = model(a, b, mask_a, mask_b)
    reverse = model(b, a, mask_b, mask_a)
    assert output.organ_logits.shape == (2, 2)
    assert output.specific_logits.shape == (2, 4)
    assert torch.allclose(output.organ_logits, reverse.organ_logits, atol=1e-6, rtol=0.0)
    assert torch.allclose(output.specific_logits, reverse.specific_logits, atol=1e-6, rtol=0.0)


def test_distillation_detaches_teacher_and_trains_student():
    torch.manual_seed(5)
    teacher = MultiModalTeacher(8, 6, 5, 3, 2, 16, 2, 4, num_heads=4)
    student = DistilledPairStudent(12, 5, 16, 2, 4, num_heads=4)
    teacher_output = teacher(_drug_batch(), _drug_batch())
    student_output = student(torch.randn(2, 5, 12), torch.randn(2, 5, 12))
    loss_fn = DistillationLoss(
        temperature=2.0,
        supervised_weight=2.5,
        distillation_weight=0.25,
        hierarchy_weight=0.0,
    )
    assert loss_fn.supervised_weight == 2.5
    assert loss_fn.distillation_weight == 0.25
    loss = loss_fn(
        student_output,
        teacher_output,
        organ_targets=torch.zeros(2, 2),
        specific_targets=torch.ones(2, 4),
    )
    loss.backward()
    assert loss.isfinite()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
