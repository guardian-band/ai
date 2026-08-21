import pytest
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
        "molformer_tokens": torch.randn(batch_size, 3, 6),
        "mpnn_tokens": torch.randn(batch_size, 4, 7),
        "kg_tokens": torch.randn(batch_size, 2, 5),
        "morgan_available": torch.ones(batch_size, dtype=torch.bool),
        "molformer_available": torch.ones(batch_size, dtype=torch.bool),
        "mpnn_available": torch.ones(batch_size, dtype=torch.bool),
        "kg_available": torch.ones(batch_size, dtype=torch.bool),
    }


def test_teacher_consumes_distinct_molformer_mpnn_and_hgt_modalities():
    model = MultiModalTeacher(
        morgan_dim=8,
        molformer_dim=6,
        mpnn_dim=7,
        kg_dim=5,
        molformer_token_count=3,
        mpnn_token_count=4,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
    ).eval()
    drug = {
        "morgan": torch.randn(2, 8),
        "molformer_tokens": torch.randn(2, 3, 6),
        "mpnn_tokens": torch.randn(2, 4, 7),
        "kg_tokens": torch.randn(2, 2, 5),
    }
    with torch.inference_mode():
        baseline = model(drug, drug).specific_logits
        changed = dict(drug)
        changed["mpnn_tokens"] = drug["mpnn_tokens"] + 5.0
        updated = model(changed, drug).specific_logits
    assert baseline.shape == (2, 4)
    assert not torch.allclose(baseline, updated)


def test_teacher_uses_multiple_tokens_and_is_swap_invariant():
    torch.manual_seed(3)
    model = MultiModalTeacher(
        morgan_dim=8,
        molformer_dim=6,
        mpnn_dim=7,
        kg_dim=5,
        molformer_token_count=3,
        mpnn_token_count=4,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
        modality_summary_token_count=2,
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
    changed["molformer_tokens"][:, 1, :] += 10.0
    changed_output = model(changed, drug_b)
    assert not torch.allclose(output.specific_logits, changed_output.specific_logits)


def test_teacher_handles_all_missing_modalities_with_finite_logits():
    model = MultiModalTeacher(
        morgan_dim=8,
        molformer_dim=6,
        mpnn_dim=7,
        kg_dim=5,
        molformer_token_count=3,
        mpnn_token_count=4,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
        modality_summary_token_count=2,
    ).eval()
    drug = _drug_batch()
    for key in (
        "morgan_available",
        "molformer_available",
        "mpnn_available",
        "kg_available",
    ):
        drug[key].fill_(False)
    output = model(drug, drug)
    assert torch.isfinite(output.organ_logits).all()
    assert torch.isfinite(output.specific_logits).all()


def test_teacher_enabled_modalities_disable_signal_parameters_and_diagnostics():
    torch.manual_seed(17)
    model = MultiModalTeacher(
        morgan_dim=8,
        molformer_dim=6,
        mpnn_dim=7,
        kg_dim=5,
        molformer_token_count=3,
        mpnn_token_count=4,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
        enabled_modalities=("molformer",),
    )
    model.eval()
    drug_a = _drug_batch()
    drug_b = _drug_batch()
    baseline = model(drug_a, drug_b)
    changed = {key: value.clone() for key, value in drug_a.items()}
    changed["mpnn_tokens"] += 100.0
    changed["kg_tokens"] -= 100.0
    updated = model(changed, drug_b)
    assert torch.equal(baseline.specific_logits, updated.specific_logits)
    assert set(model.validation_variants(drug_a, drug_b)) == {"baseline", "molformer", "combined"}
    diagnostics = model.diagnostics()
    assert set(diagnostics["gates"]) == {"molformer"}
    assert set(diagnostics["coverage"]) == {"molformer"}
    loss = updated.specific_logits.square().mean()
    loss.backward()
    disabled_parameter_names = (
        "mpnn",
        "kg",
    )
    for name, parameter in model.named_parameters():
        if any(f"{modality}_" in name or f".{modality}." in name for modality in disabled_parameter_names):
            assert parameter.requires_grad is False
            assert parameter.grad is None


def test_teacher_morgan_only_has_no_auxiliary_variants_or_gradients():
    model = MultiModalTeacher(
        morgan_dim=8,
        molformer_dim=6,
        mpnn_dim=7,
        kg_dim=5,
        molformer_token_count=3,
        mpnn_token_count=4,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
        enabled_modalities=[],
    )
    assert model.enabled_modalities == ()
    assert model.validation_variants(_drug_batch(), _drug_batch()).keys() == {"baseline"}
    assert model.auxiliary_parameters() == []


def test_teacher_cache_encoding_matches_student_schema_and_uses_modalities():
    torch.manual_seed(31)
    teacher = MultiModalTeacher(
        morgan_dim=8,
        molformer_dim=6,
        mpnn_dim=7,
        kg_dim=5,
        molformer_token_count=3,
        mpnn_token_count=4,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
        cache_token_count=5,
        cache_token_dim=12,
        modality_summary_token_count=2,
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
        ("molformer_tokens", "molformer_available"),
        ("mpnn_tokens", "mpnn_available"),
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
    teacher = MultiModalTeacher(
        morgan_dim=8,
        molformer_dim=6,
        mpnn_dim=7,
        kg_dim=5,
        molformer_token_count=3,
        mpnn_token_count=4,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
    )
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


def test_zero_distillation_weight_is_exactly_supervised_loss():
    torch.manual_seed(6)
    student_output = DualHeadOutput(torch.randn(2, 2), torch.randn(2, 4))
    teacher_output = DualHeadOutput(torch.randn(2, 2), torch.randn(2, 4))
    organ_targets = torch.zeros(2, 2)
    specific_targets = torch.ones(2, 4)
    loss_fn = DistillationLoss(
        temperature=2.0,
        supervised_weight=1.0,
        distillation_weight=0.0,
        hierarchy_weight=0.0,
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        student_output.organ_logits, organ_targets
    ) + torch.nn.functional.binary_cross_entropy_with_logits(
        student_output.specific_logits, specific_targets
    )
    assert torch.equal(
        loss_fn(
            student_output,
            teacher_output,
            organ_targets=organ_targets,
            specific_targets=specific_targets,
        ),
        expected,
    )


def test_teacher_resamples_each_auxiliary_modality_to_equal_summary_tokens_and_initializes_gates_near_zero():
    model = MultiModalTeacher(
        morgan_dim=8,
        molformer_dim=6,
        mpnn_dim=7,
        kg_dim=5,
        molformer_token_count=3,
        mpnn_token_count=4,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
        modality_summary_token_count=2,
    )

    summaries = model.encode_auxiliary_modalities(_drug_batch())

    assert {value.shape[1] for value in summaries.values()} == {2}
    assert set(summaries) == {"molformer", "mpnn", "kg"}
    assert all(
        torch.allclose(value, torch.full_like(value, -4.0))
        for value in model.modality_gate_logits.values()
    )
    assert all(
        torch.sigmoid(value).max().item() < 0.02
        for value in model.modality_gate_logits.values()
    )


def test_teacher_accepts_a_less_restrictive_gate_initialization():
    model = MultiModalTeacher(
        morgan_dim=8, molformer_dim=6, mpnn_dim=7, kg_dim=5,
        molformer_token_count=3, mpnn_token_count=4, kg_token_count=2,
        hidden_dim=16, num_organ=2, num_specific=4, num_heads=4,
        enabled_modalities=("mpnn",), modality_gate_init_logit=-2.0,
    )
    assert torch.allclose(
        model.modality_gate_logits["mpnn"],
        torch.full((2,), -2.0),
    )


def test_teacher_zero_auxiliary_gates_equal_morgan_baseline_exactly_and_swap_invariant():
    torch.manual_seed(101)
    model = MultiModalTeacher(
        morgan_dim=8,
        molformer_dim=6,
        mpnn_dim=7,
        kg_dim=5,
        molformer_token_count=3,
        mpnn_token_count=4,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
        modality_summary_token_count=2,
    ).eval()
    for value in model.modality_gate_logits.values():
        value.data.fill_(-float("inf"))
    drug_a, drug_b = _drug_batch(), _drug_batch()

    output = model(drug_a, drug_b)
    baseline = model.morgan_baseline_logits(drug_a, drug_b)
    reverse = model(drug_b, drug_a)

    assert torch.equal(output.organ_logits, baseline.organ_logits)
    assert torch.equal(output.specific_logits, baseline.specific_logits)
    assert torch.equal(output.organ_logits, reverse.organ_logits)
    assert torch.equal(output.specific_logits, reverse.specific_logits)


def test_teacher_initial_logits_are_close_to_morgan_baseline():
    torch.manual_seed(103)
    model = MultiModalTeacher(
        morgan_dim=8,
        molformer_dim=6,
        mpnn_dim=7,
        kg_dim=5,
        molformer_token_count=3,
        mpnn_token_count=4,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
        modality_summary_token_count=2,
    ).eval()
    drug_a, drug_b = _drug_batch(), _drug_batch()

    output = model(drug_a, drug_b)
    baseline = model.morgan_baseline_logits(drug_a, drug_b)

    assert (output.organ_logits - baseline.organ_logits).abs().max().item() < 0.1
    assert (output.specific_logits - baseline.specific_logits).abs().max().item() < 0.1


def test_teacher_single_modality_variants_are_isolated_from_other_modalities():
    torch.manual_seed(104)
    model = MultiModalTeacher(
        morgan_dim=8,
        molformer_dim=6,
        mpnn_dim=7,
        kg_dim=5,
        molformer_token_count=3,
        mpnn_token_count=4,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
        modality_summary_token_count=2,
    ).eval()
    drug_a, drug_b = _drug_batch(), _drug_batch()
    changed_a = {key: value.clone() for key, value in drug_a.items()}
    changed_b = {key: value.clone() for key, value in drug_b.items()}
    changed_a["kg_tokens"] += 10.0
    changed_b["kg_tokens"] += 10.0

    original = model.validation_variants(drug_a, drug_b)
    changed = model.validation_variants(changed_a, changed_b)

    for name in ("baseline", "molformer", "mpnn"):
        assert torch.equal(original[name].specific_logits, changed[name].specific_logits)
    assert not torch.equal(original["kg"].specific_logits, changed["kg"].specific_logits)
    assert not torch.equal(original["combined"].specific_logits, changed["combined"].specific_logits)

    model.modality_gate_logits["kg"].data.fill_(-float("inf"))
    gated_original = model(drug_a, drug_b)
    gated_changed = model(changed_a, changed_b)
    assert torch.equal(gated_original.specific_logits, gated_changed.specific_logits)


def test_teacher_exposes_gate_and_modality_coverage_diagnostics_and_gradients_reach_auxiliaries():
    torch.manual_seed(102)
    model = MultiModalTeacher(
        morgan_dim=8,
        molformer_dim=6,
        mpnn_dim=7,
        kg_dim=5,
        molformer_token_count=3,
        mpnn_token_count=4,
        kg_token_count=2,
        hidden_dim=16,
        num_organ=2,
        num_specific=4,
        num_heads=4,
        modality_summary_token_count=2,
    )
    output = model(_drug_batch(), _drug_batch())
    (output.organ_logits.square().mean() + output.specific_logits.square().mean()).backward()
    diagnostics = model.diagnostics()

    assert set(diagnostics["gates"]) == {"molformer", "mpnn", "kg"}
    assert set(diagnostics["coverage"]) == {"molformer", "mpnn", "kg"}
    assert all(not value.requires_grad for value in diagnostics["gates"].values())
    assert all(
        model.modality_projections[name][0].weight.grad is not None
        for name in ("molformer", "mpnn", "kg")
    )


def test_teacher_validates_modality_dropout():
    with pytest.raises(ValueError, match="modality_dropout"):
        MultiModalTeacher(
            morgan_dim=8,
            molformer_dim=6,
            mpnn_dim=7,
            kg_dim=5,
            molformer_token_count=3,
            mpnn_token_count=4,
            kg_token_count=2,
            hidden_dim=16,
            num_organ=2,
            num_specific=4,
            num_heads=4,
            modality_dropout=1.0,
        )
