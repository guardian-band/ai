import pytest

from scripts.freeze_warm_teacher import MODEL_ID, select_warm_teacher


def _candidate(seed, score, *, mode="fused", scenario="warm_pair"):
    return {
        "seed": seed,
        "validation_macro_ap": score,
        "selected_mode": mode,
        "scenario": scenario,
        "run_model_id": MODEL_ID,
        "checkpoint_sha256": str(seed),
    }


def test_selects_only_fused_warm_candidate_by_validation():
    selected = select_warm_teacher(
        [
            _candidate(42, 0.49),
            _candidate(101, 0.51, mode="baseline"),
            _candidate(2024, 0.50),
            _candidate(7, 0.90, scenario="cold_1"),
        ]
    )
    assert selected["seed"] == 2024


def test_rejects_cohort_without_fused_warm_candidate():
    with pytest.raises(ValueError, match="selected fused"):
        select_warm_teacher([_candidate(42, 0.5, mode="baseline")])
