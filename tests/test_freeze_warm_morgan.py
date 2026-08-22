import json
from pathlib import Path

from scripts.freeze_warm_morgan import freeze_paired_morgan


def test_freeze_paired_morgan_records_validation_only_provenance(tmp_path: Path):
    teacher_release = tmp_path / "teacher"
    teacher_run = teacher_release / "run"
    teacher_run.mkdir(parents=True)
    (teacher_run / "checkpoint_baseline.pt").write_bytes(b"paired-morgan")
    (teacher_run / "teacher_validation_selection.json").write_text(
        json.dumps({"baseline": {"macro_ap": 0.49}})
    )
    (teacher_release / "frozen_teacher_manifest.json").write_text(
        json.dumps(
            {
                "winner": {
                    "scenario": "warm_pair",
                    "seed": 42,
                    "validation_macro_ap": 0.508,
                }
            }
        )
    )

    result = freeze_paired_morgan(teacher_release, tmp_path / "morgan")

    assert result["validation_macro_ap"] == 0.49
    assert result["source_teacher_validation_macro_ap"] == 0.508
    assert result["test_metrics_used_for_selection"] is False
    assert result["source_baseline_checkpoint_sha256"] == result["frozen_checkpoint_sha256"]
    assert (tmp_path / "morgan" / "checkpoint_best.pt").read_bytes() == b"paired-morgan"
