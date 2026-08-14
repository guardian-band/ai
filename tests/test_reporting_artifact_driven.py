import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluation.reporting import (
    ReportingValidationError,
    canonicalize_aggregation_summary,
    emit_summary_csv,
)


def _cohort(model, scenario, offset=0.0):
    metrics = {}
    for name, value in {
        "macro_ap": 0.512 + offset,
        "micro_ap": 0.623 + offset,
        "macro_auroc": 0.734 + offset,
        "precision_at_k": 0.145 + offset,
        "recall_at_k": 0.256 + offset,
        "ndcg_at_k": 0.367 + offset,
    }.items():
        metrics[name] = {
            "mean": value,
            "std": 0.012,
            "lower_ci": value - 0.02,
            "upper_ci": value + 0.02,
        }
    return {
        "model_type": model,
        "benchmark_id": "fixture_bench",
        "scenario": scenario,
        "seeds": [11, 22],
        "run_ids": [f"{model}__{scenario}__seed_11__hash", f"{model}__{scenario}__seed_22__hash"],
        "metrics": metrics,
    }


def _summary():
    scenarios = ["warm_pair", "cold_1", "cold_2"]
    cohorts = [_cohort("logistic", scenario, index * 0.01) for index, scenario in enumerate(scenarios)]
    return {
        "schema_version": 1,
        "benchmark_config": "configs/benchmark.yaml",
        "expected_seeds": [11, 22],
        "expected_scenarios": scenarios,
        "cohorts": cohorts,
        "champion_rule": "highest mean macro_ap averaged across required scenarios, then higher mean micro_ap, then lexicographically smaller model_type",
        "champion": {"model_type": "logistic", "mean_macro_ap": 0.522, "mean_micro_ap": 0.633},
    }


def test_summary_validation_and_csv_are_artifact_driven(tmp_path):
    summary = canonicalize_aggregation_summary(_summary())
    csv_path = tmp_path / "summary.csv"
    emit_summary_csv(summary, csv_path)
    text = csv_path.read_text()
    assert "logistic,fixture_bench,warm_pair" in text
    assert "0.512" in text


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda s: s.update({"schema_version": 99}), "schema_version"),
        (lambda s: s.update({"expected_seeds": [11]}), "seeds"),
        (lambda s: s["cohorts"].pop(), "scenarios"),
        (lambda s: s["champion"].update({"model_type": "prevalence"}), "champion"),
        (lambda s: s["cohorts"][0]["metrics"]["macro_ap"].update({"mean": None}), "macro_ap"),
    ],
)
def test_malformed_or_incomplete_summary_is_rejected(mutator, message):
    summary = _summary()
    mutator(summary)
    with pytest.raises(ReportingValidationError, match=message):
        canonicalize_aggregation_summary(summary)


def test_pdf_cli_uses_summary_and_rejects_missing_input(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary()))
    output = tmp_path / "report.pdf"
    result = subprocess.run(
        [sys.executable, "generate_advanced_pdf_report.py", "--summary", str(summary_path), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists() and output.stat().st_size > 0

    missing_output = tmp_path / "missing.pdf"
    result = subprocess.run(
        [sys.executable, "generate_advanced_pdf_report.py", "--summary", str(tmp_path / "no.json"), "--output", str(missing_output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not missing_output.exists()


def test_pdf_contains_fixture_metrics_and_safe_claims(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary()))
    output = tmp_path / "report.pdf"
    subprocess.run(
        [sys.executable, "generate_advanced_pdf_report.py", "--summary", str(summary_path), "--output", str(output)],
        check=True,
    )
    text = subprocess.check_output(["pdftotext", str(output), "-"]).decode()
    assert "0.512000" in text
    assert "logistic" in text
    assert "warm_pair" in text and "cold_1" in text and "cold_2" in text
    assert "benchmark-selected champion under the stated rule" in text.lower()
    for forbidden in ("state-of-the-art", "actionable", "leak-free", "89.26", "0.9218"):
        assert forbidden.lower() not in text.lower()


def test_report_source_has_no_placeholder_or_fabricated_paths():
    source = Path("generate_advanced_pdf_report.py").read_text()
    reporting = Path("src/evaluation/reporting.py").read_text()
    for text in (source, reporting):
        assert "placeholder" not in text.lower()
        assert "aggregated_results.csv" not in text
        assert "89.26" not in text
        assert "state-of-the-art" not in text.lower()
