import json
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.run_advanced_colab_sweep import (
    DEFAULT_SEEDS,
    SCENARIOS,
    ensure_manifest_hierarchy,
    render_config,
)


def test_v2_template_renders_unique_run_identity_and_explicit_hgt_seed(tmp_path):
    template = Path("configs/advanced_pipeline.v2.template.yaml")
    identities = set()
    for scenario in SCENARIOS:
        for seed in DEFAULT_SEEDS:
            destination = tmp_path / f"{scenario}_{seed}.yaml"
            rendered = render_config(template, destination, scenario, seed)
            loaded = yaml.safe_load(destination.read_text())
            assert loaded == rendered
            hgt = rendered["stages"]["hgt"]
            assert hgt[hgt.index("--seed") + 1] == str(seed)
            manifest = hgt[hgt.index("--manifest") + 1]
            assert manifest == f"artifacts/benchmarks/polypharmacy_v2/{scenario}/seed_{seed}/manifest.json"
            identities.add((scenario, seed, manifest))
    assert len(identities) == 15


def test_v2_template_contains_only_per_run_stages():
    payload = yaml.safe_load(Path("configs/advanced_pipeline.v2.template.yaml").read_text())
    assert list(payload["stages"]) == [
        "hgt", "assemble", "teacher_config", "teacher", "cache", "student_config", "student"
    ]


def test_sweep_script_can_import_project_modules_when_executed_by_path(tmp_path):
    (tmp_path / "labels.json").write_text(
        json.dumps({"labels": [{"cui": "C0004144", "name": "Atelectasis"}]})
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"labels_path": "labels.json"}))
    (tmp_path / "hierarchy.json").write_text(
        json.dumps(
            {
                "organ_order": ["Respiratory & Thoracic (Lungs)"],
                "mappings": [],
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy,sys; from pathlib import Path; "
                "scope=runpy.run_path('scripts/run_advanced_colab_sweep.py'); "
                "scope['ensure_manifest_hierarchy'](Path(sys.argv[1]),Path(sys.argv[2]))"
            ),
            str(tmp_path / "hierarchy.json"),
            str(tmp_path / "manifest.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_manifest_hierarchy_is_completed_deterministically(tmp_path):
    labels = [
        {"cui": "C0004144", "name": "Atelectasis"},
        {"cui": "C0024117", "name": "Chronic obstructive pulmonary disease"},
        {"cui": "C0149871", "name": "Deep vein thrombosis"},
        {"cui": "C0428977", "name": "Bradycardia"},
    ]
    (tmp_path / "labels.json").write_text(json.dumps({"labels": labels}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"labels_path": "labels.json"}))
    hierarchy = tmp_path / "hierarchy.json"
    hierarchy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "keyword_fallback_v1",
                "organ_order": [
                    "Cardiovascular & Vascular",
                    "Respiratory & Thoracic (Lungs)",
                    "General Disorders & Systemic",
                ],
                "mappings": [],
            }
        )
    )

    assert ensure_manifest_hierarchy(hierarchy, manifest) == [item["cui"] for item in labels]
    payload = json.loads(hierarchy.read_text())
    resolved = {item["specific_cui"]: item["organ_index"] for item in payload["mappings"]}
    assert resolved == {
        "C0004144": 1,
        "C0024117": 1,
        "C0149871": 0,
        "C0428977": 0,
    }
    assert ensure_manifest_hierarchy(hierarchy, manifest) == []
