from pathlib import Path

import yaml

from scripts.run_advanced_colab_sweep import DEFAULT_SEEDS, SCENARIOS, render_config


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
