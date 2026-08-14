import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from src.training.engine import verify_manifest


def _fixture_sources(root: Path) -> tuple[Path, Path, Path]:
    drugs = pd.DataFrame(
        {
            "drugbank_id": [f"D{i:02d}" for i in range(1, 21)],
            "pubchem_compound_id": list(range(1001, 1021)),
        }
    )
    drugs_path = root / "drugs.csv"
    drugs.to_csv(drugs_path, index=False)
    side_effects_path = root / "side_effects.csv"
    pd.DataFrame(
        {
            "umls_cui_from_meddra": ["C001", "C002", "C999"],
            "side_effect_name": ["common", "rare", "oov"],
        }
    ).to_csv(side_effects_path, index=False)
    rows = []
    for i in range(1, 21):
        for j in range(i + 1, 21):
            if (i + j) % 4 != 1:
                continue
            rows.append([f"CID{i + 1000}", f"CID{j + 1000}", "C001", "common"])
            if (i + j) % 5 == 0:
                rows.append([f"CID{i + 1000}", f"CID{j + 1000}", "C002", "rare"])
    biosnap_path = root / "biosnap.csv"
    pd.DataFrame(rows, columns=["STITCH 1", "STITCH 2", "Polypharmacy Side Effect", "Side Effect Name"]).to_csv(biosnap_path, index=False)
    return drugs_path, side_effects_path, biosnap_path


def _config(root: Path, output_root: Path, **overrides):
    drugs, side_effects, biosnap = _fixture_sources(root)
    config = {
        "schema_version": 1,
        "benchmark_name": "tiny_benchmark",
        "seeds": [7, 11],
        "top_k_labels": 1,
        "paths": {
            "drugs_master": str(drugs),
            "side_effects": str(side_effects),
            "biosnap_ddi": str(biosnap),
        },
        "partitions": {
            "warm_pair": {"train_fraction": 0.5, "validation_fraction": 0.25, "test_fraction": 0.25},
            "cold_1": {"train_drug_fraction": 0.5, "validation_new_drug_fraction": 0.25, "test_new_drug_fraction": 0.25},
            "cold_2": {"train_drug_fraction": 0.5, "validation_new_drug_fraction": 0.25, "test_new_drug_fraction": 0.25},
        },
        "sampling": {
            "controls_per_positive_pair": 1,
            "primekg_degree_bins": 10,
            "molecular_similarity_bins": 5,
            "maximum_attempts_per_control": 1000,
            "pu_prior_multipliers": [1.0],
        },
        "bootstrap": {"resamples": 10, "confidence_level": 0.95, "seed": 1},
    }
    config.update(overrides)
    path = root / "benchmark.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_build_benchmark_rejects_unknown_config_key(tmp_path):
    from build_benchmark import build_benchmarks

    config_path = _config(tmp_path, tmp_path / "out")
    payload = yaml.safe_load(config_path.read_text())
    payload["unexpected"] = True
    config_path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="unknown"):
        build_benchmarks(config_path, tmp_path / "out")
    assert not list((tmp_path / "out").rglob("manifest.json"))


def test_builds_all_scenarios_and_seeds_with_verified_manifests(tmp_path):
    from build_benchmark import build_benchmarks

    config_path = _config(tmp_path, tmp_path / "out")
    output = build_benchmarks(config_path, tmp_path / "out")
    assert len(output) == 6
    for manifest_path in sorted((tmp_path / "out").rglob("manifest.json")):
        manifest = verify_manifest(str(manifest_path))
        assert manifest["benchmark_id"] == "tiny_benchmark"
        assert manifest["scenario"] in {"warm_pair", "cold_1", "cold_2"}
        assert (manifest_path.parent / manifest["pairs_path"]).exists()
        assert (manifest_path.parent / manifest["triples_path"]).exists()
        labels = json.loads((manifest_path.parent / manifest["labels_path"]).read_text())
        assert [label["index"] for label in labels["labels"]] == list(range(len(labels["labels"])))
        pairs = pd.read_parquet(manifest_path.parent / manifest["pairs_path"])
        assert pairs.columns.tolist() == ["pair_id", "drug_a", "drug_b", "split", "scenario", "observation_status", "source_positive_pair_id"]
        assert pairs["pair_id"].is_unique
        assert set(pairs["observation_status"]) <= {"observed_positive", "sampled_unlabeled"}
        split_drugs = {
            split: set(pairs.loc[pairs.split == split, "drug_a"]) | set(pairs.loc[pairs.split == split, "drug_b"])
            for split in ("train", "validation", "test")
        }
        for split in ("train", "validation", "test"):
            frame = pairs[pairs.split == split]
            assert len(frame[frame.observation_status == "sampled_unlabeled"]) == len(frame[frame.observation_status == "observed_positive"])
        if manifest["scenario"] == "warm_pair":
            assert split_drugs["validation"].issubset(split_drugs["train"])
            assert split_drugs["test"].issubset(split_drugs["train"])
        elif manifest["scenario"] == "cold_1":
            validation_new = split_drugs["validation"] - split_drugs["train"]
            test_new = split_drugs["test"] - split_drugs["train"]
            assert not test_new.intersection(split_drugs["train"] | split_drugs["validation"] - validation_new)
            assert all(((a in split_drugs["train"] and b in validation_new) or (b in split_drugs["train"] and a in validation_new) or (a in split_drugs["train"] and b in split_drugs["train"])) for a, b in pairs.loc[pairs.split == "validation", ["drug_a", "drug_b"]].itertuples(index=False, name=None))
            assert all(((a in split_drugs["train"] and b in test_new) or (b in split_drugs["train"] and a in test_new) or (a in split_drugs["train"] and b in split_drugs["train"])) for a, b in pairs.loc[pairs.split == "test", ["drug_a", "drug_b"]].itertuples(index=False, name=None))
        else:
            assert all(a in split_drugs["validation"] and b in split_drugs["validation"] for a, b in pairs.loc[pairs.split == "validation", ["drug_a", "drug_b"]].itertuples(index=False, name=None))
            assert all(a in split_drugs["test"] and b in split_drugs["test"] for a, b in pairs.loc[pairs.split == "test", ["drug_a", "drug_b"]].itertuples(index=False, name=None))


def test_build_is_deterministic_and_controls_have_new_ids(tmp_path):
    from build_benchmark import build_benchmarks

    config_path = _config(tmp_path, tmp_path / "out")
    first = build_benchmarks(config_path, tmp_path / "out")
    manifests_before = {str(path.relative_to(tmp_path / "out")): verify_manifest(str(path))["manifest_hash"] for path in (tmp_path / "out").rglob("manifest.json")}
    second = build_benchmarks(config_path, tmp_path / "out")
    manifests_after = {str(path.relative_to(tmp_path / "out")): verify_manifest(str(path))["manifest_hash"] for path in (tmp_path / "out").rglob("manifest.json")}
    assert first and second and manifests_before == manifests_after
    for path in (tmp_path / "out").rglob("manifest.json"):
        pairs = pd.read_parquet(path.parent / "pairs.parquet")
        positives = set(pairs.loc[pairs.observation_status == "observed_positive", "pair_id"])
        controls = pairs.loc[pairs.observation_status == "sampled_unlabeled"]
        assert not set(controls.pair_id).intersection(positives)
        assert controls.source_positive_pair_id.notna().all()


def test_missing_source_writes_no_manifest(tmp_path):
    from build_benchmark import build_benchmarks

    config_path = _config(tmp_path, tmp_path / "out")
    payload = yaml.safe_load(config_path.read_text())
    payload["paths"]["biosnap_ddi"] = str(tmp_path / "missing.csv")
    config_path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="does not exist"):
        build_benchmarks(config_path, tmp_path / "out")
    assert not list((tmp_path / "out").rglob("manifest.json"))


def test_smiles_similarity_uses_rdkit_tanimoto_and_manifest_counts(tmp_path):
    from build_benchmark import _build_similarity_map, build_benchmarks

    drugs, side_effects, biosnap = _fixture_sources(tmp_path)
    drug_frame = pd.read_csv(drugs)
    drug_frame["smiles"] = ["C", "CC"] + ["CCC"] * 18
    drug_frame.to_csv(drugs, index=False)
    pair_frame = pd.DataFrame({"drug_a": ["D01"], "drug_b": ["D02"]})
    similarities, status = _build_similarity_map(drug_frame, pair_frame)
    expected = DataStructs.TanimotoSimilarity(
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles("C"), 2, nBits=2048),
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles("CC"), 2, nBits=2048),
    )
    assert status == "available_smiles"
    assert similarities[("D01", "D02")] == pytest.approx(expected)

    config_path = _config(tmp_path, tmp_path / "out")
    manifests = build_benchmarks(config_path, tmp_path / "out")
    manifest = verify_manifest(str(manifests[0]))
    assert {"raw_edges", "mapped_triples", "rejected_unmapped_drug", "rejected_invalid_cui", "rejected_self_pair", "rejected_duplicates", "final_canonical_triples"} <= set(manifest["input_counts"])
    assert {"positive_pairs", "unlabeled_controls", "unique_drugs"} <= set(manifest["split_counts"]["train"])
    assert {"train", "validation_new", "test_new"} <= set(manifest["drug_set_hashes"])
    assert manifest["sampler_diagnostics"]["relaxations"] == 0


def test_build_cli_subprocess_emits_verified_manifest(tmp_path):
    import subprocess

    config_path = _config(tmp_path, tmp_path / "out")
    subprocess.run([
        ".venv/bin/python", "build_benchmark.py", "--config", str(config_path),
        "--output-root", str(tmp_path / "out"),
    ], check=True)
    manifests = sorted((tmp_path / "out").rglob("manifest.json"))
    assert len(manifests) == 6
    assert verify_manifest(str(manifests[0]))["manifest_hash"]


def test_available_smiles_controls_match_source_similarity_bin_exactly():
    from src.data.negative_sampling import calculate_similarity_bin, generate_negative_samples

    positive = pd.DataFrame([{"pair_id": "p1", "drug_a": "D1", "drug_b": "D2", "split": "train", "scenario": "warm_pair"}])
    smiles = {("D1", "D2"): 0.8, ("D1", "D3"): 0.8, ("D2", "D3"): 0.2, ("D1", "D4"): 0.8, ("D2", "D4"): 0.2}
    controls = generate_negative_samples(
        positive,
        {("D1", "D2")},
        ["D1", "D2", "D3", "D4"],
        {"D1": 0, "D2": 0, "D3": 0, "D4": 0},
        smiles,
        max_attempts=100,
        enforce_similarity=True,
    )
    row = controls.iloc[0]
    similarity = smiles.get((row.drug_a, row.drug_b), smiles.get((row.drug_b, row.drug_a)))
    assert calculate_similarity_bin(similarity) == calculate_similarity_bin(0.8)
    assert controls.attrs["sampler_diagnostics"]["relaxations"] == 0
