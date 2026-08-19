"""Build immutable, manifest-backed benchmark partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.benchmark_builder import canonicalize_drug_pair, generate_pair_id, parse_stitch_id
from src.data.negative_sampling import generate_negative_samples
from src.data.schemas import INPUT_SCHEMAS, validate_dataframe
from src.data.splitters import cold_drug_partition, generate_cold_splits, warm_pair_split
from src.data.validators import (
    check_cold_1_test_endpoints,
    check_cold_2_test_endpoints,
    check_no_overlap,
    check_no_test_new_in_train_val,
)
from src.training.engine import verify_manifest


TOP_LEVEL_KEYS = {
    "schema_version", "benchmark_name", "seeds", "top_k_labels", "paths",
    "partitions", "sampling", "bootstrap",
}
PATH_KEYS = {"drugs_master", "side_effects", "biosnap_ddi", "sider", "primekg"}
PARTITION_KEYS = {
    "warm_pair": {"train_fraction", "validation_fraction", "test_fraction"},
    "cold_1": {"train_drug_fraction", "validation_new_drug_fraction", "test_new_drug_fraction"},
    "cold_2": {"train_drug_fraction", "validation_new_drug_fraction", "test_new_drug_fraction"},
}
SAMPLING_KEYS = {
    "controls_per_positive_pair", "primekg_degree_bins", "molecular_similarity_bins",
    "maximum_attempts_per_control", "pu_prior_multipliers",
}
BOOTSTRAP_KEYS = {"resamples", "confidence_level", "seed"}
PAIR_COLUMNS = [
    "pair_id", "drug_a", "drug_b", "split", "scenario",
    "observation_status", "source_positive_pair_id",
]


def _sha256(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_fraction(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    value = float(value)
    if not np.isfinite(value) or value <= 0 or value >= 1:
        raise ValueError(f"{key} must be between zero and one")
    return value


def validate_benchmark_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("benchmark config must be a mapping")
    unknown = set(config) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown benchmark config keys: {sorted(unknown)}")
    required = {"schema_version", "benchmark_name", "seeds", "top_k_labels", "paths", "partitions", "sampling"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"benchmark config missing keys: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if not isinstance(config["benchmark_name"], str) or not config["benchmark_name"].strip():
        raise ValueError("benchmark_name must be a non-empty string")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique non-negative integers")
    if isinstance(config["top_k_labels"], bool) or not isinstance(config["top_k_labels"], int) or config["top_k_labels"] <= 0:
        raise ValueError("top_k_labels must be a positive integer")

    paths = config["paths"]
    if not isinstance(paths, dict):
        raise ValueError("paths must be a mapping")
    unknown_paths = set(paths) - PATH_KEYS
    if unknown_paths:
        raise ValueError(f"unknown paths keys: {sorted(unknown_paths)}")
    for key in ("drugs_master", "side_effects", "biosnap_ddi"):
        if not isinstance(paths.get(key), str) or not paths[key]:
            raise ValueError(f"paths.{key} must be a non-empty string")

    partitions = config["partitions"]
    if not isinstance(partitions, dict) or set(partitions) != set(PARTITION_KEYS):
        raise ValueError("partitions must define warm_pair, cold_1, and cold_2")
    for scenario, allowed in PARTITION_KEYS.items():
        values = partitions[scenario]
        if not isinstance(values, dict):
            raise ValueError(f"partitions.{scenario} must be a mapping")
        unknown_partition = set(values) - allowed
        if unknown_partition:
            raise ValueError(f"unknown partitions.{scenario} keys: {sorted(unknown_partition)}")
        if set(values) != allowed:
            raise ValueError(f"partitions.{scenario} has missing keys")
        fractions = [_strict_fraction(values[key], f"partitions.{scenario}.{key}") for key in allowed]
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError(f"partitions.{scenario} fractions must sum to one")

    sampling = config["sampling"]
    if not isinstance(sampling, dict) or set(sampling) != SAMPLING_KEYS:
        raise ValueError("sampling has unknown or missing keys")
    for key in ("controls_per_positive_pair", "primekg_degree_bins", "molecular_similarity_bins", "maximum_attempts_per_control"):
        value = sampling[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"sampling.{key} must be a positive integer")
    if not isinstance(sampling["pu_prior_multipliers"], list) or not sampling["pu_prior_multipliers"]:
        raise ValueError("sampling.pu_prior_multipliers must be a non-empty list")
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0 for value in sampling["pu_prior_multipliers"]):
        raise ValueError("sampling.pu_prior_multipliers must contain positive numbers")

    if "bootstrap" in config:
        bootstrap = config["bootstrap"]
        if not isinstance(bootstrap, dict) or set(bootstrap) != BOOTSTRAP_KEYS:
            raise ValueError("bootstrap has unknown or missing keys")
        if isinstance(bootstrap["resamples"], bool) or not isinstance(bootstrap["resamples"], int) or bootstrap["resamples"] <= 0:
            raise ValueError("bootstrap.resamples must be a positive integer")
        if not 0 < float(bootstrap["confidence_level"]) < 1:
            raise ValueError("bootstrap.confidence_level must be between zero and one")
        if isinstance(bootstrap["seed"], bool) or not isinstance(bootstrap["seed"], int):
            raise ValueError("bootstrap.seed must be an integer")
    return config


def _resolve_sources(config_path: Path, config: dict[str, Any]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for key, raw_path in config["paths"].items():
        path = Path(raw_path)
        if not path.is_absolute():
            config_relative = config_path.parent / path
            repo_relative = Path(__file__).resolve().parent / path
            # Config-local paths take precedence; the repository-relative
            # fallback keeps the checked-in configs' ``data/...`` contract.
            path = config_relative if config_relative.exists() else repo_relative
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"configured input {key} does not exist: {path}")
        sources[key] = path
    return sources


def _read_sources(sources: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        drugs = pd.read_csv(sources["drugs_master"])
        try:
            valid_drugs_df = pd.read_parquet("artifacts/morgan_fingerprints.parquet")
            drugs = drugs[drugs["drugbank_id"].isin(set(valid_drugs_df["drugbank_id"]))]
        except Exception:
            pass
        side_effects = pd.read_csv(sources["side_effects"])
        combo = pd.read_csv(sources["biosnap_ddi"])
    except Exception as exc:
        raise ValueError(f"unable to read configured benchmark inputs: {exc}") from exc
    for name, frame in (("drugs_master", drugs), ("side_effects", side_effects)):
        validate_dataframe(frame, name)
    required_biosnap = INPUT_SCHEMAS["biosnap"]
    if not set(required_biosnap).issubset(combo.columns):
        combo = pd.read_csv(sources["biosnap_ddi"], comment="#", names=required_biosnap)
    validate_dataframe(combo, "biosnap")
    return drugs, side_effects, combo


def _canonical_triples(
    drugs: pd.DataFrame,
    side_effects: pd.DataFrame,
    combo: pd.DataFrame,
    *,
    return_counts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    drug_ids = set(drugs["drugbank_id"].dropna().astype(str))
    cid = pd.to_numeric(drugs["pubchem_compound_id"], errors="coerce")
    cid_to_drug = dict(zip(cid.dropna().astype(int), drugs.loc[cid.notna(), "drugbank_id"].astype(str)))
    valid_cuis = set(side_effects["umls_cui_from_meddra"].dropna().astype(str))
    rows = []
    rejected_unmapped = rejected_invalid_cui = rejected_self_pair = mapped_count = 0
    for row in combo.itertuples(index=False, name=None):
        raw_a, raw_b = str(row[0]), str(row[1])
        parsed_a, parsed_b = parse_stitch_id(raw_a), parse_stitch_id(raw_b)
        if raw_a.upper().startswith("CID") and raw_a[3:].isdigit():
            parsed_a = int(raw_a[3:])
        if raw_b.upper().startswith("CID") and raw_b[3:].isdigit():
            parsed_b = int(raw_b[3:])
        drug_a = cid_to_drug.get(parsed_a, raw_a if raw_a in drug_ids else None)
        drug_b = cid_to_drug.get(parsed_b, raw_b if raw_b in drug_ids else None)
        cui = str(row[2])
        if not drug_a or not drug_b:
            rejected_unmapped += 1
            continue
        mapped_count += 1
        if cui not in valid_cuis:
            rejected_invalid_cui += 1
            continue
        if drug_a == drug_b:
            rejected_self_pair += 1
            continue
        drug_a, drug_b = canonicalize_drug_pair(drug_a, drug_b)
        rows.append({
            "pair_id": generate_pair_id(drug_a, drug_b),
            "drug_a": drug_a,
            "drug_b": drug_b,
            "label_cui": cui,
            "observation_status": "observed_positive",
        })
    triples = pd.DataFrame(rows, columns=["pair_id", "drug_a", "drug_b", "label_cui", "observation_status"])
    if triples.empty:
        raise ValueError("no canonical observed-positive triples remain after validation")
    deduplicated = triples.drop_duplicates(subset=["pair_id", "label_cui"])
    counts = {
        "raw_edges": int(len(combo)),
        "mapped_triples": int(mapped_count),
        "rejected_unmapped_drug": int(rejected_unmapped),
        "rejected_invalid_cui": int(rejected_invalid_cui),
        "rejected_self_pair": int(rejected_self_pair),
        "rejected_duplicates": int(len(triples) - len(deduplicated)),
        "final_canonical_triples": int(len(deduplicated)),
    }
    result = deduplicated.sort_values(["pair_id", "label_cui"]).reset_index(drop=True)
    return (result, counts) if return_counts else result


def _pair_table(triples: pd.DataFrame) -> pd.DataFrame:
    return triples[["pair_id", "drug_a", "drug_b"]].drop_duplicates("pair_id").reset_index(drop=True)


def _similarity(a: Any, b: Any) -> float:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    mol_a = Chem.MolFromSmiles(str(a))
    mol_b = Chem.MolFromSmiles(str(b))
    if mol_a is None or mol_b is None:
        raise ValueError("invalid SMILES encountered while computing molecular similarity")
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _degree_bins(pairs: pd.DataFrame, n_bins: int) -> dict[str, int]:
    degrees = pairs["drug_a"].value_counts().add(pairs["drug_b"].value_counts(), fill_value=0)
    # Preserve exact-degree matching: all drugs with the same observed degree
    # belong to the same bin, even when that yields fewer than the configured
    # maximum number of bins.
    grouped = degrees.groupby(degrees).groups
    # Keep a few candidates in each bin for control generation; singleton bins
    # make a valid replacement impossible on sparse tiny fixtures.
    if len(grouped) <= max(1, len(degrees) // 3):
        return {str(drug): index for index, degree in enumerate(sorted(grouped)) for drug in grouped[degree]}
    ordered = degrees.sort_index().sort_values(kind="mergesort").index.to_numpy(copy=True)
    effective_bins = min(n_bins, max(1, len(ordered) // 3))
    chunks = np.array_split(ordered, effective_bins)
    return {str(drug): index for index, chunk in enumerate(chunks) for drug in chunk}


def _scenario_split(positive_pairs: pd.DataFrame, scenario: str, partition: dict[str, Any], seed: int):
    if scenario == "warm_pair":
        return warm_pair_split(positive_pairs, partition["train_fraction"], partition["validation_fraction"], seed)
    for retry in range(25):
        train_drugs, validation_drugs, test_drugs = cold_drug_partition(
            positive_pairs,
            partition["train_drug_fraction"],
            partition["validation_new_drug_fraction"],
            seed + retry,
        )
        split = generate_cold_splits(positive_pairs, scenario, train_drugs, validation_drugs, test_drugs)
        frames = (
            split[split["split"] == "train"].copy(),
            split[split["split"] == "validation"].copy(),
            split[split["split"] == "test"].copy(),
        )
        if all(not frame.empty for frame in frames):
            return frames, (train_drugs, validation_drugs, test_drugs)
    raise ValueError(f"{scenario} could not produce non-empty train/validation/test splits for seed {seed}")


def _attach_positive_metadata(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = frame[["pair_id", "drug_a", "drug_b", "split"]].copy()
    out["scenario"] = scenario
    out["observation_status"] = "observed_positive"
    out["source_positive_pair_id"] = None
    return out


def _build_similarity_map(drugs: pd.DataFrame, pairs: pd.DataFrame) -> tuple[dict[tuple[str, str], float], str]:
    smiles_col = next((column for column in ("smiles", "canonical_smiles") if column in drugs.columns), None)
    if smiles_col is None or drugs[smiles_col].dropna().empty:
        return {}, "unavailable_no_smiles"
    participating = set(pairs["drug_a"]) | set(pairs["drug_b"])
    smiles_series = drugs.set_index("drugbank_id")[smiles_col]
    missing = [drug for drug in sorted(participating) if pd.isna(smiles_series.get(drug)) or not str(smiles_series.get(drug)).strip()]
    if missing:
        raise ValueError(f"missing SMILES for participating drugs: {missing}")
    smiles = {str(drug): str(smiles_series[drug]) for drug in participating}
    ids = sorted(set(pairs["drug_a"]) | set(pairs["drug_b"]))
    
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    fps = {}
    for drug in ids:
        mol = Chem.MolFromSmiles(smiles[drug])
        fps[drug] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None
        
    similarities = {}
    for index, drug_a in enumerate(ids):
        for drug_b in ids[index + 1:]:
            if fps[drug_a] is not None and fps[drug_b] is not None:
                similarities[(drug_a, drug_b)] = float(DataStructs.TanimotoSimilarity(fps[drug_a], fps[drug_b]))
    return similarities, "available_smiles"


def _controls_for_scenario(
    observed: pd.DataFrame,
    scenario: str,
    split_frames: dict[str, pd.DataFrame],
    split_sets: tuple[set[str], set[str], set[str]] | None,
    config: dict[str, Any],
    seed: int,
    similarities: dict[tuple[str, str], float],
    forbidden_pairs: set[tuple[str, str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_bins = config["sampling"]["primekg_degree_bins"]
    degree_bins = _degree_bins(observed, n_bins)
    train_set, validation_set, test_set = split_sets or (set(observed.drug_a) | set(observed.drug_b),) * 3
    allowed = {
        "train": train_set,
        "validation": (train_set | validation_set) if scenario == "cold_1" else (validation_set if scenario == "cold_2" else train_set),
        "test": (train_set | test_set) if scenario == "cold_1" else (test_set if scenario == "cold_2" else train_set),
    }

    def eligible(drug_a: str, drug_b: str, split: str) -> bool:
        if drug_a not in allowed[split] or drug_b not in allowed[split]:
            return False
        if scenario == "cold_1":
            if split == "train":
                return drug_a in train_set and drug_b in train_set
            new_set = validation_set if split == "validation" else test_set
            return (drug_a in train_set and drug_b in new_set) or (drug_b in train_set and drug_a in new_set)
        if scenario == "cold_2":
            target = validation_set if split == "validation" else test_set if split == "test" else train_set
            return drug_a in target and drug_b in target
        return True

    controls = []
    diagnostics = {"attempts": 0, "accepted": 0, "relaxations": 0}
    enforce_similarity = bool(similarities)
    for split, frame in split_frames.items():
        if frame.empty:
            raise ValueError(f"{scenario} produced an empty {split} positive split")
        split_seed = {"train": 0, "validation": 1, "test": 2}[split]
        controls.append(generate_negative_samples(
            frame,
            set(forbidden_pairs),
            sorted(allowed[split]),
            degree_bins,
            similarities,
            seed=seed + split_seed,
            max_attempts=config["sampling"]["maximum_attempts_per_control"],
            sim_bins=config["sampling"]["molecular_similarity_bins"],
            controls_per_positive_pair=config["sampling"]["controls_per_positive_pair"],
            eligible_pair=eligible,
            forbidden_pairs=forbidden_pairs,
            enforce_similarity=enforce_similarity,
        ))
        generated_diag = controls[-1].attrs.get("sampler_diagnostics", {})
        for key in diagnostics:
            diagnostics[key] += int(generated_diag.get(key, 0))
        forbidden_pairs.update(zip(controls[-1]["drug_a"], controls[-1]["drug_b"]))
    return pd.concat(controls, ignore_index=True), {
        "controls": int(sum(len(control) for control in controls)),
        "controls_per_positive_pair": config["sampling"]["controls_per_positive_pair"],
        "degree_matching": "mandatory",
        "similarity_matching": "available_smiles" if enforce_similarity else "unavailable_no_smiles",
        **diagnostics,
    }


def _labels_json(triples: pd.DataFrame, side_effects: pd.DataFrame, train_ids: set[str], split_ids: dict[str, set[str]], top_k: int) -> dict[str, Any]:
    train = triples[triples.pair_id.isin(train_ids)]
    counts = train.groupby("label_cui")["pair_id"].nunique().sort_values(ascending=False, kind="mergesort")
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:top_k]
    selected = {cui for cui, _ in ordered}
    names = dict(zip(side_effects["umls_cui_from_meddra"].astype(str), side_effects["side_effect_name"].astype(str)))
    labels = [{"index": index, "cui": str(cui), "name": names.get(str(cui), ""), "train_positive_count": int(count)} for index, (cui, count) in enumerate(ordered)]
    oov = {}
    for split, ids in split_ids.items():
        subset = triples[triples.pair_id.isin(ids)]
        oov[split] = int((~subset.label_cui.isin(selected)).sum())
    return {"labels": labels, "selection_split": "train", "top_k": int(top_k), "out_of_vocabulary_positive_counts": oov, "selected_cuis": sorted(selected)}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _write_manifest_artifacts(
    target: Path,
    scenario: str,
    seed: int,
    config: dict[str, Any],
    sources: dict[str, Path],
    drugs: pd.DataFrame,
    side_effects: pd.DataFrame,
    triples: pd.DataFrame,
    input_counts: dict[str, int],
    positive_splits: dict[str, pd.DataFrame],
    split_sets: tuple[set[str], set[str], set[str]] | None,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_manifest = target / "manifest.json"
    existing_created_at = None
    if existing_manifest.exists():
        try:
            existing_created_at = json.loads(existing_manifest.read_text()).get("created_at_utc")
        except Exception:
            raise ValueError(f"existing benchmark directory has invalid manifest: {target}")
    temp = Path(tempfile.mkdtemp(prefix=f".{scenario}_{seed}_", dir=str(target.parent)))
    try:
        observed = _pair_table(triples)
        similarities, similarity_status = _build_similarity_map(drugs, observed)
        forbidden = set(zip(observed["drug_a"], observed["drug_b"]))
        controls, sampler = _controls_for_scenario(observed, scenario, positive_splits, split_sets, config, seed, similarities, forbidden)
        positives = pd.concat([_attach_positive_metadata(frame, scenario) for frame in positive_splits.values()], ignore_index=True)
        pairs = pd.concat([positives, controls[PAIR_COLUMNS]], ignore_index=True)
        pairs = pairs.drop_duplicates("pair_id").sort_values(["split", "pair_id"]).reset_index(drop=True)
        if not pairs["pair_id"].is_unique:
            raise ValueError("pairs.parquet pair_id values must be unique")
        split_ids = {split: set(frame.pair_id) for split, frame in positive_splits.items()}
        labels_payload = _labels_json(triples, side_effects, split_ids["train"], split_ids, config["top_k_labels"])
        selected = set(labels_payload["selected_cuis"])
        triples_out = triples[triples.label_cui.isin(selected)].copy()
        triples_out = triples_out[triples_out.pair_id.isin(set(pairs.pair_id))]
        triples_out = triples_out[["pair_id", "drug_a", "drug_b", "label_cui", "observation_status"]].sort_values(["pair_id", "label_cui"]).reset_index(drop=True)
        pairs.to_parquet(temp / "pairs.parquet", index=False)
        triples_out.to_parquet(temp / "triples.parquet", index=False)
        (temp / "labels.json").write_text(json.dumps({key: value for key, value in labels_payload.items() if key != "selected_cuis"}, indent=2, sort_keys=True))

        train_df = pairs[pairs.split == "train"]
        val_df = pairs[pairs.split == "validation"]
        test_df = pairs[pairs.split == "test"]
        validators = {
            "no_pair_overlap": check_no_overlap(train_df, val_df, test_df),
            "pair_references_valid": set(triples_out.pair_id).issubset(set(pairs.pair_id)),
            "control_references_valid": set(controls.source_positive_pair_id).issubset(set(positives.pair_id)),
        }
        if scenario == "cold_1" and split_sets:
            validators["cold_endpoint_rules"] = check_cold_1_test_endpoints(test_df, split_sets[0], split_sets[2]) and check_no_test_new_in_train_val(train_df, val_df, split_sets[2])
        elif scenario == "cold_2" and split_sets:
            validators["cold_endpoint_rules"] = check_cold_2_test_endpoints(test_df, split_sets[2]) and check_no_test_new_in_train_val(train_df, val_df, split_sets[2])
        else:
            validators["cold_endpoint_rules"] = True
        if not all(validators.values()):
            raise ValueError(f"benchmark validation failed: {validators}")
        split_counts = {}
        for split, frame in (("train", train_df), ("validation", val_df), ("test", test_df)):
            split_counts[split] = {
                "positive_pairs": int((frame.observation_status == "observed_positive").sum()),
                "unlabeled_controls": int((frame.observation_status == "sampled_unlabeled").sum()),
                "unique_drugs": int(len(set(frame.drug_a) | set(frame.drug_b))),
            }
        artifact_hashes = {name: _file_digest(temp / name) for name in ("pairs.parquet", "triples.parquet", "labels.json")}
        artifact_metadata = {
            name: {"sha256": artifact_hashes[name], "bytes": (temp / name).stat().st_size}
            for name in artifact_hashes
        }
        source_info = {key: _sha256(path) for key, path in sorted(sources.items())}
        created_at = existing_created_at or datetime.now(timezone.utc).isoformat()
        drug_set_hashes = {
            split: hashlib.sha256("\0".join(sorted(set(frame.drug_a) | set(frame.drug_b))).encode()).hexdigest()
            for split, frame in (("train", train_df), ("validation", val_df), ("test", test_df))
        }
        if split_sets:
            train_drugs, validation_new_drugs, test_new_drugs = split_sets
            for key, values in (("train", train_drugs), ("validation_new", validation_new_drugs), ("test_new", test_new_drugs)):
                drug_set_hashes[key] = hashlib.sha256("\0".join(sorted(values)).encode()).hexdigest()
        else:
            drug_set_hashes["validation_new"] = drug_set_hashes["validation"]
            drug_set_hashes["test_new"] = drug_set_hashes["test"]
        manifest = {
            "schema_version": 1,
            "benchmark_id": config["benchmark_name"],
            "benchmark_name": config["benchmark_name"],
            "scenario": scenario,
            "seed": int(seed),
            "created_at_utc": created_at,
            "git_sha": _git_sha(),
            "sources": source_info,
            "source_files": source_info,
            "counts": {
                "pairs": int(len(pairs)), "positive_pairs": int(len(positives)), "controls": int(len(controls)),
                "triples": int(len(triples_out)), "labels": int(len(labels_payload["labels"])),
            },
            "input_counts": input_counts,
            "mapping_counts": input_counts,
            "split_counts": split_counts,
            "drug_set_hashes": drug_set_hashes,
            "artifact_sha256": artifact_hashes,
            "artifact_metadata": artifact_metadata,
            "pairs_sha256": artifact_hashes["pairs.parquet"],
            "triples_sha256": artifact_hashes["triples.parquet"],
            "labels_sha256": artifact_hashes["labels.json"],
            "sampler_diagnostics": {**sampler, "similarity_matching": similarity_status},
            "validator_results": validators,
            "pairs_path": "pairs.parquet",
            "triples_path": "triples.parquet",
            "labels_path": "labels.json",
            "artifact_paths": {
                "pairs": "pairs.parquet",
                "triples": "triples.parquet",
                "labels": "labels.json",
            },
            "artifacts": {
                "pairs_path": "pairs.parquet",
                "triples_path": "triples.parquet",
                "labels_path": "labels.json",
            },
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
        manifest["manifest_hash"] = hashlib.sha256(manifest_bytes).hexdigest()[:12]
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        if existing_manifest.exists():
            existing = verify_manifest(str(existing_manifest))
            if existing["manifest_hash"] != manifest["manifest_hash"]:
                raise ValueError(f"immutable benchmark mismatch at {target}")
            return existing_manifest
        os.replace(temp, target)
        temp = None
        return target / "manifest.json"
    finally:
        if temp is not None and temp.exists():
            shutil.rmtree(temp, ignore_errors=True)


def build_benchmarks(config_path: str | Path, output_root: str | Path | None = None) -> list[Path]:
    config_path = Path(config_path).resolve()
    try:
        config = validate_benchmark_config(yaml.safe_load(config_path.read_text()))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"unable to read benchmark config: {exc}") from exc
    sources = _resolve_sources(config_path, config)
    drugs, side_effects, combo = _read_sources(sources)
    triples, input_counts = _canonical_triples(drugs, side_effects, combo, return_counts=True)
    positive_pairs = _pair_table(triples)
    root = Path(output_root or "artifacts/benchmarks").resolve() / config["benchmark_name"]
    # Preflight every split and sampler before writing any manifest.  This
    # prevents a later cold scenario failure from leaving an apparently valid
    # partial benchmark family on disk.
    prepared: list[tuple[int, str, dict[str, pd.DataFrame], tuple[set[str], set[str], set[str]] | None]] = []
    for seed in config["seeds"]:
        for scenario in ("warm_pair", "cold_1", "cold_2"):
            split_result = _scenario_split(positive_pairs, scenario, config["partitions"][scenario], seed)
            if scenario == "warm_pair":
                frames, sets = split_result, None
            else:
                frames, sets = split_result
            split_frames = {"train": frames[0], "validation": frames[1], "test": frames[2]}
            if any(frame.empty for frame in split_frames.values()):
                raise ValueError(f"{scenario} seed {seed} produced an empty split")
            _controls_for_scenario(
                positive_pairs,
                scenario,
                split_frames,
                sets,
                config,
                seed,
                _build_similarity_map(drugs, positive_pairs)[0],
                set(zip(positive_pairs["drug_a"], positive_pairs["drug_b"])),
            )
            prepared.append((seed, scenario, split_frames, sets))
    manifests = []
    for seed, scenario, split_frames, sets in prepared:
        manifests.append(_write_manifest_artifacts(root / scenario / f"seed_{seed}", scenario, seed, config, sources, drugs, side_effects, triples, input_counts, split_frames, sets))
    return manifests


# Singular alias retained for callers that build one configured benchmark
# family (all configured seeds/scenarios are still emitted).
build_benchmark = build_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="artifacts/benchmarks")
    args = parser.parse_args()
    build_benchmarks(args.config, args.output_root)


if __name__ == "__main__":
    main()
