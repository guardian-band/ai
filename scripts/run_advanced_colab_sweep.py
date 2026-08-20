"""Run scenario/seed advanced experiments with hash-verified Drive boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STAGES = ("hgt", "assemble", "teacher_config", "teacher", "cache", "student_config", "student")
SCENARIOS = ("warm_pair", "cold_1", "cold_2")
DEFAULT_SEEDS = (42, 101, 2024, 27182, 31415)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(paths: Iterable[Path]) -> list[Path]:
    output: list[Path] = []
    for path in paths:
        if path.is_dir():
            output.extend(sorted(item for item in path.rglob("*") if item.is_file()))
        elif path.is_file():
            output.append(path)
        else:
            raise FileNotFoundError(f"missing stage output: {path}")
    return output


def render_config(template: Path, destination: Path, scenario: str, seed: int) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported scenario: {scenario}")
    payload = yaml.safe_load(template.read_text())

    def render(value):
        if isinstance(value, str):
            return value.format(scenario=scenario, seed=seed)
        if isinstance(value, list):
            return [render(item) for item in value]
        if isinstance(value, dict):
            return {key: render(item) for key, item in value.items()}
        return value

    rendered = render(payload)
    hgt = rendered["stages"]["hgt"]
    seed_index = hgt.index("--seed")
    if hgt[seed_index + 1] != str(seed):
        raise ValueError("rendered HGT seed does not match the run seed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(rendered, sort_keys=False))
    return rendered


def _manifest(scenario: str, seed: int) -> tuple[Path, dict]:
    path = ROOT / "artifacts" / "benchmarks" / "polypharmacy_v2" / scenario / f"seed_{seed}" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing v2 manifest: {path}")
    payload = json.loads(path.read_text())
    if payload.get("scenario") != scenario or payload.get("seed") != seed:
        raise ValueError(f"manifest identity mismatch: {path}")
    return path, payload


def ensure_manifest_hierarchy(hierarchy_path: Path, manifest_path: Path) -> list[str]:
    """Extend the versioned fallback hierarchy for every selected label."""
    from src.features.meddra_hierarchy import map_side_effect_to_soc

    manifest = json.loads(manifest_path.read_text())
    labels_path = Path(manifest["labels_path"])
    if not labels_path.is_absolute():
        labels_path = manifest_path.parent / labels_path
    labels_payload = json.loads(labels_path.read_text())
    labels = labels_payload.get("labels", labels_payload)
    hierarchy = json.loads(hierarchy_path.read_text())
    organ_order = hierarchy["organ_order"]
    mapped = {str(item["specific_cui"]) for item in hierarchy["mappings"]}
    added: list[str] = []
    for label in labels:
        cui = str(label["cui"])
        if cui in mapped:
            continue
        organ = map_side_effect_to_soc(str(label["name"]))
        hierarchy["mappings"].append(
            {"organ_index": organ_order.index(organ), "specific_cui": cui}
        )
        mapped.add(cui)
        added.append(cui)
    if added:
        hierarchy["mappings"] = sorted(
            hierarchy["mappings"], key=lambda item: str(item["specific_cui"])
        )
        hierarchy["source"] = "keyword_fallback_v2_manifest_complete"
        hierarchy_path.write_text(json.dumps(hierarchy, indent=2, sort_keys=True))
    return added


def expected_outputs(stage: str, scenario: str, seed: int, manifest_hash: str) -> list[Path]:
    artifacts = ROOT / "artifacts"
    mapping = {
        "hgt": [artifacts / "checkpoints" / f"hgt_{scenario}_seed_{seed}.pt", artifacts / "features" / f"hgt_{scenario}_seed_{seed}.npz", artifacts / "features" / f"hgt_{scenario}_seed_{seed}.npz.sha256"],
        "assemble": [artifacts / "features" / f"advanced_{scenario}_seed_{seed}.npz", artifacts / "features" / f"advanced_{scenario}_seed_{seed}.npz.sha256"],
        "teacher_config": [artifacts / "configs" / f"teacher_{scenario}_seed_{seed}.yaml"],
        "teacher": [artifacts / "runs" / f"multimodal_teacher__{scenario}__seed_{seed}__{manifest_hash}"],
        "cache": [artifacts / "features" / f"teacher_cache_{scenario}_seed_{seed}.npz", artifacts / "features" / f"teacher_cache_{scenario}_seed_{seed}.npz.sha256"],
        "student_config": [artifacts / "configs" / f"student_{scenario}_seed_{seed}.yaml"],
        "student": [artifacts / "runs" / f"distilled_pair_student__{scenario}__seed_{seed}__{manifest_hash}"],
    }
    return mapping[stage]


def _restore(backup: Path) -> None:
    source = backup / "artifacts"
    if source.is_dir():
        shutil.copytree(source, ROOT / "artifacts", dirs_exist_ok=True)


def _marker_valid(marker: Path) -> bool:
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text())
    for relative, expected_hash in payload.get("files", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            return False
    return bool(payload.get("files"))


def _backup_stage(backup: Path, marker: Path, outputs: list[Path], scenario: str, seed: int, stage: str) -> None:
    files = _files(outputs)
    hashes: dict[str, str] = {}
    for source in files:
        relative = source.relative_to(ROOT)
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = _sha256(source)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"scenario": scenario, "seed": seed, "stage": stage, "files": hashes}, indent=2, sort_keys=True))


def _gpu_status() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def _run_live(command: list[str], *, stage: str) -> None:
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT)
    while True:
        try:
            return_code = process.wait(timeout=30)
            break
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - started) / 60
            print(
                f"HEARTBEAT {stage}: {elapsed:.1f} min | "
                f"GPU %, used MB, total MB: {_gpu_status()}",
                flush=True,
            )
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def run_one(scenario: str, seed: int, template: Path, drive_root: Path, dry_run: bool) -> None:
    config_path = ROOT / "artifacts" / "configs" / f"pipeline_v2_{scenario}_seed_{seed}.yaml"
    rendered = render_config(template, config_path, scenario, seed)
    if dry_run:
        print(f"DRY-RUN {scenario}/seed_{seed}: stages={','.join(rendered['stages'])}")
        print(f"  HGT seed argument: {rendered['stages']['hgt'][rendered['stages']['hgt'].index('--seed') + 1]}")
        return
    manifest_path, manifest = _manifest(scenario, seed)
    backup = drive_root / f"{scenario}_seed_{seed}"
    _restore(backup)
    hierarchy_path = ROOT / "artifacts" / "meddra_hierarchy.json"
    added = ensure_manifest_hierarchy(hierarchy_path, manifest_path)
    if added:
        print(
            f"HIERARCHY completed for {scenario}/seed_{seed}: {', '.join(added)}",
            flush=True,
        )
    for number, stage in enumerate(STAGES, start=1):
        marker = backup / "markers" / f"{stage}.complete.json"
        if _marker_valid(marker):
            print(f"[{number}/{len(STAGES)}] SKIP verified: {scenario}/seed_{seed}/{stage}", flush=True)
            continue
        command = [sys.executable, "-u", str(ROOT / "run_advanced_pipeline.py"), "--config", str(config_path), "--start-at", stage, "--stop-after", stage]
        print(f"[{number}/{len(STAGES)}] RUN: {scenario}/seed_{seed}/{stage}", flush=True)
        _run_live(command, stage=f"{scenario}/seed_{seed}/{stage}")
        outputs = expected_outputs(stage, scenario, seed, manifest["manifest_hash"])
        _backup_stage(backup, marker, outputs, scenario, seed, stage)
        if not _marker_valid(marker):
            raise RuntimeError(f"backup verification failed: {scenario}/seed_{seed}/{stage}")
        print(f"[{number}/{len(STAGES)}] SAVED+VERIFIED: {stage}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=ROOT / "configs" / "advanced_pipeline.v2.template.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    for scenario in args.scenarios:
        for seed in args.seeds:
            run_one(scenario, seed, args.template, args.drive_root, args.dry_run)


if __name__ == "__main__":
    main()
