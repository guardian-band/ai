"""Safe multimodal features for the advanced teacher."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any

import numpy as np


SCHEMA_VERSION = 2
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_FLOAT_FIELDS = ("morgan", "molformer_tokens", "mpnn_tokens", "kg_tokens")
_AVAILABLE_FIELDS = (
    "morgan_available",
    "molformer_available",
    "mpnn_available",
    "kg_available",
)
_MASK_FIELDS = ("molformer_padding_mask", "mpnn_padding_mask", "kg_padding_mask")
_PROVENANCE_FIELDS = (
    "morgan_provenance_hash",
    "molformer_provenance_hash",
    "mpnn_provenance_hash",
    "kg_provenance_hash",
)


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 hash")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _validate_manifest_compatibility(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("manifest_compatibility metadata must be an object")
    required = ("benchmark_id", "scenario", "seed", "manifest_hash")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(
            f"manifest_compatibility missing required fields: {', '.join(missing)}"
        )
    for key in ("benchmark_id", "scenario", "manifest_hash"):
        item = value[key]
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"manifest_compatibility {key} must be a non-empty string")
    seed = value["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("manifest_compatibility seed must be an integer")
    return copy.deepcopy(dict(value))


def _digest(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(_json_ready(metadata), sort_keys=True, separators=(",", ":")).encode()
    )
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path = path.with_name(f"{path.name}.sha256")
    payload_fd, payload_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    checksum_fd, checksum_tmp = tempfile.mkstemp(
        prefix=f".{checksum_path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(payload_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        with os.fdopen(checksum_fd, "w", encoding="ascii") as handle:
            handle.write(hashlib.sha256(payload).hexdigest() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(payload_tmp, path)
        os.replace(checksum_tmp, checksum_path)
    finally:
        for temporary in (payload_tmp, checksum_tmp):
            if os.path.exists(temporary):
                os.unlink(temporary)


@dataclass(frozen=True)
class MultimodalFeatureArtifact:
    """Immutable four-modality arrays with deterministic drug lookup."""

    drug_ids: tuple[str, ...]
    morgan: np.ndarray
    molformer_tokens: np.ndarray
    mpnn_tokens: np.ndarray
    kg_tokens: np.ndarray
    morgan_available: np.ndarray
    molformer_available: np.ndarray
    mpnn_available: np.ndarray
    kg_available: np.ndarray
    molformer_padding_mask: np.ndarray
    mpnn_padding_mask: np.ndarray
    kg_padding_mask: np.ndarray
    metadata: Mapping[str, Any]
    _id_to_index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        drug_ids = tuple(self.drug_ids)
        names = _FLOAT_FIELDS + _AVAILABLE_FIELDS + _MASK_FIELDS
        arrays = {
            name: np.array(getattr(self, name), copy=True, order="C") for name in names
        }
        metadata = _freeze(self.metadata)
        object.__setattr__(self, "drug_ids", drug_ids)
        object.__setattr__(self, "metadata", metadata)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        count = len(drug_ids)
        if count == 0 or any(
            not isinstance(item, str) or not item.strip() for item in drug_ids
        ):
            raise ValueError("drug IDs must be non-empty strings")
        if len(set(drug_ids)) != count:
            raise ValueError("duplicate drug ID in multimodal feature artifact")
        if tuple(sorted(drug_ids)) != drug_ids:
            raise ValueError("drug IDs must be sorted deterministically")
        self._validate_arrays(count)
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported multimodal feature schema version")
        for key in _PROVENANCE_FIELDS:
            _require_hash(metadata.get(key), key)
        _validate_manifest_compatibility(metadata.get("manifest_compatibility"))
        expected = metadata.get("payload_sha256")
        bare = {key: value for key, value in metadata.items() if key != "payload_sha256"}
        digest_arrays = arrays | {"drug_ids": np.asarray(drug_ids, dtype="<U")}
        if expected != _digest(digest_arrays, bare):
            raise ValueError("multimodal feature artifact checksum verification failed")
        for array in arrays.values():
            array.setflags(write=False)
        object.__setattr__(
            self,
            "_id_to_index",
            MappingProxyType({drug_id: index for index, drug_id in enumerate(drug_ids)}),
        )

    def _validate_arrays(self, count: int) -> None:
        for name in _FLOAT_FIELDS:
            array = getattr(self, name)
            expected_ndim = 2 if name == "morgan" else 3
            if (
                array.dtype != np.float32
                or array.ndim != expected_ndim
                or array.shape[0] != count
            ):
                raise ValueError(f"{name} must be float32 with shape [N, ...]")
            if any(size <= 0 for size in array.shape[1:]) or not np.isfinite(array).all():
                raise ValueError(f"{name} must have positive dimensions and finite values")
        for name in _AVAILABLE_FIELDS:
            array = getattr(self, name)
            if array.dtype != np.bool_ or array.shape != (count,):
                raise ValueError(f"{name} must be bool with shape [{count}]")
        token_fields = ("molformer_tokens", "mpnn_tokens", "kg_tokens")
        for mask_name, token_name in zip(_MASK_FIELDS, token_fields, strict=True):
            mask = getattr(self, mask_name)
            expected = (count, getattr(self, token_name).shape[1])
            if mask.dtype != np.bool_ or mask.shape != expected:
                raise ValueError(f"{mask_name} must be bool with shape {expected}")

    @classmethod
    def write(
        cls,
        path: str | Path,
        *,
        drug_ids: Sequence[str],
        morgan: np.ndarray,
        molformer_tokens: np.ndarray,
        mpnn_tokens: np.ndarray,
        kg_tokens: np.ndarray,
        morgan_available: np.ndarray,
        molformer_available: np.ndarray,
        mpnn_available: np.ndarray,
        kg_available: np.ndarray,
        molformer_padding_mask: np.ndarray,
        mpnn_padding_mask: np.ndarray,
        kg_padding_mask: np.ndarray,
        morgan_provenance_hash: str,
        molformer_provenance_hash: str,
        mpnn_provenance_hash: str,
        kg_provenance_hash: str,
        manifest_compatibility: Mapping[str, Any],
    ) -> "MultimodalFeatureArtifact":
        raw_ids = list(drug_ids)
        if any(not isinstance(item, str) or not item.strip() for item in raw_ids):
            raise ValueError("drug IDs must be non-empty strings")
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError("duplicate drug ID in multimodal feature artifact")
        ids = np.asarray(raw_ids, dtype="<U")
        arrays = {
            name: np.asarray(value)
            for name, value in {
                "morgan": morgan,
                "molformer_tokens": molformer_tokens,
                "mpnn_tokens": mpnn_tokens,
                "kg_tokens": kg_tokens,
                "morgan_available": morgan_available,
                "molformer_available": molformer_available,
                "mpnn_available": mpnn_available,
                "kg_available": kg_available,
                "molformer_padding_mask": molformer_padding_mask,
                "mpnn_padding_mask": mpnn_padding_mask,
                "kg_padding_mask": kg_padding_mask,
            }.items()
        }
        order = np.argsort(ids, kind="stable")
        ids = np.ascontiguousarray(ids[order])
        for name, array in arrays.items():
            if array.ndim == 0 or array.shape[0] != len(ids):
                raise ValueError(f"{name} must have a first dimension matching drug IDs")
            arrays[name] = np.ascontiguousarray(array[order])
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "drug_count": int(len(ids)),
            "morgan_dim": int(arrays["morgan"].shape[1])
            if arrays["morgan"].ndim == 2
            else 0,
            "molformer_token_count": int(arrays["molformer_tokens"].shape[1])
            if arrays["molformer_tokens"].ndim == 3
            else 0,
            "molformer_dim": int(arrays["molformer_tokens"].shape[2])
            if arrays["molformer_tokens"].ndim == 3
            else 0,
            "mpnn_token_count": int(arrays["mpnn_tokens"].shape[1])
            if arrays["mpnn_tokens"].ndim == 3
            else 0,
            "mpnn_dim": int(arrays["mpnn_tokens"].shape[2])
            if arrays["mpnn_tokens"].ndim == 3
            else 0,
            "kg_token_count": int(arrays["kg_tokens"].shape[1])
            if arrays["kg_tokens"].ndim == 3
            else 0,
            "kg_dim": int(arrays["kg_tokens"].shape[2])
            if arrays["kg_tokens"].ndim == 3
            else 0,
            "manifest_compatibility": _validate_manifest_compatibility(
                manifest_compatibility
            ),
        }
        for key, value in {
            "morgan_provenance_hash": morgan_provenance_hash,
            "molformer_provenance_hash": molformer_provenance_hash,
            "mpnn_provenance_hash": mpnn_provenance_hash,
            "kg_provenance_hash": kg_provenance_hash,
        }.items():
            metadata[key] = _require_hash(value, key)
        metadata["payload_sha256"] = _digest(arrays | {"drug_ids": ids}, metadata)
        cls(
            tuple(str(item) for item in ids.tolist()),
            metadata=metadata,
            **arrays,
        )
        payload = io.BytesIO()
        np.savez_compressed(
            payload,
            drug_ids=ids,
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
            **arrays,
        )
        _atomic_write(Path(path), payload.getvalue())
        return cls.load(path)

    @classmethod
    def load(cls, path: str | Path) -> "MultimodalFeatureArtifact":
        path = Path(path)
        checksum_path = path.with_name(f"{path.name}.sha256")
        try:
            payload = path.read_bytes()
            expected = checksum_path.read_text(encoding="ascii").strip()
        except OSError as exc:
            raise ValueError(f"unable to read multimodal feature artifact {path}") from exc
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("multimodal feature artifact file checksum verification failed")
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                required = {
                    "drug_ids",
                    "metadata_json",
                    *_FLOAT_FIELDS,
                    *_AVAILABLE_FIELDS,
                    *_MASK_FIELDS,
                }
                if set(archive.files) != required:
                    raise ValueError("multimodal feature artifact has an invalid field set")
                ids = np.asarray(archive["drug_ids"])
                metadata = json.loads(str(archive["metadata_json"].item()))
                arrays = {
                    name: np.asarray(archive[name])
                    for name in _FLOAT_FIELDS + _AVAILABLE_FIELDS + _MASK_FIELDS
                }
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if isinstance(exc, ValueError) and "multimodal feature" in str(exc):
                raise
            raise ValueError(f"unable to read multimodal feature artifact {path}") from exc
        if (
            ids.ndim != 1
            or ids.dtype.kind not in {"U", "S"}
            or not isinstance(metadata, dict)
        ):
            raise ValueError("multimodal feature artifact metadata or IDs are invalid")
        return cls(tuple(str(item) for item in ids.tolist()), metadata=metadata, **arrays)

    def lookup(self, drug_id: str) -> dict[str, Any]:
        key = drug_id.strip() if isinstance(drug_id, str) else ""
        if not key or key not in self._id_to_index:
            raise KeyError(f"unknown drug ID: {drug_id}")
        index = self._id_to_index[key]
        return {
            name: getattr(self, name)[index].copy()
            for name in _FLOAT_FIELDS + _AVAILABLE_FIELDS + _MASK_FIELDS
        }
