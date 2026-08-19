"""Safe, versioned precomputed inputs for the multimodal teacher."""

from __future__ import annotations

from collections.abc import Mapping
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


SCHEMA_VERSION = 1
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 hash")
    return value


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_metadata(item) for item in value)
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
        raise ValueError(f"manifest_compatibility missing required fields: {', '.join(missing)}")
    for key in ("benchmark_id", "scenario", "manifest_hash"):
        field = value[key]
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"manifest_compatibility {key} must be a non-empty string")
    seed = value["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("manifest_compatibility seed must be an integer")
    return dict(value)


def _digest(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(_json_ready(metadata), sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(arrays):
        array = arrays[name]
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MultimodalFeatureArtifact:
    """Immutable modality arrays with deterministic O(1) drug lookup."""

    drug_ids: tuple[str, ...]
    morgan: np.ndarray
    molecular_tokens: np.ndarray
    kg_tokens: np.ndarray
    morgan_available: np.ndarray
    molecular_available: np.ndarray
    kg_available: np.ndarray
    molecular_padding_mask: np.ndarray
    kg_padding_mask: np.ndarray
    metadata: Mapping[str, Any]
    _id_to_index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        arrays = {
            "morgan": np.array(self.morgan, copy=True, order="C"),
            "molecular_tokens": np.array(self.molecular_tokens, copy=True, order="C"),
            "kg_tokens": np.array(self.kg_tokens, copy=True, order="C"),
            "morgan_available": np.array(self.morgan_available, copy=True, order="C"),
            "molecular_available": np.array(self.molecular_available, copy=True, order="C"),
            "kg_available": np.array(self.kg_available, copy=True, order="C"),
            "molecular_padding_mask": np.array(self.molecular_padding_mask, copy=True, order="C"),
            "kg_padding_mask": np.array(self.kg_padding_mask, copy=True, order="C"),
        }
        metadata = _freeze_metadata(self.metadata)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "drug_ids", tuple(self.drug_ids))
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        n = len(self.drug_ids)
        if n == 0 or any(not isinstance(item, str) or not item.strip() for item in self.drug_ids):
            raise ValueError("drug IDs must be non-empty strings")
        if len(set(self.drug_ids)) != n:
            raise ValueError("duplicate drug ID in multimodal feature artifact")
        if tuple(sorted(self.drug_ids)) != self.drug_ids:
            raise ValueError("drug IDs must be sorted deterministically")
        self._validate_arrays(n)
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported multimodal feature schema version")
        for key in ("morgan_provenance_hash", "molecular_provenance_hash", "kg_provenance_hash"):
            _require_hash(self.metadata.get(key), key)
        compatibility = _validate_manifest_compatibility(self.metadata.get("manifest_compatibility"))
        expected = self.metadata.get("payload_sha256")
        digest_arrays = dict(arrays)
        digest_arrays["drug_ids"] = np.asarray(self.drug_ids, dtype="<U")
        if not isinstance(expected, str) or expected != _digest(
            digest_arrays,
            {key: value for key, value in self.metadata.items() if key != "payload_sha256"},
        ):
            raise ValueError("multimodal feature artifact checksum verification failed")
        object.__setattr__(
            self,
            "_id_to_index",
            MappingProxyType({drug_id: index for index, drug_id in enumerate(self.drug_ids)}),
        )

    def _validate_arrays(self, n: int) -> None:
        float_arrays = {
            "morgan": (self.morgan, (n, None)),
            "molecular_tokens": (self.molecular_tokens, (n, None, None)),
            "kg_tokens": (self.kg_tokens, (n, None, None)),
        }
        for name, (array, shape) in float_arrays.items():
            if array.dtype != np.float32 or array.ndim != len(shape):
                raise ValueError(f"{name} must be float32 with shape [N, ...]")
            if any(expected is not None and actual != expected for actual, expected in zip(array.shape, shape)):
                raise ValueError(f"{name} has an invalid shape")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
        if self.molecular_tokens.shape[1] <= 0 or self.molecular_tokens.shape[2] <= 0:
            raise ValueError("molecular_tokens must have positive token count and dimension")
        if self.kg_tokens.shape[1] <= 0 or self.kg_tokens.shape[2] <= 0:
            raise ValueError("kg_tokens must have positive token count and dimension")
        if self.morgan.shape[1] <= 0:
            raise ValueError("morgan must have positive feature dimension")
        bool_arrays = {
            "morgan_available": (self.morgan_available, (n,)),
            "molecular_available": (self.molecular_available, (n,)),
            "kg_available": (self.kg_available, (n,)),
            "molecular_padding_mask": (
                self.molecular_padding_mask,
                (n, self.molecular_tokens.shape[1]),
            ),
            "kg_padding_mask": (self.kg_padding_mask, (n, self.kg_tokens.shape[1])),
        }
        for name, (array, shape) in bool_arrays.items():
            if array.dtype != np.bool_ or array.shape != shape:
                raise ValueError(f"{name} must be bool with shape {shape}")

    @classmethod
    def write(
        cls,
        path: str | Path,
        drug_ids: list[str] | tuple[str, ...],
        morgan: np.ndarray,
        molecular_tokens: np.ndarray,
        kg_tokens: np.ndarray,
        morgan_available: np.ndarray,
        molecular_available: np.ndarray,
        kg_available: np.ndarray,
        molecular_padding_mask: np.ndarray,
        kg_padding_mask: np.ndarray,
        *,
        morgan_provenance_hash: str,
        molecular_provenance_hash: str,
        kg_provenance_hash: str,
        manifest_compatibility: Mapping[str, Any],
    ) -> "MultimodalFeatureArtifact":
        raw_ids = list(drug_ids)
        if any(not isinstance(item, str) or not item.strip() for item in raw_ids):
            raise ValueError("drug IDs must be non-empty strings")
        ids = np.asarray(raw_ids, dtype="<U")
        if ids.ndim != 1 or len(ids) == 0 or any(not item.strip() for item in ids.tolist()):
            raise ValueError("drug IDs must be a non-empty sequence of non-empty strings")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("duplicate drug ID in multimodal feature artifact")
        arrays = {
            "morgan": np.asarray(morgan),
            "molecular_tokens": np.asarray(molecular_tokens),
            "kg_tokens": np.asarray(kg_tokens),
            "morgan_available": np.asarray(morgan_available),
            "molecular_available": np.asarray(molecular_available),
            "kg_available": np.asarray(kg_available),
            "molecular_padding_mask": np.asarray(molecular_padding_mask),
            "kg_padding_mask": np.asarray(kg_padding_mask),
        }
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        for name, array in list(arrays.items()):
            if array.ndim == 0 or array.shape[0] != len(ids):
                raise ValueError(f"{name} must have a first dimension matching drug IDs")
            arrays[name] = np.ascontiguousarray(array[order])
        compatibility = _validate_manifest_compatibility(manifest_compatibility)
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "drug_count": int(len(ids)),
            "morgan_dim": int(arrays["morgan"].shape[1]) if arrays["morgan"].ndim == 2 else 0,
            "molecular_token_count": int(arrays["molecular_tokens"].shape[1]) if arrays["molecular_tokens"].ndim == 3 else 0,
            "molecular_dim": int(arrays["molecular_tokens"].shape[2]) if arrays["molecular_tokens"].ndim == 3 else 0,
            "kg_token_count": int(arrays["kg_tokens"].shape[1]) if arrays["kg_tokens"].ndim == 3 else 0,
            "kg_dim": int(arrays["kg_tokens"].shape[2]) if arrays["kg_tokens"].ndim == 3 else 0,
            "morgan_provenance_hash": _require_hash(morgan_provenance_hash, "morgan_provenance_hash"),
            "molecular_provenance_hash": _require_hash(molecular_provenance_hash, "molecular_provenance_hash"),
            "kg_provenance_hash": _require_hash(kg_provenance_hash, "kg_provenance_hash"),
            "manifest_compatibility": copy.deepcopy(compatibility),
        }
        metadata["payload_sha256"] = _digest(arrays | {"drug_ids": ids}, metadata)
        payload = io.BytesIO()
        np.savez_compressed(payload, drug_ids=ids, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))), **arrays)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        checksum_path = path.with_name(f"{path.name}.sha256")
        checksum_fd, checksum_temporary = tempfile.mkstemp(prefix=f".{checksum_path.name}.", suffix=".tmp", dir=path.parent)
        payload_bytes = payload.getvalue()
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            with os.fdopen(checksum_fd, "w", encoding="ascii") as handle:
                handle.write(hashlib.sha256(payload_bytes).hexdigest() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(checksum_temporary, checksum_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
            if os.path.exists(checksum_temporary):
                os.unlink(checksum_temporary)
        return cls.load(path)

    @classmethod
    def load(cls, path: str | Path) -> "MultimodalFeatureArtifact":
        path = Path(path)
        sidecar = path.with_name(f"{path.name}.sha256")
        try:
            expected = sidecar.read_text(encoding="ascii").strip()
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"unable to read multimodal feature checksum for {path}") from exc
        if expected != hashlib.sha256(payload).hexdigest():
            raise ValueError("multimodal feature artifact file checksum verification failed")
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                required = {"drug_ids", "metadata_json", "morgan", "molecular_tokens", "kg_tokens", "morgan_available", "molecular_available", "kg_available", "molecular_padding_mask", "kg_padding_mask"}
                if set(archive.files) != required:
                    raise ValueError("multimodal feature artifact has an invalid field set")
                values = {name: np.asarray(archive[name]) for name in required if name not in {"metadata_json", "drug_ids"}}
                ids = np.asarray(archive["drug_ids"])
                metadata = json.loads(str(archive["metadata_json"].item()))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if isinstance(exc, ValueError) and "multimodal feature" in str(exc):
                raise
            raise ValueError(f"unable to read multimodal feature artifact {path}") from exc
        if ids.ndim != 1 or ids.dtype.kind not in {"U", "S"} or not isinstance(metadata, dict):
            raise ValueError("multimodal feature artifact metadata or IDs are invalid")
        return cls(tuple(str(item) for item in ids.tolist()), metadata=metadata, **values)

    def lookup(self, drug_id: str) -> dict[str, np.ndarray]:
        if not isinstance(drug_id, str) or not drug_id.strip() or drug_id.strip() not in self._id_to_index:
            raise KeyError(f"unknown drug ID: {drug_id}")
        index = self._id_to_index[drug_id.strip()]
        return {
            "morgan": self.morgan[index].copy(),
            "molecular_tokens": self.molecular_tokens[index].copy(),
            "kg_tokens": self.kg_tokens[index].copy(),
            "morgan_available": self.morgan_available[index].copy(),
            "molecular_available": self.molecular_available[index].copy(),
            "kg_available": self.kg_available[index].copy(),
            "molecular_padding_mask": self.molecular_padding_mask[index].copy(),
            "kg_padding_mask": self.kg_padding_mask[index].copy(),
        }
