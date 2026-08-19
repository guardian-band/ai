"""Safe, versioned per-drug token feature artifacts."""

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


SCHEMA_VERSION = 1
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 hash")
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _digest(
    drug_ids: np.ndarray,
    tokens: np.ndarray,
    padding_mask: np.ndarray,
    available: np.ndarray,
    metadata: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(_json_ready(metadata), sort_keys=True, separators=(",", ":")).encode())
    for name, array in (
        ("drug_ids", drug_ids),
        ("tokens", tokens),
        ("padding_mask", padding_mask),
        ("available", available),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes())
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
class TokenFeatureArtifact:
    """Immutable token arrays with file and content provenance verification."""

    drug_ids: tuple[str, ...]
    tokens: np.ndarray
    padding_mask: np.ndarray
    available: np.ndarray
    metadata: Mapping[str, Any]
    _id_to_index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        drug_ids = tuple(self.drug_ids)
        tokens = np.array(self.tokens, copy=True, order="C")
        padding_mask = np.array(self.padding_mask, copy=True, order="C")
        available = np.array(self.available, copy=True, order="C")
        metadata = _freeze(self.metadata)
        object.__setattr__(self, "drug_ids", drug_ids)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "padding_mask", padding_mask)
        object.__setattr__(self, "available", available)
        object.__setattr__(self, "metadata", metadata)

        count = len(drug_ids)
        if count == 0 or any(not isinstance(item, str) or not item.strip() for item in drug_ids):
            raise ValueError("drug IDs must be non-empty strings")
        if len(set(drug_ids)) != count:
            raise ValueError("duplicate drug ID in token feature artifact")
        if tuple(sorted(drug_ids)) != drug_ids:
            raise ValueError("drug IDs must be sorted deterministically")
        if tokens.dtype != np.float32 or tokens.ndim != 3 or tokens.shape[0] != count:
            raise ValueError("tokens must be float32 with shape [N, token_count, token_dim]")
        if tokens.shape[1] <= 0 or tokens.shape[2] <= 0 or not np.isfinite(tokens).all():
            raise ValueError("tokens must have positive dimensions and finite values")
        if padding_mask.dtype != np.bool_ or padding_mask.shape != tokens.shape[:2]:
            raise ValueError("padding_mask must be bool with shape [N, token_count]")
        if available.dtype != np.bool_ or available.shape != (count,):
            raise ValueError("available must be bool with shape [N]")
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported token feature artifact schema version")
        producer = metadata.get("producer")
        if not isinstance(producer, str) or not producer.strip():
            raise ValueError("token feature producer must be a non-empty string")
        if not isinstance(metadata.get("producer_config"), Mapping):
            raise ValueError("producer_config must be an object")
        _require_hash(metadata.get("source_sha256"), "source_sha256")
        checkpoint_hash = metadata.get("checkpoint_sha256")
        if checkpoint_hash is not None:
            _require_hash(checkpoint_hash, "checkpoint_sha256")
        expected = metadata.get("provenance_sha256")
        bare_metadata = {
            key: value for key, value in metadata.items() if key != "provenance_sha256"
        }
        actual = _digest(
            np.asarray(drug_ids, dtype="<U"), tokens, padding_mask, available, bare_metadata
        )
        if expected != actual:
            raise ValueError("token feature artifact content checksum verification failed")
        for array in (tokens, padding_mask, available):
            array.setflags(write=False)
        object.__setattr__(
            self,
            "_id_to_index",
            MappingProxyType({drug_id: index for index, drug_id in enumerate(drug_ids)}),
        )

    @property
    def provenance_sha256(self) -> str:
        return str(self.metadata["provenance_sha256"])

    @classmethod
    def write(
        cls,
        path: str | Path,
        drug_ids: Sequence[str],
        tokens: np.ndarray,
        padding_mask: np.ndarray,
        available: np.ndarray,
        *,
        producer: str,
        producer_config: Mapping[str, Any],
        source_sha256: str,
        checkpoint_sha256: str | None = None,
    ) -> "TokenFeatureArtifact":
        raw_ids = list(drug_ids)
        if any(not isinstance(item, str) or not item.strip() for item in raw_ids):
            raise ValueError("drug IDs must be non-empty strings")
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError("duplicate drug ID in token feature artifact")
        ids = np.asarray(raw_ids, dtype="<U")
        values = {
            "tokens": np.asarray(tokens),
            "padding_mask": np.asarray(padding_mask),
            "available": np.asarray(available),
        }
        order = np.argsort(ids, kind="stable")
        ids = np.ascontiguousarray(ids[order])
        for name, array in values.items():
            if array.ndim == 0 or array.shape[0] != len(ids):
                raise ValueError(f"{name} must have a first dimension matching drug IDs")
            values[name] = np.ascontiguousarray(array[order])
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "producer": producer,
            "producer_config": copy.deepcopy(dict(producer_config)),
            "source_sha256": _require_hash(source_sha256, "source_sha256"),
            "checkpoint_sha256": (
                None
                if checkpoint_sha256 is None
                else _require_hash(checkpoint_sha256, "checkpoint_sha256")
            ),
            "drug_count": int(len(ids)),
            "token_count": int(values["tokens"].shape[1]) if values["tokens"].ndim == 3 else 0,
            "token_dim": int(values["tokens"].shape[2]) if values["tokens"].ndim == 3 else 0,
        }
        metadata["provenance_sha256"] = _digest(
            ids,
            values["tokens"],
            values["padding_mask"],
            values["available"],
            metadata,
        )
        cls(
            tuple(str(item) for item in ids.tolist()),
            values["tokens"],
            values["padding_mask"],
            values["available"],
            metadata,
        )
        payload = io.BytesIO()
        np.savez_compressed(
            payload,
            drug_ids=ids,
            tokens=values["tokens"],
            padding_mask=values["padding_mask"],
            available=values["available"],
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
        )
        _atomic_write(Path(path), payload.getvalue())
        return cls.load(path)

    @classmethod
    def load(cls, path: str | Path) -> "TokenFeatureArtifact":
        path = Path(path)
        checksum_path = path.with_name(f"{path.name}.sha256")
        try:
            payload = path.read_bytes()
            expected = checksum_path.read_text(encoding="ascii").strip()
        except OSError as exc:
            raise ValueError(f"unable to read token feature artifact {path}") from exc
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("token feature artifact file checksum verification failed")
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                required = {
                    "drug_ids",
                    "tokens",
                    "padding_mask",
                    "available",
                    "metadata_json",
                }
                if set(archive.files) != required:
                    raise ValueError("token feature artifact has an invalid field set")
                ids = np.asarray(archive["drug_ids"])
                tokens = np.asarray(archive["tokens"])
                padding_mask = np.asarray(archive["padding_mask"])
                available = np.asarray(archive["available"])
                metadata = json.loads(str(archive["metadata_json"].item()))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if isinstance(exc, ValueError) and "token feature" in str(exc):
                raise
            raise ValueError(f"unable to read token feature artifact {path}") from exc
        if ids.ndim != 1 or ids.dtype.kind not in {"U", "S"} or not isinstance(metadata, dict):
            raise ValueError("token feature artifact metadata or IDs are invalid")
        return cls(
            tuple(str(item) for item in ids.tolist()),
            tokens,
            padding_mask,
            available,
            metadata,
        )

    def lookup(self, drug_id: str) -> dict[str, Any]:
        key = drug_id.strip() if isinstance(drug_id, str) else ""
        if not key or key not in self._id_to_index:
            raise KeyError(f"unknown drug ID: {drug_id}")
        index = self._id_to_index[key]
        return {
            "tokens": self.tokens[index].copy(),
            "padding_mask": self.padding_mask[index].copy(),
            "available": bool(self.available[index]),
        }
