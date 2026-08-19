"""Versioned, non-pickle cached latent-token artifacts for CPU inference."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from collections.abc import Iterable, Mapping
import copy
import hashlib
import io
import json
from pathlib import Path
import os
import re
import tempfile
from typing import Any
from types import MappingProxyType

import numpy as np
import torch


SCHEMA_VERSION = 1


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 hash")
    return value


def _resolve_feature_artifact_hash(
    multimodal_feature_artifact_hash: str | None,
    modality_provenance_hash: str | None,
) -> str:
    """Resolve the exact feature-artifact hash, retaining a compatibility alias."""

    if multimodal_feature_artifact_hash is not None and modality_provenance_hash is not None:
        if multimodal_feature_artifact_hash != modality_provenance_hash:
            raise ValueError("multimodal feature artifact hashes disagree")
    value = multimodal_feature_artifact_hash or modality_provenance_hash
    return _require_hash(value, "multimodal_feature_artifact_hash")


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


def _canonical_digest(
    drug_ids: np.ndarray,
    tokens: np.ndarray,
    availability: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(_json_ready(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name, array in (("drug_ids", drug_ids), ("tokens", tokens), ("availability", availability)):
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _serialize(
    drug_ids: np.ndarray,
    tokens: np.ndarray,
    availability: np.ndarray,
    metadata: dict[str, Any],
) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        drug_ids=drug_ids,
        tokens=tokens,
        availability=availability,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )
    return buffer.getvalue()


@dataclass(frozen=True)
class CachedTokenArtifact:
    """Validated latent tokens with O(1) drug-ID lookup."""

    drug_ids: tuple[str, ...]
    tokens: np.ndarray
    availability: np.ndarray
    metadata: Mapping[str, Any]
    _id_to_index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        tokens = np.array(self.tokens, copy=True, order="C")
        availability = np.array(self.availability, copy=True, order="C")
        tokens.setflags(write=False)
        availability.setflags(write=False)
        metadata = _freeze_metadata(self.metadata)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "drug_ids", tuple(self.drug_ids))
        if self.tokens.ndim != 3 or self.tokens.dtype != np.float32:
            raise ValueError("cached tokens must be float32 with shape [drugs, tokens, dimensions]")
        if self.availability.dtype != np.bool_ or self.availability.shape != self.tokens.shape[:2]:
            raise ValueError("availability must be bool with shape [drugs, tokens]")
        if len(self.drug_ids) != self.tokens.shape[0]:
            raise ValueError("drug ID count does not match cached token rows")
        if len(set(self.drug_ids)) != len(self.drug_ids) or any(not drug_id for drug_id in self.drug_ids):
            raise ValueError("cached drug IDs must be unique and non-empty")
        if not np.isfinite(self.tokens).all():
            raise ValueError("cached tokens must contain only finite values")
        if tuple(sorted(self.drug_ids)) != self.drug_ids:
            raise ValueError("cached drug IDs must be sorted deterministically")
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported cached-token schema version")
        _require_hash(self.metadata.get("teacher_provenance_hash"), "teacher_provenance_hash")
        _require_hash(self.metadata.get("teacher_checkpoint_hash"), "teacher_checkpoint_hash")
        _require_hash(self.metadata.get("teacher_config_hash"), "teacher_config_hash")
        _require_hash(self.metadata.get("modality_provenance_hash"), "modality_provenance_hash")
        expected = self.metadata.get("payload_sha256")
        if not isinstance(expected, str) or expected != _canonical_digest(
            np.asarray(self.drug_ids, dtype="<U"), self.tokens, self.availability, {
                key: value for key, value in self.metadata.items() if key != "payload_sha256"
            }
        ):
            raise ValueError("cached-token artifact checksum verification failed")
        object.__setattr__(
            self,
            "_id_to_index",
            MappingProxyType({drug_id: index for index, drug_id in enumerate(self.drug_ids)}),
        )

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def token_dim(self) -> int:
        return int(self.tokens.shape[2])

    @classmethod
    def write(
        cls,
        path: str | Path,
        drug_ids: list[str] | tuple[str, ...],
        tokens: np.ndarray,
        availability: np.ndarray,
        *,
        teacher_provenance_hash: str,
        multimodal_feature_artifact_hash: str | None = None,
        modality_provenance_hash: str | None = None,
        teacher_checkpoint_hash: str | None = None,
        teacher_config_hash: str | None = None,
    ) -> "CachedTokenArtifact":
        path = Path(path)
        ids = np.asarray([str(drug_id).strip() for drug_id in drug_ids], dtype="<U")
        if ids.ndim != 1 or len(ids) == 0 or any(not drug_id for drug_id in ids.tolist()):
            raise ValueError("drug id values must be a non-empty sequence of non-empty strings")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("duplicate drug ID in cached-token artifact")
        if not isinstance(tokens, np.ndarray) or tokens.ndim != 3:
            raise ValueError("tokens must be a 3D numpy array")
        if tokens.dtype != np.float32:
            raise ValueError("tokens must use float32")
        if not np.isfinite(tokens).all():
            raise ValueError("tokens must contain only finite values")
        if not isinstance(availability, np.ndarray) or availability.dtype != np.bool_:
            raise ValueError("availability must be a boolean numpy array")
        if availability.shape != tokens.shape[:2]:
            raise ValueError("availability must have shape [drugs, tokens]")
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        tokens = np.ascontiguousarray(tokens[order])
        availability = np.ascontiguousarray(availability[order])
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "drug_count": int(len(ids)),
            "token_count": int(tokens.shape[1]),
            "token_dim": int(tokens.shape[2]),
            "teacher_provenance_hash": _require_hash(
                teacher_provenance_hash, "teacher_provenance_hash"
            ),
            "teacher_checkpoint_hash": _require_hash(
                teacher_checkpoint_hash or teacher_provenance_hash,
                "teacher_checkpoint_hash",
            ),
            "teacher_config_hash": _require_hash(
                teacher_config_hash or teacher_provenance_hash,
                "teacher_config_hash",
            ),
            "modality_provenance_hash": _resolve_feature_artifact_hash(
                multimodal_feature_artifact_hash, modality_provenance_hash
            ),
        }
        metadata["payload_sha256"] = _canonical_digest(
            ids,
            tokens,
            availability,
            metadata,
        )
        payload = _serialize(ids, tokens, availability, metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        checksum_path = path.with_name(f"{path.name}.sha256")
        checksum_fd, checksum_temporary = tempfile.mkstemp(
            prefix=f".{checksum_path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            with os.fdopen(checksum_fd, "w", encoding="ascii") as handle:
                handle.write(hashlib.sha256(payload).hexdigest())
                handle.write("\n")
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
    def load(cls, path: str | Path) -> "CachedTokenArtifact":
        path = Path(path)
        checksum_path = path.with_name(f"{path.name}.sha256")
        try:
            expected_file_checksum = checksum_path.read_text(encoding="ascii").strip()
            actual_file_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"unable to verify cached-token artifact checksum for {path}") from exc
        if expected_file_checksum != actual_file_checksum:
            raise ValueError("cached-token artifact file checksum verification failed")
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = set(archive.files)
                required = {"drug_ids", "tokens", "availability", "metadata_json"}
                if keys != required:
                    raise ValueError("cached-token artifact has an invalid field set")
                ids = np.asarray(archive["drug_ids"])
                tokens = np.asarray(archive["tokens"])
                availability = np.asarray(archive["availability"])
                raw_metadata = archive["metadata_json"].item()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, ValueError) and "cached-token" in str(exc):
                raise
            raise ValueError(f"unable to read cached-token artifact {path}: {exc}") from exc
        if ids.ndim != 1 or ids.dtype.kind not in {"U", "S"}:
            raise ValueError("cached-token artifact drug_ids must be a string vector")
        try:
            metadata = json.loads(str(raw_metadata))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("cached-token artifact metadata is invalid") from exc
        if not isinstance(metadata, dict):
            raise ValueError("cached-token artifact metadata must be an object")
        artifact = cls(tuple(str(item) for item in ids.tolist()), tokens, availability, metadata)
        if metadata.get("drug_count") != len(artifact.drug_ids):
            raise ValueError("cached-token artifact drug_count is inconsistent")
        if metadata.get("token_count") != artifact.token_count or metadata.get("token_dim") != artifact.token_dim:
            raise ValueError("cached-token artifact shape metadata is inconsistent")
        return artifact

    def lookup(self, drug_id: str) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(drug_id, str) or not drug_id.strip():
            raise KeyError("unknown or empty drug ID")
        index = self._id_to_index.get(drug_id.strip())
        if index is None:
            raise KeyError(f"unknown drug ID: {drug_id}")
        return self.tokens[index].copy(), self.availability[index].copy()

    def lookup_pair(
        self, drug_a: str, drug_b: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tokens_a, available_a = self.lookup(drug_a)
        tokens_b, available_b = self.lookup(drug_b)
        return tokens_a, available_a, tokens_b, available_b


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def build_cached_token_artifact(
    teacher: torch.nn.Module,
    batches: Iterable[Mapping[str, Any] | tuple[Iterable[str], Mapping[str, Any]]],
    path: str | Path,
    *,
    teacher_checkpoint_hash: str,
    teacher_config_hash: str,
    multimodal_feature_artifact_hash: str | None = None,
    modality_provenance_hash: str | None = None,
) -> CachedTokenArtifact:
    """Build a validated cache by exporting the teacher's learned resampler.

    ``multimodal_feature_artifact_hash`` must be the SHA-256 of the exact
    ``MultimodalFeatureArtifact`` file used to produce the inputs.  A batch is either
    ``{"drug_ids": [...], "inputs": drug_mapping}`` or a
    two-tuple ``(drug_ids, drug_mapping)``.  The teacher must already be in
    ``eval()`` mode; only inference-mode encoder calls are made.
    """

    if not isinstance(teacher, torch.nn.Module):
        raise TypeError("teacher must be a torch module")
    if teacher.training:
        raise ValueError("teacher must be in eval() mode before cache export")
    encoder = getattr(teacher, "encode_drug_for_cache", None)
    if not callable(encoder):
        raise ValueError("teacher must expose encode_drug_for_cache")
    feature_hash = _resolve_feature_artifact_hash(
        multimodal_feature_artifact_hash, modality_provenance_hash
    )
    try:
        device = next(teacher.parameters()).device
    except StopIteration as exc:
        raise ValueError("teacher must have parameters to determine its device") from exc

    all_ids: list[str] = []
    all_tokens: list[np.ndarray] = []
    seen: set[str] = set()
    for batch in batches:
        if isinstance(batch, Mapping):
            drug_ids = batch.get("drug_ids")
            inputs = batch.get("inputs")
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            drug_ids, inputs = batch
        else:
            raise ValueError("each cache batch must contain drug_ids and inputs")
        if isinstance(drug_ids, (str, bytes)) or drug_ids is None:
            raise ValueError("cache batch drug_ids must be a sequence of IDs")
        ids = list(drug_ids)
        if not ids:
            raise ValueError("cache batch must contain at least one drug ID")
        normalized_ids: list[str] = []
        for drug_id in ids:
            if not isinstance(drug_id, str) or not drug_id.strip():
                raise ValueError("cache batch contains an empty or invalid drug id")
            normalized = drug_id.strip()
            if normalized in seen:
                raise ValueError(f"duplicate drug ID during cache export: {normalized}")
            seen.add(normalized)
            normalized_ids.append(normalized)
        if not isinstance(inputs, Mapping):
            raise ValueError("cache batch inputs must be a modality mapping")
        with torch.inference_mode():
            encoded = encoder(_move_to_device(inputs, device))
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 3:
            raise ValueError("teacher cache encoder must return a 3D tensor")
        expected_count = getattr(teacher, "cache_token_count", None)
        expected_dim = getattr(teacher, "cache_token_dim", None)
        if encoded.shape[0] != len(normalized_ids) or encoded.shape[1:] != (
            expected_count,
            expected_dim,
        ):
            raise ValueError("teacher cache encoder output does not match drug IDs or cache schema")
        encoded = encoded.detach().to(device="cpu")
        if not torch.isfinite(encoded).all():
            raise ValueError("teacher cache encoder emitted non-finite values")
        all_ids.extend(normalized_ids)
        all_tokens.append(encoded.numpy().astype(np.float32, copy=True))
    if not all_ids or not all_tokens:
        raise ValueError("cache export requires at least one drug batch")
    tokens = np.concatenate(all_tokens, axis=0)
    availability = np.ones(tokens.shape[:2], dtype=np.bool_)
    CachedTokenArtifact.write(
        path,
        all_ids,
        tokens,
        availability,
        teacher_provenance_hash=teacher_checkpoint_hash,
        teacher_checkpoint_hash=teacher_checkpoint_hash,
        teacher_config_hash=teacher_config_hash,
        multimodal_feature_artifact_hash=feature_hash,
    )
    return CachedTokenArtifact.load(path)
