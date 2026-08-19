"""Pinned MolFormer token extraction without import-time downloads."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.features.token_feature_artifact import TokenFeatureArtifact


DEFAULT_MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class MolFormerTokenProducer:
    """Extract final hidden-state tokens from one pinned MolFormer revision."""

    def __init__(
        self,
        tokenizer: Any,
        model: torch.nn.Module,
        *,
        model_id: str,
        revision: str,
        max_length: int = 128,
        output_token_count: int = 8,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if _COMMIT_PATTERN.fullmatch(revision) is None:
            raise ValueError("revision must be a pinned 40-character lowercase commit hash")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if (
            isinstance(output_token_count, bool)
            or not isinstance(output_token_count, int)
            or output_token_count <= 0
        ):
            raise ValueError("output_token_count must be a positive integer")
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.model_id = model_id
        self.revision = revision
        self.max_length = max_length
        self.output_token_count = output_token_count
        self.device = torch.device(device)
        self.model.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        *,
        revision: str,
        allow_remote_code: bool,
        model_id: str = DEFAULT_MODEL_ID,
        max_length: int = 128,
        output_token_count: int = 8,
        device: str | torch.device = "cpu",
        local_files_only: bool = False,
    ) -> "MolFormerTokenProducer":
        if _COMMIT_PATTERN.fullmatch(revision) is None:
            raise ValueError("revision must be a pinned 40-character lowercase commit hash")
        if allow_remote_code is not True:
            raise ValueError(
                "MolFormer requires custom Hugging Face code; set allow_remote_code=True explicitly"
            )
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        common = {
            "revision": revision,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
        model = AutoModelForMaskedLM.from_pretrained(model_id, **common)
        return cls(
            tokenizer,
            model,
            model_id=model_id,
            revision=revision,
            max_length=max_length,
            output_token_count=output_token_count,
            device=device,
        )

    def encode(self, smiles: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        values = list(smiles)
        if not values or any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError("smiles must be a non-empty sequence of non-empty strings")
        encoded = self.tokenizer(
            values,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "attention_mask" not in encoded:
            raise RuntimeError("MolFormer tokenizer did not return attention_mask")
        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("MolFormer model did not return hidden states")
        tokens = hidden_states[-1].detach().to(dtype=torch.float32, device="cpu")
        padding_mask = ~model_inputs["attention_mask"].to(dtype=torch.bool, device="cpu")
        if tokens.ndim != 3 or tokens.shape[:2] != padding_mask.shape:
            raise RuntimeError("MolFormer emitted an invalid token tensor shape")
        tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        pooled = tokens.new_zeros(
            (tokens.shape[0], self.output_token_count, tokens.shape[2])
        )
        pooled_mask = torch.ones(
            (tokens.shape[0], self.output_token_count), dtype=torch.bool
        )
        for batch_index in range(tokens.shape[0]):
            valid_tokens = tokens[batch_index, ~padding_mask[batch_index]]
            count = valid_tokens.shape[0]
            for token_index in range(self.output_token_count):
                start = token_index * count // self.output_token_count
                stop = (token_index + 1) * count // self.output_token_count
                if stop > start:
                    pooled[batch_index, token_index] = valid_tokens[start:stop].mean(dim=0)
                    pooled_mask[batch_index, token_index] = False
        if not torch.isfinite(pooled).all():
            raise RuntimeError("MolFormer emitted non-finite token values")
        return pooled.numpy(), pooled_mask.numpy()


def export_molformer_token_artifact(
    producer: MolFormerTokenProducer,
    *,
    drug_ids: Sequence[str],
    smiles: Sequence[str],
    output_path: str | Path,
    source_sha256: str,
    batch_size: int = 32,
) -> TokenFeatureArtifact:
    """Canonicalize SMILES, extract pinned MolFormer tokens, and persist them."""

    ids = list(drug_ids)
    values = list(smiles)
    if len(ids) != len(values) or not ids:
        raise ValueError("drug_ids and smiles must be non-empty aligned sequences")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    from rdkit import Chem

    canonical: list[str] = []
    for drug_id, value in zip(ids, values, strict=True):
        if not isinstance(drug_id, str) or not drug_id.strip():
            raise ValueError("drug IDs must be non-empty strings")
        molecule = Chem.MolFromSmiles(value) if isinstance(value, str) else None
        if molecule is None:
            raise ValueError(f"invalid SMILES for drug {drug_id}")
        canonical.append(Chem.MolToSmiles(molecule, canonical=True))
    token_batches: list[np.ndarray] = []
    mask_batches: list[np.ndarray] = []
    for start in range(0, len(ids), batch_size):
        tokens, padding_mask = producer.encode(canonical[start : start + batch_size])
        token_batches.append(tokens)
        mask_batches.append(padding_mask)
    tokens = np.ascontiguousarray(np.concatenate(token_batches, axis=0), dtype=np.float32)
    padding_mask = np.ascontiguousarray(np.concatenate(mask_batches, axis=0), dtype=bool)
    return TokenFeatureArtifact.write(
        output_path,
        ids,
        tokens,
        padding_mask,
        np.ones(len(ids), dtype=bool),
        producer="molformer",
        producer_config={
            "model_id": producer.model_id,
            "revision": producer.revision,
            "max_length": producer.max_length,
            "output_token_count": producer.output_token_count,
        },
        source_sha256=source_sha256,
    )
