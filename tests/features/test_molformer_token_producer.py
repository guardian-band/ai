from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.features.molformer_token_producer import (
    MolFormerTokenProducer,
    export_molformer_token_artifact,
)


class FakeTokenizer:
    def __call__(self, smiles, **kwargs):
        assert kwargs["padding"] == "max_length"
        assert kwargs["truncation"] is True
        length = kwargs["max_length"]
        batch = len(smiles)
        input_ids = torch.arange(batch * length).reshape(batch, length)
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])[:batch, :length]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask, output_hidden_states, return_dict):
        assert output_hidden_states and return_dict
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        return SimpleNamespace(hidden_states=(hidden - 1, hidden))


def test_molformer_producer_returns_token_embeddings_and_padding_mask():
    producer = MolFormerTokenProducer(
        FakeTokenizer(),
        FakeModel(),
        model_id="ibm-research/MoLFormer-XL-both-10pct",
        revision="1" * 40,
        max_length=4,
        output_token_count=2,
        device="cpu",
    )
    tokens, padding = producer.encode(["CCO", "CC"])
    assert tokens.shape == (2, 2, 3)
    assert tokens.dtype == np.float32
    assert padding.tolist() == [[False, False], [False, False]]
    assert np.all(tokens[padding] == 0.0)


def test_molformer_loader_requires_pinned_revision_and_explicit_remote_code():
    with pytest.raises(ValueError, match="40-character"):
        MolFormerTokenProducer.from_pretrained(revision="main", allow_remote_code=True)
    with pytest.raises(ValueError, match="allow_remote_code"):
        MolFormerTokenProducer.from_pretrained(revision="1" * 40, allow_remote_code=False)


def test_molformer_export_writes_safe_artifact_in_batches(tmp_path):
    producer = MolFormerTokenProducer(
        FakeTokenizer(),
        FakeModel(),
        model_id="ibm-research/MoLFormer-XL-both-10pct",
        revision="1" * 40,
        max_length=4,
        output_token_count=2,
        device="cpu",
    )
    artifact = export_molformer_token_artifact(
        producer,
        drug_ids=["D2", "D1"],
        smiles=["CC", "OCC"],
        output_path=tmp_path / "molformer.npz",
        source_sha256="a" * 64,
        batch_size=1,
    )
    assert artifact.drug_ids == ("D1", "D2")
    assert artifact.tokens.shape == (2, 2, 3)
    assert artifact.metadata["producer_config"]["revision"] == "1" * 40


def test_molformer_export_marks_invalid_smiles_unavailable(tmp_path):
    producer = MolFormerTokenProducer(
        FakeTokenizer(),
        FakeModel(),
        model_id="ibm-research/MoLFormer-XL-both-10pct",
        revision="1" * 40,
        max_length=4,
        output_token_count=2,
        device="cpu",
    )
    artifact = export_molformer_token_artifact(
        producer,
        drug_ids=["D3", "D1", "D2"],
        smiles=["", "CC", None],
        output_path=tmp_path / "molformer-missing.npz",
        source_sha256="b" * 64,
        batch_size=1,
    )

    assert artifact.drug_ids == ("D1", "D2", "D3")
    assert artifact.available.tolist() == [True, False, False]
    assert np.all(artifact.tokens[1:] == 0.0)
    assert np.all(artifact.padding_mask[1:])
