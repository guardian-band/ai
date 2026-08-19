import pytest
import torch

from src.models.molecular_mpnn import (
    MolecularMPNN,
    collate_molecular_graphs,
    export_mpnn_token_artifact,
    featurize_smiles,
    load_mpnn_checkpoint,
    masked_atom_loss,
    save_mpnn_checkpoint,
)


def test_smiles_featurizer_is_canonical_and_bidirectional():
    first = featurize_smiles("OCC")
    second = featurize_smiles("CCO")
    assert torch.equal(first.atom_features, second.atom_features)
    assert torch.equal(first.edge_index, second.edge_index)
    assert first.edge_index.shape[0] == 2
    assert first.edge_index.shape[1] == first.bond_features.shape[0]
    assert first.edge_index.shape[1] == 4
    with pytest.raises(ValueError, match="invalid SMILES"):
        featurize_smiles("not-a-smiles")


def test_mpnn_returns_fixed_masked_atom_tokens_and_supports_pretraining():
    torch.manual_seed(4)
    batch = collate_molecular_graphs(
        [featurize_smiles("CCO"), featurize_smiles("c1ccccc1")]
    )
    model = MolecularMPNN(hidden_dim=16, layers=2, token_count=5)
    tokens, padding_mask = model.encode_tokens(batch)
    assert tokens.shape == (2, 5, 16)
    assert padding_mask.shape == (2, 5)
    assert padding_mask[0].tolist() == [False, False, False, True, True]
    assert torch.all(tokens[padding_mask] == 0)
    loss = masked_atom_loss(model, batch, mask_probability=0.5, generator_seed=8)
    loss.backward()
    assert loss.isfinite()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mpnn_is_permutation_invariant_to_batch_order():
    torch.manual_seed(5)
    model = MolecularMPNN(hidden_dim=12, layers=2, token_count=4).eval()
    first = collate_molecular_graphs([featurize_smiles("CC"), featurize_smiles("CCO")])
    second = collate_molecular_graphs([featurize_smiles("CCO"), featurize_smiles("CC")])
    with torch.inference_mode():
        first_tokens, _ = model.encode_tokens(first)
        second_tokens, _ = model.encode_tokens(second)
    assert torch.allclose(first_tokens[0], second_tokens[1], atol=1e-6)
    assert torch.allclose(first_tokens[1], second_tokens[0], atol=1e-6)


def test_mpnn_checkpoint_and_token_export_are_reproducible(tmp_path):
    torch.manual_seed(12)
    model = MolecularMPNN(hidden_dim=8, layers=2, token_count=4).eval()
    checkpoint = tmp_path / "mpnn.pt"
    checkpoint_hash = save_mpnn_checkpoint(
        model,
        checkpoint,
        source_sha256="a" * 64,
        training_config={"epochs": 1, "objective": "masked_atom_identity"},
    )
    loaded, metadata = load_mpnn_checkpoint(checkpoint, expected_sha256=checkpoint_hash)
    artifact = export_mpnn_token_artifact(
        loaded,
        drug_ids=["D2", "D1"],
        smiles=["CCO", "CC"],
        output_path=tmp_path / "mpnn.npz",
        source_sha256="a" * 64,
        checkpoint_sha256=checkpoint_hash,
        batch_size=1,
    )
    assert metadata["model_config"] == {"hidden_dim": 8, "layers": 2, "token_count": 4}
    assert artifact.drug_ids == ("D1", "D2")
    assert artifact.tokens.shape == (2, 4, 8)


def test_mpnn_export_marks_invalid_smiles_unavailable(tmp_path):
    torch.manual_seed(13)
    model = MolecularMPNN(hidden_dim=8, layers=1, token_count=4).eval()
    artifact = export_mpnn_token_artifact(
        model,
        drug_ids=["D3", "D1", "D2"],
        smiles=[None, "CC", ""],
        output_path=tmp_path / "mpnn-missing.npz",
        source_sha256="c" * 64,
        checkpoint_sha256="d" * 64,
        batch_size=1,
    )

    assert artifact.drug_ids == ("D1", "D2", "D3")
    assert artifact.available.tolist() == [True, False, False]
    assert (artifact.tokens[1:] == 0).all()
    assert artifact.padding_mask[1:].all()
