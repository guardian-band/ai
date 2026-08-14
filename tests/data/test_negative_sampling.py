import pytest
import pandas as pd
from src.data.negative_sampling import generate_negative_samples, calculate_similarity_bin

def test_calculate_similarity_bin():
    assert calculate_similarity_bin(0.0) == 0
    assert calculate_similarity_bin(0.19) == 0
    assert calculate_similarity_bin(0.20) == 1
    assert calculate_similarity_bin(0.99) == 4
    assert calculate_similarity_bin(1.0) == 4

def test_generate_negative_samples():
    positive_pairs = pd.DataFrame([
        {'pair_id': 'p1', 'drug_a': 'D1', 'drug_b': 'D2', 'label_cui': 'C1', 'split': 'train', 'scenario': 'warm_pair'},
        {'pair_id': 'p2', 'drug_a': 'D3', 'drug_b': 'D4', 'label_cui': 'C2', 'split': 'train', 'scenario': 'warm_pair'},
    ])
    
    known_positives = {('D1', 'D2'), ('D3', 'D4')}
    allowed_drugs = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    
    # Degrees: D1:0, D2:0, D3:1, D4:1, D5:0, D6:1
    drug_degrees = {'D1': 0, 'D2': 0, 'D3': 1, 'D4': 1, 'D5': 0, 'D6': 1}
    
    # Similarities: all 0 for simplicity
    drug_sims = {}
    
    controls = generate_negative_samples(
        positive_pairs, known_positives, allowed_drugs,
        drug_degrees, drug_sims, max_attempts=100
    )
    
    assert len(controls) == 2
    
    # Ensure no overlap with known positives
    for _, row in controls.iterrows():
        da, db = row['drug_a'], row['drug_b']
        assert (da, db) not in known_positives
        assert da != db
        
    # Ensure properties carried over
    assert set(controls['split']) == {'train'}
    assert set(controls['scenario']) == {'warm_pair'}
    assert controls['pair_id'].is_unique
    assert all(controls['pair_id'] != controls['source_positive_pair_id'])
