import pytest
import pandas as pd
from src.data.benchmark_builder import canonicalize_drug_pair, generate_pair_id, parse_stitch_id

def test_canonicalize_drug_pair():
    assert canonicalize_drug_pair('DB001', 'DB002') == ('DB001', 'DB002')
    assert canonicalize_drug_pair('DB002', 'DB001') == ('DB001', 'DB002')
    
def test_generate_pair_id():
    # reversed input pairs produce identical pair_id
    id1 = generate_pair_id('DB001', 'DB002')
    id2 = generate_pair_id('DB002', 'DB001')
    assert id1 == id2
    assert len(id1) == 24

def test_parse_stitch_id():
    assert parse_stitch_id('CID0000123') == 123
    assert parse_stitch_id('CID1000123') == 123
    assert parse_stitch_id('123') == 123
    assert parse_stitch_id('invalid') is None

def test_select_labels_mutation():
    from src.data.benchmark_builder import select_labels
    
    # 3 train pairs, 2 val pairs
    data = [
        {'pair_id': '1', 'drug_a': 'A', 'drug_b': 'B', 'label_cui': 'C1'},
        {'pair_id': '2', 'drug_a': 'C', 'drug_b': 'D', 'label_cui': 'C1'},
        {'pair_id': '3', 'drug_a': 'E', 'drug_b': 'F', 'label_cui': 'C2'},
        {'pair_id': '4', 'drug_a': 'G', 'drug_b': 'H', 'label_cui': 'C2'},
        {'pair_id': '5', 'drug_a': 'I', 'drug_b': 'J', 'label_cui': 'C3'},
    ]
    df_original = pd.DataFrame(data)
    train_ids = {'1', '2', '3'} # pair_id 1,2,3 are train. So C1 has 2, C2 has 1, C3 has 0.
    
    selected_1, meta_1 = select_labels(df_original, train_ids, top_k=2)
    
    assert len(selected_1) == 2
    assert selected_1.iloc[0]['label_cui'] == 'C1'
    assert selected_1.iloc[1]['label_cui'] == 'C2'
    
    # Mutate the val/test pairs (pair 4 and 5)
    df_mutated = df_original.copy()
    df_mutated.loc[df_mutated['pair_id'] == '4', 'label_cui'] = 'C3'
    df_mutated.loc[df_mutated['pair_id'] == '5', 'label_cui'] = 'C1'
    
    selected_2, meta_2 = select_labels(df_mutated, train_ids, top_k=2)
    
    import json
    assert json.dumps(meta_1['labels'], sort_keys=True) == json.dumps(meta_2['labels'], sort_keys=True)
    # The selection process is entirely invariant to val/test labels.
