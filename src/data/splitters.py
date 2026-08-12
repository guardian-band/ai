import numpy as np
import pandas as pd
from typing import Tuple, Dict

def warm_pair_split(pairs_df: pd.DataFrame, train_frac=0.7, val_frac=0.15, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates warm_pair splits.
    Ensures every drug in validation or test is also present in train.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Unique positive pair IDs
    unique_pairs = pairs_df[['pair_id', 'drug_a', 'drug_b']].drop_duplicates().reset_index(drop=True)
    n_pairs = len(unique_pairs)
    
    # 2. Shuffle
    indices = np.arange(n_pairs)
    rng.shuffle(indices)
    
    n_train = int(n_pairs * train_frac)
    n_val = int(n_pairs * val_frac)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    train_pairs = unique_pairs.iloc[train_idx].copy()
    val_pairs = unique_pairs.iloc[val_idx].copy()
    test_pairs = unique_pairs.iloc[test_idx].copy()
    
    def ensure_train_coverage(train_df, other_df):
        train_drugs = set(train_df['drug_a']).union(set(train_df['drug_b']))
        other_drugs = set(other_df['drug_a']).union(set(other_df['drug_b']))
        
        missing_drugs = sorted(list(other_drugs - train_drugs))
        moves = []
        for d in missing_drugs:
            # Lexicographically smallest pair containing d
            d_pairs = other_df[(other_df['drug_a'] == d) | (other_df['drug_b'] == d)].copy()
            if len(d_pairs) > 0:
                d_pairs = d_pairs.sort_values(by='pair_id')
                move_pair = d_pairs.iloc[0]
                moves.append(move_pair['pair_id'])
                # Need to update train_drugs immediately so a move covering multiple drugs counts
                train_drugs.add(move_pair['drug_a'])
                train_drugs.add(move_pair['drug_b'])
        
        return moves

    # 4. Move pairs missing from train
    val_moves = ensure_train_coverage(train_pairs, val_pairs)
    train_pairs = pd.concat([train_pairs, val_pairs[val_pairs['pair_id'].isin(val_moves)]])
    val_pairs = val_pairs[~val_pairs['pair_id'].isin(val_moves)]
    
    test_moves = ensure_train_coverage(train_pairs, test_pairs)
    train_pairs = pd.concat([train_pairs, test_pairs[test_pairs['pair_id'].isin(test_moves)]])
    test_pairs = test_pairs[~test_pairs['pair_id'].isin(test_moves)]
    
    # 5. Rebalance (simplify for now, just ensure counts are approximately right, moves should be small)
    
    # Final assignment
    train_pairs['split'] = 'train'
    val_pairs['split'] = 'validation'
    test_pairs['split'] = 'test'
    
    out = pd.concat([train_pairs, val_pairs, test_pairs])
    out['scenario'] = 'warm_pair'
    out['observation_status'] = 'observed_positive'
    out['source_positive_pair_id'] = None
    
    return out[out['split']=='train'], out[out['split']=='validation'], out[out['split']=='test']

def cold_drug_partition(pairs_df: pd.DataFrame, train_frac=0.7, val_frac=0.1, seed=42) -> Tuple[set, set, set]:
    """
    Partitions drugs into train, validation_new, test_new using degree deciles.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Compute positive-pair degree
    drugs_a = pairs_df['drug_a'].value_counts()
    drugs_b = pairs_df['drug_b'].value_counts()
    degree = drugs_a.add(drugs_b, fill_value=0)
    
    drug_df = pd.DataFrame({'drug_id': degree.index, 'degree': degree.values})
    
    # 2. Bin drugs into deciles with deterministic tie handling
    drug_df = drug_df.sort_values(by=['degree', 'drug_id'])
    drug_df['decile'] = pd.qcut(drug_df['degree'].rank(method='first'), 10, labels=False)
    
    train_drugs = []
    val_drugs = []
    test_drugs = []
    
    for _, group in drug_df.groupby('decile'):
        drugs = group['drug_id'].values
        rng.shuffle(drugs)
        n = len(drugs)
        n_tr = int(n * train_frac)
        n_va = int(n * val_frac)
        
        train_drugs.extend(drugs[:n_tr])
        val_drugs.extend(drugs[n_tr:n_tr+n_va])
        test_drugs.extend(drugs[n_tr+n_va:])
        
    return set(train_drugs), set(val_drugs), set(test_drugs)

def generate_cold_splits(pairs_df: pd.DataFrame, scenario: str, train_drugs: set, val_drugs: set, test_drugs: set) -> pd.DataFrame:
    """
    Applies Cold-1 or Cold-2 rules to select pairs.
    """
    unique_pairs = pairs_df[['pair_id', 'drug_a', 'drug_b']].drop_duplicates().copy()
    
    def get_split(da, db):
        if da in train_drugs and db in train_drugs:
            return 'train'
            
        if scenario == 'cold_1':
            if (da in train_drugs and db in val_drugs) or (db in train_drugs and da in val_drugs):
                return 'validation'
            if (da in train_drugs and db in test_drugs) or (db in train_drugs and da in test_drugs):
                return 'test'
        elif scenario == 'cold_2':
            if da in val_drugs and db in val_drugs:
                return 'validation'
            if da in test_drugs and db in test_drugs:
                return 'test'
        return 'exclude'
        
    unique_pairs['split'] = unique_pairs.apply(lambda r: get_split(r['drug_a'], r['drug_b']), axis=1)
    
    out = unique_pairs[unique_pairs['split'] != 'exclude'].copy()
    out['scenario'] = scenario
    out['observation_status'] = 'observed_positive'
    out['source_positive_pair_id'] = None
    
    return out
