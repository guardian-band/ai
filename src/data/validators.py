import pandas as pd
from typing import Set

def check_no_overlap(train_pairs: pd.DataFrame, val_pairs: pd.DataFrame, test_pairs: pd.DataFrame) -> bool:
    train_ids = set(train_pairs['pair_id'])
    val_ids = set(val_pairs['pair_id'])
    test_ids = set(test_pairs['pair_id'])
    
    return len(train_ids.intersection(val_ids)) == 0 and \
           len(train_ids.intersection(test_ids)) == 0 and \
           len(val_ids.intersection(test_ids)) == 0

def check_cold_1_test_endpoints(test_pairs: pd.DataFrame, train_drugs: Set[str], test_new_drugs: Set[str]) -> bool:
    for _, row in test_pairs.iterrows():
        da, db = row['drug_a'], row['drug_b']
        c1 = da in train_drugs and db in test_new_drugs
        c2 = db in train_drugs and da in test_new_drugs
        if not (c1 or c2):
            return False
    return True

def check_cold_2_test_endpoints(test_pairs: pd.DataFrame, test_new_drugs: Set[str]) -> bool:
    for _, row in test_pairs.iterrows():
        da, db = row['drug_a'], row['drug_b']
        if da not in test_new_drugs or db not in test_new_drugs:
            return False
    return True

def check_no_test_new_in_train_val(train_pairs: pd.DataFrame, val_pairs: pd.DataFrame, test_new_drugs: Set[str]) -> bool:
    train_drugs = set(train_pairs['drug_a']).union(set(train_pairs['drug_b']))
    val_drugs = set(val_pairs['drug_a']).union(set(val_pairs['drug_b']))
    
    return len(train_drugs.intersection(test_new_drugs)) == 0 and \
           len(val_drugs.intersection(test_new_drugs)) == 0
