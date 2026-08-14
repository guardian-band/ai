import pytest
import pandas as pd
from src.data.splitters import warm_pair_split, cold_drug_partition, generate_cold_splits
from src.data.validators import (
    check_no_overlap,
    check_cold_1_test_endpoints,
    check_cold_2_test_endpoints,
    check_no_test_new_in_train_val
)

@pytest.fixture
def dummy_pairs():
    # 5 drugs, some pairs
    data = []
    pid = 1
    for i in range(1, 6):
        for j in range(i+1, 6):
            data.append({
                'pair_id': str(pid),
                'drug_a': f"D{i}",
                'drug_b': f"D{j}",
                'label_cui': 'C001'
            })
            pid += 1
    return pd.DataFrame(data)

def test_warm_pair_split(dummy_pairs):
    tr, va, te = warm_pair_split(dummy_pairs, seed=42)
    assert check_no_overlap(tr, va, te)
    
    # Assert every val/test drug is in train
    tr_drugs = set(tr['drug_a']).union(set(tr['drug_b']))
    va_drugs = set(va['drug_a']).union(set(va['drug_b']))
    te_drugs = set(te['drug_a']).union(set(te['drug_b']))
    
    assert va_drugs.issubset(tr_drugs)
    assert te_drugs.issubset(tr_drugs)

def test_cold_partitions(dummy_pairs):
    # we need more drugs to make deciles work
    # let's generate 20 drugs
    data = []
    pid = 1
    for i in range(1, 21):
        for j in range(i+1, i+3):
            if j <= 20:
                data.append({
                    'pair_id': str(pid),
                    'drug_a': f"D{i}",
                    'drug_b': f"D{j}",
                    'label_cui': 'C001'
                })
                pid += 1
    big_dummy = pd.DataFrame(data)
    
    tr_d, va_d, te_d = cold_drug_partition(big_dummy, seed=42)
    
    assert len(tr_d.intersection(va_d)) == 0
    assert len(tr_d.intersection(te_d)) == 0
    assert len(va_d.intersection(te_d)) == 0
    
    c1 = generate_cold_splits(big_dummy, 'cold_1', tr_d, va_d, te_d)
    tr_c1 = c1[c1['split'] == 'train']
    va_c1 = c1[c1['split'] == 'validation']
    te_c1 = c1[c1['split'] == 'test']
    
    assert check_no_overlap(tr_c1, va_c1, te_c1)
    assert check_no_test_new_in_train_val(tr_c1, va_c1, te_d)
    assert check_cold_1_test_endpoints(te_c1, tr_d, te_d)

    c2 = generate_cold_splits(big_dummy, 'cold_2', tr_d, va_d, te_d)
    tr_c2 = c2[c2['split'] == 'train']
    va_c2 = c2[c2['split'] == 'validation']
    te_c2 = c2[c2['split'] == 'test']
    
    assert check_no_overlap(tr_c2, va_c2, te_c2)
    assert check_no_test_new_in_train_val(tr_c2, va_c2, te_d)
    assert check_cold_2_test_endpoints(te_c2, te_d)


def test_cold_drug_partition_handles_small_drug_sets_without_qcut_failure():
    pairs = pd.DataFrame(
        [
            {"pair_id": f"p{i}", "drug_a": f"D{i}", "drug_b": f"D{i + 1}", "label_cui": "C001"}
            for i in range(1, 6)
        ]
    )
    train, validation, test = cold_drug_partition(pairs, train_frac=0.5, val_frac=0.25, seed=7)
    assert train and validation and test
    assert not (train & validation or train & test or validation & test)


def test_cold_drug_partition_apportions_each_degree_stratum_deterministically():
    rows = []
    pid = 0
    for base, degree in [(0, 1), (10, 2), (20, 3)]:
        for offset in range(6):
            drug = f"D{base + offset:02d}"
            for edge in range(degree):
                other = f"X{base + offset}_{edge}"
                rows.append({"pair_id": f"p{pid}", "drug_a": drug, "drug_b": other, "label_cui": "C001"})
                pid += 1
    frame = pd.DataFrame(rows)
    first = cold_drug_partition(frame, train_frac=0.5, val_frac=0.25, seed=9)
    second = cold_drug_partition(frame, train_frac=0.5, val_frac=0.25, seed=9)
    assert first == second
    degree = frame["drug_a"].value_counts().add(frame["drug_b"].value_counts(), fill_value=0)
    for group in [set(degree[degree == value].index) for value in sorted(degree.unique()) if (degree == value).sum() >= 3]:
        assert group & first[0]
        assert group & first[1]
        assert group & first[2]
