import os
import hashlib
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from src.training.reproducibility import repository_root
from src.data.schemas import validate_dataframe

def canonicalize_drug_pair(da: str, db: str) -> Tuple[str, str]:
    """Returns drugs in lexicographical order (smaller, larger)."""
    return (da, db) if da <= db else (db, da)

def generate_pair_id(drug_a: str, drug_b: str) -> str:
    """Generates canonical SHA-256 hash of the pair, taking first 24 hex chars."""
    if not isinstance(drug_a, str) or not isinstance(drug_b, str):
        return None
    # Ensure they are sorted before hashing just in case
    d1, d2 = canonicalize_drug_pair(drug_a, drug_b)
    pair_string = f"{d1}\0{d2}".encode('utf-8')
    return hashlib.sha256(pair_string).hexdigest()[:24]

def parse_stitch_id(stitch_str) -> int:
    """Parses various STITCH/CID formats into integer CIDs."""
    if pd.isna(stitch_str) or not isinstance(stitch_str, str):
        return None
    cleaned = stitch_str.replace("CID000", "").replace("CID00", "").replace("CID0", "").replace("CID1", "").replace("CID", "")
    try:
        return int(cleaned)
    except ValueError:
        return None

def build_canonical_triples(
    drugs_master_path: str,
    side_effects_path: str,
    biosnap_path: str,
    output_parquet: str
) -> Dict[str, int]:
    """
    Builds the canonical observed-positive triples table.
    Returns a dictionary of counts and rejection reasons.
    """
    # 1. Load data
    drugs_df = pd.read_csv(drugs_master_path)
    side_eff_df = pd.read_csv(side_effects_path, usecols=['umls_cui_from_meddra', 'side_effect_name'])
    combo_df = pd.read_csv(biosnap_path, comment='#', names=['STITCH 1', 'STITCH 2', 'Polypharmacy Side Effect', 'Side Effect Name'])
    
    # 2. Validate columns
    validate_dataframe(drugs_df, "drugs_master")
    validate_dataframe(side_eff_df, "side_effects")
    validate_dataframe(combo_df, "biosnap")
    
    stats = {"raw_edges": len(combo_df)}
    
    # Extract DrugBank mappings
    drugs_df['pubchem_cid_clean'] = pd.to_numeric(drugs_df['pubchem_compound_id'], errors='coerce')
    valid_db_ids = set(drugs_df['drugbank_id'].dropna())
    cid_to_drugbank = dict(zip(drugs_df['pubchem_cid_clean'].dropna().astype(int), drugs_df['drugbank_id']))
    valid_cuis = set(side_eff_df['umls_cui_from_meddra'].dropna())
    
    # Parse STITCH and map
    combo_df['cid_1'] = combo_df['STITCH 1'].apply(parse_stitch_id)
    combo_df['cid_2'] = combo_df['STITCH 2'].apply(parse_stitch_id)
    
    # Map to DrugBank ID
    combo_df['db_1'] = combo_df['cid_1'].map(cid_to_drugbank)
    combo_df['db_2'] = combo_df['cid_2'].map(cid_to_drugbank)
    
    # Fallback to direct DrugBank ID if STITCH format was actually a DB ID
    combo_df['db_1'] = combo_df['db_1'].fillna(combo_df['STITCH 1'].where(combo_df['STITCH 1'].isin(valid_db_ids)))
    combo_df['db_2'] = combo_df['db_2'].fillna(combo_df['STITCH 2'].where(combo_df['STITCH 2'].isin(valid_db_ids)))
    
    # Rejection: Unmapped drugs
    unmapped_drug1 = combo_df['db_1'].isna()
    unmapped_drug2 = combo_df['db_2'].isna()
    stats["rejected_unmapped_drug"] = int((unmapped_drug1 | unmapped_drug2).sum())
    
    df_mapped = combo_df[~(unmapped_drug1 | unmapped_drug2)].copy()
    
    # Rejection: Invalid CUI
    invalid_cui = ~df_mapped['Polypharmacy Side Effect'].isin(valid_cuis)
    stats["rejected_invalid_cui"] = int(invalid_cui.sum())
    df_mapped = df_mapped[~invalid_cui].copy()
    
    # Rejection: Self-pairs
    self_pairs = df_mapped['db_1'] == df_mapped['db_2']
    stats["rejected_self_pairs"] = int(self_pairs.sum())
    df_mapped = df_mapped[~self_pairs].copy()
    
    # 3. Canonicalize pairs
    pair_a = np.minimum(df_mapped['db_1'].values, df_mapped['db_2'].values)
    pair_b = np.maximum(df_mapped['db_1'].values, df_mapped['db_2'].values)
    
    df_mapped['drug_a'] = pair_a
    df_mapped['drug_b'] = pair_b
    df_mapped['label_cui'] = df_mapped['Polypharmacy Side Effect']
    
    # 4. Deduplicate
    pre_dedup_len = len(df_mapped)
    df_mapped = df_mapped.drop_duplicates(subset=['drug_a', 'drug_b', 'label_cui'])
    stats["rejected_duplicates"] = int(pre_dedup_len - len(df_mapped))
    
    # Generate pair_id and observation_status
    df_mapped['pair_id'] = df_mapped.apply(lambda row: generate_pair_id(row['drug_a'], row['drug_b']), axis=1)
    df_mapped['observation_status'] = 'observed_positive'
    
    # 5. Persist
    final_df = df_mapped[['pair_id', 'drug_a', 'drug_b', 'label_cui', 'observation_status']]
    final_df = final_df.sort_values(by=['pair_id', 'label_cui']).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    final_df.to_parquet(output_parquet, index=False)
    
    stats["final_canonical_triples"] = int(len(final_df))
    return stats

def select_labels(canonical_triples: pd.DataFrame, train_pair_ids: set, top_k: int = 100) -> Tuple[pd.DataFrame, Dict]:
    """
    Selects top_k labels strictly from the train pairs.
    Returns the ordered labels dataframe and metadata dictionary.
    """
    train_triples = canonical_triples[canonical_triples['pair_id'].isin(train_pair_ids)]
    
    # Count unique train pairs per CUI
    counts = train_triples.groupby('label_cui')['pair_id'].nunique().reset_index()
    counts.columns = ['label_cui', 'train_positive_count']
    
    # Sort by descending count, then ascending CUI
    counts = counts.sort_values(by=['train_positive_count', 'label_cui'], ascending=[False, True]).reset_index(drop=True)
    
    selected = counts.head(top_k).copy()
    selected['index'] = np.arange(len(selected))
    
    selected_cuis = set(selected['label_cui'])
    
    # OOV counts (labels present in total but not in selected train)
    val_test_triples = canonical_triples[~canonical_triples['pair_id'].isin(train_pair_ids)]
    
    train_oov = train_triples[~train_triples['label_cui'].isin(selected_cuis)]
    val_test_oov = val_test_triples[~val_test_triples['label_cui'].isin(selected_cuis)]
    
    metadata = {
        "selection_split": "train",
        "top_k": top_k,
        "labels": selected.rename(columns={'label_cui': 'cui'}).to_dict('records'),
        "out_of_vocabulary_positive_counts": {
            "train": int(len(train_oov)),
            "val_test": int(len(val_test_oov))
        }
    }
    
    return selected, metadata
