import os
import gzip
import urllib.request
import pandas as pd
import numpy as np

BIOSNAP_URL = "https://snap.stanford.edu/biodata/datasets/10017/files/ChChSe-Decagon_polypharmacy.csv.gz"

def download_biosnap(dest_dir="data/external"):
    os.makedirs(dest_dir, exist_ok=True)
    gz_path = os.path.join(dest_dir, "ChChSe-Decagon_polypharmacy.csv.gz")
    csv_path = os.path.join(dest_dir, "ChChSe-Decagon_polypharmacy.csv")
    
    if not os.path.exists(csv_path):
        if not os.path.exists(gz_path):
            print(f"Downloading BioSNAP dataset from {BIOSNAP_URL}...")
            # Use curl command to bypass SSL cert issues if python urllib fails
            os.system(f"curl -k -L -o {gz_path} {BIOSNAP_URL}")
            print("Download completed.")
        
        print("Extracting ChChSe-Decagon_polypharmacy.csv.gz...")
        os.system(f"gunzip -k -f {gz_path}")
        print("Extraction completed.")
    else:
        print("BioSNAP dataset already exists at", csv_path)
    return csv_path

def load_and_map_dataset(
    drugs_master_path="data/raw/drugs_master.csv",
    side_effects_path="data/raw/side_effects.csv",
    biosnap_path="data/external/ChChSe-Decagon_polypharmacy.csv",
    top_k_side_effects=100
):
    print("Loading local metadata...")
    drugs_df = pd.read_csv(drugs_master_path)
    side_eff_df = pd.read_csv(side_effects_path, usecols=['umls_cui_from_meddra', 'side_effect_name'])
    
    # Extract valid DrugBank IDs and PubChem CIDs
    drugs_df['pubchem_cid_clean'] = pd.to_numeric(drugs_df['pubchem_compound_id'], errors='coerce')
    cid_to_drugbank = dict(zip(drugs_df['pubchem_cid_clean'].dropna().astype(int), drugs_df['drugbank_id']))
    
    valid_drugbank_ids = set(drugs_df['drugbank_id'].dropna().unique())
    valid_cuis = set(side_eff_df['umls_cui_from_meddra'].dropna().unique())
    
    print(f"Loaded {len(drugs_df)} drugs ({len(cid_to_drugbank)} with PubChem CID) and {len(valid_cuis)} side effect CUIs.")
    
    print("Loading BioSNAP interaction edges...")
    combo_df = pd.read_csv(biosnap_path, comment='#', names=['STITCH 1', 'STITCH 2', 'Polypharmacy Side Effect', 'Side Effect Name'])
    print(f"Raw BioSNAP dataset contains {len(combo_df)} edges.")
    print("Columns:", list(combo_df.columns))
    
    def parse_stitch_id(stitch_str):
        if not isinstance(stitch_str, str):
            return None
        cleaned = stitch_str.replace("CID000", "").replace("CID00", "").replace("CID0", "").replace("CID1", "").replace("CID", "")
        try:
            return int(cleaned)
        except ValueError:
            return None

    combo_df['cid_1'] = combo_df['STITCH 1'].apply(parse_stitch_id)
    combo_df['cid_2'] = combo_df['STITCH 2'].apply(parse_stitch_id)
    
    combo_df['db_id_1'] = combo_df['cid_1'].map(cid_to_drugbank)
    combo_df['db_id_2'] = combo_df['cid_2'].map(cid_to_drugbank)
    
    combo_df['db_id_1'] = combo_df['db_id_1'].fillna(combo_df['STITCH 1'].where(combo_df['STITCH 1'].isin(valid_drugbank_ids)))
    combo_df['db_id_2'] = combo_df['db_id_2'].fillna(combo_df['STITCH 2'].where(combo_df['STITCH 2'].isin(valid_drugbank_ids)))
    
    mapped_df = combo_df.dropna(subset=['db_id_1', 'db_id_2']).copy()
    mapped_df = mapped_df[mapped_df['Polypharmacy Side Effect'].isin(valid_cuis)].copy()
    
    print(f"Mapped BioSNAP dataset to local DrugBank & CUI IDs: {len(mapped_df)} edges remaining.")
    
    # Canonical ordering: (min(A,B), max(A,B))
    pair_a = np.minimum(mapped_df['db_id_1'].values, mapped_df['db_id_2'].values)
    pair_b = np.maximum(mapped_df['db_id_1'].values, mapped_df['db_id_2'].values)
    
    mapped_df['drug_a'] = pair_a
    mapped_df['drug_b'] = pair_b
    mapped_df['cui'] = mapped_df['Polypharmacy Side Effect']
    
    mapped_df = mapped_df[mapped_df['drug_a'] != mapped_df['drug_b']].copy()
    mapped_df = mapped_df.drop_duplicates(subset=['drug_a', 'drug_b', 'cui']).copy()
    
    print(f"Unique mapped (drug_a, drug_b, cui) triples: {len(mapped_df)}")
    
    top_cuis = mapped_df['cui'].value_counts().head(top_k_side_effects).index.tolist()
    filtered_df = mapped_df[mapped_df['cui'].isin(top_cuis)].copy()
    
    print(f"Filtered to Top {top_k_side_effects} side effects: {len(filtered_df)} triples remaining.")
    print(f"Unique drug pairs in dataset: {len(filtered_df[['drug_a', 'drug_b']].drop_duplicates())}")
    
    return filtered_df, top_cuis

if __name__ == "__main__":
    csv_file = download_biosnap()
