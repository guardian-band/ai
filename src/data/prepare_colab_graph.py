import pandas as pd
import os
import json
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def generate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((0,), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros(n_bits, dtype=np.float32)

def main():
    print("Preparing safe PrimeKG graph for Colab...")
    primekg_path = "data/external/primekg_kg.csv"
    output_graph_path = "data/processed/colab_primekg_safe.parquet"
    
    df = pd.read_csv(primekg_path, low_memory=False)
    print(f"Original PrimeKG edges: {len(df)}")
    
    # Strictly filter leakage relations
    leakage_relations = {
        "drug_drug",
        "drug_effect",
        "contraindication",
        "indication"
    }
    safe_df = df[~df['relation'].isin(leakage_relations)].copy()
    print(f"Filtered edges: {len(safe_df)} (Removed {len(df) - len(safe_df)} leakage edges)")
    
    os.makedirs("data/processed", exist_ok=True)
    safe_df.to_parquet(output_graph_path, index=False)
    print(f"Saved safe graph to {output_graph_path}")
    
    # 2. Export Morgan features so Colab doesn't need RDKit
    print("Generating Morgan Fingerprints for all drugs...")
    drugs_df = pd.read_csv("data/raw/drugs_master.csv")
    
    features = []
    for _, row in drugs_df.iterrows():
        drug_id = row['drugbank_id']
        smiles = row['smiles']
        fp = generate_morgan_fingerprint(smiles)
        features.append({"drugbank_id": drug_id, "morgan_2048": fp.tolist()})
        
    features_df = pd.DataFrame(features)
    out_features = "data/processed/colab_morgan_features.parquet"
    features_df.to_parquet(out_features, index=False)
    print(f"Saved Morgan features to {out_features}")
    print("Done!")

if __name__ == "__main__":
    main()
