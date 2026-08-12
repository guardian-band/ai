import os
from src.training.reproducibility import repository_root

import json
import pandas as pd

def integrate_chembl(chembl_path, output_dir):
    print("Parsing and indexing ChEMBL 34 dataset (2.4M molecules)...")
    os.makedirs(output_dir, exist_ok=True)
    
    total_molecules = 0
    unique_smiles_count = 0
    sample_molecules = []
    
    unique_smiles_set = set()
    
    # Process in chunks
    for chunk_idx, chunk in enumerate(pd.read_csv(chembl_path, sep='\t', chunksize=500000, usecols=['chembl_id', 'canonical_smiles'])):
        chunk_clean = chunk.dropna(subset=['canonical_smiles'])
        total_molecules += len(chunk_clean)
        
        if len(sample_molecules) < 10:
            sample_molecules.extend(chunk_clean.head(5).to_dict(orient='records'))
            
        unique_smiles_set.update(chunk_clean['canonical_smiles'].head(100000))
        
    summary = {
        "dataset_name": "EMBL-EBI ChEMBL 34 Bioactive Compounds",
        "total_bioactive_molecules": total_molecules,
        "format": "ChEMBL ID + Canonical SMILES",
        "sample_entries": sample_molecules[:5]
    }
    
    summary_path = os.path.join(output_dir, "chembl34_metadata_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print(f"ChEMBL 34 successfully indexed! Total Bioactive Molecules: {total_molecules:,}")
    return summary

if __name__ == "__main__":
    base_dir = str(repository_root()) + ""
    chembl_file = os.path.join(base_dir, "data/external/chembl_34_chemreps.txt")
    out_dir = os.path.join(base_dir, "artifacts")
    integrate_chembl(chembl_file, out_dir)
