import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
from pathlib import Path

def generate_morgan_fingerprints(input_csv: str, output_parquet: str, radius: int = 2, n_bits: int = 2048):
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Check for smiles column
    smiles_col = None
    for col in ["smiles", "canonical_smiles", "smiles_simple"]:
        if col in df.columns:
            smiles_col = col
            break
            
    if not smiles_col:
        raise ValueError("No SMILES column found in dataset!")
        
    print(f"Using SMILES column: {smiles_col}")
    
    records = []
    missing_smiles = 0
    parse_errors = 0
    
    for idx, row in df.iterrows():
        drug_id = row['drugbank_id']
        smiles = row[smiles_col]
        
        if pd.isna(smiles) or not str(smiles).strip():
            missing_smiles += 1
            continue
            
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                parse_errors += 1
                continue
                
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            # Convert to float numpy array
            arr = np.zeros((0,), dtype=np.float32)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            
            records.append({
                "drugbank_id": str(drug_id),
                "morgan_fingerprint": arr.tolist() # saving as list of floats for parquet
            })
            
        except Exception as e:
            parse_errors += 1
            
    print(f"Total drugs: {len(df)}")
    print(f"Missing SMILES: {missing_smiles}")
    print(f"Parse errors: {parse_errors}")
    print(f"Successfully generated: {len(records)}")
    
    if not records:
        raise ValueError("Failed to generate any fingerprints!")
        
    out_df = pd.DataFrame(records)
    
    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/drugs_master.csv")
    parser.add_argument("--output", default="artifacts/morgan_fingerprints.parquet")
    args = parser.parse_args()
    
    generate_morgan_fingerprints(args.input, args.output)
