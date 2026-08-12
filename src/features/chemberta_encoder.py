import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

def extract_chemberta_embeddings(
    drugs_master_path="/Users/acelyayildiz/.gemini/antigravity/scratch/polypharmacy_ai/data/raw/drugs_master.csv",
    existing_features_path="/Users/acelyayildiz/.gemini/antigravity/scratch/polypharmacy_ai/artifacts/drug_features.parquet",
    output_path="/Users/acelyayildiz/.gemini/antigravity/scratch/polypharmacy_ai/artifacts/drug_features_chemberta.parquet",
    model_name="DeepChem/ChemBERTa-77M-MTR",
    batch_size=128
):
    print("="*60)
    print("EXTRACTING CHEMBERTA MOLECULAR TRANSFORMER EMBEDDINGS")
    print("="*60)
    
    start_time = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Compute Device for ChemBERTa Inference: {device}")
    
    print(f"Loading {model_name} Tokenizer and Transformer Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    hidden_dim = model.config.hidden_size
    print(f"ChemBERTa Model Hidden Dimension: {hidden_dim}")
    
    drugs_df = pd.read_csv(drugs_master_path)
    print(f"Loaded {len(drugs_df)} total drugs from master list.")
    
    valid_mask = drugs_df['smiles'].notna() & (drugs_df['smiles'].astype(str).str.strip() != '')
    valid_drugs = drugs_df[valid_mask].copy()
    print(f"Found {len(valid_drugs)} drugs with valid SMILES strings.")
    
    smiles_list = valid_drugs['smiles'].tolist()
    drug_ids = valid_drugs['drugbank_id'].tolist()
    
    all_embeddings = []
    print("\nExtracting ChemBERTa embeddings in batches on GPU...")
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i + batch_size]
            inputs = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
            all_embeddings.append(mean_pooled)
            
    all_embeddings = np.vstack(all_embeddings)
    print(f"Extracted ChemBERTa embeddings matrix shape: {all_embeddings.shape}")
    
    emb_dict = {drug_ids[i]: all_embeddings[i] for i in range(len(drug_ids))}
    
    print(f"\nMerging ChemBERTa embeddings with existing biological features from {existing_features_path}...")
    existing_df = pd.read_parquet(existing_features_path)
    
    chem_cols = [f"chem_{i}" for i in range(hidden_dim)]
    chem_data = []
    mean_fallback = np.mean(all_embeddings, axis=0)
    for db_id in existing_df['drugbank_id']:
        if db_id in emb_dict:
            chem_data.append(emb_dict[db_id])
        else:
            chem_data.append(mean_fallback)
            
    chem_df = pd.DataFrame(chem_data, columns=chem_cols)
    merged_df = pd.concat([existing_df, chem_df], axis=1)
    
    merged_df.to_parquet(output_path, index=False)
    print(f"Saved ChemBERTa-enriched feature matrix to {output_path}")
    print(f"Total Columns: {len(merged_df.columns)} (Total Features: {len(merged_df.columns) - 2})")
    print(f"Execution Time: {time.time() - start_time:.1f} seconds")
    return output_path

if __name__ == "__main__":
    extract_chemberta_embeddings()
