import pandas as pd
import numpy as np

# Load Morgan features
morgan_df = pd.read_parquet("data/processed/colab_morgan_features.parquet")
print(f"Loaded Morgan features: {len(morgan_df)} drugs")

# Load GraphSAGE embeddings
graphsage_embs = np.load("artifacts/embeddings/GraphSAGE_cold_1_seed_42.npy")
print(f"Loaded GraphSAGE embeddings shape: {graphsage_embs.shape}")

# Load the drug mapping we verified earlier
mapping_df = pd.read_parquet("artifacts/embeddings/GraphSAGE_cold_1_seed_42_drug_mapping.parquet")
print(f"Loaded drug mapping: {len(mapping_df)} drugs")

# Combine them based on the verified mapping
combined_features = []
drugbank_ids = []

for i, row in mapping_df.iterrows():
    drug_id = row['drugbank_id']
    idx = row['node_index']
    
    # Get Morgan fingerprint from morgan_df for this drug
    morgan_fp = morgan_df[morgan_df['drugbank_id'] == drug_id]['morgan_2048'].values[0]
    
    # Get GraphSAGE embedding for this index
    graphsage_fp = graphsage_embs[idx]
    
    # Concatenate (2048 + 128 = 2176)
    combined = np.concatenate([morgan_fp, graphsage_fp])
    assert len(combined) == 2176
    
    combined_features.append(combined)
    drugbank_ids.append(drug_id)

# Save to a new parquet file matching the required format
df_combined = pd.DataFrame({
    'drugbank_id': drugbank_ids,
    'morgan_fingerprint': combined_features  # We keep this column name so ManifestPolypharmacyDataset can load it
})

out_path = "artifacts/morgan_graphsage_features.parquet"
df_combined.to_parquet(out_path, index=False)
print(f"Saved combined features to {out_path}")
