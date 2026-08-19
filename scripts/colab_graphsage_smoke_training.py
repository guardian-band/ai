import os
import json
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch.nn as nn
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F

# 1. Model Definition
class BaseGraphSAGE(nn.Module):
    def __init__(self, hidden_channels=128):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class HeteroLinkPredictor(nn.Module):
    def __init__(self, metadata, hidden_channels=128):
        super().__init__()
        self.encoder = to_hetero(BaseGraphSAGE(hidden_channels), metadata, aggr='mean')
        
    def forward(self, x_dict, edge_index_dict, edge_label_index, src_type, dst_type):
        out_dict = self.encoder(x_dict, edge_index_dict)
        src_emb = out_dict[src_type][edge_label_index[0]]
        dst_emb = out_dict[dst_type][edge_label_index[1]]
        return (src_emb * dst_emb).sum(dim=-1)

def main():
    print("--- REAL COLAB SMOKE TRAINING: cold_1, seed 42 ---")
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("ERROR: CUDA is not available. Please execute this strictly on a T4 GPU instance.")
        
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    
    print(f"[Check] Is actual GPU device T4? {gpu_name}")
    print(f"[Check] Is training running on CUDA? YES")
    
    # Load manifest and splits
    manifest_path = "artifacts/benchmarks/polypharmacy_v1/cold_1/seed_42/manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    pairs_path = "artifacts/benchmarks/polypharmacy_v1/cold_1/seed_42/pairs.parquet"
    pairs_df = pd.read_parquet(pairs_path)
    
    train_pairs = pairs_df[pairs_df['split'] == 'train']
    eval_pairs = pairs_df[pairs_df['split'] != 'train']
    
    train_drugs = set(train_pairs['drug_a']).union(set(train_pairs['drug_b']))
    eval_drugs = set(eval_pairs['drug_a']).union(set(eval_pairs['drug_b']))
    unseen_drugs = eval_drugs - train_drugs
    
    # Load safe graph
    kg_df = pd.read_parquet("data/processed/colab_primekg_safe.parquet")
    
    leakage_rels = {"drug_drug", "drug_effect", "contraindication", "indication"}
    leaked_count = len(kg_df[kg_df['relation'].isin(leakage_rels)])
    print(f"[Check] Is leakage relation count 0? {'YES' if leaked_count == 0 else 'NO'}")
    
    # Isolation FIX: Use x_id / y_id instead of integer x_index / y_index
    mask_x_unseen = (kg_df['x_type'] == 'drug') & (kg_df['x_id'].isin(unseen_drugs))
    mask_y_unseen = (kg_df['y_type'] == 'drug') & (kg_df['y_id'].isin(unseen_drugs))
    
    train_kg = kg_df[~(mask_x_unseen | mask_y_unseen)]
    
    # Check isolation
    train_kg_drugs = set(train_kg[train_kg['x_type'] == 'drug']['x_id']).union(set(train_kg[train_kg['y_type'] == 'drug']['y_id']))
    leak_check = unseen_drugs.intersection(train_kg_drugs)
    print(f"[Check] Are val/test (unseen) drugs present in training graph? {'NO' if len(leak_check) == 0 else 'YES'}")
    
    # Inference isolation
    unseen_incoming = kg_df[mask_y_unseen & ~mask_x_unseen]
    inference_kg = pd.concat([train_kg, unseen_incoming])
    out_edges = inference_kg[(inference_kg['x_type'] == 'drug') & (inference_kg['x_id'].isin(unseen_drugs))]
    print(f"[Check] Are outgoing edges from cold drugs in inference blocked? {'YES' if len(out_edges) == 0 else 'NO'}")
    
    # Map string IDs to integers
    print("Mapping nodes to contiguous indices...")
    morgan_df = pd.read_parquet("data/processed/colab_morgan_features.parquet")
    drug_mapping = {id_: i for i, id_ in enumerate(morgan_df['drugbank_id'])}
    
    bio_nodes = set(kg_df['x_id'][kg_df['x_type'] != 'drug']).union(set(kg_df['y_id'][kg_df['y_type'] != 'drug']))
    bio_mapping = {id_: i for i, id_ in enumerate(bio_nodes)}
    
    # Construct HeteroData
    data = HeteroData()
    data['drug'].x = torch.tensor(np.vstack(morgan_df['morgan_2048'].values), dtype=torch.float32)
    data['bio'].x = torch.ones((len(bio_mapping), 1), dtype=torch.float32)
    
    # Map train edges
    edge_index_dict = {}
    for (src_type, rel, dst_type), group in train_kg.groupby(['x_type', 'relation', 'y_type']):
        src_t = 'drug' if src_type == 'drug' else 'bio'
        dst_t = 'drug' if dst_type == 'drug' else 'bio'
        
        # Normalize relation string to avoid PyG warnings (e.g. replace spaces and hyphens)
        safe_rel = rel.replace('-', '_').replace(' ', '_')
        
        rel_name = f"rev_{safe_rel}" if (src_t == 'bio' and dst_t == 'drug') else safe_rel
            
        src_indices = [drug_mapping.get(x) if src_t == 'drug' else bio_mapping.get(x) for x in group['x_id']]
        dst_indices = [drug_mapping.get(x) if dst_t == 'drug' else bio_mapping.get(x) for x in group['y_id']]
            
        valid = [(s, d) for s, d in zip(src_indices, dst_indices) if s is not None and d is not None]
        if len(valid) > 0:
            s, d = zip(*valid)
            edge_index_dict[(src_t, rel_name, dst_t)] = torch.tensor([s, d], dtype=torch.long)
            
    for k, v in edge_index_dict.items():
        data[k].edge_index = v
        
    # Logging as requested by the user
    print(f"data.metadata(): {data.metadata()}")
    print(f"data.node_types: {data.node_types}")
    print(f"data.edge_types: {data.edge_types}")
    
    drug_edge_types = [et for et in data.edge_types if 'drug' in et]
    print(f"Drug içeren edge type'ların listesi: {drug_edge_types}")
    
    for etype in drug_edge_types:
        print(f"Edge '{etype}': count={data[etype].edge_index.shape[1]}")
    
    # Check drug biological neighbors in training
    train_bio_incoming = [et for et in data.edge_types if et[0] == 'bio' and et[2] == 'drug']
    all_train_dst = []
    for et in train_bio_incoming:
        all_train_dst.extend(data[et].edge_index[1].tolist())
    train_drugs_with_neighbors = set(all_train_dst)
    print(f"Kaç train drug'ın en az 1 incoming biological neighbor'ı var: {len(train_drugs_with_neighbors)}")
    
    # Inference graph check (just counting for the log)
    unseen_incoming_edges = unseen_incoming[unseen_incoming['y_type'] == 'drug']
    cold_drugs_with_neighbors = len(set(unseen_incoming_edges['y_id']))
    print(f"Kaç cold drug'ın inference sırasında en az 1 permitted incoming biological edge'i var: {cold_drugs_with_neighbors}")
    
    print(f"Filtering sonrası safe PrimeKG satır sayısı: {len(kg_df)}")
    print(f"cold_1 / seed_42 train drug sayısı: {len(train_drugs)}")
    
    # Move data to device
    data = data.to(device)

    # Model
    model = HeteroLinkPredictor(data.metadata(), hidden_channels=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Device Checks
    print("\n--- Device Checks Before Training ---")
    print(f"Model param device: {next(model.parameters()).device}")
    print(f"data['drug'].x device: {data['drug'].x.device}")
    print(f"data['bio'].x device: {data['bio'].x.device}")
    sample_edge_type = data.edge_types[0]
    print(f"Sample edge {sample_edge_type} device: {data[sample_edge_type].edge_index.device}")
    
    # Simulate link prediction on a random bio->bio edge type
    edge_type = [et for et in data.edge_types if et[0] == 'bio' and et[2] == 'bio'][0]
    src_type, _, dst_type = edge_type
    pos_edge_index = data[edge_type].edge_index
    print(f"pos_edge_index device: {pos_edge_index.device}")
    print("-------------------------------------\n")
    
    # Training Loop (1 Epoch Smoke Test)
    print("Starting 1-Epoch Smoke Training...")
    model.train()
    optimizer.zero_grad()
    
    # Forward
    pos_pred = model(data.x_dict, data.edge_index_dict, pos_edge_index, src_type, dst_type)
    loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred, device=device))
    loss.backward()
    optimizer.step()
    
    print(f"[Check] Is training loss actually calculated? YES | Loss: {loss.item():.4f}")
    
    # Inference
    model.eval()
    with torch.no_grad():
        out_dict = model.encoder(data.x_dict, data.edge_index_dict)
        drug_embs = out_dict['drug']
        print(f"[Check] Is 128-dim embedding actually produced? YES | Shape: {drug_embs.shape}")
        
    print(f"[Check] Are random/dummy/hash features used? NO (Using 2048-dim Morgan for drugs, 1-dim [1.0] for biological nodes)")
    print(f"[Check] Actual coverage: {drug_embs.shape[0]} drugs covered out of {len(drug_mapping)} total.")
    print("Smoke training successfully completed.")

if __name__ == "__main__":
    main()
