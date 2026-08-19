import os
import json
import torch
import pandas as pd
import numpy as np
import argparse
from torch_geometric.data import HeteroData
import torch.nn as nn
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
import torch.amp

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    
    scenario = args.scenario
    seed = args.seed
    
    print(f"--- REAL COLAB FULL GRAPH SAGE TRAINING: {scenario}, seed {seed} ---")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Executing on CPU (for testing only).")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"[Check] GPU device: {torch.cuda.get_device_name(0)}")
    
    manifest_path = f"artifacts/benchmarks/polypharmacy_v1/{scenario}/seed_{seed}/manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    pairs_path = f"artifacts/benchmarks/polypharmacy_v1/{scenario}/seed_{seed}/pairs.parquet"
    pairs_df = pd.read_parquet(pairs_path)
    
    train_pairs = pairs_df[pairs_df['split'] == 'train']
    eval_pairs = pairs_df[pairs_df['split'] != 'train']
    
    train_drugs = set(train_pairs['drug_a']).union(set(train_pairs['drug_b']))
    eval_drugs = set(eval_pairs['drug_a']).union(set(eval_pairs['drug_b']))
    
    # In warm_pair, all eval drugs are structurally part of the train_drugs distribution,
    # so we do not hide any node. For cold scenarios, unseen_drugs are strictly hidden.
    if scenario == "warm_pair":
        unseen_drugs = set()
        print("warm_pair: Universal encoder mode. No drugs hidden.")
    else:
        unseen_drugs = eval_drugs - train_drugs
        print(f"{scenario}: Hiding {len(unseen_drugs)} unseen eval drugs.")
    
    # Load safe graph
    kg_df = pd.read_parquet("data/processed/colab_primekg_safe.parquet")
    
    # Isolation
    mask_x_unseen = (kg_df['x_type'] == 'drug') & (kg_df['x_id'].isin(unseen_drugs))
    mask_y_unseen = (kg_df['y_type'] == 'drug') & (kg_df['y_id'].isin(unseen_drugs))
    
    train_kg = kg_df[~(mask_x_unseen | mask_y_unseen)]
    
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
        
    print(f"Data constructed. Total drug nodes: {data['drug'].x.shape[0]}, Total bio nodes: {data['bio'].x.shape[0]}")
    
    epochs = 10
    learning_rate = 0.001
    weight_decay = 1e-4
    hidden_channels = 128
    batch_size = 8192
    num_neighbors = [10, 10]
    
    model = HeteroLinkPredictor(data.metadata(), hidden_channels=hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    print("\n--- Device Checks Before Training ---")
    print(f"Model param device: {next(model.parameters()).device}")
    print(f"Batch Size: {batch_size}, Fanout: {num_neighbors}")
    
    loss_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        
        for edge_type in data.edge_types:
            if data[edge_type].edge_index.shape[1] == 0:
                continue
            
            src_type, _, dst_type = edge_type
            loader = LinkNeighborLoader(
                data,
                num_neighbors=num_neighbors,
                edge_label_index=(edge_type, data[edge_type].edge_index),
                edge_label=torch.ones(data[edge_type].edge_index.size(1)),
                batch_size=batch_size,
                shuffle=True,
                neg_sampling_ratio=1.0,
            )
            
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                batch_edge_label_index = batch[edge_type].edge_label_index
                batch_edge_label = batch[edge_type].edge_label
                
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        pred = model(batch.x_dict, batch.edge_index_dict, batch_edge_label_index, src_type, dst_type)
                        loss = F.binary_cross_entropy_with_logits(pred, batch_edge_label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(batch.x_dict, batch.edge_index_dict, batch_edge_label_index, src_type, dst_type)
                    loss = F.binary_cross_entropy_with_logits(pred, batch_edge_label)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
                
        epoch_loss = total_loss / max(1, total_batches)
        loss_history.append(epoch_loss)
        
        if device.type == 'cuda':
            gpu_mem_alloc = torch.cuda.memory_allocated() / (1024**2)
            gpu_mem_res = torch.cuda.memory_reserved() / (1024**2)
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.4f} | Batches: {total_batches} | GPU Mem: {gpu_mem_alloc:.1f}MB alloc / {gpu_mem_res:.1f}MB res")
        else:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.4f} | Batches: {total_batches}")
    
    print("Extracting frozen GraphSAGE embeddings...")
    
    # For inference, add back unseen incoming edges
    unseen_incoming = kg_df[mask_y_unseen & ~mask_x_unseen]
    inference_edge_index_dict = {}
    inference_kg = pd.concat([train_kg, unseen_incoming])
    
    for (src_type, rel, dst_type), group in inference_kg.groupby(['x_type', 'relation', 'y_type']):
        src_t = 'drug' if src_type == 'drug' else 'bio'
        dst_t = 'drug' if dst_type == 'drug' else 'bio'
        safe_rel = rel.replace('-', '_').replace(' ', '_')
        rel_name = f"rev_{safe_rel}" if (src_t == 'bio' and dst_t == 'drug') else safe_rel
        src_indices = [drug_mapping.get(x) if src_t == 'drug' else bio_mapping.get(x) for x in group['x_id']]
        dst_indices = [drug_mapping.get(x) if dst_t == 'drug' else bio_mapping.get(x) for x in group['y_id']]
        valid = [(s, d) for s, d in zip(src_indices, dst_indices) if s is not None and d is not None]
        if len(valid) > 0:
            s, d = zip(*valid)
            inference_edge_index_dict[(src_t, rel_name, dst_t)] = torch.tensor([s, d], dtype=torch.long)
            
    inference_data = HeteroData()
    inference_data['drug'].x = data['drug'].x.clone().to(device)
    inference_data['bio'].x = data['bio'].x.clone().to(device)
    for k, v in inference_edge_index_dict.items():
        inference_data[k].edge_index = v.to(device)
        
    model.eval()
    with torch.no_grad():
        out_dict = model.encoder(inference_data.x_dict, inference_data.edge_index_dict)
        drug_embs = out_dict['drug'].cpu().numpy()
        
    # In warm_pair scenario, this embedding is universal, so we label it as such
    run_name = f"GraphSAGE_universal_encoder" if scenario == "warm_pair" else f"GraphSAGE_{scenario}_seed_{seed}"
    run_artifact_dir = f"artifacts/runs/{run_name}"
    
    os.makedirs(run_artifact_dir, exist_ok=True)
    os.makedirs("artifacts/embeddings", exist_ok=True)
    
    out_path = f"artifacts/embeddings/{run_name}.npy"
    np.save(out_path, drug_embs)
    print(f"Saved frozen GraphSAGE embeddings to {out_path}")
    
    # Save deterministic mapping
    mapping_df = pd.DataFrame(list(drug_mapping.items()), columns=['drugbank_id', 'node_index'])
    mapping_path = f"artifacts/embeddings/{run_name}_drug_mapping.parquet"
    mapping_df.to_parquet(mapping_path, index=False)
    
    ckpt_path = os.path.join(run_artifact_dir, "checkpoint_best.pt")
    torch.save(model.state_dict(), ckpt_path)
    
    config_info = {
        "model_type": "graphsage",
        "scenario": scenario,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hidden_channels": hidden_channels,
        "batch_size": batch_size,
        "num_neighbors": num_neighbors,
        "loss_history": loss_history
    }
    
    config_path = os.path.join(run_artifact_dir, "config.resolved.json")
    with open(config_path, "w") as f:
        json.dump(config_info, f, indent=2)

if __name__ == "__main__":
    main()
