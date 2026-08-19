import os
import json
import torch
import pandas as pd
import numpy as np
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
    print("--- REAL COLAB FULL GRAPH SAGE TRAINING: cold_1, seed 42 ---")
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("ERROR: CUDA is not available. Please execute this strictly on a T4 GPU instance.")
        
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[Check] GPU device: {gpu_name}")
    
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
    
    # Isolation FIX
    mask_x_unseen = (kg_df['x_type'] == 'drug') & (kg_df['x_id'].isin(unseen_drugs))
    mask_y_unseen = (kg_df['y_type'] == 'drug') & (kg_df['y_id'].isin(unseen_drugs))
    
    train_kg = kg_df[~(mask_x_unseen | mask_y_unseen)]
    
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
    
    # Model and Hyperparameters (from model_morgan_mlp.yaml)
    epochs = 10
    learning_rate = 0.001
    weight_decay = 1e-4
    hidden_channels = 128
    batch_size = 8192
    num_neighbors = [10, 10]
    
    model = HeteroLinkPredictor(data.metadata(), hidden_channels=hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler('cuda')
    
    print("\n--- Device Checks Before Training ---")
    print(f"Model param device: {next(model.parameters()).device}")
    print("Using AMP (Automatic Mixed Precision): YES")
    print(f"Batch Size: {batch_size}, Fanout: {num_neighbors}")
    
    # Training Loop
    print(f"Starting Full GraphSAGE Training for {epochs} epochs...")
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        
        total_loss = 0
        total_batches = 0
        
        # We iterate through all edge types and use LinkNeighborLoader for each
        for edge_type in data.edge_types:
            # Skip empty edge types
            if data[edge_type].edge_index.shape[1] == 0:
                continue
            
            src_type, _, dst_type = edge_type
            
            # Create a loader for the current edge type
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
                
                # The label and edge_label_index are stored inside the batch object
                # for the specific edge_type we created the loader on
                batch_edge_label_index = batch[edge_type].edge_label_index
                batch_edge_label = batch[edge_type].edge_label
                
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    pred = model(batch.x_dict, batch.edge_index_dict, batch_edge_label_index, src_type, dst_type)
                    loss = F.binary_cross_entropy_with_logits(pred, batch_edge_label)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                total_batches += 1
                
        epoch_loss = total_loss / max(1, total_batches)
        loss_history.append(epoch_loss)
        
        # Log GPU memory
        gpu_mem_alloc = torch.cuda.memory_allocated() / (1024**2)
        gpu_mem_res = torch.cuda.memory_reserved() / (1024**2)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.4f} | Batches: {total_batches} | GPU Mem: {gpu_mem_alloc:.1f}MB alloc / {gpu_mem_res:.1f}MB res")
    
    # Save GraphSAGE Frozen Embeddings
    print("Training complete. Extracting frozen GraphSAGE embeddings...")
    
    # For inference, we need to add back the unseen incoming edges
    unseen_incoming = kg_df[mask_y_unseen & ~mask_x_unseen]
    
    # Add unseen incoming to a NEW data object for inference
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
        
    print(f"Produced final drug embeddings shape: {drug_embs.shape}")
    
    # Save to disk
    run_artifact_dir = "artifacts/runs/GraphSAGE_cold_1_seed_42"
    os.makedirs(run_artifact_dir, exist_ok=True)
    os.makedirs("artifacts/embeddings", exist_ok=True)
    
    out_path = "artifacts/embeddings/GraphSAGE_cold_1_seed_42.npy"
    np.save(out_path, drug_embs)
    print(f"Saved frozen GraphSAGE embeddings to {out_path}")
    
    ckpt_path = os.path.join(run_artifact_dir, "checkpoint_best.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved trained GraphSAGE model checkpoint to {ckpt_path}")
    
    config_info = {
        "model_type": "graphsage",
        "scenario": "cold_1",
        "seed": 42,
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
    print(f"Saved training configuration and loss logs to {config_path}")

if __name__ == "__main__":
    main()
