import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss
from tqdm import tqdm

from src.models.gnn_decagon import HybridDualHeadDecagonGNN
from src.features.smiles_encoder import build_drug_features
from src.data_loader import load_and_map_dataset

def compute_auprc(y_true, y_pred_prob):
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    return auc(recall, precision)

def compute_precision_at_k(y_true_matrix, y_pred_prob_matrix, k=5):
    precisions = []
    for i in range(len(y_true_matrix)):
        top_k_indices = np.argsort(y_pred_prob_matrix[i])[-k:]
        relevant_retrieved = y_true_matrix[i, top_k_indices].sum()
        precisions.append(relevant_retrieved / float(k))
    return np.mean(precisions)

class PolypharmacyDataset(Dataset):
    def __init__(self, pairs, y_matrix, drug_feature_dict):
        self.pairs = pairs
        self.y_matrix = torch.tensor(y_matrix, dtype=torch.float32)
        self.drug_feature_dict = drug_feature_dict
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        da, db = self.pairs[idx]
        feat_a = torch.tensor(self.drug_feature_dict[da], dtype=torch.float32)
        feat_b = torch.tensor(self.drug_feature_dict[db], dtype=torch.float32)
        target = self.y_matrix[idx]
        return feat_a, feat_b, target

def train_single_model(seed, in_features, num_relations, train_loader, test_loader, device, epochs=15):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = HybridDualHeadDecagonGNN(
        in_features=in_features,
        hidden_dim=256,
        embedding_dim=128,
        num_relations=num_relations,
        dropout=0.15
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        for feat_a, feat_b, target in train_loader:
            feat_a, feat_b, target = feat_a.to(device), feat_b.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(feat_a, feat_b, return_logits=True)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    # Inference on Test Set
    model.eval()
    all_preds = []
    with torch.no_grad():
        for feat_a, feat_b, _ in test_loader:
            feat_a, feat_b = feat_a.to(device), feat_b.to(device)
            probs = model(feat_a, feat_b, return_logits=False)
            all_preds.append(probs.cpu().numpy())
            
    return model, np.vstack(all_preds)

def main():
    base_dir = "/Users/acelyayildiz/.gemini/antigravity/scratch/polypharmacy_ai"
    raw_dir = os.path.join(base_dir, "data/raw")
    art_dir = os.path.join(base_dir, "artifacts")
    splits_dir = os.path.join(art_dir, "splits")
    
    # 1. Build Rich Hybrid Drug Features
    print("Extracting Rich Hybrid Molecular Features...")
    drug_feature_dict, in_features = build_drug_features(
        drugs_master_path=os.path.join(raw_dir, "drugs_master.csv"),
        indications_path=os.path.join(raw_dir, "indications.csv"),
        n_bits=512
    )
    
    # Save updated drug_features.parquet
    drug_feat_rows = []
    for db_id, feat in drug_feature_dict.items():
        row_dict = {'drugbank_id': db_id}
        for idx, val in enumerate(feat):
            row_dict[f'f_{idx}'] = val
        drug_feat_rows.append(row_dict)
    pd.DataFrame(drug_feat_rows).to_parquet(os.path.join(art_dir, "drug_features.parquet"), index=False)
    print("Saved rich hybrid drug_features.parquet to artifacts.")
    
    side_effects_df = pd.read_csv(os.path.join(art_dir, "side_effect_labels.csv"))
    top_cuis = side_effects_df['cui'].tolist()
    num_relations = len(top_cuis)
    
    train_pairs_df = pd.read_csv(os.path.join(splits_dir, "train_pairs.csv"))
    test_pairs_df = pd.read_csv(os.path.join(splits_dir, "test_pairs.csv"))
    
    # Re-map valid multi-label matrices
    mapped_triples, _ = load_and_map_dataset(
        drugs_master_path=os.path.join(raw_dir, "drugs_master.csv"),
        side_effects_path=os.path.join(raw_dir, "side_effects.csv"),
        biosnap_path=os.path.join(base_dir, "data/external/ChChSe-Decagon_polypharmacy.csv"),
        top_k_side_effects=100
    )
    cui_to_idx = {cui: idx for idx, cui in enumerate(top_cuis)}
    pair_grouped = mapped_triples.groupby(['drug_a', 'drug_b'])['cui'].apply(list).to_dict()
    
    def build_y_matrix(pairs_df):
        y_mat = []
        pairs_list = []
        for _, row in pairs_df.iterrows():
            da, db = row['drug_a'], row['drug_b']
            pairs_list.append((da, db))
            y_vec = np.zeros(num_relations, dtype=np.float32)
            if (da, db) in pair_grouped:
                for c in pair_grouped[(da, db)]:
                    if c in cui_to_idx:
                        y_vec[cui_to_idx[c]] = 1.0
            y_mat.append(y_vec)
        return np.array(pairs_list), np.array(y_mat)

    train_pairs, y_train = build_y_matrix(train_pairs_df)
    test_pairs, y_test = build_y_matrix(test_pairs_df)
    
    train_dataset = PolypharmacyDataset(train_pairs, y_train, drug_feature_dict)
    test_dataset = PolypharmacyDataset(test_pairs, y_test, drug_feature_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using PyTorch compute device: {device}")
    
    # 2. Train 3-Model Diverse Ensemble
    seeds = [42, 101, 2024]
    ensemble_preds = []
    
    print("\n" + "="*50)
    print("TRAINING HYBRID DUAL-HEAD DECAGON GNN ENSEMBLE")
    print("="*50)
    
    for s_idx, seed in enumerate(seeds):
        print(f"--> Training Ensemble Model {s_idx+1}/{len(seeds)} (Seed: {seed})...")
        model, test_pred = train_single_model(seed, in_features, num_relations, train_loader, test_loader, device, epochs=15)
        ensemble_preds.append(test_pred)
        if s_idx == 0:
            torch.save(model.state_dict(), os.path.join(art_dir, "champion_hybrid_decagon_gnn.pt"))
            
    # Soft-Voting Average
    final_test_preds = np.mean(ensemble_preds, axis=0)
    
    # Calculate Final Evaluation Metrics
    auprc_scores = []
    brier_scores = []
    for k in range(num_relations):
        y_true_k = y_test[:, k]
        y_pred_k = final_test_preds[:, k]
        if y_true_k.sum() > 0:
            auprc_scores.append(compute_auprc(y_true_k, y_pred_k))
            brier_scores.append(brier_score_loss(y_true_k, y_pred_k))
            
    macro_auprc = np.mean(auprc_scores)
    mean_brier = np.mean(brier_scores)
    p_at_5 = compute_precision_at_k(y_test, final_test_preds, k=5)
    
    print("\n" + "*"*50)
    print("🏆 FINAL CHAMPION HYBRID DUAL-HEAD DECAGON GNN RESULTS:")
    print(f"  Macro AUPRC:      {macro_auprc:.4f}")
    print(f"  Mean Brier Score: {mean_brier:.4f}")
    print(f"  Precision@5:      {p_at_5:.4f}")
    print("*"*50)

if __name__ == "__main__":
    main()
