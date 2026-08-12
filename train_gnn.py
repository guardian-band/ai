import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss
from tqdm import tqdm

from src.models.gnn_decagon import OptimizedDecagonGNN, FocalLoss
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

def main():
    base_dir = "/Users/acelyayildiz/.gemini/antigravity/scratch/polypharmacy_ai"
    raw_dir = os.path.join(base_dir, "data/raw")
    art_dir = os.path.join(base_dir, "artifacts")
    splits_dir = os.path.join(art_dir, "splits")
    
    print("Loading prepared dataset artifacts...")
    drug_feat_df = pd.read_parquet(os.path.join(art_dir, "drug_features.parquet"))
    side_effects_meta = pd.read_csv(os.path.join(raw_dir, "side_effects.csv"))
    side_effects_df = pd.read_csv(os.path.join(art_dir, "side_effect_labels.csv"))
    
    train_pairs_df = pd.read_csv(os.path.join(splits_dir, "train_pairs.csv"))
    test_pairs_df = pd.read_csv(os.path.join(splits_dir, "test_pairs.csv"))
    cold_test_df = pd.read_csv(os.path.join(splits_dir, "cold_drug_test.csv"))
    
    # Extract 768-dim BioBERT side effect embeddings
    emb_cols = [str(i) for i in range(768)]
    embedding_dict = {}
    for _, row in side_effects_meta.iterrows():
        cui = row['umls_cui_from_meddra']
        emb = row[emb_cols].values.astype(np.float32)
        embedding_dict[cui] = emb
        
    top_cuis = side_effects_df['cui'].tolist()
    num_relations = len(top_cuis)
    
    pretrained_rel_embs = []
    for cui in top_cuis:
        if cui in embedding_dict:
            pretrained_rel_embs.append(embedding_dict[cui])
        else:
            pretrained_rel_embs.append(np.random.normal(0, 0.1, 768).astype(np.float32))
    pretrained_rel_embs = torch.tensor(np.array(pretrained_rel_embs), dtype=torch.float32)
    print(f"Loaded {len(pretrained_rel_embs)} BioBERT 768-dim side-effect embeddings.")
    
    feat_cols = [c for c in drug_feat_df.columns if c.startswith('f_')]
    in_features = len(feat_cols)
    
    drug_feature_dict = {}
    for _, row in drug_feat_df.iterrows():
        drug_feature_dict[row['drugbank_id']] = row[feat_cols].values.astype(np.float32)
        
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
    
    model = OptimizedDecagonGNN(
        in_features=in_features,
        hidden_dim=256,
        embedding_dim=128,
        num_relations=num_relations,
        pretrained_rel_embeddings=pretrained_rel_embs,
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    print("\n" + "="*50)
    print("TRAINING OPTIMIZED DECAGON GNN (BioBERT Embeddings + Focal Loss)")
    print("="*50)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for feat_a, feat_b, target in train_loader:
            feat_a, feat_b, target = feat_a.to(device), feat_b.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits = model(feat_a, feat_b, return_logits=True)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(target)
            
        scheduler.step()
        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch:02d}/{epochs:02d} - Training Focal Loss: {avg_loss:.4f} (LR: {scheduler.get_last_lr()[0]:.6f})")
        
    # Evaluate Optimized Model on Test Set
    print("\nEvaluating Optimized Decagon GNN on Test Set...")
    model.eval()
    all_test_preds = []
    all_test_targets = []
    
    with torch.no_grad():
        for feat_a, feat_b, target in test_loader:
            feat_a, feat_b = feat_a.to(device), feat_b.to(device)
            logits = model(feat_a, feat_b, return_logits=True)
            probs = torch.sigmoid(logits)
            all_test_preds.append(probs.cpu().numpy())
            all_test_targets.append(target.numpy())
            
    test_preds = np.vstack(all_test_preds)
    test_targets = np.vstack(all_test_targets)
    
    auprc_scores = []
    brier_scores = []
    for k in range(num_relations):
        y_true_k = test_targets[:, k]
        y_pred_k = test_preds[:, k]
        if y_true_k.sum() > 0:
            auprc_scores.append(compute_auprc(y_true_k, y_pred_k))
            brier_scores.append(brier_score_loss(y_true_k, y_pred_k))
            
    macro_auprc = np.mean(auprc_scores)
    mean_brier = np.mean(brier_scores)
    p_at_5 = compute_precision_at_k(test_targets, test_preds, k=5)
    
    print("\n" + "*"*50)
    print("OPTIMIZED DECAGON GNN EVALUATION RESULTS:")
    print(f"  Macro AUPRC:      {macro_auprc:.4f}")
    print(f"  Mean Brier Score: {mean_brier:.4f}")
    print(f"  Precision@5:      {p_at_5:.4f}")
    print("*"*50)
    
    torch.save(model.state_dict(), os.path.join(art_dir, "optimized_decagon_gnn.pt"))
    print("Saved optimized model weights to artifacts/optimized_decagon_gnn.pt.")

if __name__ == "__main__":
    main()
