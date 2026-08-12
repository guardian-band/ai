import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, accuracy_score, brier_score_loss

from src.models.chemberta_cross_attention_gnn import ChemBERTaCrossAttentionGNN
from src.features.meddra_hierarchy import build_meddra_hierarchical_mapping
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

def build_sider_mono_drug_priors(sider_path, drugs_master_path, top_cuis):
    print("Building SIDER 4.1 Mono-Drug Prior Knowledge Vectors...")
    drugs_df = pd.read_csv(drugs_master_path)
    sider_df = pd.read_csv(sider_path, sep='\t', header=None, names=['stitch_flat', 'stitch_stereo', 'cui_raw', 'meddra_type', 'cui', 'name'])
    
    drugs_df['cid'] = pd.to_numeric(drugs_df['pubchem_compound_id'], errors='coerce')
    cid_to_db = dict(zip(drugs_df['cid'].dropna().astype(int), drugs_df['drugbank_id']))
    
    def parse_stitch(s):
        if not isinstance(s, str): return None
        s = s.replace('CID1', '').replace('CID0', '').lstrip('0')
        try: return int(s)
        except: return None
        
    sider_df['cid'] = sider_df['stitch_flat'].apply(parse_stitch)
    sider_df['db_id'] = sider_df['cid'].map(cid_to_db)
    
    cui_to_idx = {cui: idx for idx, cui in enumerate(top_cuis)}
    num_cuis = len(top_cuis)
    
    sider_grouped = sider_df.dropna(subset=['db_id']).groupby('db_id')['cui'].apply(set).to_dict()
    
    drug_sider_dict = {}
    for db_id in drugs_df['drugbank_id']:
        vec = np.zeros(num_cuis, dtype=np.float32)
        if db_id in sider_grouped:
            for c in sider_grouped[db_id]:
                if c in cui_to_idx:
                    vec[cui_to_idx[c]] = 1.0
        drug_sider_dict[db_id] = vec
        
    print(f"Constructed SIDER Mono-Drug Prior vectors for {len(drug_sider_dict)} drugs.")
    return drug_sider_dict

class ChemBERTaCrossAttentionDataset(Dataset):
    def __init__(self, pairs, y_organ_mat, y_spec_mat, drug_feature_dict, sider_prior_dict):
        self.pairs = pairs
        self.y_organ = torch.tensor(y_organ_mat, dtype=torch.float32)
        self.y_spec = torch.tensor(y_spec_mat, dtype=torch.float32)
        self.drug_feature_dict = drug_feature_dict
        self.sider_prior_dict = sider_prior_dict
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        da, db = self.pairs[idx]
        feat_a = torch.tensor(self.drug_feature_dict[da], dtype=torch.float32)
        feat_b = torch.tensor(self.drug_feature_dict[db], dtype=torch.float32)
        
        prior_a = self.sider_prior_dict.get(da, np.zeros(self.y_spec.shape[1], dtype=np.float32))
        prior_b = self.sider_prior_dict.get(db, np.zeros(self.y_spec.shape[1], dtype=np.float32))
        pair_prior = torch.tensor(np.maximum(prior_a, prior_b), dtype=torch.float32)
        
        return feat_a, feat_b, pair_prior, self.y_organ[idx], self.y_spec[idx]

def main():
    start_time = time.time()
    base_dir = "/Users/acelyayildiz/.gemini/antigravity/scratch/polypharmacy_ai"
    raw_dir = os.path.join(base_dir, "data/raw")
    art_dir = os.path.join(base_dir, "artifacts")
    splits_dir = os.path.join(art_dir, "splits")
    
    print("="*65)
    print("TRAINING CHEMBERTA + MULTI-HEAD CROSS-ATTENTION + ADDITIVE FUSION")
    print("Unified Drug Transformers & Bio-Molecular Deep Graph Architecture")
    print("="*65)
    
    drug_feat_df = pd.read_parquet(os.path.join(art_dir, "drug_features_chemberta.parquet"))
    side_effects_df = pd.read_csv(os.path.join(art_dir, "side_effect_labels.csv"))
    train_pairs_df = pd.read_csv(os.path.join(splits_dir, "train_pairs.csv"))
    test_pairs_df = pd.read_csv(os.path.join(splits_dir, "test_pairs.csv"))
    
    feat_cols = [c for c in drug_feat_df.columns if c.startswith('f_') or c.startswith('chem_')]
    in_features = len(feat_cols)
    print(f"Total Enhanced Input Features per Drug: {in_features}")
    
    drug_feature_dict = {row['drugbank_id']: row[feat_cols].values.astype(np.float32) for _, row in drug_feat_df.iterrows()}
    
    top_cuis = side_effects_df['cui'].tolist()
    num_specific = len(top_cuis)
    cui_to_spec_idx = {cui: idx for idx, cui in enumerate(top_cuis)}
    
    # 1. SIDER Priors
    sider_prior_dict = build_sider_mono_drug_priors(
        sider_path=os.path.join(base_dir, "data/external/meddra_all_se.tsv"),
        drugs_master_path=os.path.join(raw_dir, "drugs_master.csv"),
        top_cuis=top_cuis
    )
    
    # 2. MedDRA Hierarchical Mapping
    cui_to_soc, soc_categories = build_meddra_hierarchical_mapping(os.path.join(raw_dir, "side_effects.csv"))
    soc_to_idx = {soc: idx for idx, soc in enumerate(soc_categories)}
    num_socs = len(soc_categories)
    
    spec_to_soc_idx = []
    for cui in top_cuis:
        soc_name = cui_to_soc.get(cui, 'General Disorders & Systemic')
        spec_to_soc_idx.append(soc_to_idx.get(soc_name, 0))
    spec_to_soc_idx = np.array(spec_to_soc_idx, dtype=np.int64)
    
    mapped_triples, _ = load_and_map_dataset(
        drugs_master_path=os.path.join(raw_dir, "drugs_master.csv"),
        side_effects_path=os.path.join(raw_dir, "side_effects.csv"),
        biosnap_path=os.path.join(base_dir, "data/external/ChChSe-Decagon_polypharmacy.csv"),
        top_k_side_effects=None
    )
    pair_grouped_all = mapped_triples.groupby(['drug_a', 'drug_b'])['cui'].apply(list).to_dict()
    
    def build_matrices(pairs_df):
        pairs_list = []
        y_organ_mat = []
        y_spec_mat = []
        for _, row in pairs_df.iterrows():
            da, db = row['drug_a'], row['drug_b']
            pairs_list.append((da, db))
            y_org = np.zeros(num_socs, dtype=np.float32)
            y_sp = np.zeros(num_specific, dtype=np.float32)
            if (da, db) in pair_grouped_all:
                for c in pair_grouped_all[(da, db)]:
                    if c in cui_to_soc and cui_to_soc[c] in soc_to_idx:
                        y_org[soc_to_idx[cui_to_soc[c]]] = 1.0
                    if c in cui_to_spec_idx:
                        y_sp[cui_to_spec_idx[c]] = 1.0
            y_organ_mat.append(y_org)
            y_spec_mat.append(y_sp)
        return np.array(pairs_list), np.array(y_organ_mat), np.array(y_spec_mat)

    train_pairs, y_train_org, y_train_spec = build_matrices(train_pairs_df)
    test_pairs, y_test_org, y_test_spec = build_matrices(test_pairs_df)
    
    train_dataset = ChemBERTaCrossAttentionDataset(train_pairs, y_train_org, y_train_spec, drug_feature_dict, sider_prior_dict)
    test_dataset = ChemBERTaCrossAttentionDataset(test_pairs, y_test_org, y_test_spec, drug_feature_dict, sider_prior_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Compute Acceleration: {device}")
    
    model = ChemBERTaCrossAttentionGNN(
        in_features=in_features,
        hidden_dim=256,
        embedding_dim=128,
        num_socs=num_socs,
        num_specific=num_specific,
        soc_indices_map=spec_to_soc_idx,
        dropout=0.15
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    epochs = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Positive weights for specific side effects to reward rare positives in BCE
    pos_counts = y_train_spec.sum(axis=0)
    neg_counts = len(y_train_spec) - pos_counts
    pos_weights = torch.tensor(np.clip(neg_counts / np.maximum(pos_counts, 1.0), 1.0, 5.0), dtype=torch.float32).to(device)
    
    criterion_org = nn.BCEWithLogitsLoss()
    criterion_sp = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    print("\nStarting Cross-Attention GPU Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for feat_a, feat_b, pair_prior, target_org, target_sp in train_loader:
            feat_a, feat_b, pair_prior = feat_a.to(device), feat_b.to(device), pair_prior.to(device)
            target_org, target_sp = target_org.to(device), target_sp.to(device)
            
            optimizer.zero_grad()
            logits_org, logits_sp = model(feat_a, feat_b, sider_priors=pair_prior, return_logits=True)
            
            loss = criterion_org(logits_org, target_org) + criterion_sp(logits_sp, target_sp)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target_org)
            
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            avg_loss = total_loss / len(train_dataset)
            print(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {avg_loss:.4f} (LR: {scheduler.get_last_lr()[0]:.6f})")
            
    # Evaluation on Test Pairs
    print("\nEvaluating ChemBERTa Cross-Attention Model on Test Pairs...")
    model.eval()
    all_org_preds, all_spec_preds = [], []
    with torch.no_grad():
        for feat_a, feat_b, pair_prior, _, _ in test_loader:
            feat_a, feat_b, pair_prior = feat_a.to(device), feat_b.to(device), pair_prior.to(device)
            prob_org, prob_sp = model(feat_a, feat_b, sider_priors=pair_prior, return_logits=False)
            all_org_preds.append(prob_org.cpu().numpy())
            all_spec_preds.append(prob_sp.cpu().numpy())
            
    test_org_preds = np.vstack(all_org_preds)
    test_spec_preds = np.vstack(all_spec_preds)
    
    # 1. Organ Level
    org_auroc, org_acc, org_auprc = [], [], []
    for k in range(num_socs):
        y_t = y_test_org[:, k]
        y_p = test_org_preds[:, k]
        if y_t.sum() > 0:
            org_auroc.append(roc_auc_score(y_t, y_p))
            org_auprc.append(compute_auprc(y_t, y_p))
            org_acc.append(accuracy_score(y_t, (y_p >= 0.4).astype(int)))
            
    # 2. Specific Side Effects
    sp_auroc, sp_acc, sp_auprc, sp_brier = [], [], [], []
    for k in range(num_specific):
        y_t = y_test_spec[:, k]
        y_p = test_spec_preds[:, k]
        if y_t.sum() > 0:
            sp_auroc.append(roc_auc_score(y_t, y_p))
            sp_auprc.append(compute_auprc(y_t, y_p))
            sp_acc.append(accuracy_score(y_t, (y_p >= 0.3).astype(int)))
            sp_brier.append(brier_score_loss(y_t, y_p))
            
    p_at_5 = compute_precision_at_k(y_test_spec, test_spec_preds, k=5)
    elapsed = time.time() - start_time
    
    print("\n" + "*"*65)
    print("🏆 CHEMBERTA + CROSS-ATTENTION AI MODEL - EVALUATION REPORT")
    print("="*65)
    print(f"⏱️ Total Execution & Training Time: {elapsed:.1f} seconds")
    print("-----------------------------------------------------------------")
    print("📌 SEVİYE 1: 15 MEDDRA ORGAN SİSTEMİ RİSK DEĞERLENDİRMESİ:")
    print(f"  Organ Sistemi Sınıflandırma Doğruluğu:   {np.mean(org_acc)*100:.2f}%")
    print(f"  Organ Sistemi AUROC (Ayırt Edicilik):    {np.mean(org_auroc)*100:.2f}%")
    print(f"  Organ Sistemi Macro AUPRC (Sıralama):    {np.mean(org_auprc):.4f}")
    print("-----------------------------------------------------------------")
    print("📌 SEVİYE 2: SPESİFİK YAN ETKİ TEŞHİSİ (Top 100 Klinik Teşhis):")
    print(f"  Spesifik Yan Etki Doğruluğu:             {np.mean(sp_acc)*100:.2f}%")
    print(f"  Spesifik Yan Etki AUROC:                 {np.mean(sp_auroc)*100:.2f}%")
    print(f"  Spesifik Yan Etki Macro AUPRC (Sıralama):{np.mean(sp_auprc):.4f}")
    print(f"  Spesifik Yan Etki Precision@5:           {p_at_5:.4f}")
    print("*"*65)
    
    model_save_path = os.path.join(art_dir, "champion_chemberta_cross_attention_gnn.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nSaved Champion ChemBERTa Cross-Attention Model to {model_save_path}")

if __name__ == "__main__":
    main()
