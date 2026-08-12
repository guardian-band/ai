import os
import shutil
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss
from tqdm import tqdm

from src.data_loader import download_biosnap, load_and_map_dataset
from src.features.smiles_encoder import build_drug_features
from src.features.pair_features import build_symmetric_pair_features

def compute_auprc(y_true, y_pred_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    return auc(recall, precision)

def main():
    base_dir = "/Users/acelyayildiz/.gemini/antigravity/scratch/polypharmacy_ai"
    raw_dir = os.path.join(base_dir, "data/raw")
    ext_dir = os.path.join(base_dir, "data/external")
    art_dir = os.path.join(base_dir, "artifacts")
    splits_dir = os.path.join(art_dir, "splits")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ext_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    
    # Copy raw CSVs from archive
    src_archive = "/Users/acelyayildiz/Downloads/archive"
    for fname in ["drugs_master.csv", "indications.csv", "side_effects.csv"]:
        src_path = os.path.join(src_archive, fname)
        dst_path = os.path.join(raw_dir, fname)
        if not os.path.exists(dst_path) and os.path.exists(src_path):
            print(f"Copying {fname} to raw data directory...")
            shutil.copy(src_path, dst_path)
            
    # 1. Download & Load BioSNAP dataset
    biosnap_path = download_biosnap(dest_dir=ext_dir)
    triples_df, top_cuis = load_and_map_dataset(
        drugs_master_path=os.path.join(raw_dir, "drugs_master.csv"),
        side_effects_path=os.path.join(raw_dir, "side_effects.csv"),
        biosnap_path=biosnap_path,
        top_k_side_effects=100
    )
    
    # Save side_effect_labels.csv
    side_effects_meta = pd.read_csv(os.path.join(raw_dir, "side_effects.csv"))
    cui_to_name = dict(zip(side_effects_meta['umls_cui_from_meddra'], side_effects_meta['side_effect_name']))
    
    labels_df = pd.DataFrame({
        'cui': top_cuis,
        'side_effect_name': [cui_to_name.get(c, "Unknown") for c in top_cuis],
        'frequency': [ (triples_df['cui'] == c).sum() for c in top_cuis ]
    })
    labels_df.to_csv(os.path.join(art_dir, "side_effect_labels.csv"), index=False)
    print("Saved side_effect_labels.csv to artifacts.")
    
    # 2. Build Drug Features (SMILES Morgan Fingerprints + Indications)
    drug_feature_dict, single_dim = build_drug_features(
        drugs_master_path=os.path.join(raw_dir, "drugs_master.csv"),
        indications_path=os.path.join(raw_dir, "indications.csv"),
        n_bits=1024
    )
    
    # 3. Create Multi-Label Pair Dataset
    print("Constructing Multi-Label Pair Dataset...")
    cui_to_idx = {cui: idx for idx, cui in enumerate(top_cuis)}
    
    pair_grouped = triples_df.groupby(['drug_a', 'drug_b'])['cui'].apply(list).reset_index()
    print(f"Total positive drug pairs: {len(pair_grouped)}")
    
    # Filter pairs where both drugs have features
    valid_pairs = []
    y_labels = []
    
    for _, row in pair_grouped.iterrows():
        da, db = row['drug_a'], row['drug_b']
        if da in drug_feature_dict and db in drug_feature_dict:
            valid_pairs.append((da, db))
            y_vec = np.zeros(len(top_cuis), dtype=np.float32)
            for c in row['cui']:
                if c in cui_to_idx:
                    y_vec[cui_to_idx[c]] = 1.0
            y_labels.append(y_vec)
            
    valid_pairs = np.array(valid_pairs)
    y_labels = np.array(y_labels)
    print(f"Valid positive drug pairs with complete features: {len(valid_pairs)}")
    
    # 4. Controlled Negative Sampling (Unlabeled pairs as negatives)
    print("Generating Controlled Negative Samples...")
    all_drugs = list(drug_feature_dict.keys())
    pos_pair_set = set(tuple(p) for p in valid_pairs)
    
    neg_pairs = []
    np.random.seed(42)
    target_neg_count = len(valid_pairs)
    
    while len(neg_pairs) < target_neg_count:
        d1, d2 = np.random.choice(all_drugs, size=2, replace=False)
        da, db = min(d1, d2), max(d1, d2)
        if (da, db) not in pos_pair_set:
            neg_pairs.append((da, db))
            pos_pair_set.add((da, db)) # prevent duplicate negs
            
    neg_pairs = np.array(neg_pairs)
    y_neg_labels = np.zeros((len(neg_pairs), len(top_cuis)), dtype=np.float32)
    
    all_pairs = np.vstack([valid_pairs, neg_pairs])
    all_y = np.vstack([y_labels, y_neg_labels])
    all_is_pos = np.array([1]*len(valid_pairs) + [0]*len(neg_pairs))
    
    print(f"Combined Dataset: {len(all_pairs)} pairs (Pos: {len(valid_pairs)}, Neg: {len(neg_pairs)})")
    
    # Save pair_side_effect_labels.parquet
    pair_labels_df = pd.DataFrame({
        'drug_a': all_pairs[:, 0],
        'drug_b': all_pairs[:, 1],
        'is_positive_interaction': all_is_pos
    })
    pair_labels_df.to_parquet(os.path.join(art_dir, "pair_side_effect_labels.parquet"), index=False)
    print("Saved pair_side_effect_labels.parquet to artifacts.")
    
    # 5. Pair-Disjoint Train/Val/Test Splitting
    print("Performing Pair-Disjoint Train/Validation/Test Split...")
    n_total = len(all_pairs)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    
    train_end = int(0.7 * n_total)
    val_end = int(0.85 * n_total)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    pd.DataFrame(pair_labels_df.iloc[train_idx]).to_csv(os.path.join(splits_dir, "train_pairs.csv"), index=False)
    pd.DataFrame(pair_labels_df.iloc[val_idx]).to_csv(os.path.join(splits_dir, "validation_pairs.csv"), index=False)
    pd.DataFrame(pair_labels_df.iloc[test_idx]).to_csv(os.path.join(splits_dir, "test_pairs.csv"), index=False)
    print("Saved split CSVs in artifacts/splits/.")
    
    # 6. Extract Symmetric Pair Features
    print("Extracting Symmetric Pair Features...")
    X_all = []
    for da, db in tqdm(all_pairs):
        v1 = drug_feature_dict[da]
        v2 = drug_feature_dict[db]
        feat = build_symmetric_pair_features(v1, v2)
        X_all.append(feat)
    X_all = np.array(X_all, dtype=np.float32)
    print(f"Pair Feature Matrix Shape: {X_all.shape}")
    
    X_train, y_train = X_all[train_idx], all_y[train_idx]
    X_test, y_test = X_all[test_idx], all_y[test_idx]
    
    # 7. Train Baseline 1: Multi-Label Logistic Regression
    print("\n" + "="*50)
    print("TRAINING BASELINE 1: Logistic Regression (Per Side Effect)")
    print("="*50)
    
    auprc_scores = []
    brier_scores = []
    
    for k in tqdm(range(len(top_cuis)), desc="Training Side-Effect Classifiers"):
        y_tr_k = y_train[:, k]
        y_te_k = y_test[:, k]
        
        if y_tr_k.sum() == 0 or y_te_k.sum() == 0:
            continue
            
        clf = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs')
        clf.fit(X_train, y_tr_k)
        
        preds_prob = clf.predict_proba(X_test)[:, 1]
        
        score_auprc = compute_auprc(y_te_k, preds_prob)
        score_brier = brier_score_loss(y_te_k, preds_prob)
        
        auprc_scores.append(score_auprc)
        brier_scores.append(score_brier)
        
    macro_auprc = np.mean(auprc_scores)
    mean_brier = np.mean(brier_scores)
    
    print("\n" + "*"*50)
    print("BASELINE 1 EVALUATION RESULTS:")
    print(f"  Macro AUPRC:     {macro_auprc:.4f}")
    print(f"  Mean Brier Score: {mean_brier:.4f}")
    print("*"*50)

if __name__ == "__main__":
    main()
