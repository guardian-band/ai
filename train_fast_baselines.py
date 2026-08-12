import os
import shutil
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss
from joblib import Parallel, delayed
from tqdm import tqdm

from src.data_loader import download_biosnap, load_and_map_dataset
from src.features.smiles_encoder import build_drug_features
from src.features.pair_features import build_symmetric_pair_features

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

def train_eval_single_sgd(k, X_tr, y_tr_k, X_te, y_te_k):
    if y_tr_k.sum() < 5 or y_te_k.sum() == 0:
        return k, None, None, np.zeros(len(X_te))
        
    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, max_iter=100, tol=1e-3, random_state=42)
    clf.fit(X_tr, y_tr_k)
    
    probs = clf.predict_proba(X_te)[:, 1]
    auprc = compute_auprc(y_te_k, probs)
    brier = brier_score_loss(y_te_k, probs)
    
    return k, auprc, brier, probs

def train_eval_single_hgb(k, X_tr, y_tr_k, X_te, y_te_k):
    if y_tr_k.sum() < 5 or y_te_k.sum() == 0:
        return k, None, None, np.zeros(len(X_te))
        
    clf = HistGradientBoostingClassifier(max_iter=25, max_depth=5, learning_rate=0.1, random_state=42)
    clf.fit(X_tr, y_tr_k)
    
    probs = clf.predict_proba(X_te)[:, 1]
    auprc = compute_auprc(y_te_k, probs)
    brier = brier_score_loss(y_te_k, probs)
    
    return k, auprc, brier, probs

def main():
    base_dir = "/Users/acelyayildiz/.gemini/antigravity/scratch/polypharmacy_ai"
    raw_dir = os.path.join(base_dir, "data/raw")
    ext_dir = os.path.join(base_dir, "data/external")
    art_dir = os.path.join(base_dir, "artifacts")
    splits_dir = os.path.join(art_dir, "splits")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ext_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    
    # 1. Load BioSNAP dataset
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
    
    # 2. Build Drug Features
    drug_feature_dict, single_dim = build_drug_features(
        drugs_master_path=os.path.join(raw_dir, "drugs_master.csv"),
        indications_path=os.path.join(raw_dir, "indications.csv"),
        n_bits=512
    )
    
    # Save drug_features.parquet
    drug_feat_rows = []
    for db_id, feat in drug_feature_dict.items():
        row_dict = {'drugbank_id': db_id}
        for idx, val in enumerate(feat):
            row_dict[f'f_{idx}'] = val
        drug_feat_rows.append(row_dict)
    pd.DataFrame(drug_feat_rows).to_parquet(os.path.join(art_dir, "drug_features.parquet"), index=False)
    print("Saved drug_features.parquet to artifacts.")
    
    # 3. Create Multi-Label Pair Dataset
    cui_to_idx = {cui: idx for idx, cui in enumerate(top_cuis)}
    pair_grouped = triples_df.groupby(['drug_a', 'drug_b'])['cui'].apply(list).reset_index()
    
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
    
    # 4. Controlled Negative Sampling
    all_drugs = list(drug_feature_dict.keys())
    pos_pair_set = set(tuple(p) for p in valid_pairs)
    neg_pairs = []
    np.random.seed(42)
    
    while len(neg_pairs) < len(valid_pairs):
        d1, d2 = np.random.choice(all_drugs, size=2, replace=False)
        da, db = min(d1, d2), max(d1, d2)
        if (da, db) not in pos_pair_set:
            neg_pairs.append((da, db))
            pos_pair_set.add((da, db))
            
    neg_pairs = np.array(neg_pairs)
    y_neg_labels = np.zeros((len(neg_pairs), len(top_cuis)), dtype=np.float32)
    
    all_pairs = np.vstack([valid_pairs, neg_pairs])
    all_y = np.vstack([y_labels, y_neg_labels])
    all_is_pos = np.array([1]*len(valid_pairs) + [0]*len(neg_pairs))
    
    # Save pair_side_effect_labels.parquet
    pair_labels_df = pd.DataFrame({
        'drug_a': all_pairs[:, 0],
        'drug_b': all_pairs[:, 1],
        'is_positive_interaction': all_is_pos
    })
    pair_labels_df.to_parquet(os.path.join(art_dir, "pair_side_effect_labels.parquet"), index=False)
    print("Saved pair_side_effect_labels.parquet to artifacts.")
    
    # 5. Split Strategies: Pair-Disjoint & Cold-Drug
    print("Creating Splits (Pair-Disjoint & Cold-Drug)...")
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
    
    unique_drugs = np.unique(all_pairs)
    cold_test_drugs = set(np.random.choice(unique_drugs, size=int(0.15 * len(unique_drugs)), replace=False))
    cold_mask = [ (p[0] in cold_test_drugs) or (p[1] in cold_test_drugs) for p in all_pairs ]
    cold_test_df = pair_labels_df[cold_mask]
    cold_test_df.to_csv(os.path.join(splits_dir, "cold_drug_test.csv"), index=False)
    print(f"Saved splits to artifacts/splits/ (Cold-drug test pairs: {len(cold_test_df)}).")
    
    # 6. Extract Symmetric Pair Features
    print("Extracting Symmetric Pair Features...")
    X_all = []
    for da, db in tqdm(all_pairs):
        v1 = drug_feature_dict[da]
        v2 = drug_feature_dict[db]
        feat = build_symmetric_pair_features(v1, v2)
        X_all.append(feat)
    X_all = np.array(X_all, dtype=np.float32)
    
    X_train, y_train = X_all[train_idx], all_y[train_idx]
    X_test, y_test = X_all[test_idx], all_y[test_idx]
    
    # 7. Baseline 1: Fast SGD Logistic Regression
    print("\n" + "="*50)
    print("TRAINING BASELINE 1: Fast SGD Logistic Regression (Parallel)")
    print("="*50)
    
    sgd_results = Parallel(n_jobs=-1)(
        delayed(train_eval_single_sgd)(k, X_train, y_train[:, k], X_test, y_test[:, k])
        for k in tqdm(range(len(top_cuis)), desc="Parallel Training SGD")
    )
    
    b1_auprc_list = []
    b1_brier_list = []
    test_pred_matrix_b1 = np.zeros_like(y_test)
    
    for res in sgd_results:
        if res[1] is not None:
            k, auprc, brier, probs = res
            b1_auprc_list.append(auprc)
            b1_brier_list.append(brier)
            test_pred_matrix_b1[:, k] = probs
            
    b1_macro_auprc = np.mean(b1_auprc_list)
    b1_mean_brier = np.mean(b1_brier_list)
    b1_p_at_5 = compute_precision_at_k(y_test, test_pred_matrix_b1, k=5)
    
    print("\n" + "*"*50)
    print("BASELINE 1 (Logistic Regression SGD) RESULTS:")
    print(f"  Macro AUPRC:      {b1_macro_auprc:.4f}")
    print(f"  Mean Brier Score: {b1_mean_brier:.4f}")
    print(f"  Precision@5:      {b1_p_at_5:.4f}")
    print("*"*50)
    
    # 8. Baseline 2: Fast HistGradientBoosting Classifier
    print("\n" + "="*50)
    print("TRAINING BASELINE 2: HistGradientBoosting (Gradient Trees)")
    print("="*50)
    
    hgb_results = Parallel(n_jobs=-1)(
        delayed(train_eval_single_hgb)(k, X_train, y_train[:, k], X_test, y_test[:, k])
        for k in tqdm(range(len(top_cuis)), desc="Parallel Training HistGradientBoosting")
    )
    
    b2_auprc_list = []
    b2_brier_list = []
    test_pred_matrix_b2 = np.zeros_like(y_test)
    
    for res in hgb_results:
        if res[1] is not None:
            k, auprc, brier, probs = res
            b2_auprc_list.append(auprc)
            b2_brier_list.append(brier)
            test_pred_matrix_b2[:, k] = probs
            
    b2_macro_auprc = np.mean(b2_auprc_list)
    b2_mean_brier = np.mean(b2_brier_list)
    b2_p_at_5 = compute_precision_at_k(y_test, test_pred_matrix_b2, k=5)
    
    print("\n" + "*"*50)
    print("BASELINE 2 (HistGradientBoosting) RESULTS:")
    print(f"  Macro AUPRC:      {b2_macro_auprc:.4f}")
    print(f"  Mean Brier Score: {b2_mean_brier:.4f}")
    print(f"  Precision@5:      {b2_p_at_5:.4f}")
    print("*"*50)

if __name__ == "__main__":
    main()
