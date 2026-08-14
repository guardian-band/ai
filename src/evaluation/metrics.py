import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    brier_score_loss,
    confusion_matrix,
    ndcg_score
)
from typing import Dict, Any, List

def calculate_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    """Calculates ECE with equal-width probability bins."""
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        # Include 0.0 in the first bin
        if bin_lower == 0.0:
            in_bin = in_bin | (y_prob == 0.0)
            
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)

def calculate_ranking_metrics_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int = 5) -> Dict[str, float]:
    """Calculates Precision@k, Recall@k, NDCG@k per pair."""
    # Assuming y_true and y_prob are matrices [n_pairs, n_labels]
    n_pairs = y_true.shape[0]
    
    precisions = []
    recalls = []
    ndcgs = []
    
    for i in range(n_pairs):
        yt = y_true[i]
        yp = y_prob[i]
        
        valid = ~np.isnan(yt)
        if np.sum(yt[valid]) == 0:
            continue
            
        yt_v = yt[valid]
        yp_v = yp[valid]
        
        # Sort by predicted probability
        sorted_indices = np.argsort(-yp_v)
        yt_sorted = yt_v[sorted_indices]
        
        top_k = yt_sorted[:k]
        
        precisions.append(np.sum(top_k) / k)
        recalls.append(np.sum(top_k) / np.sum(yt_v))
        
        # NDCG@k
        ndcgs.append(ndcg_score([yt_v], [yp_v], k=k))
        
    if not precisions:
        return {"precision_at_k": float('nan'), "recall_at_k": float('nan'), "ndcg_at_k": float('nan')}
        
    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs))
    }

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates sens, spec, ppv, npv, f1."""
    if len(y_true) == 0:
        return {}
        
    # Extract values from confusion matrix
    # labels=[0,1] forces 2x2 even if one class is missing
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    npv = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')
    
    return {
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": f1
    }

def compute_all_metrics(
    y_true: pd.DataFrame, 
    y_prob: pd.DataFrame, 
    thresholds: Dict[str, float],
    train_prevalences: Dict[str, float]
) -> Dict[str, Any]:
    """Computes all required metrics for Phase 5."""
    
    results = {}
    
    y_true_mat = y_true.values
    y_prob_mat = y_prob.values
    
    valid_mask = ~np.isnan(y_true_mat)
    yt_flat = y_true_mat[valid_mask]
    yp_flat = y_prob_mat[valid_mask]
    
    # Micro metrics
    if len(np.unique(yt_flat)) == 2:
        results["micro_ap"] = average_precision_score(yt_flat, yp_flat)
        results["micro_auroc"] = roc_auc_score(yt_flat, yp_flat)
        results["brier_score"] = brier_score_loss(yt_flat, yp_flat)
        results["ece_15"] = calculate_expected_calibration_error(yt_flat, yp_flat)
    else:
        results["micro_ap"] = float('nan')
        results["micro_auroc"] = float('nan')
        results["brier_score"] = float('nan')
        results["ece_15"] = float('nan')
        
    # Ranking
    ranking = calculate_ranking_metrics_at_k(y_true_mat, y_prob_mat, k=5)
    results.update(ranking)
    
    # Macro metrics and per-label
    label_aps = []
    label_aurocs = []
    
    per_label_metrics = {}
    undefined_labels = []
    
    for col in y_true.columns:
        # Keep truth/probability rows aligned when either side contains a
        # missing value.  Dropping each series independently can silently pair
        # one pair's truth with another pair's probability.
        aligned = y_true[col].notna() & y_prob[col].notna()
        yt = y_true.loc[aligned, col].values
        yp = y_prob.loc[aligned, col].values
        
        if len(np.unique(yt)) < 2:
            undefined_labels.append(col)
            continue
            
        ap = average_precision_score(yt, yp)
        auroc = roc_auc_score(yt, yp)
        
        label_aps.append(ap)
        label_aurocs.append(auroc)
        
        prev = train_prevalences.get(col, float('nan'))
        lift = ap / prev if prev > 0 else float('nan')
        
        # Classification
        th = thresholds.get(col, 0.5)
        if isinstance(th, dict):
            th = th.get('threshold', 0.5)
            
        y_pred = (yp >= th).astype(int)
        clf_mets = calculate_classification_metrics(yt, y_pred)
        
        per_label_metrics[col] = {
            "ap": ap,
            "auroc": auroc,
            "ap_lift": lift,
            **clf_mets
        }
        
    results["macro_ap"] = np.mean(label_aps) if label_aps else float('nan')
    results["macro_auroc"] = np.mean(label_aurocs) if label_aurocs else float('nan')
    # Expose the per-label values for auditable persistence.  Keep the
    # evaluator's scalar output unchanged for existing callers.
    results["per_label_metrics"] = per_label_metrics
    
    results["undefined_labels_count"] = len(undefined_labels)
    results["undefined_labels"] = undefined_labels
    
    return results
