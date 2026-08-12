import numpy as np
import pandas as pd
from typing import Dict, Any

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_ppv: float = 0.5
) -> Dict[str, Any]:
    """
    Finds the threshold that maximizes recall subject to PPV >= min_ppv.
    If no threshold meets min_ppv, maximizes F1.
    """
    candidates = np.unique(y_prob)
    candidates = np.concatenate([[0.0], candidates, [1.0]])
    candidates = np.sort(np.unique(candidates))
    
    best_thresh = 0.0
    best_rule = "max_f1"
    max_metric = -1.0
    
    # Pre-calculate to avoid loop if possible, but loop over unique is usually fast enough for validation set
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # thresholds from precision_recall_curve might not include 0 and 1 explicitly in the exact way we want,
    # but it's close enough. The plan says "unique validation probabilities plus 0 and 1".
    
    # We will just evaluate all unique candidates manually for exactness to the plan
    best_f1 = -1.0
    best_f1_thresh = 0.0
    
    for th in candidates:
        preds = (y_prob >= th).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * (ppv * rec) / (ppv + rec) if (ppv + rec) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = th
            
        if ppv >= min_ppv:
            if rec > max_metric:
                max_metric = rec
                best_thresh = th
                best_rule = f"max_recall_at_ppv_{min_ppv}"
                
    if max_metric == -1.0:
        # No candidate met PPV constraint
        best_thresh = best_f1_thresh
        best_rule = "max_f1"
        
    return {
        "threshold": float(best_thresh),
        "rule": best_rule
    }

def select_thresholds(
    val_y_true: pd.DataFrame, 
    val_y_prob: pd.DataFrame, 
    min_ppv: float = 0.5
) -> Dict[str, Any]:
    """
    val_y_true, val_y_prob: DataFrames where columns are label_cuis.
    Returns per-label threshold dict, and falls back to micro F1 if <10 positives.
    """
    thresholds = {}
    
    # Calculate global threshold using micro F1
    y_true_flat = val_y_true.values.flatten()
    y_prob_flat = val_y_prob.values.flatten()
    
    # Only consider valid entries if there are NaNs
    valid_mask = ~np.isnan(y_true_flat)
    y_true_flat = y_true_flat[valid_mask]
    y_prob_flat = y_prob_flat[valid_mask]
    
    global_res = find_best_threshold(y_true_flat, y_prob_flat, min_ppv=0.0) # For micro, we just use max F1
    global_thresh = global_res["threshold"]
    
    for label in val_y_true.columns:
        yt = val_y_true[label].dropna().values
        yp = val_y_prob[label].dropna().values
        
        pos_count = np.sum(yt == 1)
        
        if pos_count < 10:
            thresholds[label] = {
                "threshold": global_thresh,
                "rule": "global_micro_f1_due_to_low_support",
                "positive_count": int(pos_count)
            }
        else:
            res = find_best_threshold(yt, yp, min_ppv=min_ppv)
            res["positive_count"] = int(pos_count)
            thresholds[label] = res
            
    return thresholds
