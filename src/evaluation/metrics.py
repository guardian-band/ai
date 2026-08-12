import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, brier_score_loss

def compute_macro_average_precision(y_true, y_pred_prob):
    """
    Computes Macro Average Precision (AP) across labels using scikit-learn average_precision_score.
    Handles labels with zero positives or zero negatives transparently.
    """
    aps = []
    num_labels = y_true.shape[1]
    for k in range(num_labels):
        yt = y_true[:, k]
        yp = y_pred_prob[:, k]
        if yt.sum() > 0 and (len(yt) - yt.sum()) > 0:
            ap = average_precision_score(yt, yp)
            aps.append(ap)
    return np.mean(aps) if len(aps) > 0 else 0.0

def compute_micro_average_precision(y_true, y_pred_prob):
    """
    Computes Micro Average Precision (AP) across all flattened label instances.
    """
    yt_flat = y_true.ravel()
    yp_flat = y_pred_prob.ravel()
    if yt_flat.sum() > 0:
        return average_precision_score(yt_flat, yp_flat)
    return 0.0

def compute_precision_at_k(y_true_matrix, y_pred_prob_matrix, k=5):
    """
    Computes mean Precision@K for pair-level multi-label predictions.
    """
    precisions = []
    for i in range(len(y_true_matrix)):
        top_k_indices = np.argsort(y_pred_prob_matrix[i])[-k:]
        relevant_retrieved = y_true_matrix[i, top_k_indices].sum()
        precisions.append(relevant_retrieved / float(k))
    return float(np.mean(precisions))

def compute_recall_at_k(y_true_matrix, y_pred_prob_matrix, k=5):
    """
    Computes mean Recall@K for pair-level multi-label predictions.
    """
    recalls = []
    for i in range(len(y_true_matrix)):
        total_positives = y_true_matrix[i].sum()
        if total_positives > 0:
            top_k_indices = np.argsort(y_pred_prob_matrix[i])[-k:]
            relevant_retrieved = y_true_matrix[i, top_k_indices].sum()
            recalls.append(relevant_retrieved / float(total_positives))
    return float(np.mean(recalls)) if len(recalls) > 0 else 0.0

def compute_ndcg_at_k(y_true_matrix, y_pred_prob_matrix, k=5):
    """
    Computes mean NDCG@K for pair-level multi-label predictions.
    """
    ndcgs = []
    for i in range(len(y_true_matrix)):
        yt = y_true_matrix[i]
        yp = y_pred_prob_matrix[i]
        if yt.sum() > 0:
            top_k_indices = np.argsort(yp)[-k:][::-1]
            dcg = np.sum((2 ** yt[top_k_indices] - 1) / np.log2(np.arange(2, k + 2)))
            
            ideal_top_k = np.argsort(yt)[-k:][::-1]
            idcg = np.sum((2 ** yt[ideal_top_k] - 1) / np.log2(np.arange(2, k + 2)))
            
            if idcg > 0:
                ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if len(ndcgs) > 0 else 0.0

def compute_macro_auroc(y_true, y_pred_prob):
    """
    Computes Macro AUROC across labels where both classes are present.
    """
    aurocs = []
    num_labels = y_true.shape[1]
    for k in range(num_labels):
        yt = y_true[:, k]
        yp = y_pred_prob[:, k]
        if yt.sum() > 0 and (len(yt) - yt.sum()) > 0:
            aurocs.append(roc_auc_score(yt, yp))
    return float(np.mean(aurocs)) if len(aurocs) > 0 else 0.0

def compute_comprehensive_metrics(y_true, y_pred_prob, k=5):
    """
    Returns an authoritative metrics dictionary covering Macro/Micro AP, Precision@K,
    Recall@K, NDCG@K, AUROC, and Brier Score.
    """
    macro_ap = float(compute_macro_average_precision(y_true, y_pred_prob))
    micro_ap = float(compute_micro_average_precision(y_true, y_pred_prob))
    p_at_k = compute_precision_at_k(y_true, y_pred_prob, k=k)
    r_at_k = compute_recall_at_k(y_true, y_pred_prob, k=k)
    ndcg_at_k = compute_ndcg_at_k(y_true, y_pred_prob, k=k)
    macro_auroc = compute_macro_auroc(y_true, y_pred_prob)
    
    brier = float(brier_score_loss(y_true.ravel(), y_pred_prob.ravel()))
    
    return {
        "macro_ap": macro_ap,
        "micro_ap": micro_ap,
        f"precision_at_{k}": p_at_k,
        f"recall_at_{k}": r_at_k,
        f"ndcg_at_{k}": ndcg_at_k,
        "macro_auroc": macro_auroc,
        "brier_score": brier
    }
