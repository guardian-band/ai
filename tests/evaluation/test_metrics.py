import pytest
import numpy as np
import pandas as pd
from src.evaluation.metrics import (
    calculate_expected_calibration_error,
    calculate_ranking_metrics_at_k,
    calculate_classification_metrics,
    compute_all_metrics
)
from src.evaluation.thresholds import find_best_threshold, select_thresholds
from src.evaluation.calibration import fit_temperature
from src.evaluation.bootstrap import calculate_bootstrap_ci

def test_calculate_expected_calibration_error():
    y_true = np.array([1, 1, 0, 0, 1])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.4])
    
    ece = calculate_expected_calibration_error(y_true, y_prob, bins=2)
    assert ece >= 0.0

def test_calculate_classification_metrics():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    
    mets = calculate_classification_metrics(y_true, y_pred)
    assert mets['sensitivity'] == 0.5
    assert mets['specificity'] == 0.5
    assert mets['ppv'] == 0.5
    assert mets['npv'] == 0.5
    assert mets['f1'] == 0.5

def test_find_best_threshold():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.6, 0.4, 0.1])
    
    res = find_best_threshold(y_true, y_prob, min_ppv=1.0)
    assert res['threshold'] == 0.6 # at 0.6: tp=2, fp=0, ppv=1.0, rec=1.0
    
    # Check fallback to max F1 when PPV constraint impossible
    y_true_2 = np.array([1, 0])
    y_prob_2 = np.array([0.4, 0.6]) # predicted wrong
    res2 = find_best_threshold(y_true_2, y_prob_2, min_ppv=0.8)
    # The best F1 is probably 0, thresh doesn't matter much but it shouldn't crash
    assert "threshold" in res2

def test_fit_temperature():
    logits = np.array([2.0, 1.0, -1.0, -2.0])
    labels = np.array([1, 1, 0, 0])
    temp = fit_temperature(logits, labels)
    assert temp > 0.0

def test_bootstrap():
    data = pd.DataFrame({
        'pair_id': ['1', '2', '3', '4'],
        'y_true': [1, 1, 0, 0],
        'y_prob': [0.9, 0.8, 0.2, 0.1]
    })
    
    def dummy_metric(df):
        from sklearn.metrics import roc_auc_score
        yt = df['y_true'].values
        yp = df['y_prob'].values
        if len(np.unique(yt)) < 2:
            return float('nan')
        return roc_auc_score(yt, yp)
        
    res = calculate_bootstrap_ci(data, dummy_metric, resamples=10)
    assert "mean" in res
    assert res['lower_ci'] <= res['upper_ci']
