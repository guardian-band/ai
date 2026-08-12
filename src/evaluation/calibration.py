import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score
import numpy as np

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize temperature to 1.0 (log(1.0) = 0.0)
        self.log_temp = nn.Parameter(torch.zeros(1))
        
    def forward(self, logits):
        temp = torch.clamp(torch.exp(self.log_temp), 0.05, 10.0)
        return logits / temp

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Fits a single temperature scale to logits using NLL on validation set.
    """
    valid_mask = ~np.isnan(labels)
    t_logits = torch.tensor(logits[valid_mask], dtype=torch.float32)
    t_labels = torch.tensor(labels[valid_mask], dtype=torch.float32)
    
    if len(t_logits) == 0:
        return 1.0
        
    # Baseline
    bce = nn.BCEWithLogitsLoss()
    initial_nll = bce(t_logits, t_labels).item()
    initial_ap = average_precision_score(t_labels.numpy(), t_logits.numpy())
    
    scaler = TemperatureScaler()
    optimizer = optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=100)
    
    def eval_closure():
        optimizer.zero_grad()
        scaled = scaler(t_logits)
        loss = bce(scaled, t_labels)
        loss.backward()
        return loss
        
    optimizer.step(eval_closure)
    
    final_temp = torch.clamp(torch.exp(scaler.log_temp), 0.05, 10.0).item()
    
    # Check if NLL improved and macro AP unchanged
    with torch.no_grad():
        scaled_logits = scaler(t_logits)
        final_nll = bce(scaled_logits, t_labels).item()
        
    # Calibration does not change AP because it's a monotonic transformation,
    # but we double check as requested.
    final_ap = average_precision_score(t_labels.numpy(), scaled_logits.numpy())
    
    if final_nll < initial_nll and abs(final_ap - initial_ap) < 1e-10:
        return final_temp
        
    return 1.0

def calibrate_logits(val_logits: dict, val_labels: dict) -> dict:
    """
    Expects dicts with keys 'organ' and 'specific' containing DataFrames or numpy arrays.
    Returns calibrated logits and the fitted temperatures.
    """
    temps = {}
    calibrated_logits = {}
    
    for level in ['organ', 'specific']:
        if level in val_logits and level in val_labels:
            logits_flat = val_logits[level].values.flatten()
            labels_flat = val_labels[level].values.flatten()
            
            temp = fit_temperature(logits_flat, labels_flat)
            temps[level] = temp
            
            calibrated = val_logits[level] / temp
            calibrated_logits[level] = calibrated
            
    return calibrated_logits, temps
