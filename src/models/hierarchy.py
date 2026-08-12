import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class HierarchyLoss(nn.Module):
    """
    Penalizes predictions where a specific side effect probability 
    is greater than its parent organ probability.
    hierarchy_loss = mean(relu(sigmoid(specific) - sigmoid(parent)))
    """
    def __init__(self, mapping: Dict[int, int]):
        """
        mapping: maps specific_label_idx -> parent_label_idx
        """
        super().__init__()
        self.mapping = mapping
        
        # Pre-compute indices for fast tensor operations
        self.specific_indices = list(mapping.keys())
        self.parent_indices = [mapping[i] for i in self.specific_indices]
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.specific_indices:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        probs = torch.sigmoid(logits)
        
        specific_probs = probs[:, self.specific_indices]
        parent_probs = probs[:, self.parent_indices]
        
        # relu(p_specific - p_parent) -> >0 if specific > parent
        violations = F.relu(specific_probs - parent_probs)
        
        return torch.mean(violations)
