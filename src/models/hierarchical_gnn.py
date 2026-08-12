import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalMedDRAGNN(nn.Module):
    """
    Hierarchical Multi-Task Polypharmacy Graph Neural Network:
    Level 1: 15 MedDRA Major Organ System Classes (SOCs) - High confidence organ risk
    Level 2: 100 Specific Polypharmacy Side Effects - Fine-grained relational prediction
    """
    def __init__(self, in_features, hidden_dim=256, embedding_dim=128, num_socs=15, num_specific=100, dropout=0.2):
        super(HierarchicalMedDRAGNN, self).__init__()
        
        # 1. Shared Bio-Molecular Drug Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
        
        pair_input_dim = embedding_dim * 4
        
        # 2. Level 1: Organ System Class (SOC) Multi-Label Head
        self.organ_head = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_socs)
        )
        
        # 3. Level 2: Decagon Multi-Relational Tensor Factorization Head
        self.num_specific = num_specific
        self.rel_matrices = nn.Parameter(torch.Tensor(num_specific, embedding_dim))
        nn.init.xavier_uniform_(self.rel_matrices)
        
        self.specific_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_specific)
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_a, x_b, return_logits=True):
        # Encode drug pair
        h_a = self.encoder(x_a) # (N, D)
        h_b = self.encoder(x_b) # (N, D)
        
        sum_f = h_a + h_b
        diff_f = torch.abs(h_a - h_b)
        prod_f = h_a * h_b
        pair_feat = torch.cat([sum_f, diff_f, prod_f, (h_a + h_b)/2.0], dim=1) # (N, 4D)
        
        # Level 1: Organ System Predictions
        organ_logits = self.organ_head(pair_feat) # (N, num_socs)
        
        # Level 2: Specific Side Effect Predictions
        h_a_exp = h_a.unsqueeze(1)
        h_b_exp = h_b.unsqueeze(1)
        m_exp = self.rel_matrices.unsqueeze(0)
        tensor_logits = torch.sum(h_a_exp * m_exp * h_b_exp, dim=2)
        
        mlp_logits = self.specific_mlp(pair_feat)
        w = torch.sigmoid(self.alpha)
        specific_logits = w * tensor_logits + (1.0 - w) * mlp_logits
        
        if return_logits:
            return organ_logits, specific_logits
        return torch.sigmoid(organ_logits), torch.sigmoid(specific_logits)
