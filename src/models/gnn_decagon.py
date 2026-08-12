import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridDualHeadDecagonGNN(nn.Module):
    """
    State-of-the-Art Hybrid Dual-Head Polypharmacy GNN Architecture.
    Combines:
    1. Multi-Relational Tensor Factorization Head (Decagon h_A^T * M_r * h_B)
    2. Symmetric Bilinear Interaction MLP Head (Pair-level cross interaction)
    """
    def __init__(self, in_features, hidden_dim=256, embedding_dim=128, num_relations=100, dropout=0.2):
        super(HybridDualHeadDecagonGNN, self).__init__()
        
        # 1. Deep Feature Encoder with Residual Connections & LayerNorm
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
        
        # 2. Decagon Multi-Relational Tensor Head
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.rel_matrices = nn.Parameter(torch.Tensor(num_relations, embedding_dim))
        nn.init.xavier_uniform_(self.rel_matrices)
        
        # 3. Bilinear Symmetric Pair MLP Head
        pair_input_dim = embedding_dim * 4 # [h_a, h_b, |h_a - h_b|, h_a * h_b]
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relations)
        )
        
        # Learnable balancing parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_a, x_b, return_logits=True):
        # Encode drug features
        h_a = self.encoder(x_a) # (N, D)
        h_b = self.encoder(x_b) # (N, D)
        
        # Head 1: Tensor Factorization
        h_a_exp = h_a.unsqueeze(1) # (N, 1, D)
        h_b_exp = h_b.unsqueeze(1) # (N, 1, D)
        m_exp = self.rel_matrices.unsqueeze(0) # (1, R, D)
        tensor_logits = torch.sum(h_a_exp * m_exp * h_b_exp, dim=2) # (N, R)
        
        # Head 2: Symmetric Bilinear Pair MLP
        sum_f = h_a + h_b
        diff_f = torch.abs(h_a - h_b)
        prod_f = h_a * h_b
        pair_feat = torch.cat([sum_f, diff_f, prod_f, (h_a + h_b)/2.0], dim=1) # (N, 4D)
        mlp_logits = self.pair_mlp(pair_feat) # (N, R)
        
        # Combined dual-head logits
        w = torch.sigmoid(self.alpha)
        combined_logits = w * tensor_logits + (1.0 - w) * mlp_logits
        
        if return_logits:
            return combined_logits
        return torch.sigmoid(combined_logits)
