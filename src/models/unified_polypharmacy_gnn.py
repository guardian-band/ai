import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedPolypharmacyGNN(nn.Module):
    """
    Unified Master Polypharmacy AI Model:
    Combines DrugBank, PrimeKG, ChEMBL, and BioSNAP knowledge into a single Hierarchical Multi-Task GNN.
    
    Level 1: 15 MedDRA System Organ Classes (SOCs) - Macro Organ System Toxicity Risk
    Level 2: 100 Specific Clinical Side Effects - Fine-Grained Decagon Tensor Factorization
    """
    def __init__(self, in_features, hidden_dim=256, embedding_dim=128, num_socs=15, num_specific=100, dropout=0.15):
        super(UnifiedPolypharmacyGNN, self).__init__()
        
        # 1. Deep Residual Bio-Molecular Drug Encoder
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        pair_input_dim = embedding_dim * 4 # [h_a, h_b, |h_a - h_b|, h_a * h_b]
        
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
        self.embedding_dim = embedding_dim
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

    def encode_drug(self, x):
        h1 = F.gelu(self.norm1(self.fc1(x)))
        h1 = self.dropout(h1)
        
        h2 = F.gelu(self.norm2(self.fc2(h1))) + h1 # Residual connection
        h2 = self.dropout(h2)
        
        out = F.gelu(self.norm3(self.fc3(h2)))
        return out

    def forward(self, x_a, x_b, return_logits=True):
        # 1. Encode both drugs to latent biological manifold
        h_a = self.encode_drug(x_a) # (N, D)
        h_b = self.encode_drug(x_b) # (N, D)
        
        # Build symmetric cross-interaction features
        sum_f = h_a + h_b
        diff_f = torch.abs(h_a - h_b)
        prod_f = h_a * h_b
        pair_feat = torch.cat([sum_f, diff_f, prod_f, (h_a + h_b)/2.0], dim=1) # (N, 4D)
        
        # 2. Level 1: Organ System Predictions (15 MedDRA SOCs)
        organ_logits = self.organ_head(pair_feat) # (N, num_socs)
        
        # 3. Level 2: Fine-Grained Side Effect Predictions (Top 100 Specific Conditions)
        h_a_exp = h_a.unsqueeze(1) # (N, 1, D)
        h_b_exp = h_b.unsqueeze(1) # (N, 1, D)
        m_exp = self.rel_matrices.unsqueeze(0) # (1, R, D)
        tensor_logits = torch.sum(h_a_exp * m_exp * h_b_exp, dim=2) # (N, R)
        
        mlp_logits = self.specific_mlp(pair_feat)
        w = torch.sigmoid(self.alpha)
        specific_logits = w * tensor_logits + (1.0 - w) * mlp_logits
        
        if return_logits:
            return organ_logits, specific_logits
        return torch.sigmoid(organ_logits), torch.sigmoid(specific_logits)
