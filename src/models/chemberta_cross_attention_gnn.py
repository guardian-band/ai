import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttentionBlock(nn.Module):
    """
    Bidirectional Multi-Head Cross-Attention between Drug A and Drug B.
    Captures fine-grained atom-group interactions and cross-molecular affinities.
    """
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super(MultiHeadCrossAttentionBlock, self).__init__()
        self.attn_a_to_b = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_b_to_a = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, h_a, h_b):
        # h_a, h_b: (B, D) -> unsqueeze to (B, 1, D) for sequence length 1
        q_a = h_a.unsqueeze(1)
        q_b = h_b.unsqueeze(1)
        
        # Drug A attends to Drug B
        attn_out_a, _ = self.attn_a_to_b(query=q_a, key=q_b, value=q_b)
        h_a_cross = self.norm1(h_a + attn_out_a.squeeze(1))
        
        # Drug B attends to Drug A
        attn_out_b, _ = self.attn_b_to_a(query=q_b, key=q_a, value=q_a)
        h_b_cross = self.norm2(h_b + attn_out_b.squeeze(1))
        
        return h_a_cross, h_b_cross

class ChemBERTaCrossAttentionGNN(nn.Module):
    """
    State-of-the-Art Polypharmacy AI Architecture combining:
    1. ChemBERTa Transformer Molecular Knowledge
    2. Multi-Head Cross-Attention Drug Pair Interactions
    3. Multi-Relational Decagon Tensor Factorization Head
    4. Top-Down Log-Odds Additive Organ & SIDER Prior Fusion
    """
    def __init__(self, in_features, hidden_dim=256, embedding_dim=128, num_socs=15, num_specific=100, soc_indices_map=None, dropout=0.15):
        super(ChemBERTaCrossAttentionGNN, self).__init__()
        
        self.num_socs = num_socs
        self.num_specific = num_specific
        self.embedding_dim = embedding_dim
        
        # 1. Deep Bio-Molecular Feature Projector
        self.projector = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
        
        # 2. Multi-Head Cross-Attention Layer
        self.cross_attn = MultiHeadCrossAttentionBlock(embed_dim=embedding_dim, num_heads=4, dropout=dropout)
        
        pair_input_dim = embedding_dim * 4
        
        # 3. Level 1: 15 MedDRA Organ Systems Head
        self.organ_head = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_socs)
        )
        
        # 4. Level 2: Specific Side Effects Tensor & MLP Head
        self.rel_matrices = nn.Parameter(torch.Tensor(num_specific, embedding_dim))
        nn.init.xavier_uniform_(self.rel_matrices)
        
        self.specific_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_specific)
        )
        
        # Mapping from specific side effect -> parent organ system
        if soc_indices_map is not None:
            self.register_buffer('spec_to_soc_idx', torch.tensor(soc_indices_map, dtype=torch.long))
        else:
            self.register_buffer('spec_to_soc_idx', torch.zeros(num_specific, dtype=torch.long))
            
        # Learnable Log-Odds Additive Fusion weights
        self.alpha_organ = nn.Parameter(torch.tensor(0.5))
        self.gamma_sider = nn.Parameter(torch.tensor(0.5))
        self.w_tensor = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_a, x_b, sider_priors=None, return_logits=True):
        # 1. Project to latent bio-molecular manifold
        h_a = self.projector(x_a) # (B, D)
        h_b = self.projector(x_b) # (B, D)
        
        # 2. Cross-Attention
        h_a_cross, h_b_cross = self.cross_attn(h_a, h_b)
        
        # Symmetric Cross-Pair Feature Vector
        sum_f = h_a_cross + h_b_cross
        diff_f = torch.abs(h_a_cross - h_b_cross)
        prod_f = h_a_cross * h_b_cross
        pair_feat = torch.cat([sum_f, diff_f, prod_f, (h_a_cross + h_b_cross)/2.0], dim=1) # (B, 4D)
        
        # 3. Level 1: Organ System Logits
        organ_logits = self.organ_head(pair_feat) # (B, 15)
        
        # 4. Level 2: Base Tensor Factorization & MLP Logits
        h_a_exp = h_a_cross.unsqueeze(1)
        h_b_exp = h_b_cross.unsqueeze(1)
        m_exp = self.rel_matrices.unsqueeze(0)
        tensor_logits = torch.sum(h_a_exp * m_exp * h_b_exp, dim=2) # (B, 100)
        
        mlp_logits = self.specific_mlp(pair_feat) # (B, 100)
        w = torch.sigmoid(self.w_tensor)
        base_specific_logits = w * tensor_logits + (1.0 - w) * mlp_logits
        
        # 5. Log-Odds Additive Fusion (Organ Logits + SIDER Priors)
        parent_organ_logits = organ_logits[:, self.spec_to_soc_idx] # (B, 100)
        alpha = torch.sigmoid(self.alpha_organ) * 0.8
        
        if sider_priors is not None:
            gamma = torch.sigmoid(self.gamma_sider) * 0.8
            final_specific_logits = base_specific_logits + alpha * parent_organ_logits + gamma * sider_priors
        else:
            final_specific_logits = base_specific_logits + alpha * parent_organ_logits
            
        if return_logits:
            return organ_logits, final_specific_logits
        return torch.sigmoid(organ_logits), torch.sigmoid(final_specific_logits)
