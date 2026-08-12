import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Polypharmacy Side-Effect Ranking.
    Prevents negative-gradient dominance while maintaining high recall on rare side effects.
    """
    def __init__(self, gamma_neg=2.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        # x: logits (N, C), y: binary targets (N, C)
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric focusing
        if self.gamma_pos > 0:
            los_pos *= ((1.0 - xs_pos) ** self.gamma_pos)
        if self.gamma_neg > 0:
            los_neg *= ((x_sigmoid) ** self.gamma_neg)

        loss = - (los_pos + los_neg)
        return loss.mean()

class AdvancedGatedPolypharmacyGNN(nn.Module):
    """
    State-of-the-Art Hierarchical Gated Polypharmacy GNN:
    - Level 1: 15 MedDRA Organ Systems
    - Level 2: 100 Specific Side Effects with Top-Down Organ Gating & SIDER Mono-Drug Prior Injection
    """
    def __init__(self, in_features, hidden_dim=256, embedding_dim=128, num_socs=15, num_specific=100, soc_indices_map=None, dropout=0.15):
        super(AdvancedGatedPolypharmacyGNN, self).__init__()
        
        self.num_socs = num_socs
        self.num_specific = num_specific
        self.embedding_dim = embedding_dim
        
        # 1. Deep Bio-Molecular Feature Encoder
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
        
        # 2. Level 1: Organ System Risk Head
        self.organ_head = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_socs)
        )
        
        # 3. Level 2: Specific Side Effects Tensor & MLP Head
        self.rel_matrices = nn.Parameter(torch.Tensor(num_specific, embedding_dim))
        nn.init.xavier_uniform_(self.rel_matrices)
        
        self.specific_mlp = nn.Sequential(
            nn.Linear(pair_input_dim + num_specific, hidden_dim), # Injects SIDER Mono-Drug Priors
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_specific)
        )
        
        # Mapping from specific side effect index -> parent organ system index
        if soc_indices_map is not None:
            self.register_buffer('spec_to_soc_idx', torch.tensor(soc_indices_map, dtype=torch.long))
        else:
            self.register_buffer('spec_to_soc_idx', torch.zeros(num_specific, dtype=torch.long))
            
        # Learnable gating weights
        self.gate_weight = nn.Parameter(torch.tensor(0.6))
        self.prior_weight = nn.Parameter(torch.tensor(0.4))
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_a, x_b, sider_priors=None, return_logits=True):
        # 1. Encode drugs
        h_a = self.encoder(x_a) # (N, D)
        h_b = self.encoder(x_b) # (N, D)
        
        sum_f = h_a + h_b
        diff_f = torch.abs(h_a - h_b)
        prod_f = h_a * h_b
        pair_feat = torch.cat([sum_f, diff_f, prod_f, (h_a + h_b)/2.0], dim=1) # (N, 4D)
        
        # Level 1 Organ Predictions
        organ_logits = self.organ_head(pair_feat) # (N, num_socs)
        organ_probs = torch.sigmoid(organ_logits)
        
        # Level 2 Specific Predictions
        if sider_priors is None:
            sider_priors = torch.zeros(x_a.size(0), self.num_specific, device=x_a.device)
            
        pair_with_priors = torch.cat([pair_feat, sider_priors], dim=1)
        
        h_a_exp = h_a.unsqueeze(1)
        h_b_exp = h_b.unsqueeze(1)
        m_exp = self.rel_matrices.unsqueeze(0)
        tensor_logits = torch.sum(h_a_exp * m_exp * h_b_exp, dim=2)
        
        mlp_logits = self.specific_mlp(pair_with_priors)
        w = torch.sigmoid(self.alpha)
        raw_specific_logits = w * tensor_logits + (1.0 - w) * mlp_logits
        
        # 4. Top-Down Hierarchical Organ Gating
        # Extract parent organ probability for each specific side effect
        parent_organ_probs = organ_probs[:, self.spec_to_soc_idx] # (N, num_specific)
        g_w = torch.sigmoid(self.gate_weight)
        
        # Gated Specific Probability
        spec_probs_base = torch.sigmoid(raw_specific_logits)
        gated_probs = (parent_organ_probs ** g_w) * (spec_probs_base ** (1.0 - g_w))
        
        # Prior boosting
        p_w = torch.sigmoid(self.prior_weight)
        final_spec_probs = (1.0 - p_w * 0.3) * gated_probs + (p_w * 0.3) * (sider_priors * parent_organ_probs)
        final_spec_probs = final_spec_probs.clamp(1e-7, 1.0 - 1e-7)
        
        if return_logits:
            final_spec_logits = torch.log(final_spec_probs / (1.0 - final_spec_probs))
            return organ_logits, final_spec_logits
        return organ_probs, final_spec_probs
