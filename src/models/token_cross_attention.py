import torch
import torch.nn as nn

class TokenCrossAttention(nn.Module):
    """
    Applies cross-attention between token-level representations of Drug A and Drug B.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Cross attention layers
        self.attn_a_to_b = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_b_to_a = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Layer norms
        self.norm1_a = nn.LayerNorm(embed_dim)
        self.norm1_b = nn.LayerNorm(embed_dim)
        
        # Feed forwards
        self.ff_a = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ff_b = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.norm2_a = nn.LayerNorm(embed_dim)
        self.norm2_b = nn.LayerNorm(embed_dim)
        
    def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor, mask_a=None, mask_b=None):
        """
        tokens_a, tokens_b: [batch, seq_len, embed_dim]
        mask_a, mask_b: [batch, seq_len] boolean padding masks (True for padding)
        """
        # A attends to B
        attended_a, _ = self.attn_a_to_b(tokens_a, tokens_b, tokens_b, key_padding_mask=mask_b)
        out_a = self.norm1_a(tokens_a + attended_a)
        
        # B attends to A
        attended_b, _ = self.attn_b_to_a(tokens_b, tokens_a, tokens_a, key_padding_mask=mask_a)
        out_b = self.norm1_b(tokens_b + attended_b)
        
        # FF
        out_a = self.norm2_a(out_a + self.ff_a(out_a))
        out_b = self.norm2_b(out_b + self.ff_b(out_b))
        
        # Pool (e.g. mean pooling over valid tokens)
        if mask_a is not None:
            mask_a_float = (~mask_a).float().unsqueeze(-1)
            pooled_a = (out_a * mask_a_float).sum(dim=1) / mask_a_float.sum(dim=1).clamp(min=1e-9)
        else:
            pooled_a = out_a.mean(dim=1)
            
        if mask_b is not None:
            mask_b_float = (~mask_b).float().unsqueeze(-1)
            pooled_b = (out_b * mask_b_float).sum(dim=1) / mask_b_float.sum(dim=1).clamp(min=1e-9)
        else:
            pooled_b = out_b.mean(dim=1)
            
        return pooled_a, pooled_b
