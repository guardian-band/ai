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
        # MultiheadAttention returns NaNs when every key in a row is masked.
        # Use a zero sentinel key for those rows; pooling still excludes every
        # padded query, and the result is finite and independent of pad values.
        safe_tokens_b, safe_mask_b = self._safe_keys(tokens_b, mask_b)
        safe_tokens_a, safe_mask_a = self._safe_keys(tokens_a, mask_a)

        # A attends to B
        attended_a, _ = self.attn_a_to_b(tokens_a, safe_tokens_b, safe_tokens_b, key_padding_mask=safe_mask_b)
        out_a = self.norm1_a(tokens_a + attended_a)
        
        # B attends to A
        attended_b, _ = self.attn_b_to_a(tokens_b, safe_tokens_a, safe_tokens_a, key_padding_mask=safe_mask_a)
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

    @staticmethod
    def _safe_keys(tokens: torch.Tensor, mask: torch.Tensor | None):
        if mask is None:
            return tokens, None
        if mask.ndim != 2 or mask.shape[:2] != tokens.shape[:2]:
            raise ValueError("padding mask must have shape [batch, sequence]")
        safe_mask = mask.clone()
        all_masked = safe_mask.all(dim=1)
        if all_masked.any():
            safe_tokens = tokens.clone()
            safe_tokens[all_masked] = 0.0
            safe_mask[all_masked] = False
            return safe_tokens, safe_mask
        return tokens, safe_mask
