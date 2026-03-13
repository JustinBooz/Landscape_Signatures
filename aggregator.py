"""
Aggregator Head for Landscape Signature Extraction.

Architecture:
    GeM Pooling -> MLP Projector -> L2 Normalization

The final output is an L2-normalised 1D Landscape Signature vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM)
    Operates over the length dimension N of dense patch tokens [B, N, D].
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # x shape: [B, N, D]
        # We pool over the spatial/sequence dimension N (dim=1)
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = x.mean(dim=1)
        x = x.pow(1.0 / self.p)
        return x

class AggregatorHead(nn.Module):
    """
    Full aggregation stack applied on top of the dense patch tokens from the encoder.

    Args:
        embed_dim:   Input dimensionality (DINOv3 output patch dim, e.g. 1280 for ViT-H)
        out_dim:     Final signature / projection dimension (e.g. 1280)
    """
    def __init__(
        self,
        embed_dim: int = 1280,
        out_dim: int = 1280,
    ):
        super().__init__()
        # Robust pooling over dense tokens
        self.pool = GeMPooling(p=3.0)
        
        # 2-layer MLP projector
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim), # Batch-independent LayerNorm
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim, bias=False)
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, N, D] — Dense spatial tokens from the ViT encoder
        Returns:
            L2-normalised vector [B, out_dim]
        """
        assert patch_tokens.dim() == 3, f"Expected patch_tokens to be 3D [B, N, D], got {patch_tokens.dim()}D"
        
        # 1. Pool over spatial dimension
        x = self.pool(patch_tokens)             # [B, embed_dim]
        
        # 2. Linear projection
        x = self.projector(x)                   # [B, out_dim]
            
        # 3. L2 Normalize for InfoNCE
        return F.normalize(x, dim=-1, p=2)      # [B, out_dim]
