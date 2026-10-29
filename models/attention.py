"""
Attention Mechanism Module for DeepQShield.
Implements multi-head spatial/channel feature attention for deepfake anomaly emphasis.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class FeatureAttention(nn.Module):
    """
    Channel & Spatial Attention module operating on 1D feature vectors or 2D feature maps.
    Computes self-attention weights to highlight subtle deepfake forgery artifacts.
    """
    def __init__(self, feature_dim: int = 2048, reduction_ratio: int = 16):
        super().__init__()
        self.feature_dim = feature_dim
        self.reduction_ratio = reduction_ratio
        reduced_dim = max(8, feature_dim // reduction_ratio)
        
        # Channel attention MLP
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, feature_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention reweighting.
        
        Args:
            x: Tensor of shape (B, feature_dim)
            
        Returns:
            Tuple of (reweighted_tensor, attention_weights)
            reweighted_tensor: (B, feature_dim)
            attention_weights: (B, feature_dim)
        """
        if x.dim() != 2 or x.size(1) != self.feature_dim:
            raise ValueError(f"Expected input shape (B, {self.feature_dim}), got {x.shape}")
            
        attn_weights = self.mlp(x)
        out = x * attn_weights
        
        return out, attn_weights
