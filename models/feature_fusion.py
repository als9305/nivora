"""
Feature Fusion Module for DeepQShield.
Combines backbone ResNeXt features (2048-dim) with 2-stage LWE lattice features (256-dim)
into a 2304-dim concatenated representation and projects to 2048-dim fused features.
"""

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    Feature Fusion Layer matching DeepQShield paper.
    Input: Backbone features (B, 2048) and LWE features (B, 256).
    Output: Fused feature representation (B, 2048).
    """
    def __init__(self, backbone_dim: int = 2048, lwe_dim: int = 256, fused_dim: int = 2048):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.lwe_dim = lwe_dim
        self.fused_dim = fused_dim
        
        # Projection mapping concatenated (2048 + 256 = 2304) -> fused_dim (2048)
        concat_dim = backbone_dim + lwe_dim
        self.fusion_linear = nn.Linear(concat_dim, fused_dim)
        self.bn = nn.BatchNorm1d(fused_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual gating network
        self.gate = nn.Sequential(
            nn.Linear(concat_dim, fused_dim),
            nn.Sigmoid()
        )

    def forward(self, backbone_feats: torch.Tensor, lwe_feats: torch.Tensor) -> torch.Tensor:
        """
        Fuse backbone features (B, 2048) and LWE features (B, 256).
        
        Args:
            backbone_feats: Tensor of shape (B, 2048)
            lwe_feats: Tensor of shape (B, 256)
            
        Returns:
            Fused Tensor of shape (B, 2048)
        """
        if backbone_feats.size(1) != self.backbone_dim:
            raise ValueError(f"Expected backbone dim {self.backbone_dim}, got {backbone_feats.size(1)}")
        if lwe_feats.size(1) != self.lwe_dim:
            raise ValueError(f"Expected LWE dim {self.lwe_dim}, got {lwe_feats.size(1)}")
            
        # Concatenation: (B, 2304)
        concat_feats = torch.cat([backbone_feats, lwe_feats], dim=1)
        
        # Linear projection: (B, 2048)
        projected = self.fusion_linear(concat_feats)
        projected = self.bn(projected)
        projected = self.relu(projected)
        
        # Gated fusion
        gate_weight = self.gate(concat_feats)
        fused = gate_weight * projected + (1.0 - gate_weight) * backbone_feats
        
        return fused
