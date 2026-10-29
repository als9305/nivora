"""
Learning With Errors (LWE) / Lattice-Based Feature Enhancement Layer.
Refactored to strictly match the DeepQShield paper specification:
- TWO sequential lattice projection layers
- Dimension 256 each
- Controlled LWE lattice noise std = 0.1
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class LWELayer(nn.Module):
    """
    Two-Stage LWE (Learning With Errors) / Lattice-Based Feature Enhancement Module.
    
    Paper Specification:
        - Layer 1: 2048 -> 256 lattice projection with LWE noise std = 0.1
        - Layer 2: 256 -> 256 lattice projection with LWE noise std = 0.1
        
    Mathematical Form for Stage k:
        Y_k = LayerNorm(SiLU(X_k @ A_k + e_k + X_k @ S_k))
        where e_k ~ N(0, 0.1^2)
    """
    def __init__(
        self,
        in_features: int = 2048,
        lattice_dim: int = 256,
        noise_std: float = 0.1,
        seed: Optional[int] = 42
    ):
        super().__init__()
        self.in_features = in_features
        self.lattice_dim = lattice_dim
        self.noise_std = noise_std
        
        # Generator for deterministic lattice basis initialization
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        
        # --- Stage 1 Lattice Layer (2048 -> 256) ---
        a1 = torch.empty(in_features, lattice_dim)
        if gen is not None:
            nn.init.orthogonal_(a1, generator=gen)
        else:
            nn.init.orthogonal_(a1)
        self.register_buffer("lattice_matrix1", a1)
        self.perturbation1 = nn.Parameter(torch.zeros(in_features, lattice_dim))
        nn.init.kaiming_uniform_(self.perturbation1, a=math.sqrt(5))
        self.norm1 = nn.LayerNorm(lattice_dim)
        
        # --- Stage 2 Lattice Layer (256 -> 256) ---
        a2 = torch.empty(lattice_dim, lattice_dim)
        if gen is not None:
            nn.init.orthogonal_(a2, generator=gen)
        else:
            nn.init.orthogonal_(a2)
        self.register_buffer("lattice_matrix2", a2)
        self.perturbation2 = nn.Parameter(torch.zeros(lattice_dim, lattice_dim))
        nn.init.kaiming_uniform_(self.perturbation2, a=math.sqrt(5))
        self.norm2 = nn.LayerNorm(lattice_dim)
        
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhance backbone features (B, 2048) through two sequential 256-dim lattice layers.
        
        Args:
            x: Input backbone feature tensor (B, 2048)
            
        Returns:
            Lattice-enhanced feature tensor (B, 256)
        """
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"Expected input shape (B, {self.in_features}), got {x.shape}")
            
        # --- Stage 1 (2048 -> 256) ---
        proj1 = torch.matmul(x, self.lattice_matrix1) + 0.1 * torch.matmul(x, self.perturbation1)
        if self.training and self.noise_std > 0:
            noise1 = torch.randn_like(proj1) * self.noise_std
            proj1 = proj1 + noise1
        h1 = self.act(self.norm1(proj1))
        
        # --- Stage 2 (256 -> 256) ---
        proj2 = torch.matmul(h1, self.lattice_matrix2) + 0.1 * torch.matmul(h1, self.perturbation2)
        if self.training and self.noise_std > 0:
            noise2 = torch.randn_like(proj2) * self.noise_std
            proj2 = proj2 + noise2
        out = self.act(self.norm2(proj2))
        
        return out
