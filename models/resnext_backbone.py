"""
ResNeXt50_32x4d Backbone Feature Extractor module for DeepQShield.
Implements the feature extraction backbone based on torchvision resnext50_32x4d.
"""

import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from typing import Optional, Dict, Any


class ResNeXtBackbone(nn.Module):
    """
    ResNeXt50_32x4d feature extraction backbone.
    Outputs a 2048-dimensional spatial feature map or pooled representation.
    """
    def __init__(self, pretrained: bool = True, freeze_early_layers: bool = False):
        super().__init__()
        weights = ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        base_model = resnext50_32x4d(weights=weights)
        
        # Extract features up to layer4 and global average pooling
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        self.avgpool = base_model.avgpool
        self.feature_dim = 2048
        
        if freeze_early_layers:
            for param in [self.conv1, self.bn1, self.layer1, self.layer2]:
                for p in param.parameters():
                    p.requires_grad = False

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 2048-dim feature representation from 224x224 input image tensors.
        
        Args:
            x: Tensor of shape (B, 3, 224, 224)
            
        Returns:
            Tensor of shape (B, 2048)
        """
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected input shape (B, 3, H, W), got {x.shape}")
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if x.size(1) != self.feature_dim:
            raise ValueError(f"Expected output feature dim {self.feature_dim}, got {x.size(1)}")
            
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)
