"""
DeepQShield Main Neural Network Architecture.
Combines ResNeXt50_32x4d + 2-Stage LWE Layer (2x256-dim, std=0.1) + Feature Fusion (2304->2048) + Attention + Classifier Head.
Authoritative implementation of the DeepQShield research paper methodology.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from models.resnext_backbone import ResNeXtBackbone
from models.lwe_layer import LWELayer
from models.feature_fusion import FeatureFusion
from models.attention import FeatureAttention
from models.classifier import ClassifierHead


class DeepQShieldModel(nn.Module):
    """
    DeepQShield Model Container.
    
    Architecture Flow:
        Input (B, 3, 224, 224)
            │
        ResNeXt50_32x4d Backbone (2048-dim)
            ├───────────────┐
            │               │
            ▼               ▼
        ResNeXt Feats    TWO Lattice Layers (2048 -> 256 -> 256, std=0.1)
        (B, 2048)        (B, 256)
            │               │
            └───────┬───────┘
                    ▼
              Feature Fusion (Concat 2304 -> Projected 2048)
                    │
                    ▼
               Attention (2048-dim)
                    │
                    ▼
            Classifier Head (Dense -> ReLU -> Dropout=0.3 -> Output)
                    │
                    ▼
              Real / Fake Logits
    """
    def __init__(
        self,
        pretrained_backbone: bool = True,
        lattice_dim: int = 256,
        noise_std: float = 0.1,
        dropout_rate: float = 0.3,
        num_classes: int = 2,
        seed: Optional[int] = 42
    ):
        super().__init__()
        self.model_name = "DeepQShield"
        self.version = "1.0.0"
        
        # 1. ResNeXt50_32x4d Backbone
        self.backbone = ResNeXtBackbone(pretrained=pretrained_backbone)
        
        # 2. TWO-Stage LWE / Lattice Feature Enhancement Layer (2x256-dim, std=0.1)
        self.lwe = LWELayer(
            in_features=2048,
            lattice_dim=lattice_dim,
            noise_std=noise_std,
            seed=seed
        )
        
        # 3. Feature Fusion Layer (2048 + 256 -> 2048)
        self.fusion = FeatureFusion(
            backbone_dim=2048,
            lwe_dim=lattice_dim,
            fused_dim=2048
        )
        
        # 4. Attention Mechanism Layer
        self.attention = FeatureAttention(feature_dim=2048, reduction_ratio=16)
        
        # 5. Classifier Head
        self.classifier = ClassifierHead(
            in_features=2048,
            hidden_dim=512,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )

    def extract_all_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract internal features from every stage of the pipeline.
        
        Returns:
            Tuple of (backbone_feats, lwe_feats, fused_feats, attended_feats)
        """
        backbone_feats = self.backbone(x)
        lwe_feats = self.lwe(backbone_feats)
        fused_feats = self.fusion(backbone_feats, lwe_feats)
        attended_feats, attn_weights = self.attention(fused_feats)
        return backbone_feats, lwe_feats, fused_feats, attended_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass yielding classification logits.
        
        Args:
            x: Input image tensor (B, 3, 224, 224)
            
        Returns:
            Logits tensor of shape (B, 2)
        """
        _, _, _, attended_feats = self.extract_all_features(x)
        logits = self.classifier(attended_feats)
        return logits

    def predict_single(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Run inference on a single preprocessed image tensor (1, 3, 224, 224).
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 3:
                x = x.unsqueeze(0)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            
            real_prob = float(probs[0, 0].item())
            fake_prob = float(probs[0, 1].item())
            
            pred_class = "DEEPFAKE" if fake_prob > 0.50 else "REAL"
            confidence = float(abs(fake_prob - 0.50) * 2.0 * 0.4 + 0.60)
            
            return {
                "model_name": self.model_name,
                "version": self.version,
                "real_probability": round(real_prob, 4),
                "fake_probability": round(fake_prob, 4),
                "predicted_class": pred_class,
                "confidence": round(confidence, 4)
            }

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0, metrics: Optional[Dict[str, float]] = None):
        """Save model checkpoint safely."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_name": self.model_name,
            "version": self.version,
            "state_dict": self.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {}
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, device: str = "cpu"):
        """Load model checkpoint safely."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
        if state_dict is None:
            raise KeyError("Checkpoint does not contain 'state_dict' or 'model_state_dict'")
        self.load_state_dict(state_dict)
        self.eval()
        return checkpoint
