"""
Classification Head Module for DeepQShield.
Implements Dense -> ReLU -> Dropout (0.3) -> Binary Classifier (Real / Fake).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class ClassifierHead(nn.Module):
    """
    DeepQShield Classifier Head.
    Input: Fused and attended feature vector (2048-dim).
    Output: Logits or binary classification probability.
    """
    def __init__(self, in_features: int = 2048, hidden_dim: int = 512, dropout_rate: float = 0.3, num_classes: int = 2):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute classification logits.
        
        Args:
            x: Feature tensor of shape (B, 2048)
            
        Returns:
            Logits tensor of shape (B, 2)
        """
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"Expected shape (B, {self.in_features}), got {x.shape}")
            
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Inference mode structured prediction output.
        
        Args:
            x: Feature tensor of shape (1, 2048) or (B, 2048)
            
        Returns:
            Dict containing real_probability, fake_probability, predicted_class, confidence
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            
            # Index 0: REAL, Index 1: DEEPFAKE
            real_prob = float(probs[0, 0].item())
            fake_prob = float(probs[0, 1].item())
            
            pred_class = "fake" if fake_prob > 0.50 else "real"
            confidence = float(abs(fake_prob - 0.50) * 2.0)
            
            return {
                "real_probability": round(real_prob, 4),
                "fake_probability": round(fake_prob, 4),
                "predicted_class": pred_class,
                "confidence": round(confidence, 4)
            }
