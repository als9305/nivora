"""
Focal Loss and Label Smoothing Loss Functions for DeepQShield.
Implements binary and multi-class Focal Loss with label smoothing based on research paper.
Paper Parameters: Focal Loss alpha = 2.0, gamma = 2.0, label smoothing epsilon = 0.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing for DeepQShield.
    
    Paper Specification:
        alpha = 2.0
        gamma = 2.0
        label_smoothing = 0.1
        
    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 2.0,
        label_smoothing: float = 0.1,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            logits: Predicted class logits (B, C) or (B,)
            targets: Target ground truth labels (B,) with class indices (0 or 1)
            
        Returns:
            Computed scalar loss tensor
        """
        num_classes = logits.size(1) if logits.dim() > 1 else 2
        
        if logits.dim() == 1 or logits.size(1) == 1:
            logits = torch.cat([-logits.unsqueeze(1), logits.unsqueeze(1)], dim=1)
            
        # Apply label smoothing to targets
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        # Focal weight term: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma
        
        # Class weighting alpha (alpha = 2.0 as per paper)
        alpha_factor = torch.full_like(logits, self.alpha)
        
        loss = -alpha_factor * focal_weight * smooth_targets * log_probs
        loss = torch.sum(loss, dim=1)
        
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss
