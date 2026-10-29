"""
Validation Pipeline for DeepQShield.
Evaluates trained model against validation dataset and logs performance metrics.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.metrics import compute_metrics


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu"
) -> Tuple[float, Dict[str, float]]:
    """
    Run evaluation loop over validation data split.
    
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_fake_probs = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)
            
            probs = F.softmax(logits, dim=-1)
            fake_probs = probs[:, 1].cpu().tolist()
            
            all_targets.extend(targets.cpu().tolist())
            all_fake_probs.extend(fake_probs)
            
    avg_loss = total_loss / max(1, len(val_loader.dataset))
    metrics = compute_metrics(all_targets, all_fake_probs)
    return avg_loss, metrics
