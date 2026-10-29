"""
Training Pipeline for DeepQShield.
Executes PyTorch training loop using AdamW, CosineAnnealingWarmRestarts (T0=10, eta_min=1e-6), Focal Loss (alpha=2, gamma=2), and checkpointing.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.deepqshield import DeepQShieldModel
from training.dataset import create_dataloader
from training.augmentations import get_train_transform, get_validation_transform
from training.losses import FocalLoss
from training.validate import validate_epoch
from training.checkpointing import save_checkpoint


def train_deepqshield(
    data_dir: str = "data",
    output_dir: str = "checkpoints",
    reports_dir: str = "reports",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Execute DeepQShield training and validation pipeline according to exact paper specs.
    
    Returns:
        Dict containing training history, validation metrics, and held-out test metrics.
    """
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Initialize Dataloaders for train, val, and test splits
    train_transform = get_train_transform()
    val_transform = get_validation_transform()
    
    train_loader = create_dataloader(data_dir, split="train", batch_size=batch_size, shuffle=True, transform=train_transform)
    val_loader = create_dataloader(data_dir, split="val", batch_size=batch_size, shuffle=False, transform=val_transform)
    test_loader = create_dataloader(data_dir, split="test", batch_size=batch_size, shuffle=False, transform=val_transform)
    
    print(f"Data Loaders Initialized: Train={len(train_loader.dataset)} frames, Val={len(val_loader.dataset)} frames, Test={len(test_loader.dataset)} frames")
    
    # 2. Instantiate Model with ImageNet Pretrained ResNeXt50_32x4d Backbone
    model = DeepQShieldModel(pretrained_backbone=True, lattice_dim=256, noise_std=0.1, seed=seed)
    model.to(device)
    
    # 3. Loss, Optimizer, Scheduler matching paper parameters
    criterion = FocalLoss(gamma=2.0, alpha=2.0, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    best_val_auc = -1.0
    best_metrics = {}
    history = []
    
    total_steps = len(train_loader) * epochs
    print(f"Starting DeepQShield Training: {epochs} Epochs ({total_steps} total steps), Batch Size={batch_size}, Device={device}...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device=device)
        
        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"]
        }
        history.append(epoch_info)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']*100:.1f}% | Val AUC: {val_metrics['roc_auc']:.4f}")
        
        # Save best model checkpoint based on val_roc_auc
        if val_metrics["roc_auc"] >= best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_metrics = val_metrics
            best_path = os.path.join(output_dir, "deepqshield_best.pth")
            save_checkpoint(model, optimizer, epoch, val_metrics, best_path)
            
    print(f"\nTraining Complete! Best Val ROC-AUC: {best_val_auc:.4f}")
    
    # 4. Final Evaluation on Held-Out Test Split using Best Checkpoint
    best_path = os.path.join(output_dir, "deepqshield_best.pth")
    if os.path.exists(best_path):
        model.load_checkpoint(best_path, device=device)
        
    test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device=device)
    print("\n================ Held-Out Test Set Evaluation ================")
    print(f"  Test Loss:       {test_loss:.4f}")
    print(f"  Test Accuracy:   {test_metrics['accuracy']*100:.2f}%")
    print(f"  Test Precision:  {test_metrics['precision']:.4f}")
    print(f"  Test Recall:     {test_metrics['recall']:.4f}")
    print(f"  Test F1-Score:   {test_metrics['f1']:.4f}")
    print(f"  Test ROC-AUC:    {test_metrics['roc_auc']:.4f}")
    print("==============================================================")
    
    # Save training history & test evaluation json
    with open(os.path.join(reports_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
        
    test_report = {
        "test_loss": round(test_loss, 4),
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "epochs_trained": epochs,
        "total_training_steps": total_steps,
        "checkpoint_path": os.path.abspath(best_path)
    }
    with open(os.path.join(reports_dir, "test_evaluation.json"), "w") as f:
        json.dump(test_report, f, indent=2)
        
    return {"history": history, "best_metrics": best_metrics, "test_metrics": test_metrics}


if __name__ == "__main__":
    train_deepqshield(epochs=50, batch_size=32)
