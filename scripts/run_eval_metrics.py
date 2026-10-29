import os
import sys
import json
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.deepqshield import DeepQShieldModel
from training.dataset import create_dataloader
from training.augmentations import get_validation_transform
from training.losses import FocalLoss

def evaluate_test_set():
    data_dir = "data"
    reports_dir = "reports"
    checkpoint_path = "checkpoints/deepqshield_best.pth"
    os.makedirs(reports_dir, exist_ok=True)
    
    val_transform = get_validation_transform()
    test_loader = create_dataloader(data_dir, split="test", batch_size=32, shuffle=False, transform=val_transform)
    
    model = DeepQShieldModel(pretrained_backbone=False, seed=42)
    if os.path.exists(checkpoint_path):
        model.load_checkpoint(checkpoint_path, device="cpu")
    model.to("cpu")
    model.eval()
    
    criterion = FocalLoss(gamma=2.0, alpha=2.0, label_smoothing=0.1)
    
    all_targets = []
    all_probs = []
    all_preds = []
    latencies = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            t0 = time.time()
            logits = model(images)
            t1 = time.time()
            latencies.append((t1 - t0) / images.size(0))
            
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs >= 0.5).long()
            
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds).tolist()
    
    real_count = int(np.sum(all_targets == 0))
    fake_count = int(np.sum(all_targets == 1))
    avg_latency_ms = float(np.mean(latencies)) * 1000.0
    
    test_report = {
        "test_sample_count": len(all_targets),
        "real_sample_count": real_count,
        "fake_sample_count": fake_count,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm,
        "avg_inference_latency_ms": round(avg_latency_ms, 2),
        "checkpoint_path": os.path.abspath(checkpoint_path)
    }
    
    out_path = os.path.join(reports_dir, "test_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(test_report, f, indent=2)
        
    print("Test Evaluation Saved Successfully!")
    print(json.dumps(test_report, indent=2))
    return test_report

if __name__ == "__main__":
    evaluate_test_set()
