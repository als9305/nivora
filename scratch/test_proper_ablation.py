import os
import sys
import time
import csv
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, "d:/projects/DeepGuard")

from models.deepqshield import DeepQShieldModel
from training.dataset import create_dataloader
from training.augmentations import get_validation_transform
from training.metrics import compute_metrics


class AblatedModelWrapper(nn.Module):
    def __init__(self, full_model: DeepQShieldModel, mode: str):
        super().__init__()
        self.model = full_model
        self.mode = mode # 'full', 'no_attention', 'no_fusion_attention', 'baseline_only'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backbone_feats = self.model.backbone(x) # (B, 2048)
        
        if self.mode == "full":
            # D. Full DeepQShield (Backbone + LWE + Fusion + Attention + Classifier)
            lwe_feats = self.model.lwe(backbone_feats) # (B, 256)
            fused_feats = self.model.fusion(backbone_feats, lwe_feats) # (B, 2048)
            attended_feats, _ = self.model.attention(fused_feats) # (B, 2048)
            return self.model.classifier(attended_feats)
            
        elif self.mode == "no_attention":
            # C. ResNeXt + LWE + Fusion (Bypassing Attention)
            lwe_feats = self.model.lwe(backbone_feats) # (B, 256)
            fused_feats = self.model.fusion(backbone_feats, lwe_feats) # (B, 2048)
            return self.model.classifier(fused_feats)
            
        elif self.mode == "no_fusion_attention":
            # B. ResNeXt + LWE (Bypassing Fusion & Attention)
            lwe_feats = self.model.lwe(backbone_feats) # (B, 256)
            # Create a zero-LWE fusion pass (pure LWE perturbation on backbone)
            fused_feats = self.model.fusion(torch.zeros_like(backbone_feats), lwe_feats) # (B, 2048)
            return self.model.classifier(fused_feats)
            
        elif self.mode == "baseline_only":
            # A. Baseline ResNeXt50 (Bypassing LWE, Fusion & Attention)
            # Pass raw backbone features directly to classifier
            return self.model.classifier(backbone_feats)
            
        else:
            raise ValueError(f"Unknown ablation mode: {self.mode}")


def test_ablation():
    checkpoint_path = "checkpoints/deepqshield_best.pth"
    full_model = DeepQShieldModel(pretrained_backbone=False, seed=42)
    full_model.load_checkpoint(checkpoint_path, device="cpu")
    full_model.eval()
    
    val_transform = get_validation_transform()
    loader = create_dataloader("data", split="test", batch_size=32, shuffle=False, transform=val_transform)
    
    configs = [
        ("A. Baseline ResNeXt50 (Bypass LWE+Fusion+Attention)", AblatedModelWrapper(full_model, "baseline_only")),
        ("B. ResNeXt + LWE (Bypass Fusion+Attention)", AblatedModelWrapper(full_model, "no_fusion_attention")),
        ("C. ResNeXt + LWE + Fusion (Bypass Attention)", AblatedModelWrapper(full_model, "no_attention")),
        ("D. Full DeepQShield (Backbone+LWE+Fusion+Attention)", AblatedModelWrapper(full_model, "full"))
    ]
    
    print("=================================================================")
    print(" RE-RUNNING PROPER ABLATION STUDY USING TRAINED CHECKPOINT WEIGHTS")
    print("=================================================================\n")
    
    for name, wrapper in configs:
        wrapper.eval()
        all_targets = []
        all_probs = []
        latencies = []
        
        for images, targets in loader:
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = wrapper(images)
                probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
            t1 = time.perf_counter()
            
            latencies.append((t1 - t0) * 1000.0 / images.size(0))
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)
            
        m = compute_metrics(all_targets, all_probs)
        avg_lat = round(float(np.mean(latencies)), 2)
        print(f"[{name}]")
        print(f"    Acc: {m['accuracy']*100:.2f}% | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1']:.4f} | AUC: {m['roc_auc']:.4f} | Latency: {avg_lat} ms/frame\n")

if __name__ == "__main__":
    test_ablation()
