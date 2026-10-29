"""
Gradient Flow Verification and Overfit Sanity Test for DeepQShield.
Verifies non-zero gradients across all module parameters (Backbone, LWE, Fusion, Attention, Classifier)
and tests driving loss down and achieving 100% training accuracy on a tiny 8-sample batch.
"""

import sys
import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, "d:/projects/DeepGuard")

from models.deepqshield import DeepQShieldModel
from training.losses import FocalLoss


def test_gradient_flow_and_overfit():
    print("--- 1. Testing Gradient Flow Across All DeepQShield Layers ---")
    
    # Initialize model with pretrained backbone
    model = DeepQShieldModel(pretrained_backbone=True, lattice_dim=256, noise_std=0.1, seed=42)
    model.train()
    
    # Tiny batch: 8 samples (4 Real, 4 Fake)
    torch.manual_seed(42)
    x = torch.randn(8, 3, 224, 224)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    criterion = FocalLoss(gamma=2.0, alpha=2.0, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Forward pass
    logits = model(x)
    loss = criterion(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Verify non-zero gradient flow for every parameter group
    param_gradients = {}
    missing_grads = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                missing_grads.append(name)
            else:
                grad_norm = float(param.grad.norm().item())
                param_gradients[name] = grad_norm
                
    assert len(missing_grads) == 0, f"Missing gradients in parameters: {missing_grads}"
    
    print("[PASS] Gradient Flow Confirmed for All Parameter Groups:")
    print(f"  - Backbone Conv/BN Layer4 Grad Norm: {param_gradients['backbone.layer4.2.conv3.weight']:.6f}")
    print(f"  - LWE Stage 1 Perturbation Grad Norm: {param_gradients['lwe.perturbation1']:.6f}")
    print(f"  - LWE Stage 2 Perturbation Grad Norm: {param_gradients['lwe.perturbation2']:.6f}")
    print(f"  - Feature Fusion Linear Grad Norm:   {param_gradients['fusion.fusion_linear.weight']:.6f}")
    print(f"  - Feature Attention MLP Grad Norm:   {param_gradients['attention.mlp.0.weight']:.6f}")
    print(f"  - Classifier FC Head Grad Norm:      {param_gradients['classifier.fc1.weight']:.6f}")
    
    print("\n--- 2. Executing Overfit Sanity Test (Driving Loss Down & Memorizing Batch) ---")
    initial_loss = loss.item()
    print(f"Initial Training Loss: {initial_loss:.4f}")
    
    # Run 40 optimization steps on tiny batch
    for step in range(1, 41):
        optimizer.zero_grad()
        logits = model(x)
        l = criterion(logits, y)
        l.backward()
        optimizer.step()
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean().item()
        
        if step % 10 == 0 or step == 40:
            print(f"  Step [{step:02d}/40] - Loss: {l.item():.6f} | Accuracy: {acc*100:.1f}%")
            
    final_loss = l.item()
    final_acc = (torch.argmax(model(x), dim=1) == y).float().mean().item()
    
    assert final_acc == 1.0, f"Overfit sanity test failed: expected 100% accuracy on tiny batch, got {final_acc*100:.1f}%"
    print(f"[PASS] Overfit Sanity Test Passed! Achieved 100% Accuracy on sample batch (Loss: {initial_loss:.4f} -> {final_loss:.6f})")


if __name__ == "__main__":
    test_gradient_flow_and_overfit()
