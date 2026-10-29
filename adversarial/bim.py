"""
Basic Iterative Method (BIM / I-FGSM) Adversarial Attack.
Iterative gradient step attack clamped within epsilon L_infinity bound.
"""

import torch
import torch.nn as nn


def bim_attack(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    epsilon: float = 0.0313725, # 8/255
    alpha: float = 0.0078431,   # 2/255
    steps: int = 10
) -> torch.Tensor:
    """
    Generate adversarial images using BIM (Iterative FGSM).
    """
    model.eval()
    ori_images = images.clone().detach()
    perturbed_images = images.clone().detach()
    
    for _ in range(steps):
        perturbed_images.requires_grad = True
        logits = model(perturbed_images)
        loss = criterion(logits, targets)
        
        model.zero_grad()
        loss.backward()
        
        grad = perturbed_images.grad.data
        perturbed_images = perturbed_images.detach() + alpha * grad.sign()
        
        # L_inf perturbation projection
        eta = torch.clamp(perturbed_images - ori_images, min=-epsilon, max=epsilon)
        perturbed_images = torch.clamp(ori_images + eta, min=0.0, max=1.0)
        
    return perturbed_images.detach()
