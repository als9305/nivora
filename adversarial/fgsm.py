"""
Fast Gradient Sign Method (FGSM) Adversarial Attack.
Computes standard one-step adversarial perturbation on PyTorch image tensors.
"""

import torch
import torch.nn as nn


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    epsilon: float = 0.0313725 # 8/255
) -> torch.Tensor:
    """
    Generate adversarial images using FGSM.
    
    Formula:
        x_adv = x + epsilon * sign(grad_x(loss(model(x), y)))
    """
    images = images.clone().detach().requires_grad_(True)
    model.eval()
    
    logits = model(images)
    loss = criterion(logits, targets)
    
    model.zero_grad()
    loss.backward()
    
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    return perturbed_images.detach()
