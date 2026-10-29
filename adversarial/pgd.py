"""
Projected Gradient Descent (PGD) Adversarial Attack.
Implements multi-step PGD attack with random initialization for robust evaluation.
"""

import torch
import torch.nn as nn


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    epsilon: float = 0.0313725, # 8/255
    alpha: float = 0.0078431,   # 2/255
    steps: int = 10
) -> torch.Tensor:
    """
    Generate adversarial images using PGD with random noise start.
    
    Formula:
        x_0 = x + Uniform(-epsilon, epsilon)
        x_{t+1} = Clip_{x, epsilon}(x_t + alpha * sign(grad_x_t(loss)))
    """
    model.eval()
    ori_images = images.clone().detach()
    
    # Random start within epsilon ball
    random_noise = torch.FloatTensor(images.shape).uniform_(-epsilon, epsilon).to(images.device)
    perturbed_images = torch.clamp(ori_images + random_noise, min=0.0, max=1.0)
    
    for _ in range(steps):
        perturbed_images.requires_grad = True
        logits = model(perturbed_images)
        loss = criterion(logits, targets)
        
        model.zero_grad()
        loss.backward()
        
        grad = perturbed_images.grad.data
        perturbed_images = perturbed_images.detach() + alpha * grad.sign()
        
        # Project back into epsilon L_inf ball and valid image range [0, 1]
        eta = torch.clamp(perturbed_images - ori_images, min=-epsilon, max=epsilon)
        perturbed_images = torch.clamp(ori_images + eta, min=0.0, max=1.0)
        
    return perturbed_images.detach()
