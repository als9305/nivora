"""
Adversarial Evaluation Subsystem for DeepQShield.
Evaluates model performance under Clean, FGSM, BIM, and PGD attack vectors.
Compares Baseline ResNeXt vs ResNeXt+LWE vs Full DeepQShield.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adversarial.fgsm import fgsm_attack
from adversarial.bim import bim_attack
from adversarial.pgd import pgd_attack
from training.losses import FocalLoss
from training.metrics import compute_metrics


def evaluate_adversarial_robustness(
    model: nn.Module,
    dataloader: DataLoader,
    epsilon: float = 0.0313725,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate a model across Clean, FGSM, BIM, and PGD perturbed inputs.
    
    Returns:
        Dict mapping attack_name -> accuracy / metrics
    """
    model.eval()
    model.to(device)
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    
    results = {
        "clean_accuracy": 0.0,
        "fgsm_accuracy": 0.0,
        "bim_accuracy": 0.0,
        "pgd_accuracy": 0.0,
        "clean_auc": 0.0,
        "pgd_auc": 0.0
    }
    
    eval_types = ["clean", "fgsm", "bim", "pgd"]
    eval_metrics = {}
    
    for attack in eval_types:
        all_targets = []
        all_probs = []
        
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            if attack == "clean":
                adv_images = images
            elif attack == "fgsm":
                adv_images = fgsm_attack(model, images, targets, criterion, epsilon=epsilon)
            elif attack == "bim":
                adv_images = bim_attack(model, images, targets, criterion, epsilon=epsilon, steps=5)
            elif attack == "pgd":
                adv_images = pgd_attack(model, images, targets, criterion, epsilon=epsilon, steps=5)
            else:
                adv_images = images
                
            with torch.no_grad():
                logits = model(adv_images)
                probs = F.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                
            all_targets.extend(targets.cpu().tolist())
            all_probs.extend(probs)
            
        metrics = compute_metrics(all_targets, all_probs)
        eval_metrics[f"{attack}_accuracy"] = metrics["accuracy"]
        eval_metrics[f"{attack}_f1"] = metrics["f1"]
        eval_metrics[f"{attack}_roc_auc"] = metrics["roc_auc"]
        
    results["clean_accuracy"] = eval_metrics["clean_accuracy"]
    results["fgsm_accuracy"] = eval_metrics["fgsm_accuracy"]
    results["bim_accuracy"] = eval_metrics["bim_accuracy"]
    results["pgd_accuracy"] = eval_metrics["pgd_accuracy"]
    results["clean_auc"] = eval_metrics["clean_roc_auc"]
    results["pgd_auc"] = eval_metrics["pgd_roc_auc"]
    
    return results
