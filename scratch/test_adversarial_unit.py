"""
Unit test for Adversarial Attack & Evaluation Engine (FGSM, BIM, PGD).
"""

import sys
import torch

sys.path.insert(0, "d:/projects/DeepGuard")

from models.deepqshield import DeepQShieldModel
from training.losses import FocalLoss
from adversarial.fgsm import fgsm_attack
from adversarial.bim import bim_attack
from adversarial.pgd import pgd_attack


def test_adversarial_attacks():
    print("Testing Adversarial Attacks (FGSM, BIM, PGD)...")
    
    model = DeepQShieldModel(pretrained_backbone=False, seed=42)
    model.eval()
    
    images = torch.randn(2, 3, 224, 224).clamp(0, 1)
    targets = torch.tensor([0, 1])
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    
    # 1. Test FGSM
    adv_fgsm = fgsm_attack(model, images, targets, criterion, epsilon=0.031)
    assert adv_fgsm.shape == images.shape, "FGSM tensor shape mismatch"
    diff_fgsm = torch.max(torch.abs(adv_fgsm - images)).item()
    assert diff_fgsm <= 0.032, f"FGSM perturbation bound violated: {diff_fgsm}"
    print(f"[PASS] FGSM Attack (Max Perturbation: {diff_fgsm:.4f})")
    
    # 2. Test BIM
    adv_bim = bim_attack(model, images, targets, criterion, epsilon=0.031, steps=3)
    assert adv_bim.shape == images.shape, "BIM tensor shape mismatch"
    diff_bim = torch.max(torch.abs(adv_bim - images)).item()
    assert diff_bim <= 0.032, f"BIM perturbation bound violated: {diff_bim}"
    print(f"[PASS] BIM Attack (Max Perturbation: {diff_bim:.4f})")
    
    # 3. Test PGD
    adv_pgd = pgd_attack(model, images, targets, criterion, epsilon=0.031, steps=3)
    assert adv_pgd.shape == images.shape, "PGD tensor shape mismatch"
    diff_pgd = torch.max(torch.abs(adv_pgd - images)).item()
    assert diff_pgd <= 0.032, f"PGD perturbation bound violated: {diff_pgd}"
    print(f"[PASS] PGD Attack (Max Perturbation: {diff_pgd:.4f})")
    
    print("\nALL ADVERSARIAL ATTACK UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_adversarial_attacks()
