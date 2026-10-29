"""
Automated Adversarial Robustness Benchmarking Script.
Evaluates FGSM, BIM, and PGD attacks on Baseline ResNeXt vs Full DeepQShield on held-out test data.
Uses trained checkpoint weights across all model variants.
"""

import os
import sys
import csv
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.deepqshield import DeepQShieldModel
from scripts.run_ablation import AblatedModelWrapper
from adversarial.evaluation import evaluate_adversarial_robustness
from training.dataset import create_dataloader
from training.augmentations import get_validation_transform


def run_adversarial_benchmark(data_dir: str = "data", output_dir: str = "reports", checkpoint_path: str = "checkpoints/deepqshield_best.pth"):
    os.makedirs(output_dir, exist_ok=True)
    val_transform = get_validation_transform()
    loader = create_dataloader(data_dir, split="test", batch_size=16, shuffle=False, transform=val_transform)
    
    deepqshield = DeepQShieldModel(pretrained_backbone=False, seed=42)
    if os.path.exists(checkpoint_path):
        deepqshield.load_checkpoint(checkpoint_path)
    deepqshield.eval()
    
    baseline = AblatedModelWrapper(deepqshield, "baseline_only")
    baseline.eval()
    
    print(f"Executing Adversarial Robustness Evaluation (Clean, FGSM, BIM, PGD) on {len(loader.dataset)} Held-Out Test Frames...", flush=True)
    
    baseline_res = evaluate_adversarial_robustness(baseline, loader, epsilon=0.03137)
    deepqshield_res = evaluate_adversarial_robustness(deepqshield, loader, epsilon=0.03137)
    
    rows = [
        {
            "Model": "Baseline ResNeXt50",
            "Clean_Acc": baseline_res["clean_accuracy"],
            "FGSM_Acc": baseline_res["fgsm_accuracy"],
            "BIM_Acc": baseline_res["bim_accuracy"],
            "PGD_Acc": baseline_res["pgd_accuracy"],
            "Clean_AUC": baseline_res["clean_auc"],
            "PGD_AUC": baseline_res["pgd_auc"]
        },
        {
            "Model": "Full DeepQShield (LWE+Fusion+Attn)",
            "Clean_Acc": deepqshield_res["clean_accuracy"],
            "FGSM_Acc": deepqshield_res["fgsm_accuracy"],
            "BIM_Acc": deepqshield_res["bim_accuracy"],
            "PGD_Acc": deepqshield_res["pgd_accuracy"],
            "Clean_AUC": deepqshield_res["clean_auc"],
            "PGD_AUC": deepqshield_res["pgd_auc"]
        }
    ]
    
    report_path = os.path.join(output_dir, "adversarial_results.csv")
    with open(report_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Saved Adversarial Benchmark Report: {report_path}")
    for r in rows:
        print(f"[{r['Model']}] Clean Acc: {r['Clean_Acc']} | FGSM Acc: {r['FGSM_Acc']} | BIM Acc: {r['BIM_Acc']} | PGD Acc: {r['PGD_Acc']} | PGD AUC: {r['PGD_AUC']}")


if __name__ == "__main__":
    run_adversarial_benchmark()
