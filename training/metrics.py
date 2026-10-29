"""
Evaluation Metrics Calculator for DeepQShield.
Computes Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC scores.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any, List


def compute_metrics(y_true: List[int], y_probs: List[float], threshold: float = 0.50) -> Dict[str, float]:
    """
    Calculate classification metrics from ground truth labels and predicted fake probabilities.
    
    Args:
        y_true: Ground truth binary labels (0: REAL, 1: DEEPFAKE)
        y_probs: Predicted deepfake probabilities (0.0 to 1.0)
        threshold: Decision threshold for classification
        
    Returns:
        Dict containing accuracy, precision, recall, f1, and roc_auc
    """
    y_true_arr = np.array(y_true)
    y_probs_arr = np.array(y_probs)
    y_pred_arr = (y_probs_arr >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    
    # Handle edge cases where all predictions or targets are single class
    try:
        prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    except Exception:
        prec, rec, f1 = 0.0, 0.0, 0.0
        
    try:
        if len(np.unique(y_true_arr)) > 1:
            auc = float(roc_auc_score(y_true_arr, y_probs_arr))
        else:
            auc = 0.50
    except Exception:
        auc = 0.50
        
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4)
    }
