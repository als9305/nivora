"""
Benchmark & Threshold Calibration Framework for Module 2 Face Identity Verification.
Uses strict Calibration (Dev) vs Untouched Test split to prevent threshold data leakage.
- Calibration Set: dev split of face frame identities (genuine_user_05 & real_webcam_face dev frames)
- Untouched Test Set: test split of face frame identities (genuine_user_05 & real_webcam_face test frames)
"""

import os
import sys
import json
import glob
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.face_recognition import face_engine_instance


def get_embeddings_for_files(files):
    embs = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            try:
                emb = face_engine_instance.extract_embedding(img)
                embs.append(emb)
            except Exception:
                pass
    return embs


def generate_pairs(embs_u05, embs_webcam):
    genuine = [face_engine_instance.compute_cosine_similarity(e1, e2) for i, e1 in enumerate(embs_u05) for e2 in embs_u05[i+1:]] + \
              [face_engine_instance.compute_cosine_similarity(e1, e2) for i, e1 in enumerate(embs_webcam) for e2 in embs_webcam[i+1:]]
    impostor = [face_engine_instance.compute_cosine_similarity(e1, e2) for e1 in embs_u05 for e2 in embs_webcam]
    return genuine, impostor


def run_benchmark():
    print("==========================================================================")
    print("  NIVORA MODULE 2: FACE IDENTITY BENCHMARK (STRICT CALIBRATION SPLIT)     ")
    print("==========================================================================")

    files_u05 = sorted(glob.glob(os.path.join(root_dir, "data", "test", "real", "genuine_user_05_*.jpg")))
    files_webcam = sorted(glob.glob(os.path.join(root_dir, "data", "test", "real", "real_webcam_face_*.jpg")))

    dev_u05, test_u05 = files_u05[:7], files_u05[7:]
    dev_web, test_web = files_webcam[:25], files_webcam[25:]

    embs_dev_u05 = get_embeddings_for_files(dev_u05)
    embs_dev_web = get_embeddings_for_files(dev_web)

    embs_test_u05 = get_embeddings_for_files(test_u05)
    embs_test_web = get_embeddings_for_files(test_web)

    # 1. Calibration (Dev) Pair Generation
    gen_dev, imp_dev = generate_pairs(embs_dev_u05, embs_dev_web)
    y_dev_true = [1] * len(gen_dev) + [0] * len(imp_dev)
    y_dev_scores = gen_dev + imp_dev

    fpr_d, tpr_d, thresh_d = roc_curve(y_dev_true, y_dev_scores)
    fnr_d = 1 - tpr_d
    eer_idx = np.nanargmin(np.absolute(fnr_d - fpr_d))
    calib_tau = float(thresh_d[eer_idx])

    print(f"Calibration Set (Dev):")
    print(f"  - Dev Genuine Pairs : {len(gen_dev)} (Mean = {np.mean(gen_dev):.4f} ± {np.std(gen_dev):.4f})")
    print(f"  - Dev Impostor Pairs: {len(imp_dev)} (Mean = {np.mean(imp_dev):.4f} ± {np.std(imp_dev):.4f})")
    print(f"  -> CALIBRATED DECISION THRESHOLD (tau*): {calib_tau:.4f}")

    # Production default threshold is set to 0.65 for high-precision security
    prod_tau = 0.65

    # 2. Untouched Held-Out Test Evaluation
    gen_test, imp_test = generate_pairs(embs_test_u05, embs_test_web)
    y_test_true = [1] * len(gen_test) + [0] * len(imp_test)
    y_test_scores = gen_test + imp_test

    y_test_pred = [1 if s >= prod_tau else 0 for s in y_test_scores]

    acc = float(accuracy_score(y_test_true, y_test_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_true, y_test_pred, average='binary', zero_division=0)
    
    fpr_t, tpr_t, _ = roc_curve(y_test_true, y_test_scores)
    test_auc = float(auc(fpr_t, tpr_t))

    fnr_t = 1 - tpr_t
    eer_test_idx = np.nanargmin(np.absolute(fnr_t - fpr_t))
    test_eer = float(fpr_t[eer_test_idx])

    far_01_idx = np.where(fpr_t <= 0.01)[0]
    tar_at_far_01 = float(tpr_t[far_01_idx[-1]]) if len(far_01_idx) > 0 else float(tpr_t[0])

    print("\n--------------------------------------------------------------------------")
    print(f" HELD-OUT TEST EVALUATION RESULTS (Evaluated at Production tau = {prod_tau:.2f}):")
    print(f"  - Test Genuine Pairs            : {len(gen_test)} (Mean = {np.mean(gen_test):.4f} ± {np.std(gen_test):.4f})")
    print(f"  - Test Impostor Pairs           : {len(imp_test)} (Mean = {np.mean(imp_test):.4f} ± {np.std(imp_test):.4f})")
    print(f"  - Test Accuracy                 : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  - Test Precision                : {prec:.4f} ({prec*100:.2f}%)")
    print(f"  - Test Recall                   : {rec:.4f} ({rec*100:.2f}%)")
    print(f"  - Test F1-Score                 : {f1:.4f}")
    print(f"  - Test ROC-AUC                  : {test_auc:.4f}")
    print(f"  - Test EER                      : {test_eer:.4f} ({test_eer*100:.2f}%)")
    print(f"  - Test TAR @ FAR=0.01           : {tar_at_far_01:.4f} ({tar_at_far_01*100:.2f}%)")
    print("--------------------------------------------------------------------------")

    report = {
        "model": "InceptionResnetV1_VGGFace2",
        "embedding_dim": 512,
        "calibration_split": {
            "dev_genuine_pairs": len(gen_dev),
            "dev_impostor_pairs": len(imp_dev),
            "dev_genuine_mean": round(float(np.mean(gen_dev)), 4),
            "dev_impostor_mean": round(float(np.mean(imp_dev)), 4),
            "dev_eer_threshold_tau": round(calib_tau, 4)
        },
        "held_out_test_split": {
            "test_genuine_pairs": len(gen_test),
            "test_impostor_pairs": len(imp_test),
            "test_genuine_similarity_mean": round(float(np.mean(gen_test)), 4),
            "test_impostor_similarity_mean": round(float(np.mean(imp_test)), 4)
        },
        "metrics_at_production_tau": {
            "production_tau": prod_tau,
            "accuracy": round(acc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(test_auc, 4),
            "eer": round(test_eer, 4),
            "tar_at_far_0.01": round(tar_at_far_01, 4)
        }
    }

    report_path = os.path.join(root_dir, "reports", "face_identity_benchmark.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved updated audit benchmark report to: {report_path}")
    return report


if __name__ == "__main__":
    run_benchmark()
