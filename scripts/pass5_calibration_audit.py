"""
Module 4 — Pass 5 Final Calibration & Generalization Audit Script.

Executes:
1. Audit of the 5 false-rejected Bona Fide TEST samples (from Pass 4 baseline tau=0.7000).
2. DEV-Only Threshold Calibration (Grid search over tau in [0.05, 0.95]).
3. HELD-OUT TEST Evaluation at DEV-calibrated threshold(s).
4. Objective Rule Check for Module 4 Lock Decision:
   - ROC-AUC >= 0.80
   - ACER <= 20%
   - BPCER <= 20%
   - APCER <= 20%
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import huggingface_hub

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor
from scripts.benchmark_voice_auth import fetch_librispeech_samples, fetch_asvspoof_pad_samples


def run_pass5_audit():
    print("=" * 70)
    print("MODULE 4 — PASS 5: FINAL CALIBRATION + GENERALIZATION AUDIT")
    print("=" * 70)

    # 1. Fetch Datasets
    pad_data = fetch_asvspoof_pad_samples()
    libri_dict = fetch_librispeech_samples()

    # Split ASVspoof DEV (40%) and TEST (60%)
    dev_pad_samples = {}
    test_pad_samples = {}
    for cat, samples in pad_data.items():
        n_dev = int(len(samples) * 0.4)
        dev_pad_samples[cat] = samples[:n_dev]
        test_pad_samples[cat] = samples[n_dev:]

    # LibriSpeech DEV vs TEST speakers
    spk_ids = sorted(list(libri_dict.keys()))
    half_spks = len(spk_ids) // 2
    dev_spk_ids = spk_ids[:half_spks]
    test_spk_ids = spk_ids[half_spks:]

    libri_dev_samples = []
    for spk in dev_spk_ids:
        for b in libri_dict[spk][:3]:
            libri_dev_samples.append(b)

    libri_test_samples = []
    for spk in test_spk_ids:
        for b in libri_dict[spk][:3]:
            libri_test_samples.append(b)

    # Combine Bona Fide DEV (ASVspoof authentic DEV + LibriSpeech DEV)
    dev_bonafide = dev_pad_samples['BONAFIDE'] + libri_dev_samples
    dev_attack = dev_pad_samples['REPLAY_ATTACK'] + dev_pad_samples['SYNTHETIC_TTS'] + dev_pad_samples['VOICE_CLONE']

    # TEST Bona Fide (Untouched Pass 4 TEST setup: ASVspoof authentic TEST + LibriSpeech TEST)
    test_bonafide = test_pad_samples['BONAFIDE']  # n=28 raw in pad_data -> 17 in TEST
    test_attack_per_cat = {
        'REPLAY_ATTACK': test_pad_samples['REPLAY_ATTACK'],
        'SYNTHETIC_TTS': test_pad_samples['SYNTHETIC_TTS'],
        'VOICE_CLONE': test_pad_samples['VOICE_CLONE']
    }

    print(f"\nDataset Splits:")
    print(f"  DEV Bona Fide: {len(dev_bonafide)} samples (ASVspoof: {len(dev_pad_samples['BONAFIDE'])}, LibriSpeech: {len(libri_dev_samples)})")
    print(f"  DEV Spoof:     {len(dev_attack)} samples (Replay: {len(dev_pad_samples['REPLAY_ATTACK'])}, TTS: {len(dev_pad_samples['SYNTHETIC_TTS'])}, VC: {len(dev_pad_samples['VOICE_CLONE'])})")
    print(f"  TEST Bona Fide: {len(test_bonafide)} samples")
    print(f"  TEST Spoof:     {sum(len(v) for v in test_attack_per_cat.values())} samples (Replay: {len(test_attack_per_cat['REPLAY_ATTACK'])}, TTS: {len(test_attack_per_cat['SYNTHETIC_TTS'])}, VC: {len(test_attack_per_cat['VOICE_CLONE'])})")

    # =========================================================================
    # STEP 1: AUDIT THE 5 FALSE-REJECTED BONAFIDE TEST SAMPLES AT TAU = 0.7000
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1 — AUDIT FALSE-REJECTED BONAFIDE TEST SAMPLES (Baseline tau = 0.7000)")
    print("=" * 70)

    test_bonafide_audit = []
    false_rejects = []

    for i, s in enumerate(test_bonafide):
        res = voice_auth_engine.process_voice_pipeline(s)
        p_live = res['liveness_score']
        qual = res['audio_quality']
        acous = res.get('acoustic_metrics', {})

        sample_info = {
            "index": i + 1,
            "duration_sec": qual.get("duration_sec", 0.0),
            "rms_energy": qual.get("rms_energy", 0.0),
            "clipping_ratio": qual.get("clipping_ratio", 0.0),
            "f0_mean_hz": acous.get("f0_mean_hz", 0.0),
            "f0_std_hz": acous.get("f0_std_hz", 0.0),
            "high_freq_ratio": acous.get("high_frequency_ratio", 0.0),
            "zero_crossing_rate": acous.get("zero_crossing_rate", 0.0),
            "spectral_centroid_hz": acous.get("spectral_centroid_hz", 0.0),
            "p_live": p_live,
            "verdict_tau_070": "LIVE" if p_live >= 0.7000 else "REJECTED_SPOOF"
        }
        test_bonafide_audit.append(sample_info)

        if p_live < 0.7000:
            false_rejects.append(sample_info)

    print(f"\nFound {len(false_rejects)} False Rejects out of {len(test_bonafide)} Bona Fide TEST samples at tau = 0.7000:")
    for fr in false_rejects:
        print(f"  - Sample #{fr['index']}: p_live={fr['p_live']:.4f} | duration={fr['duration_sec']}s | RMS={fr['rms_energy']} | F0_std={fr['f0_std_hz']} | HF_ratio={fr['high_freq_ratio']} | ZCR={fr['zero_crossing_rate']}")

    # =========================================================================
    # STEP 2: DEV-ONLY CALIBRATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2 — DEV-ONLY CALIBRATION")
    print("=" * 70)

    dev_bonafide_scores = [voice_auth_engine.process_voice_pipeline(s)['liveness_score'] for s in dev_bonafide]
    dev_attack_scores = [voice_auth_engine.process_voice_pipeline(s)['liveness_score'] for s in dev_attack]

    print(f"DEV BONAFIDE Scores (n={len(dev_bonafide_scores)}): mean={np.mean(dev_bonafide_scores):.4f}, median={np.median(dev_bonafide_scores):.4f}, std={np.std(dev_bonafide_scores):.4f}, min={np.min(dev_bonafide_scores):.4f}, max={np.max(dev_bonafide_scores):.4f}")
    print(f"DEV SPOOF Scores    (n={len(dev_attack_scores)}): mean={np.mean(dev_attack_scores):.4f}, median={np.median(dev_attack_scores):.4f}, std={np.std(dev_attack_scores):.4f}, min={np.min(dev_attack_scores):.4f}, max={np.max(dev_attack_scores):.4f}")

    # Threshold Search Grid on DEV
    taus = np.linspace(0.05, 0.95, 181)
    dev_grid_results = []
    best_dev_tau = 0.50
    min_dev_acer = 1.0

    tau_apcer_10 = None
    tau_apcer_15 = None
    tau_apcer_20 = None

    for t in taus:
        apcer_d = sum(1 for s in dev_attack_scores if s >= t) / float(max(1, len(dev_attack_scores)))
        bpcer_d = sum(1 for s in dev_bonafide_scores if s < t) / float(max(1, len(dev_bonafide_scores)))
        acer_d = (apcer_d + bpcer_d) / 2.0

        res = {"tau": round(t, 4), "apcer": round(apcer_d, 4), "bpcer": round(bpcer_d, 4), "acer": round(acer_d, 4)}
        dev_grid_results.append(res)

        if acer_d < min_dev_acer:
            min_dev_acer = acer_d
            best_dev_tau = t

        if tau_apcer_10 is None and apcer_d <= 0.10:
            tau_apcer_10 = t
        if tau_apcer_15 is None and apcer_d <= 0.15:
            tau_apcer_15 = t
        if tau_apcer_20 is None and apcer_d <= 0.20:
            tau_apcer_20 = t

    print(f"\nCalibrated DEV Thresholds:")
    print(f"  - Optimal DEV Threshold (Min ACER): tau* = {best_dev_tau:.4f} (DEV ACER = {min_dev_acer:.4f})")
    print(f"  - DEV Threshold for APCER <= 10%:  tau = {tau_apcer_10:.4f}" if tau_apcer_10 else "  - DEV Threshold for APCER <= 10%: N/A")
    print(f"  - DEV Threshold for APCER <= 15%:  tau = {tau_apcer_15:.4f}" if tau_apcer_15 else "  - DEV Threshold for APCER <= 15%: N/A")
    print(f"  - DEV Threshold for APCER <= 20%:  tau = {tau_apcer_20:.4f}" if tau_apcer_20 else "  - DEV Threshold for APCER <= 20%: N/A")

    # =========================================================================
    # STEP 3: EVALUATE UNTOUCHED HELD-OUT TEST SET AT DEV-CALIBRATED THRESHOLDS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3 — EVALUATE UNTOUCHED HELD-OUT TEST SET")
    print("=" * 70)

    test_bonafide_scores = [s['p_live'] for s in test_bonafide_audit]
    test_attack_scores = {}
    for cat, samples in test_attack_per_cat.items():
        test_attack_scores[cat] = [voice_auth_engine.process_voice_pipeline(s)['liveness_score'] for s in samples]

    all_test_attacks = []
    for cat, scores in test_attack_scores.items():
        all_test_attacks.extend(scores)

    # Compute ROC-AUC and EER on TEST
    thresholds_fine = np.linspace(0.0, 1.0, 1000)
    apcers_fine = [sum(1 for s in all_test_attacks if s >= t) / float(max(1, len(all_test_attacks))) for t in thresholds_fine]
    bpcers_fine = [sum(1 for s in test_bonafide_scores if s < t) / float(max(1, len(test_bonafide_scores))) for t in thresholds_fine]

    eer_idx = np.argmin(np.abs(np.array(apcers_fine) - np.array(bpcers_fine)))
    test_eer = float((apcers_fine[eer_idx] + bpcers_fine[eer_idx]) / 2.0)

    tpr_list = [1.0 - apc for apc in apcers_fine]
    fpr_list = bpcers_fine
    sorted_pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted, tpr_sorted = zip(*sorted_pairs)
    test_roc_auc = float(np.trapezoid(tpr_sorted, fpr_sorted))
    test_roc_auc = max(0.0, min(1.0, test_roc_auc))

    eval_results = {}
    eval_taus = [("DEV Min-ACER", best_dev_tau), ("Pass 4 Baseline", 0.7000)]
    if tau_apcer_20 and tau_apcer_20 not in [best_dev_tau, 0.7000]:
        eval_taus.append(("DEV APCER<=20%", tau_apcer_20))
    if tau_apcer_15 and tau_apcer_15 not in [best_dev_tau, 0.7000]:
        eval_taus.append(("DEV APCER<=15%", tau_apcer_15))

    for label, tau_val in eval_taus:
        bpcer_t = sum(1 for s in test_bonafide_scores if s < tau_val) / float(max(1, len(test_bonafide_scores)))
        apcer_t = sum(1 for s in all_test_attacks if s >= tau_val) / float(max(1, len(all_test_attacks)))
        acer_t = (bpcer_t + apcer_t) / 2.0

        per_cat_apcer = {
            cat: sum(1 for s in scores if s >= tau_val) / float(max(1, len(scores)))
            for cat, scores in test_attack_scores.items()
        }

        # Confusion Matrix
        tp = sum(1 for s in test_bonafide_scores if s >= tau_val)  # Live correctly identified
        fn = sum(1 for s in test_bonafide_scores if s < tau_val)   # Live rejected
        tn = sum(1 for s in all_test_attacks if s < tau_val)       # Spoof correctly blocked
        fp = sum(1 for s in all_test_attacks if s >= tau_val)      # Spoof accepted

        eval_results[label] = {
            "tau": round(tau_val, 4),
            "bpcer": round(bpcer_t, 4),
            "apcer": round(apcer_t, 4),
            "acer": round(acer_t, 4),
            "eer": round(test_eer, 4),
            "roc_auc": round(test_roc_auc, 4),
            "replay_apcer": round(per_cat_apcer['REPLAY_ATTACK'], 4),
            "tts_apcer": round(per_cat_apcer['SYNTHETIC_TTS'], 4),
            "voice_clone_apcer": round(per_cat_apcer['VOICE_CLONE'], 4),
            "confusion_matrix": {
                "live_correct_tp": tp,
                "live_rejected_fn": fn,
                "spoof_blocked_tn": tn,
                "spoof_accepted_fp": fp
            }
        }

        print(f"\n--- Operating Point: {label} (tau = {tau_val:.4f}) ---")
        print(f"  BPCER:      {bpcer_t:.4f} ({bpcer_t*100:.2f}%) [{fn}/{len(test_bonafide_scores)} rejected]")
        print(f"  APCER:      {apcer_t:.4f} ({apcer_t*100:.2f}%) [{fp}/{len(all_test_attacks)} accepted]")
        print(f"  ACER:       {acer_t:.4f} ({acer_t*100:.2f}%)")
        print(f"  ROC-AUC:    {test_roc_auc:.4f}")
        print(f"  EER:        {test_eer:.4f}")
        print(f"  Replay APCER:  {per_cat_apcer['REPLAY_ATTACK']:.4f}")
        print(f"  TTS APCER:     {per_cat_apcer['SYNTHETIC_TTS']:.4f}")
        print(f"  Clone APCER:   {per_cat_apcer['VOICE_CLONE']:.4f}")

    # =========================================================================
    # STEP 4: OBJECTIVE LOCK DECISION CHECK
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4 — OBJECTIVE DECISION CHECK")
    print("=" * 70)

    # Use the DEV Min-ACER threshold results for primary lock check
    primary_eval = eval_results["DEV Min-ACER"]
    print(f"Primary Evaluation Threshold: tau* = {primary_eval['tau']}")
    print(f"  Criteria 1 — ROC-AUC >= 0.80: {primary_eval['roc_auc']} -> {'PASS' if primary_eval['roc_auc'] >= 0.80 else 'FAIL'}")
    print(f"  Criteria 2 — ACER <= 20%:    {primary_eval['acer']*100:.2f}% -> {'PASS' if primary_eval['acer'] <= 0.20 else 'FAIL'}")
    print(f"  Criteria 3 — BPCER <= 20%:   {primary_eval['bpcer']*100:.2f}% -> {'PASS' if primary_eval['bpcer'] <= 0.20 else 'FAIL'}")
    print(f"  Criteria 4 — APCER <= 20%:   {primary_eval['apcer']*100:.2f}% -> {'PASS' if primary_eval['apcer'] <= 0.20 else 'FAIL'}")

    lock_passed = (
        primary_eval['roc_auc'] >= 0.80 and
        primary_eval['acer'] <= 0.20 and
        primary_eval['bpcer'] <= 0.20 and
        primary_eval['apcer'] <= 0.20
    )

    module_status = "LOCKED" if lock_passed else "UNLOCKED"
    print(f"\nOBJECTIVE DECISION RESULT: MODULE 4 STATUS: {module_status}")

    audit_output = {
        "module_status": module_status,
        "false_rejects_audit": false_rejects,
        "dev_calibration": {
            "optimal_tau": round(best_dev_tau, 4),
            "optimal_acer": round(min_dev_acer, 4),
            "tau_apcer_10": round(tau_apcer_10, 4) if tau_apcer_10 else None,
            "tau_apcer_15": round(tau_apcer_15, 4) if tau_apcer_15 else None,
            "tau_apcer_20": round(tau_apcer_20, 4) if tau_apcer_20 else None,
        },
        "eval_results": eval_results,
        "lock_criteria_evaluation": {
            "roc_auc_pass": bool(primary_eval['roc_auc'] >= 0.80),
            "acer_pass": bool(primary_eval['acer'] <= 0.20),
            "bpcer_pass": bool(primary_eval['bpcer'] <= 0.20),
            "apcer_pass": bool(primary_eval['apcer'] <= 0.20),
            "overall_lock_passed": lock_passed
        }
    }

    out_file = "reports/pass5_calibration_audit.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(audit_output, f, indent=2)

    print(f"Saved Pass 5 audit results to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_pass5_audit()
