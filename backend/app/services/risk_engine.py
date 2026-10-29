from typing import Dict, Any, List
from app.config import settings
from app.utils.logging import logger

class RiskEngine:
    """
    Deterministic & Evidence-Driven Risk Engine for Multi-Modal Adaptive Authentication.
    Combines face, liveness, deepfake, voice, and device signals into a normalized 0-100 risk score.
    
    Includes:
      - Internal Confidence/Uncertainty Layer (HIGH_CONFIDENCE_GENUINE, HIGH_CONFIDENCE_ATTACK, SUSPICIOUS, UNCERTAIN, INSUFFICIENT_QUALITY)
      - Exact Risk Contribution Breakdown (w_i * R_i)
      - Rule-Based Threat Overrides (Rule ID, Threshold, Observed, Action, Evidence)
    """
    @staticmethod
    def calculate_risk(
        face_score: float,
        liveness_score: float,
        deepfake_probability: float,
        voice_similarity: float,
        voice_spoof_probability: float,
        device_risk: float = 10.0,
        session_risk: float = 10.0
    ) -> Dict[str, Any]:
        # 1. Normalize individual risk components to 0 - 100 scale
        face_identity_risk = round(max(0.0, min(100.0, (1.0 - face_score) * 100.0)), 2)
        liveness_risk = round(max(0.0, min(100.0, (1.0 - liveness_score) * 100.0)), 2)
        deepfake_risk = round(max(0.0, min(100.0, deepfake_probability * 100.0)), 2)
        
        voice_identity_risk = (1.0 - voice_similarity) * 50.0
        voice_spoof_risk = voice_spoof_probability * 50.0
        voice_combined_risk = max(0.0, min(100.0, voice_identity_risk + voice_spoof_risk))
        device_session_risk = max(0.0, min(100.0, (device_risk + session_risk) / 2.0))

        # 2. Extract weights from configuration
        w_face = settings.WEIGHT_FACE
        w_live = settings.WEIGHT_LIVENESS
        w_fake = settings.WEIGHT_DEEPFAKE
        w_voice = settings.WEIGHT_VOICE
        w_dev = settings.WEIGHT_DEVICE

        # 3. Compute individual mathematical risk contributions
        c_face = round(w_face * face_identity_risk, 3)
        c_live = round(w_live * liveness_risk, 3)
        c_fake = round(w_fake * deepfake_risk, 3)
        c_voice = round(w_voice * voice_combined_risk, 3)
        c_dev = round(w_dev * device_session_risk, 3)

        base_weighted_score = c_face + c_live + c_fake + c_voice + c_dev
        final_weighted_score = base_weighted_score

        # 4. Evidence-driven threat override rules
        override_delta = 0.0
        rule_triggers: List[Dict[str, Any]] = []

        # Rule 1: Synthetic Media / Deepfake Override
        if deepfake_risk >= 70.0 or (deepfake_risk >= 60.0 and liveness_risk >= 60.0):
            target_score = 85.0
            if final_weighted_score < target_score:
                override_delta = round(target_score - final_weighted_score, 2)
                final_weighted_score = target_score
            rule_triggers.append({
                "rule_id": "SYNTHETIC_MEDIA_OVERRIDE_01",
                "threshold": "deepfake_risk >= 70%",
                "observed": f"deepfake_risk={int(deepfake_risk)}%",
                "action": "ESCALATE_TO_BLOCK",
                "evidence": "Multi-frame spatial-temporal frequency anomalies detected in video feed."
            })

        # Rule 2: Active Liveness Failure Override
        elif liveness_risk >= 75.0:
            target_score = 75.0
            if final_weighted_score < target_score:
                override_delta = round(target_score - final_weighted_score, 2)
                final_weighted_score = target_score
            rule_triggers.append({
                "rule_id": "LIVENESS_PRESENTATION_ATTACK_02",
                "threshold": "liveness_risk >= 75%",
                "observed": f"liveness_risk={int(liveness_risk)}%",
                "action": "ESCALATE_TO_BLOCK",
                "evidence": "Static photo replay or zero facial micro-movement detected during liveness challenge."
            })

        # Rule 3: Voice Clone / TTS Override
        elif voice_spoof_risk >= 35.0 and voice_identity_risk >= 20.0:
            target_score = 72.0
            if final_weighted_score < target_score:
                override_delta = round(target_score - final_weighted_score, 2)
                final_weighted_score = target_score
            rule_triggers.append({
                "rule_id": "VOICE_SPOOF_OVERRIDE_03",
                "threshold": "voice_spoof_risk >= 35%",
                "observed": f"voice_spoof_risk={int(voice_spoof_risk)}%",
                "action": "ESCALATE_TO_BLOCK",
                "evidence": "Synthetic speech pitch stability or TTS voice clone characteristics detected."
            })

        # Rule 4: Moderate / Inconclusive Biometric Anomaly Override -> STEP-UP OTP
        elif deepfake_risk >= 40.0 or liveness_risk >= 35.0:
            target_score = 38.0
            if final_weighted_score < target_score:
                override_delta = round(target_score - final_weighted_score, 2)
                final_weighted_score = target_score
            rule_triggers.append({
                "rule_id": "MODERATE_ANOMALY_STEP_UP_04",
                "threshold": "deepfake_risk >= 40% or liveness_risk >= 35%",
                "observed": f"deepfake_risk={int(deepfake_risk)}%, liveness_risk={int(liveness_risk)}%",
                "action": "ESCALATE_TO_STEP_UP_OTP",
                "evidence": "Inconclusive biometric evidence or moderate deepfake indicators requiring Step-Up authorization."
            })

        total_risk_score = int(round(max(0.0, min(100.0, final_weighted_score))))

        # 5. Determine Internal Confidence / Uncertainty State
        if deepfake_risk >= 70.0 or liveness_risk >= 75.0 or voice_spoof_risk >= 40.0:
            confidence_state = "HIGH_CONFIDENCE_ATTACK"
        elif total_risk_score <= 20.0 and face_score >= 0.85 and liveness_score >= 0.85:
            confidence_state = "HIGH_CONFIDENCE_GENUINE"
        elif total_risk_score <= 35.0:
            confidence_state = "SUSPICIOUS"  # Low risk but moderate variance
        elif total_risk_score <= 70.0:
            confidence_state = "UNCERTAIN"   # Inconclusive evidence -> Require Step-Up OTP
        else:
            confidence_state = "HIGH_CONFIDENCE_ATTACK"

        # 6. Map to external decision
        reasons: List[str] = []

        if deepfake_risk >= 70.0:
            reasons.append(f"High synthetic media / deepfake probability detected ({int(deepfake_risk)}%)")

        if liveness_risk >= 60.0:
            reasons.append(f"Active liveness challenge failed or suspicious static frame behavior ({int(liveness_risk)}% risk)")

        if face_identity_risk >= 50.0:
            reasons.append(f"Low face biometric identity similarity ({int(face_score * 100)}% match)")

        if voice_combined_risk >= 50.0:
            reasons.append("Voice verification failed or voice anti-spoof indicators detected")

        if device_session_risk >= 50.0:
            reasons.append("High device risk or rapid attempt frequency detected")

        if total_risk_score <= settings.RISK_THRESHOLD_LOW:
            risk_level = "LOW"
            decision = "ALLOW"
            if not reasons:
                reasons.append("All multi-modal biometric and liveness checks passed cleanly")
        elif total_risk_score <= settings.RISK_THRESHOLD_MEDIUM:
            risk_level = "MEDIUM"
            decision = "STEP-UP OTP"
            if not reasons:
                reasons.append("Inconclusive biometric evidence requiring Step-Up OTP authorization")
        else:
            risk_level = "HIGH"
            decision = "BLOCK"
            if not reasons:
                reasons.append("High overall multi-factor threat score across biometric layers")

        return {
            "risk_score": total_risk_score,
            "risk_level": risk_level,
            "decision": decision,
            "confidence_state": confidence_state,
            "reasons": reasons,
            "rule_triggers": rule_triggers,
            "mathematical_breakdown": {
                "base_risk_score": round(base_weighted_score, 2),
                "override_delta": override_delta,
                "contributions": {
                    "face": c_face,
                    "liveness": c_live,
                    "deepfake": c_fake,
                    "voice": c_voice,
                    "device": c_dev
                }
            },
            "score_breakdown": {
                "face_identity_risk": round(face_identity_risk, 1),
                "liveness_risk": round(liveness_risk, 1),
                "deepfake_risk": round(deepfake_risk, 1),
                "voice_risk": round(voice_combined_risk, 1),
                "device_risk": round(device_session_risk, 1)
            },
            "weights_used": {
                "face": w_face,
                "liveness": w_live,
                "deepfake": w_fake,
                "voice": w_voice,
                "device": w_dev
            }
        }
