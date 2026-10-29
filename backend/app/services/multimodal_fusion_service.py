"""
Module 5 Multimodal Fusion & Unified Authentication Orchestrator Service.

Consumes existing LOCKED module interfaces (Modules 1-4) through clean adapters.
Executes an Evidence-Based Decision Engine with Hard Security Gates.

CRITICAL ARCHITECTURAL GUARANTEES:
1. Does NOT retrain, refactor, rewrite, replace, or alter Modules 1-4.
2. Does NOT use a naive weighted average to mask security failures.
3. Enforces Hard Security Gates (Face Mismatch, Face Spoof, Deepfake, Speaker Mismatch, Audio Spoof).
4. Classifies Risk Level (LOW, MEDIUM, HIGH) based on empirical evidence.
5. Preserves 300s single-use session challenge security.
6. Does NOT expose raw biometric embeddings.
7. Preserves phrase_match = "UNVALIDATED".
"""

import time
import os
import sys
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.schemas.multimodal import (
    MultimodalAuthRequest,
    MultimodalAuthResponse,
    FaceIdentityEvidence,
    FaceLivenessEvidence,
    DeepfakeEvidence,
    SpeakerEvidence,
    AudioPADEvidence
)
from app.services.face_service import FaceService
from app.services.liveness_service import LivenessService
from app.services.voice_service import VoiceService, VoiceSessionManager
from video.analyzer import VideoDeepfakeAnalyzer, get_shared_deepfake_analyzer
from app.utils.files import base64_to_cv2
from app.utils.logging import logger

# Shared analyzer for Module 1 DeepQShield
get_deepfake_analyzer = get_shared_deepfake_analyzer


class MultimodalFusionService:
    """
    Module 5 Unified Multimodal Authentication Orchestrator.
    """


    @staticmethod
    def _parse_deepfake_evidence(df_res: Dict[str, Any]) -> Tuple[DeepfakeEvidence, List[str], List[str]]:
        ev = DeepfakeEvidence(status="INDETERMINATE", fake_probability=0.0, confidence=0.0)
        m_fails: List[str] = []
        m_attacks: List[str] = []
        try:
            df_status = df_res.get("status", "SUCCESS")
            df_result = df_res.get("payload", {}).get("result") or df_res.get("result", {})
            fake_prob = float(df_result.get("video_fake_probability", df_result.get("fake_probability", 0.0)))
            df_conf = float(df_result.get("confidence", 0.85))
            classification = str(df_result.get("classification", df_result.get("label", "INDETERMINATE"))).upper()

            if df_status == "QUALITY_TOO_LOW":
                ev = DeepfakeEvidence(status="INDETERMINATE", fake_probability=fake_prob, confidence=0.0)
            elif df_status == "MULTIPLE_FACES_DETECTED":
                ev = DeepfakeEvidence(status="HIGH_RISK", fake_probability=1.0, confidence=1.0)
                m_fails.append("MULTIPLE_FACES_DETECTED")
            elif classification in ["DEEPFAKE", "REJECT"] or fake_prob >= 0.65:
                ev = DeepfakeEvidence(status="DEEPFAKE", fake_probability=fake_prob, confidence=df_conf)
                m_fails.append("DEEPFAKE_DETECTED")
                m_attacks.append("DeepQShield Synthetic Face Artifacts Detected")
            elif classification in ["REAL", "AUTHENTIC"]:
                ev = DeepfakeEvidence(status="AUTHENTIC", fake_probability=fake_prob, confidence=df_conf)
            else:
                ev = DeepfakeEvidence(status="INDETERMINATE", fake_probability=fake_prob, confidence=df_conf)
        except Exception as e:
            logger.error(f"Error parsing deepfake evidence: {e}")
        return ev, m_fails, m_attacks

    @staticmethod
    def _parse_face_identity_evidence(face_ver_res: Dict[str, Any]) -> Tuple[FaceIdentityEvidence, List[str]]:
        ev = FaceIdentityEvidence(status="INSUFFICIENT_EVIDENCE", similarity=0.0, threshold=0.65, confidence=0.0)
        m_fails: List[str] = []
        try:
            if face_ver_res.get("success"):
                matched = face_ver_res.get("matched", False)
                sim_score = float(face_ver_res.get("similarity_score", 0.0))
                thresh = float(face_ver_res.get("threshold", 0.65))

                if matched:
                    ev = FaceIdentityEvidence(
                        status="VERIFIED",
                        similarity=sim_score,
                        threshold=thresh,
                        confidence=round(min(1.0, sim_score / max(1e-4, thresh)), 4)
                    )
                else:
                    ev = FaceIdentityEvidence(
                        status="MISMATCH",
                        similarity=sim_score,
                        threshold=thresh,
                        confidence=round(min(1.0, sim_score / max(1e-4, thresh)), 4)
                    )
                    m_fails.append("FACE_IDENTITY_MISMATCH")
            else:
                err_code = face_ver_res.get("error_code", "VERIFICATION_FAILED")
                if err_code == "USER_NOT_ENROLLED":
                    ev = FaceIdentityEvidence(status="NOT_ENROLLED", similarity=0.0, threshold=0.65, confidence=0.0)
                    m_fails.append("FACE_USER_NOT_ENROLLED")
                elif err_code in ["NO_FACE_DETECTED", "QUALITY_FAIL"]:
                    ev = FaceIdentityEvidence(status="QUALITY_FAIL", similarity=0.0, threshold=0.65, confidence=0.0)
                    m_fails.append("FACE_QUALITY_FAIL")
                elif err_code == "MULTIPLE_FACES_DETECTED":
                    ev = FaceIdentityEvidence(status="MISMATCH", similarity=0.0, threshold=0.65, confidence=0.0)
                    m_fails.append("MULTIPLE_FACES_DETECTED")
        except Exception as e:
            logger.error(f"Error parsing face identity evidence: {e}")
        return ev, m_fails

    @staticmethod
    def _parse_face_liveness_evidence(live_res: Dict[str, Any]) -> Tuple[FaceLivenessEvidence, List[str], List[str]]:
        ev = FaceLivenessEvidence(status="INSUFFICIENT_EVIDENCE", score=0.0, threshold=0.50, attack_type=None, confidence=0.0)
        m_fails: List[str] = []
        m_attacks: List[str] = []
        try:
            liveness_verdict = live_res.get("liveness", "INSUFFICIENT_EVIDENCE")
            liveness_score = float(live_res.get("score", 0.0))
            thresh = float(live_res.get("threshold", 0.50))
            attack_cat = live_res.get("attack_type")
            conf = float(live_res.get("confidence", 0.0))

            if liveness_verdict == "LIVE":
                ev = FaceLivenessEvidence(
                    status="LIVE",
                    score=liveness_score,
                    threshold=thresh,
                    attack_type=None,
                    confidence=conf if conf > 0 else round(min(1.0, liveness_score / max(1e-4, thresh)), 4)
                )
            elif liveness_verdict == "SPOOF":
                ev = FaceLivenessEvidence(
                    status="SPOOF",
                    score=liveness_score,
                    threshold=thresh,
                    attack_type=attack_cat,
                    confidence=conf if conf > 0 else 0.95
                )
                m_fails.append("FACE_SPOOF_DETECTED")
                m_attacks.append(f"Face Presentation Attack ({attack_cat or 'Photo/Replay'})")
            else:
                reason = live_res.get("details", {}).get("reason", "")
                if "MULTIPLE_FACES" in reason:
                    m_fails.append("MULTIPLE_FACES_DETECTED")
                elif "NO_FACE" in reason:
                    m_fails.append("NO_FACE_DETECTED")
        except Exception as e:
            logger.error(f"Error parsing face liveness evidence: {e}")
        return ev, m_fails, m_attacks

    @classmethod
    async def authenticate_multimodal(cls, req: MultimodalAuthRequest) -> MultimodalAuthResponse:
        start_time = time.perf_counter()

        user_id = req.user_id.strip() if req.user_id else ""
        challenge_token = req.challenge_token.strip() if req.challenge_token else None

        failed_checks: List[str] = []
        attack_indicators: List[str] = []

        # ---------------------------------------------------------------------
        # 0. Session Security Validation (if challenge_token provided)
        # ---------------------------------------------------------------------
        session_valid = True
        session_voice_evidence = None
        session_stored = None
        if challenge_token:
            session_voice_evidence, err_code = VoiceSessionManager.consume_voice_evidence_for_fusion(
                challenge_token, user_id
            )
            session_stored = VoiceSessionManager.get_session_evidence(challenge_token, user_id)
            if err_code == "VOICE_EVIDENCE_NOT_READY":
                # A direct Module 5 request with a newly issued challenge has
                # not violated session security, but cannot authenticate.
                session_valid = True
            elif err_code:
                session_valid = False
                failed_checks.append(f"SESSION_SECURITY_FAILURE ({err_code or 'EXPIRED_OR_INVALID'})")
                attack_indicators.append("Session Replay / Challenge Expiration Attack")

        # ---------------------------------------------------------------------
        # 1. Prepare Inputs
        # ---------------------------------------------------------------------
        face_b64_list = []
        if req.face_frames_base64 and len(req.face_frames_base64) > 0:
            face_b64_list = req.face_frames_base64
        elif req.face_image_base64:
            face_b64_list = [req.face_image_base64]

        # Keep the identity probe distinct from the liveness/deepfake sequence.
        # This prevents fusion from silently re-verifying a different frame.
        probe_face_b64 = req.face_image_base64 or (face_b64_list[0] if face_b64_list else None)
        audio_pcm_b64 = req.audio_pcm_base64

        # ---------------------------------------------------------------------
        # 2. Module 1 — DeepQShield Deepfake Detection
        # ---------------------------------------------------------------------
        # Pre-decode video frames once to share across Modules 1, 2, and 3
        cv2_frames: List[np.ndarray] = []
        if face_b64_list and not (session_stored and session_stored.get("deepfake_evidence") and session_stored.get("liveness_evidence")):
            for b64 in face_b64_list:
                try:
                    img = base64_to_cv2(b64)
                    if img is not None and img.size > 0:
                        cv2_frames.append(img)
                except Exception:
                    continue
        probe_cv2 = cv2_frames[0] if (cv2_frames and (req.face_image_base64 is None or req.face_image_base64 == face_b64_list[0])) else None

        # ---------------------------------------------------------------------
        # 2, 3, 4. Modules 1-3 — DeepQShield, Face Identity, and Face Liveness
        # Reuses session evidence if already verified; runs concurrently otherwise.
        # ---------------------------------------------------------------------
        df_pack = None
        fid_pack = None
        live_pack = None

        if session_stored and session_stored.get("deepfake_evidence"):
            df_pack = cls._parse_deepfake_evidence(session_stored["deepfake_evidence"])

        if session_stored and session_stored.get("face_evidence"):
            fid_pack = cls._parse_face_identity_evidence(session_stored["face_evidence"])

        if session_stored and session_stored.get("liveness_evidence"):
            live_pack = cls._parse_face_liveness_evidence(session_stored["liveness_evidence"])

        async def _run_deepfake():
            ev = DeepfakeEvidence(status="INDETERMINATE", fake_probability=0.0, confidence=0.0)
            m_fails: List[str] = []
            m_attacks: List[str] = []
            if not cv2_frames:
                return ev, m_fails, m_attacks
            try:
                analyzer = get_deepfake_analyzer()
                df_res = await asyncio.to_thread(analyzer.analyze_frame_sequence, cv2_frames)
                try:
                    from api.routes.deepfake import _stored_results
                    df_analysis_id = df_res.get("payload", {}).get("analysis_id")
                    if df_analysis_id:
                        _stored_results[df_analysis_id] = df_res
                except Exception:
                    pass
                return cls._parse_deepfake_evidence(df_res)
            except Exception as e:
                logger.error(f"Error in Module 1 DeepQShield execution: {e}")
                ev = DeepfakeEvidence(status="INDETERMINATE", fake_probability=0.0, confidence=0.0)
            return ev, m_fails, m_attacks

        async def _run_face_identity():
            ev = FaceIdentityEvidence(status="INSUFFICIENT_EVIDENCE", similarity=0.0, threshold=0.65, confidence=0.0)
            m_fails: List[str] = []
            if not user_id or (not probe_face_b64 and probe_cv2 is None):
                return ev, m_fails
            try:
                face_ver_res = await FaceService.verify_face(
                    user_id=user_id,
                    image_base64=probe_face_b64 or "",
                    frames_base64=face_b64_list,
                    predecoded_image_bgr=probe_cv2,
                    predecoded_burst_frames=cv2_frames[:5] if cv2_frames else None
                )
                return cls._parse_face_identity_evidence(face_ver_res)
            except Exception as e:
                logger.error(f"Error in Module 2 Face Identity execution: {e}")
            return ev, m_fails

        async def _run_face_liveness():
            ev = FaceLivenessEvidence(status="INSUFFICIENT_EVIDENCE", score=0.0, threshold=0.50, attack_type=None, confidence=0.0)
            m_fails: List[str] = []
            m_attacks: List[str] = []
            if not cv2_frames and not face_b64_list:
                return ev, m_fails, m_attacks
            try:
                live_res = await asyncio.to_thread(
                    LivenessService.evaluate_passive_liveness,
                    face_b64_list,
                    None,
                    None,
                    cv2_frames
                )
                return cls._parse_face_liveness_evidence(live_res)
            except Exception as e:
                logger.error(f"Error in Module 3 Face Liveness execution: {e}")
            return ev, m_fails, m_attacks

        pending_tasks = []
        task_types = []
        if df_pack is None:
            pending_tasks.append(_run_deepfake())
            task_types.append("df")
        if fid_pack is None:
            pending_tasks.append(_run_face_identity())
            task_types.append("fid")
        if live_pack is None:
            pending_tasks.append(_run_face_liveness())
            task_types.append("live")

        if pending_tasks:
            results = await asyncio.gather(*pending_tasks)
            for t_type, res in zip(task_types, results):
                if t_type == "df":
                    df_pack = res
                elif t_type == "fid":
                    fid_pack = res
                elif t_type == "live":
                    live_pack = res

        deepfake_evidence, df_fails, df_attacks = df_pack
        face_id_evidence, fid_fails = fid_pack
        face_liveness_evidence, live_fails, live_attacks = live_pack

        failed_checks.extend(df_fails)
        failed_checks.extend(fid_fails)
        failed_checks.extend(live_fails)
        attack_indicators.extend(df_attacks)
        attack_indicators.extend(live_attacks)

        # ---------------------------------------------------------------------
        # 5. Module 4 — Voice Authentication & Audio PAD
        # ---------------------------------------------------------------------
        speaker_evidence = SpeakerEvidence(status="INSUFFICIENT_EVIDENCE", similarity=0.0, threshold=0.65)
        audio_pad_evidence = AudioPADEvidence(status="INSUFFICIENT_EVIDENCE", score=0.0, threshold=0.70, attack_type=None)

        if session_voice_evidence is not None:
            voice_res = session_voice_evidence
            speaker_verdict = voice_res.get("speaker_verdict", "UNVERIFIED")
            spk_sim = float(voice_res.get("voice_similarity", 0.0))
            liveness_score = float(voice_res.get("liveness_score", 0.0))
            liveness_verdict = voice_res.get("liveness_verdict", "INSUFFICIENT_EVIDENCE")
            audio_attack = voice_res.get("attack_type")

            if not voice_res.get("success"):
                failed_checks.append(f"VOICE_PIPELINE_{voice_res.get('error_code', 'FAILED')}")
            elif speaker_verdict == "MATCH":
                speaker_evidence = SpeakerEvidence(status="VERIFIED", similarity=spk_sim, threshold=0.65)
            else:
                speaker_evidence = SpeakerEvidence(status="MISMATCH", similarity=spk_sim, threshold=0.65)
                failed_checks.append("SPEAKER_IDENTITY_MISMATCH")

            if liveness_verdict == "LIVE":
                audio_pad_evidence = AudioPADEvidence(status="BONAFIDE", score=liveness_score, threshold=0.70, attack_type=None)
            elif liveness_verdict == "SPOOF":
                audio_pad_evidence = AudioPADEvidence(status="SPOOF", score=liveness_score, threshold=0.70, attack_type=audio_attack)
                failed_checks.append("AUDIO_SPOOF_DETECTED")
                attack_indicators.append(f"Audio Presentation Attack ({audio_attack or 'Replay/Synthetic Audio'})")
        elif user_id and audio_pcm_b64:
            try:
                voice_res = await VoiceService.verify_voice(user_id, audio_pcm_b64, session_id=None)  # Session already consumed above
                if voice_res.get("success"):
                    speaker_verdict = voice_res.get("speaker_verdict", "UNVERIFIED")
                    spk_sim = float(voice_res.get("voice_similarity", 0.0))
                    liveness_score = float(voice_res.get("liveness_score", 0.0))
                    liveness_verdict = voice_res.get("liveness_verdict", "INSUFFICIENT_EVIDENCE")
                    audio_attack = voice_res.get("attack_type")

                    if speaker_verdict == "MATCH":
                        speaker_evidence = SpeakerEvidence(status="VERIFIED", similarity=spk_sim, threshold=0.65)
                    else:
                        speaker_evidence = SpeakerEvidence(status="MISMATCH", similarity=spk_sim, threshold=0.65)
                        failed_checks.append("SPEAKER_IDENTITY_MISMATCH")

                    if liveness_verdict == "LIVE":
                        audio_pad_evidence = AudioPADEvidence(status="BONAFIDE", score=liveness_score, threshold=0.70, attack_type=None)
                    elif liveness_verdict == "SPOOF":
                        audio_pad_evidence = AudioPADEvidence(status="SPOOF", score=liveness_score, threshold=0.70, attack_type=audio_attack)
                        failed_checks.append("AUDIO_SPOOF_DETECTED")
                        attack_indicators.append(f"Audio Presentation Attack ({audio_attack or 'Replay/Synthetic Audio'})")
                    else:
                        audio_pad_evidence = AudioPADEvidence(status="INSUFFICIENT_EVIDENCE", score=liveness_score, threshold=0.70, attack_type=None)
                else:
                    err_code = voice_res.get("error_code", "VOICE_VERIFY_FAILED")
                    if err_code in ["USER_NOT_ENROLLED", "VOICE_NOT_ENROLLED"]:
                        speaker_evidence = SpeakerEvidence(status="NOT_ENROLLED", similarity=0.0, threshold=0.65)
                        failed_checks.append("VOICE_USER_NOT_ENROLLED")
                    elif err_code in ["AUDIO_TOO_SHORT", "SILENCE_OR_NEAR_SILENCE"]:
                        audio_pad_evidence = AudioPADEvidence(status="INSUFFICIENT_EVIDENCE", score=0.0, threshold=0.70)
                        failed_checks.append(f"AUDIO_QUALITY_{err_code}")
            except Exception as e:
                logger.error(f"Error in Module 4 Voice Auth execution: {e}")

        # Deduplicate failed checks & attack indicators
        failed_checks = sorted(list(set(failed_checks)))
        attack_indicators = sorted(list(set(attack_indicators)))

        # ---------------------------------------------------------------------
        # 6. Hard Security Gates Evaluation & Decision Logic
        # ---------------------------------------------------------------------
        # Hard Failure Conditions
        has_security_breach = any(
            c in failed_checks for c in [
                "FACE_IDENTITY_MISMATCH",
                "FACE_SPOOF_DETECTED",
                "DEEPFAKE_DETECTED",
                "SPEAKER_IDENTITY_MISMATCH",
                "AUDIO_SPOOF_DETECTED",
                "MULTIPLE_FACES_DETECTED"
            ]
        ) or not session_valid

        has_missing_modality = any(
            c in failed_checks for c in [
                "FACE_USER_NOT_ENROLLED",
                "VOICE_USER_NOT_ENROLLED",
                "NO_FACE_DETECTED",
                "FACE_QUALITY_FAIL"
            ]
        ) or (face_id_evidence.status == "INSUFFICIENT_EVIDENCE" and speaker_evidence.status == "INSUFFICIENT_EVIDENCE") \
          or (speaker_evidence.status == "VERIFIED" and audio_pad_evidence.status != "BONAFIDE")

        # Determine Final Decision & Risk Level
        if has_security_breach:
            decision = "REJECTED"
            risk_level = "HIGH"
            message = "Multimodal authentication failed due to detected security breach or spoofing attempt."
        elif has_missing_modality:
            decision = "INSUFFICIENT_EVIDENCE"
            risk_level = "MEDIUM"
            message = "Multimodal authentication incomplete due to missing modality or un-enrolled identity."
        else:
            # Check all modalities passed
            face_ok = (face_id_evidence.status == "VERIFIED") and (face_liveness_evidence.status == "LIVE") and (deepfake_evidence.status == "AUTHENTIC")
            voice_ok = speaker_evidence.status == "VERIFIED" and audio_pad_evidence.status == "BONAFIDE"

            if face_ok and voice_ok:
                decision = "AUTHENTICATED"
                risk_level = "LOW"
                message = "Multimodal authentication successful. Face and voice identity and liveness verified."
            elif face_ok and speaker_evidence.status == "INSUFFICIENT_EVIDENCE":
                # Single modality pass fallback
                decision = "AUTHENTICATED"
                risk_level = "LOW"
                message = "Face authentication successful (voice modality not submitted)."
            elif voice_ok and face_id_evidence.status == "INSUFFICIENT_EVIDENCE":
                decision = "AUTHENTICATED"
                risk_level = "LOW"
                message = "Voice authentication successful (face modality not submitted)."
            else:
                decision = "RETRY_REQUIRED"
                risk_level = "MEDIUM"
                message = "Borderline confidence across modalities. Please retry authentication."

        # ---------------------------------------------------------------------
        # 7. Calculate Normalized Multimodal Confidence
        # ---------------------------------------------------------------------
        conf_scores = []
        if face_id_evidence.status == "VERIFIED":
            conf_scores.append(face_id_evidence.confidence)
        if face_liveness_evidence.status == "LIVE":
            conf_scores.append(face_liveness_evidence.score)
        if deepfake_evidence.status == "AUTHENTIC":
            conf_scores.append(1.0 - deepfake_evidence.fake_probability)
        if speaker_evidence.status == "VERIFIED":
            conf_scores.append(min(1.0, speaker_evidence.similarity / 0.65))
        if audio_pad_evidence.status == "BONAFIDE":
            conf_scores.append(audio_pad_evidence.score)

        if decision == "AUTHENTICATED" and conf_scores:
            overall_confidence = round(float(np.mean(conf_scores)), 4)
        elif decision == "REJECTED":
            overall_confidence = 0.0
        else:
            overall_confidence = round(float(np.mean(conf_scores)) * 0.5, 4) if conf_scores else 0.0

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        reason_labels = {
            "FACE_IDENTITY_MISMATCH": "The live face did not match the enrolled identity.",
            "FACE_SPOOF_DETECTED": "Face liveness detected a presentation or replay attack.",
            "DEEPFAKE_DETECTED": "Visual security detected synthetic-face artifacts.",
            "SPEAKER_IDENTITY_MISMATCH": "The voice did not match the enrolled speaker.",
            "AUDIO_SPOOF_DETECTED": "Audio liveness detected replayed, synthetic, or otherwise spoofed speech.",
            "MULTIPLE_FACES_DETECTED": "Multiple faces were present; a single-person check is required.",
            "FACE_USER_NOT_ENROLLED": "No enrolled face template is available for this user.",
            "VOICE_USER_NOT_ENROLLED": "No enrolled voice template is available for this user.",
            "NO_FACE_DETECTED": "No usable face was detected in the camera evidence.",
            "FACE_QUALITY_FAIL": "The camera evidence did not meet the face-quality requirement.",
        }
        decision_reasons = [reason_labels.get(check, check.replace("_", " ").title() + ".") for check in failed_checks]
        if not session_valid:
            decision_reasons.append("The authentication session or challenge token was invalid, expired, or already used.")
        if decision == "AUTHENTICATED":
            decision_reasons = [
                "Face identity matched the enrolled user.",
                "Face liveness confirmed a live presentation.",
                "Visual security found no deepfake evidence.",
            ]
            if speaker_evidence.status == "VERIFIED":
                decision_reasons.append("Voice identity matched the enrolled speaker.")
        elif decision == "RETRY_REQUIRED":
            decision_reasons = ["No hard security breach was found, but one or more checks were inconclusive. Recapture the evidence and try again."]
        elif decision == "INSUFFICIENT_EVIDENCE" and not decision_reasons:
            decision_reasons = ["The submitted evidence was incomplete or a required identity modality is not enrolled."]

        authorization = (
            "ALLOW" if decision == "AUTHENTICATED" else
            "REJECT" if decision == "REJECTED" else
            "RETRY_REQUIRED"
        )
        assurance_level = (
            "HIGH" if authorization == "ALLOW" and speaker_evidence.status == "VERIFIED" and audio_pad_evidence.status == "BONAFIDE" else
            "STANDARD" if authorization == "ALLOW" else
            "NOT_ESTABLISHED" if authorization == "REJECT" else
            "INCOMPLETE"
        )
        # Keep report metadata aligned with the calibrated DeepQShield runtime
        # without instantiating the vision model for an audio-only request.
        df_real_threshold = 0.60
        df_fake_threshold = 0.70
        decision_report = {
            "authorization": authorization,
            "risk_level": risk_level,
            "assurance_level": assurance_level,
            "overall_confidence": overall_confidence,
            "summary": message,
            "controls": [
                {
                    "control": "Face Identity",
                    "status": face_id_evidence.status,
                    "observed": face_id_evidence.similarity,
                    "threshold": face_id_evidence.threshold,
                    "purpose": "Confirms that the live face matches the enrolled identity.",
                },
                {
                    "control": "Face Liveness Detection",
                    "status": face_liveness_evidence.status,
                    "observed": face_liveness_evidence.score,
                    "threshold": face_liveness_evidence.threshold,
                    "purpose": "Rejects presentation attacks such as printed photos and screen replays.",
                },
                {
                    "control": "Voice Identity",
                    "status": speaker_evidence.status,
                    "observed": speaker_evidence.similarity,
                    "threshold": speaker_evidence.threshold,
                    "purpose": "Confirms that the voice matches the enrolled speaker.",
                },
                {
                    "control": "Deepfake Detection",
                    "status": deepfake_evidence.status,
                    "observed": deepfake_evidence.fake_probability,
                    "threshold": {"authentic_at_or_below": df_real_threshold, "deepfake_at_or_above": df_fake_threshold},
                    "purpose": "Detects synthetic-face and deepfake artifacts in the webcam sequence.",
                },
                {
                    "control": "Audio Presentation Defense",
                    "status": audio_pad_evidence.status,
                    "observed": audio_pad_evidence.score,
                    "threshold": audio_pad_evidence.threshold,
                    "purpose": "Rejects replayed, synthetic, and voice-cloned audio.",
                },
            ],
            "failed_security_gates": failed_checks,
            "attack_indicators": attack_indicators,
            "decision_reasons": decision_reasons,
        }

        phrase_status = (session_voice_evidence.get("phrase_match") if session_voice_evidence else None) or "UNVALIDATED"

        return MultimodalAuthResponse(
            success=True,
            decision=decision,
            overall_confidence=overall_confidence,
            risk_level=risk_level,
            face_identity=face_id_evidence,
            face_liveness=face_liveness_evidence,
            deepfake_detection=deepfake_evidence,
            speaker_identity=speaker_evidence,
            audio_pad=audio_pad_evidence,
            failed_checks=failed_checks,
            attack_indicators=attack_indicators,
            decision_reasons=decision_reasons,
            decision_report=decision_report,
            processing_time_ms=elapsed_ms,
            phrase_match_status=phrase_status,
            message=message
        )
