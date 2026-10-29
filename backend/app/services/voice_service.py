"""
Production Voice Service for Nivora (Module 4).
Provides speaker template enrollment, AEAD template decryption, real-time speaker verification + anti-spoofing,
and short-lived anti-replay session management.
"""

import time
import base64
import os
import sys
import uuid
import secrets
import random
from typing import Dict, Any, Optional, List, Tuple
from app.database.repositories import VoiceTemplateRepository, UserRepository
from app.utils.logging import logger

# Add project root to sys.path for security module
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine
from security.biometric_crypto import decrypt_template


# Controlled English vocabulary for random 3-word challenge phrase generation
# Selected for high phonetic clarity, distinct syllabic boundaries, and ease of pronunciation
PHONETIC_WORDS = [
    "apple", "river", "green", "mango", "table", "blue", "lemon", "paper",
    "sunny", "tiger", "water", "happy", "garden", "yellow", "window", "orange",
    "rabbit", "silver", "golden", "cloud", "morning", "flower", "bright",
    "simple", "honest", "winter", "summer", "ocean", "forest", "candle"
]

import hashlib

# In-memory store for active authentication sessions
_active_voice_sessions: Dict[str, Dict[str, Any]] = {}

# In-memory store for active multi-sample enrollment sessions
_active_enrollment_sessions: Dict[str, Dict[str, Any]] = {}


class VoiceSessionManager:
    """
    Manages short-lived, one-time-use cryptographic authentication sessions & challenge phrases.
    """
    @staticmethod
    def create_session(user_id: Optional[str] = None, purpose: str = "VERIFY", ttl_seconds: int = 300) -> Dict[str, Any]:
        """
        Creates a new random authentication session with a server-generated 3-word challenge phrase.
        """
        session_id = f"vsess_{uuid.uuid4().hex[:16]}"
        chosen_words = random.sample(PHONETIC_WORDS, 3)
        challenge_phrase = f"{chosen_words[0].capitalize()} {chosen_words[1]} {chosen_words[2]}"
        
        now = time.time()
        session_doc = {
            "session_id": session_id,
            "user_id": user_id,
            "purpose": purpose,
            "challenge_phrase": challenge_phrase,
            "created_at": now,
            "expires_at": now + ttl_seconds,
            "expires_in": ttl_seconds,
            "consumed": False
        }
        _active_voice_sessions[session_id] = session_doc
        logger.info(f"Created voice session {session_id} with challenge '{challenge_phrase}'")
        return session_doc

    @staticmethod
    def validate_and_consume_session(session_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Validates session existence, expiration, and consumed status. Marks session consumed if valid.
        Returns: (session_doc, error_code)
        """
        if not session_id or session_id not in _active_voice_sessions:
            return None, "SESSION_NOT_FOUND"

        session = _active_voice_sessions[session_id]
        if session.get("consumed", False):
            return None, "SESSION_ALREADY_USED"

        if time.time() > session.get("expires_at", 0):
            return None, "SESSION_EXPIRED"

        # Mark consumed immediately (one-time consumption)
        session["consumed"] = True
        return session, None

    @staticmethod
    def store_face_evidence(session_id: str, user_id: str, result: Dict[str, Any]) -> None:
        """Attach server-generated face identity evidence to session."""
        session = _active_voice_sessions.get(session_id)
        if not session:
            return
        if not session.get("user_id"):
            session["user_id"] = user_id
        session["face_evidence"] = result

    @staticmethod
    def store_liveness_evidence(session_id: str, user_id: str, result: Dict[str, Any]) -> None:
        """Attach server-generated face liveness evidence to session."""
        session = _active_voice_sessions.get(session_id)
        if not session:
            return
        if not session.get("user_id"):
            session["user_id"] = user_id
        session["liveness_evidence"] = result

    @staticmethod
    def store_deepfake_evidence(session_id: str, user_id: str, result: Dict[str, Any]) -> None:
        """Attach server-generated deepfake analysis evidence to session."""
        session = _active_voice_sessions.get(session_id)
        if not session:
            return
        if not session.get("user_id"):
            session["user_id"] = user_id
        session["deepfake_evidence"] = result

    @staticmethod
    def get_session_evidence(session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve pre-verified session evidence without consuming voice token."""
        session = _active_voice_sessions.get(session_id)
        if not session:
            return None
        if session.get("user_id") and session.get("user_id") != user_id:
            return None
        return session

    @staticmethod
    def store_voice_evidence(session_id: str, user_id: str, result: Dict[str, Any]) -> None:
        """Attach the server-generated result of the one permitted voice probe.

        Module 5 consumes this evidence after the voice endpoint has consumed the
        challenge.  The browser never sends scores back to the fusion endpoint.
        """
        session = _active_voice_sessions.get(session_id)
        if not session or not session.get("consumed"):
            return
        if session.get("user_id") and session.get("user_id") != user_id:
            return
        session["voice_evidence"] = {
            key: result.get(key) for key in (
                "success", "error_code", "speaker_verdict", "voice_similarity",
                "liveness_verdict", "liveness_score", "attack_type", "final_verdict",
                "phrase_match", "phrase_match_score", "transcribed_text", "phrase_match_reason",
                "timings", "processing_time_ms", "voice_spoof_probability",
                "pad_pretrained", "pad_policy", "pad_score_semantic", "tau_speaker", "tau_pad",
            )
        }

    @staticmethod
    def consume_voice_evidence_for_fusion(session_id: str, user_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Return a session-bound voice result exactly once for final fusion."""
        session = _active_voice_sessions.get(session_id)
        if not session:
            return None, "SESSION_NOT_FOUND"
        if session.get("user_id") and session.get("user_id") != user_id:
            return None, "SESSION_USER_MISMATCH"
        if time.time() > session.get("expires_at", 0):
            return None, "SESSION_EXPIRED"
        if not session.get("consumed"):
            return None, "VOICE_EVIDENCE_NOT_READY"
        if session.get("fusion_consumed"):
            return None, "SESSION_ALREADY_USED"
        evidence = session.get("voice_evidence")
        if not evidence:
            # A consumed challenge without a server-produced voice result is
            # replayed or incomplete; it is never reusable by fusion.
            return None, "SESSION_ALREADY_USED"
        session["fusion_consumed"] = True
        return evidence, None


class VoiceService:
    """
    Voice Authentication Service managing AEAD encrypted speaker templates, sessions & real-time PAD.
    """

    @staticmethod
    def decode_audio_payload(audio_b64: str) -> Optional[bytes]:
        if not audio_b64:
            return None
        try:
            if "," in audio_b64:
                audio_b64 = audio_b64.split(",")[1]
            return base64.b64decode(audio_b64)
        except Exception as e:
            logger.error(f"Failed to decode base64 audio payload: {e}")
            return None

    @classmethod
    async def enroll_voice(cls, user_id: str, audio_samples_base64: List[str]) -> Dict[str, Any]:
        """
        Enrolls a speaker identity template from multiple independent speech samples (3 samples minimum).
        Extracts 192-d ECAPA-TDNN embeddings, computes centroid, AEAD encrypts, and persists template.
        Raw audio is processed in memory and discarded (never permanently stored).
        """
        start_time = time.time()
        if not user_id or not user_id.strip():
            return {
                "success": False,
                "error_code": "INVALID_USER_ID",
                "message": "A valid user_id must be provided for voice enrollment."
            }

        if not audio_samples_base64 or len(audio_samples_base64) < 3:
            return {
                "success": False,
                "error_code": "INSUFFICIENT_SAMPLES",
                "message": f"At least 3 valid voice samples are required for enrollment (received {len(audio_samples_base64) if audio_samples_base64 else 0})."
            }

        # Verify duplicate recordings are rejected
        sample_hashes = set()
        for idx, audio_b64 in enumerate(audio_samples_base64):
            raw_check = cls.decode_audio_payload(audio_b64)
            if raw_check:
                h = hashlib.sha256(raw_check).hexdigest()
                if h in sample_hashes:
                    return {
                        "success": False,
                        "error_code": "DUPLICATE_AUDIO_SAMPLE",
                        "message": f"Sample {idx + 1} is an identical duplicate. Each enrollment sample must be a fresh live recording."
                    }
                sample_hashes.add(h)

        valid_embeddings = []
        sample_reports = []

        for idx, audio_b64 in enumerate(audio_samples_base64):
            raw_bytes = cls.decode_audio_payload(audio_b64)
            if not raw_bytes:
                sample_reports.append({
                    "sample_index": idx,
                    "pass": False,
                    "error_code": "DECODE_FAILED",
                    "message": "Base64 audio payload decoding failed."
                })
                continue

            res = voice_auth_engine.process_voice_pipeline(raw_bytes)
            # Discard raw bytes immediately from memory
            del raw_bytes

            pass_status = res["success"] and res["audio_quality"].get("pass", False)
            sample_reports.append({
                "sample_index": idx,
                "pass": pass_status,
                "error_code": res.get("error_code"),
                "audio_quality": res.get("audio_quality", {})
            })

            if pass_status and res.get("probe_embedding") is not None:
                emb = res["probe_embedding"]
                if len(emb) != 192:
                    logger.warning(f"[VOICE ENROLLMENT] Skipping sample {idx}: embedding dim {len(emb)} != 192")
                    continue  # skip malformed embedding rather than crash
                valid_embeddings.append(emb)

        if len(valid_embeddings) < 3:
            return {
                "success": False,
                "error_code": "ENROLLMENT_QUALITY_FAILED",
                "message": f"At least 3 valid voice samples required. Valid samples obtained: {len(valid_embeddings)}/{len(audio_samples_base64)}.",
                "samples_processed": len(audio_samples_base64),
                "samples_valid": len(valid_embeddings),
                "sample_reports": sample_reports
            }

        # Aggregate valid sample embeddings into normalized 192-d centroid template vector
        try:
            template_embedding = voice_auth_engine.aggregate_embeddings(valid_embeddings)
            assert len(template_embedding) == 192, f"Aggregated centroid shape must be 192, got {len(template_embedding)}"
        except Exception as e:
            return {
                "success": False,
                "error_code": "AGGREGATION_ERROR",
                "message": f"Failed to aggregate voice template embedding: {str(e)}"
            }

        # Persist template in VoiceTemplateRepository (AEAD encrypted AES-256-GCM)
        template_record = {
            "model_name": "SpeechBrain_ECAPA_TDNN",
            "model_version": "SpeechBrain_ECAPA_VoxCeleb_v2",
            "embedding_dim": len(template_embedding),
            "embedding": template_embedding,
            "sample_count": len(valid_embeddings),
            "total_submitted": len(audio_samples_base64),
            "status": "ACTIVE"
        }
        template_id = await VoiceTemplateRepository.save_template(user_id, template_record)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "success": True,
            "status": "VOICE_ENROLLMENT_COMPLETE",
            "user_id": user_id,
            "template_id": template_id,
            "samples_processed": len(audio_samples_base64),
            "samples_valid": len(valid_embeddings),
            "model_name": "SpeechBrain_ECAPA_TDNN",
            "embedding_dim": 192,
            "processing_time_ms": elapsed_ms,
            "message": f"Voice profile successfully enrolled for user '{user_id}' ({len(valid_embeddings)} valid samples).",
            "sample_reports": sample_reports
        }

    @classmethod
    async def enroll_voice_sample(
        cls,
        user_id: str,
        audio_b64: str,
        sample_index: int,
        total_samples: int = 3,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Incremental step-by-step voice enrollment handler.
        Holds intermediate 192-d embeddings in memory (raw audio discarded immediately).
        Upon receiving total_samples (default 3), generates centroid, encrypts, and commits template.
        """
        if not user_id or not user_id.strip():
            return {"success": False, "error_code": "INVALID_USER_ID", "message": "Invalid user ID."}

        raw_bytes = cls.decode_audio_payload(audio_b64)
        if not raw_bytes:
            return {"success": False, "error_code": "VOICE_INVALID_AUDIO", "message": "Audio payload decode failed."}

        # Check hash against prior samples in current session
        session_key = f"{user_id}_enroll"
        if session_key not in _active_enrollment_sessions or sample_index == 1:
            _active_enrollment_sessions[session_key] = {
                "user_id": user_id,
                "embeddings": [],
                "hashes": set(),
                "created_at": time.time(),
                "total_samples": total_samples
            }

        sess = _active_enrollment_sessions[session_key]
        audio_hash = hashlib.sha256(raw_bytes).hexdigest()
        if audio_hash in sess["hashes"]:
            return {
                "success": False,
                "error_code": "DUPLICATE_AUDIO_SAMPLE",
                "message": "Duplicate recording detected. Please speak a fresh live phrase."
            }

        res = voice_auth_engine.process_voice_pipeline(raw_bytes)
        del raw_bytes  # Discard raw audio immediately

        if not res["success"] or not res["audio_quality"].get("pass", False):
            return {
                "success": False,
                "error_code": res.get("error_code", "VOICE_INVALID_AUDIO"),
                "message": res.get("message", "Audio quality check failed (too quiet, clipped, or silent)."),
                "audio_quality": res.get("audio_quality", {})
            }

        emb = res.get("probe_embedding")
        if not emb or len(emb) != 192:
            return {"success": False, "error_code": "EMBEDDING_FAILED", "message": "Speaker embedding extraction failed."}

        sess["embeddings"].append(emb)
        sess["hashes"].add(audio_hash)
        collected = len(sess["embeddings"])

        state_tag = f"VOICE_SAMPLE_{collected}"

        if collected < total_samples:
            return {
                "success": True,
                "status": state_tag,
                "sample_index": collected,
                "total_samples": total_samples,
                "message": f"Voice sample {collected} of {total_samples} captured successfully."
            }

        # Final sample received — compute centroid and persist template
        template_embedding = voice_auth_engine.aggregate_embeddings(sess["embeddings"])
        if len(template_embedding) != 192:
            return {
                "success": False,
                "error_code": "AGGREGATION_ERROR",
                "message": f"Template aggregation produced unexpected dimension {len(template_embedding)}, expected 192.",
            }

        template_record = {
            "model_name": "SpeechBrain_ECAPA_TDNN",
            "model_version": "SpeechBrain_ECAPA_VoxCeleb_v2",
            "embedding_dim": 192,
            "embedding": template_embedding,
            "sample_count": total_samples,
            "total_submitted": total_samples,
            "status": "ACTIVE"
        }
        template_id = await VoiceTemplateRepository.save_template(user_id, template_record)
        del _active_enrollment_sessions[session_key]

        return {
            "success": True,
            "status": "VOICE_ENROLLMENT_COMPLETE",
            "sample_index": total_samples,
            "total_samples": total_samples,
            "template_id": template_id,
            "enrolled": True,
            "message": f"Voice profile successfully enrolled for user '{user_id}' ({total_samples} samples)."
        }

    @classmethod
    async def verify_voice(cls, user_id: str, audio_base64: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Verifies probe audio recording against enrolled speaker template (AEAD decrypted in memory).
        Validates active session if provided.
        Returns distinct states: VOICE_NOT_ENROLLED, SPEAKER_MATCH, SPEAKER_MISMATCH, AUDIO_SPOOF, VOICE_INVALID_AUDIO.
        """
        start_time = time.time()
        if not user_id or not user_id.strip():
            return {
                "success": False,
                "error_code": "INVALID_USER_ID",
                "message": "A valid user_id must be provided for voice verification."
            }

        # Session validation if session_id provided
        session_doc = None
        if session_id:
            session_doc, err_code = VoiceSessionManager.validate_and_consume_session(session_id)
            if not session_doc:
                return {
                    "success": False,
                    "error_code": err_code or "INVALID_SESSION",
                    "status": "INVALID_SESSION",
                    "message": f"Voice session validation failed ({err_code}). Please initiate a new recording session."
                }

        # Fetch enrolled speaker template FIRST — distinguish missing enrollment from speaker mismatch
        template_doc = await VoiceTemplateRepository.get_template_by_user_id(user_id)
        user = await UserRepository.get_by_id(user_id) if not template_doc else None

        decrypted_embedding = None
        if template_doc:
            try:
                if "encrypted_template" in template_doc:
                    decrypted_embedding = decrypt_template(template_doc)
                elif "embedding" in template_doc:
                    decrypted_embedding = template_doc["embedding"]
            except Exception as e:
                logger.error(f"Voice template decryption error for user '{user_id}': {e}")
                return {
                    "success": False,
                    "error_code": "DECRYPTION_FAILED",
                    "status": "VOICE_PROCESSING_ERROR",
                    "message": f"AEAD Biometric Template Decryption Failed: {str(e)}"
                }
        elif user and user.get("voice_embedding") and len(user.get("voice_embedding")) == 192:
            decrypted_embedding = user["voice_embedding"]

        # Check template compatibility with official SpeechBrain ECAPA-TDNN VoxCeleb model
        if template_doc and template_doc.get("model_version") not in ["SpeechBrain_ECAPA_VoxCeleb_v2"]:
            logger.warning(f"[NIVORA VOICE] User '{user_id}' has incompatible voice template version: {template_doc.get('model_version')}.")
            return {
                "success": False,
                "error_code": "VOICE_TEMPLATE_INCOMPATIBLE",
                "status": "VOICE_TEMPLATE_INCOMPATIBLE",
                "speaker_verdict": "VOICE_NOT_ENROLLED",
                "liveness_verdict": "UNVERIFIED",
                "final_verdict": "REJECTED",
                "voice_similarity": 0.0,
                "message": "Enrolled voice profile was created with a legacy model version. Please re-enroll your voice in Voice Enrollment."
            }

        # Strict check: if no enrolled template exists, return distinct VOICE_NOT_ENROLLED state
        if decrypted_embedding is None or len(decrypted_embedding) != 192:
            logger.warning(f"[NIVORA VOICE] User '{user_id}' has no enrolled voice template.")
            return {
                "success": False,
                "error_code": "VOICE_NOT_ENROLLED",
                "status": "VOICE_NOT_ENROLLED",
                "speaker_verdict": "VOICE_NOT_ENROLLED",
                "liveness_verdict": "UNVERIFIED",
                "final_verdict": "REJECTED",
                "voice_similarity": 0.0,
                "message": "Voice profile not enrolled. Please complete voice enrollment."
            }

        audio_bytes = cls.decode_audio_payload(audio_base64)
        if not audio_bytes:
            return {
                "success": False,
                "error_code": "VOICE_INVALID_AUDIO",
                "status": "VOICE_INVALID_AUDIO",
                "message": "Failed to decode base64 audio payload."
            }

        # Run full pipeline via VoiceAuthEngine
        res = voice_auth_engine.process_voice_pipeline(audio_bytes, decrypted_embedding)

        # Discard decrypted embedding from memory immediately
        del decrypted_embedding

        # Map distinct domain states
        if not res["success"]:
            err_c = res.get("error_code", "VOICE_PROCESSING_ERROR")
            if err_c in ["AUDIO_TOO_SHORT", "SILENCE_OR_NEAR_SILENCE", "EXCESSIVE_AUDIO_CLIPPING", "EMPTY_OR_CORRUPT_PAYLOAD"]:
                res["status"] = "VOICE_INVALID_AUDIO"
            else:
                res["status"] = err_c
            res["speaker_verified"] = False
        else:
            if res.get("liveness_verdict") == "SPOOF":
                res["status"] = "AUDIO_SPOOF"
                res["speaker_verified"] = False
                res["message"] = "Audio could not be verified as genuine. Please record the displayed phrase again in a quiet environment."
            elif res.get("speaker_verdict") == "MATCH" and res.get("liveness_verdict") == "LIVE":
                res["status"] = "SPEAKER_MATCH"
                res["speaker_verified"] = True
                res["message"] = "Voice speaker identity and live acoustic liveness verified."
            elif res.get("speaker_verdict") == "MISMATCH":
                res["status"] = "SPEAKER_MISMATCH"
                res["speaker_verified"] = False
                res["message"] = "Voice speaker does not match enrolled biometric template."
            else:
                res["status"] = res.get("final_verdict", "UNVERIFIED")
                res["speaker_verified"] = False
                if not res.get("message"):
                    res["message"] = "Audio could not be verified as genuine. Please record the displayed phrase again in a quiet environment."

        # Phrase match evaluation via offline faster-whisper ASR with accent tolerance
        phrase_match = "UNVALIDATED"
        transcribed_text = ""
        phrase_match_score = 0.0
        phrase_match_reason = ""

        if session_doc:
            ch_phrase = session_doc.get("challenge_phrase")
            res["challenge_phrase"] = ch_phrase
            res["session_id"] = session_id
            
            decoded_sig = res.get("decoded_signal")
            if decoded_sig is not None and ch_phrase:
                try:
                    from app.services.speech_recognition_service import SpeechRecognitionService
                    phrase_res = SpeechRecognitionService.verify_challenge(
                        decoded_sig, ch_phrase, sample_rate=res.get("audio_quality", {}).get("sample_rate", 16000)
                    )
                    phrase_match = phrase_res.get("phrase_match", "UNVALIDATED")
                    phrase_match_score = phrase_res.get("phrase_match_score", 0.0)
                    transcribed_text = phrase_res.get("transcribed_text", "")
                    phrase_match_reason = phrase_res.get("reason", "")
                except Exception as asr_err:
                    logger.warning(f"ASR challenge verification failed: {asr_err}")
                    phrase_match = "INCONCLUSIVE"
                    phrase_match_reason = f"ASR engine unavailable: {asr_err}"

        res["phrase_match"] = phrase_match
        res["phrase_match_score"] = phrase_match_score
        res["transcribed_text"] = transcribed_text
        if phrase_match_reason:
            res["phrase_match_reason"] = phrase_match_reason
        res.pop("decoded_signal", None)

        # Enforce server-side policy on phrase challenge verification:
        # A spoken challenge phrase mismatch is an explicit anti-replay rejection.
        if phrase_match in ["MISMATCHED", "MISMATCH"]:
            res["final_verdict"] = "REJECTED"
            res["status"] = "PHRASE_MISMATCH"
            res["speaker_verified"] = False
            res["message"] = f"Spoken challenge phrase did not match required phrase '{ch_phrase}'."
        elif phrase_match == "MATCHED":
            if res.get("speaker_verdict") == "MATCH" and res.get("liveness_verdict") == "LIVE":
                res["final_verdict"] = "AUTHENTICATED"
                res["status"] = "SPEAKER_MATCH"
                res["speaker_verified"] = True
                res["message"] = "Voice speaker identity, live acoustic liveness, and challenge phrase verified."

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        res["user_id"] = user_id
        res["processing_time_ms"] = elapsed_ms

        # Preserve only server-computed scalar evidence for the immediately
        # following fusion call; never retain or expose biometric embeddings.
        if session_doc and session_id:
            VoiceSessionManager.store_voice_evidence(session_id, user_id, res)

        # SECURITY: Strip raw biometric embeddings from production API response
        res.pop("probe_embedding", None)
        res.pop("enrolled_embedding", None)
        # Model metadata can contain local checkpoint paths; the client only
        # receives the stable, user-safe model names exposed by diagnostics.
        res.pop("model_info", None)

        return res

    @staticmethod
    async def get_template_info(user_id: str) -> Dict[str, Any]:
        """
        Retrieves voice template status and metadata for user (never exposes raw embedding).
        """
        template_doc = await VoiceTemplateRepository.get_template_by_user_id(user_id)
        if not template_doc:
            return {
                "user_id": user_id,
                "enrolled": False,
                "template": None
            }

        return {
            "user_id": user_id,
            "enrolled": True,
            "template": {
                "template_id": template_doc.get("template_id"),
                "model_version": template_doc.get("model_version", "SpeechBrain_ECAPA_TDNN"),
                "template_version": template_doc.get("template_version", "1.0"),
                "embedding_dim": template_doc.get("embedding_dim", 192),
                "encrypted": "encrypted_template" in template_doc,
                "sample_count": template_doc.get("sample_count"),
                "created_at": template_doc.get("created_at"),
                "updated_at": template_doc.get("updated_at")
            }
        }

    @staticmethod
    async def remove_template(user_id: str) -> Dict[str, Any]:
        """
        Deletes enrolled voice template for specified user.
        """
        deleted = await VoiceTemplateRepository.delete_template(user_id)
        return {
            "success": deleted,
            "user_id": user_id,
            "message": f"Voice template deleted for user '{user_id}'." if deleted else f"No voice template found for user '{user_id}'."
        }
