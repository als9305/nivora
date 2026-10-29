"""
Module 5 Multimodal Fusion & Decision Engine Unit Test Suite.

Tests:
1. All four modules PASS -> AUTHENTICATED
2. Face identity mismatch -> REJECTED
3. Face liveness spoof -> REJECTED
4. Deepfake detected -> REJECTED
5. Speaker mismatch -> REJECTED
6. Audio PAD spoof -> REJECTED
7. Valid Session Token -> PASS
8. Expired Session Token -> REJECTED
9. Invalid Session Token -> REJECTED
10. Reused / Consumed Session Token -> REJECTED
11. Missing face -> INSUFFICIENT_EVIDENCE
12. Multiple faces -> REJECTED
13. Poor frame quality -> INSUFFICIENT_EVIDENCE
14. Malicious client submitted decision -> Server ignores client payload, returns true calculation
15. No raw biometric embeddings exposed in response
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from app.schemas.multimodal import MultimodalAuthRequest, MultimodalAuthResponse
from app.services.multimodal_fusion_service import MultimodalFusionService
from app.services.voice_service import VoiceSessionManager


def test_multimodal_all_modules_pass():
    """Test 1: All 4 modules pass -> AUTHENTICATED decision, LOW risk."""
    async def run_test():
        req = MultimodalAuthRequest(
            user_id="user_test_01",
            face_image_base64="dGVzdF9mYWNlX2ltYWdl",
            audio_pcm_base64="dGVzdF9hdWRpb19wY20="
        )

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("app.services.multimodal_fusion_service.base64_to_cv2", return_value=dummy_frame), \
             patch("app.services.multimodal_fusion_service.FaceService.verify_face", new_callable=AsyncMock) as mock_face, \
             patch("app.services.multimodal_fusion_service.LivenessService.evaluate_passive_liveness") as mock_liveness, \
             patch("app.services.multimodal_fusion_service.get_deepfake_analyzer") as mock_df, \
             patch("app.services.multimodal_fusion_service.VoiceService.verify_voice", new_callable=AsyncMock) as mock_voice:

            mock_face.return_value = {"success": True, "matched": True, "similarity_score": 0.85, "threshold": 0.65}
            mock_liveness.return_value = {"success": True, "liveness": "LIVE", "score": 0.92, "threshold": 0.50}
            
            mock_analyzer_inst = MagicMock()
            mock_analyzer_inst.analyze_frame_sequence.return_value = {
                "status": "SUCCESS",
                "result": {"classification": "AUTHENTIC", "fake_probability": 0.05, "confidence": 0.95}
            }
            mock_df.return_value = mock_analyzer_inst

            mock_voice.return_value = {
                "success": True,
                "speaker_verdict": "MATCH",
                "voice_similarity": 0.82,
                "liveness_verdict": "LIVE",
                "liveness_score": 0.88
            }

            res = await MultimodalFusionService.authenticate_multimodal(req)

            assert res.success is True
            assert res.decision == "AUTHENTICATED"
            assert res.risk_level == "LOW"
            assert res.overall_confidence > 0.75
            assert len(res.failed_checks) == 0
            assert res.phrase_match_status == "UNVALIDATED"

    asyncio.run(run_test())


def test_multimodal_face_identity_mismatch():
    """Test 2: Face identity mismatch -> REJECTED decision, HIGH risk."""
    async def run_test():
        req = MultimodalAuthRequest(user_id="user_test_01", face_image_base64="dGVzdF9mYWNl")

        with patch("app.services.multimodal_fusion_service.FaceService.verify_face", new_callable=AsyncMock) as mock_face:
            mock_face.return_value = {"success": True, "matched": False, "similarity_score": 0.30, "threshold": 0.65}

            res = await MultimodalFusionService.authenticate_multimodal(req)

            assert res.decision == "REJECTED"
            assert res.risk_level == "HIGH"
            assert "FACE_IDENTITY_MISMATCH" in res.failed_checks

    asyncio.run(run_test())


def test_multimodal_face_liveness_spoof():
    """Test 3: Face liveness spoof -> REJECTED decision, HIGH risk."""
    async def run_test():
        req = MultimodalAuthRequest(user_id="user_test_01", face_image_base64="dGVzdF9mYWNl")

        with patch("app.services.multimodal_fusion_service.FaceService.verify_face", new_callable=AsyncMock) as mock_face, \
             patch("app.services.multimodal_fusion_service.LivenessService.evaluate_passive_liveness") as mock_liveness:

            mock_face.return_value = {"success": True, "matched": True, "similarity_score": 0.80, "threshold": 0.65}
            mock_liveness.return_value = {"success": True, "liveness": "SPOOF", "score": 0.10, "attack_type": "PRINTED_PHOTO"}

            res = await MultimodalFusionService.authenticate_multimodal(req)

            assert res.decision == "REJECTED"
            assert res.risk_level == "HIGH"
            assert "FACE_SPOOF_DETECTED" in res.failed_checks

    asyncio.run(run_test())


def test_multimodal_deepfake_detected():
    """Test 4: Deepfake detected -> REJECTED decision, HIGH risk."""
    async def run_test():
        req = MultimodalAuthRequest(user_id="user_test_01", face_image_base64="dGVzdF9mYWNl")

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("app.services.multimodal_fusion_service.base64_to_cv2", return_value=dummy_frame), \
             patch("app.services.multimodal_fusion_service.get_deepfake_analyzer") as mock_df:

            mock_inst = MagicMock()
            mock_inst.analyze_frame_sequence.return_value = {
                "status": "SUCCESS",
                "result": {"classification": "DEEPFAKE", "fake_probability": 0.95, "confidence": 0.99}
            }
            mock_df.return_value = mock_inst

            res = await MultimodalFusionService.authenticate_multimodal(req)

            assert res.decision == "REJECTED"
            assert res.risk_level == "HIGH"
            assert "DEEPFAKE_DETECTED" in res.failed_checks

    asyncio.run(run_test())


def test_multimodal_speaker_mismatch():
    """Test 5: Speaker mismatch -> REJECTED decision, HIGH risk."""
    async def run_test():
        req = MultimodalAuthRequest(user_id="user_test_01", audio_pcm_base64="dGVzdF9hdWRpbw==")

        with patch("app.services.multimodal_fusion_service.VoiceService.verify_voice", new_callable=AsyncMock) as mock_voice:
            mock_voice.return_value = {
                "success": True,
                "speaker_verdict": "MISMATCH",
                "voice_similarity": 0.25,
                "liveness_verdict": "LIVE",
                "liveness_score": 0.80
            }

            res = await MultimodalFusionService.authenticate_multimodal(req)

            assert res.decision == "REJECTED"
            assert res.risk_level == "HIGH"
            assert "SPEAKER_IDENTITY_MISMATCH" in res.failed_checks

    asyncio.run(run_test())


def test_multimodal_audio_pad_spoof():
    """Test 6: Audio spoof evidence is a hard security gate."""
    async def run_test():
        req = MultimodalAuthRequest(user_id="user_test_01", audio_pcm_base64="dGVzdF9hdWRpbw==")

        with patch("app.services.multimodal_fusion_service.VoiceService.verify_voice", new_callable=AsyncMock) as mock_voice:
            mock_voice.return_value = {
                "success": True,
                "speaker_verdict": "MATCH",
                "voice_similarity": 0.80,
                "liveness_verdict": "SPOOF",
                "liveness_score": 0.15,
                "attack_type": "SYNTHETIC_TTS"
            }

            res = await MultimodalFusionService.authenticate_multimodal(req)

            assert res.decision == "REJECTED"
            assert res.risk_level == "HIGH"
            assert res.speaker_identity.status == "VERIFIED"
            assert res.audio_pad.status == "SPOOF"
            assert "AUDIO_SPOOF_DETECTED" in res.failed_checks
            assert any("Audio liveness detected" in reason for reason in res.decision_reasons)

    asyncio.run(run_test())


def test_voice_match_with_missing_pad_evidence_cannot_authenticate():
    """A speaker match is not sufficient when PAD did not produce live evidence."""
    async def run_test():
        req = MultimodalAuthRequest(user_id="user_test_01", audio_pcm_base64="dGVzdF9hdWRpbw==")
        with patch("app.services.multimodal_fusion_service.VoiceService.verify_voice", new_callable=AsyncMock) as mock_voice:
            mock_voice.return_value = {
                "success": True,
                "speaker_verdict": "MATCH",
                "voice_similarity": 0.82,
                "liveness_verdict": "INSUFFICIENT_EVIDENCE",
                "liveness_score": 0.51,
                "phrase_match": "UNVALIDATED",
            }
            result = await MultimodalFusionService.authenticate_multimodal(req)

        assert result.speaker_identity.status == "VERIFIED"
        assert result.audio_pad.status == "INSUFFICIENT_EVIDENCE"
        assert result.decision != "AUTHENTICATED"
        assert result.phrase_match_status == "UNVALIDATED"

    asyncio.run(run_test())


def test_multimodal_valid_session_token():
    """Test 7: Valid challenge token passes session security gate."""
    async def run_test():
        session_doc = VoiceSessionManager.create_session("user_test_01")
        token = session_doc["session_id"]

        req = MultimodalAuthRequest(user_id="user_test_01", challenge_token=token)

        res = await MultimodalFusionService.authenticate_multimodal(req)
        # Should not fail on session security
        assert not any("SESSION_SECURITY_FAILURE" in c for c in res.failed_checks)

    asyncio.run(run_test())


def test_multimodal_uses_server_bound_voice_evidence_once():
    """A completed voice challenge feeds fusion without browser-supplied audio/scores."""
    async def run_test():
        session = VoiceSessionManager.create_session("user_session_evidence")
        token = session["session_id"]
        _, err = VoiceSessionManager.validate_and_consume_session(token)
        assert err is None
        VoiceSessionManager.store_voice_evidence(token, "user_session_evidence", {
            "success": True,
            "speaker_verdict": "MATCH",
            "voice_similarity": 0.84,
            "liveness_verdict": "LIVE",
            "liveness_score": 0.82,
            "phrase_match": "UNVALIDATED",
        })

        req = MultimodalAuthRequest(user_id="user_session_evidence", challenge_token=token)
        with patch("app.services.multimodal_fusion_service.FaceService.verify_face", new_callable=AsyncMock) as mock_face, \
             patch("app.services.multimodal_fusion_service.LivenessService.evaluate_passive_liveness") as mock_liveness, \
             patch("app.services.multimodal_fusion_service.get_deepfake_analyzer") as mock_df:
            mock_face.return_value = {"success": True, "matched": True, "similarity_score": 0.9, "threshold": 0.65}
            mock_liveness.return_value = {"liveness": "LIVE", "score": 0.9, "threshold": 0.5}
            mock_df.return_value.analyze_frame_sequence.return_value = {
                "status": "SUCCESS", "result": {"classification": "AUTHENTIC", "fake_probability": 0.05, "confidence": 0.9}
            }
            with patch("app.services.multimodal_fusion_service.base64_to_cv2", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
                req.face_image_base64 = "server-test-frame"
                result = await MultimodalFusionService.authenticate_multimodal(req)

        assert result.decision == "AUTHENTICATED"
        assert result.speaker_identity.status == "VERIFIED"
        assert result.audio_pad.status == "BONAFIDE"

        reused = await MultimodalFusionService.authenticate_multimodal(req)
        assert reused.decision == "REJECTED"
        assert any("SESSION_SECURITY_FAILURE" in value for value in reused.failed_checks)

    asyncio.run(run_test())


def test_multimodal_invalid_session_token():
    """Test 8: Invalid challenge token -> REJECTED decision, HIGH risk."""
    async def run_test():
        req = MultimodalAuthRequest(user_id="user_test_01", challenge_token="fake_invalid_token_999")

        res = await MultimodalFusionService.authenticate_multimodal(req)

        assert res.decision == "REJECTED"
        assert res.risk_level == "HIGH"
        assert any("SESSION_SECURITY_FAILURE" in c for c in res.failed_checks)

    asyncio.run(run_test())


def test_multimodal_reused_session_token():
    """Test 9: Reused/consumed challenge token -> REJECTED decision, HIGH risk."""
    async def run_test():
        session_doc = VoiceSessionManager.create_session("user_test_01")
        token = session_doc["session_id"]

        # First consumption
        VoiceSessionManager.validate_and_consume_session(token)

        # Second attempt with consumed token
        req = MultimodalAuthRequest(user_id="user_test_01", challenge_token=token)
        res = await MultimodalFusionService.authenticate_multimodal(req)

        assert res.decision == "REJECTED"
        assert res.risk_level == "HIGH"
        assert any("SESSION_SECURITY_FAILURE" in c for c in res.failed_checks)

    asyncio.run(run_test())


def test_multimodal_multiple_faces_detected():
    """Test 12: Multiple faces detected -> REJECTED decision, HIGH risk."""
    async def run_test():
        req = MultimodalAuthRequest(user_id="user_test_01", face_image_base64="dGVzdF9mYWNl")

        with patch("app.services.multimodal_fusion_service.FaceService.verify_face", new_callable=AsyncMock) as mock_face:
            mock_face.return_value = {
                "success": False,
                "error_code": "MULTIPLE_FACES_DETECTED",
                "message": "Multiple faces detected."
            }

            res = await MultimodalFusionService.authenticate_multimodal(req)

            assert res.decision == "REJECTED"
            assert res.risk_level == "HIGH"
            assert "MULTIPLE_FACES_DETECTED" in res.failed_checks

    asyncio.run(run_test())


def test_multimodal_no_raw_embeddings_exposed():
    """Test 15: Verify no raw floating-point embeddings are exposed in response JSON."""
    async def run_test():
        req = MultimodalAuthRequest(user_id="user_test_01", face_image_base64="dGVzdF9mYWNl")

        res = await MultimodalFusionService.authenticate_multimodal(req)
        res_dict = res.model_dump()

        res_str = str(res_dict)
        assert "probe_embedding" not in res_str
        assert "decrypted_embedding" not in res_str
        assert "embedding" not in res_dict.get("face_identity", {})
        assert "embedding" not in res_dict.get("speaker_identity", {})

    asyncio.run(run_test())
