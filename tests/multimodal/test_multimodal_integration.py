"""
Module 5 Integration & Smoke Test Suite.

Includes:
1. Level 1: Deterministic REST API Route Integration Tests (POST /api/v1/authenticate/multimodal).
2. Level 2: Real Interface Smoke Test invoking actual server-side module interfaces.
"""

import pytest
import numpy as np
import cv2
import base64
from fastapi.testclient import TestClient
from backend.app.main import app
from app.services.face_service import FaceService
from app.services.liveness_service import LivenessService
from app.services.multimodal_fusion_service import MultimodalFusionService, get_deepfake_analyzer
from app.schemas.multimodal import MultimodalAuthRequest

client = TestClient(app)


def test_level1_multimodal_api_endpoint_structure():
    """Level 1 Integration Test: Verify POST /api/v1/authenticate/multimodal API response structure."""
    payload = {
        "user_id": "integration_user_01",
        "challenge_token": None,
        "face_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "audio_pcm_base64": None
    }

    response = client.post("/api/v1/authenticate/multimodal", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "decision" in data
    assert "overall_confidence" in data
    assert "risk_level" in data
    assert "face_identity" in data
    assert "face_liveness" in data
    assert "deepfake_detection" in data
    assert "speaker_identity" in data
    assert "audio_pad" in data
    assert "failed_checks" in data
    assert "attack_indicators" in data
    assert "processing_time_ms" in data
    assert data["phrase_match_status"] == "UNVALIDATED"


def test_level1_multimodal_invalid_session_rejection():
    """Level 1 Integration Test: Verify invalid session challenge token results in REJECTED / HIGH risk."""
    payload = {
        "user_id": "integration_user_01",
        "challenge_token": "invalid_fake_session_12345",
        "face_image_base64": None,
        "audio_pcm_base64": None
    }

    response = client.post("/api/v1/authenticate/multimodal", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["decision"] == "REJECTED"
    assert data["risk_level"] == "HIGH"
    assert any("SESSION_SECURITY_FAILURE" in check for check in data["failed_checks"])


def test_level2_real_module_interfaces_smoke_test():
    """Level 2 Smoke Test: Invokes real server-side module interfaces to verify end-to-end connectivity."""
    # 1. Real Module 1 DeepQShield call
    analyzer = get_deepfake_analyzer()
    dummy_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    df_res = analyzer.analyze_frame_sequence([dummy_bgr])
    assert "status" in df_res

    # 2. Real Module 3 Face Liveness call
    dummy_png = cv2.imencode('.png', dummy_bgr)[1].tobytes()
    b64_str = base64.b64encode(dummy_png).decode('utf-8')
    live_res = LivenessService.evaluate_passive_liveness([b64_str])
    assert "liveness" in live_res
    assert "score" in live_res

    # 3. Real Module 5 Fusion Call via MultimodalFusionService
    req = MultimodalAuthRequest(user_id="smoke_user_01", face_image_base64=b64_str)
    
    import asyncio
    fusion_res = asyncio.run(MultimodalFusionService.authenticate_multimodal(req))
    assert fusion_res.success is True
    assert fusion_res.decision in ["AUTHENTICATED", "REJECTED", "INSUFFICIENT_EVIDENCE", "RETRY_REQUIRED"]
    assert fusion_res.risk_level in ["LOW", "MEDIUM", "HIGH"]


@pytest.mark.anyio
async def test_multimodal_pipeline_hard_gates():
    """Verify strict Module 3 hard-gate: SPOOF or INSUFFICIENT_EVIDENCE results in REJECTED fusion without allowing access."""
    # Spoof sequence evaluation
    empty_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    live_res = LivenessService.evaluate_passive_liveness([empty_b64])
    assert live_res["liveness"] in ["SPOOF", "INSUFFICIENT_EVIDENCE"]

    # Request with spoofed liveness
    req = MultimodalAuthRequest(
        user_id="test_hard_gate_user",
        face_image_base64=empty_b64,
        face_liveness_frames=[empty_b64],
        audio_pcm_base64=None
    )
    fusion_res = await MultimodalFusionService.authenticate_multimodal(req)
    assert fusion_res.decision in ["REJECTED", "INSUFFICIENT_EVIDENCE"]
    assert fusion_res.decision != "AUTHENTICATED"

