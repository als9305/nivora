"""
API Integration Tests for DeepQShield v1 FastAPI Endpoints.
"""

import os
import sys
import base64
import numpy as np
import cv2
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend")))

from app.main import app

client = TestClient(app)


import pytest
from video.face_tracker import FaceTracker

@pytest.fixture(autouse=True)
def mock_tracker_for_dummy_frames(monkeypatch):
    def mock_track(self, frames):
        return {
            "status": "OK",
            "message": "Mock tracked",
            "face_crops": [f if (isinstance(f, np.ndarray) and f.size > 0) else np.zeros((224, 224, 3), dtype=np.uint8) for f in frames],
            "multi_face_counts": [1] * len(frames)
        }
    monkeypatch.setattr(FaceTracker, "track_sequence", mock_track)


def _make_dummy_base64_frame() -> str:
    img = (np.random.rand(224, 224, 3) * 255).astype('uint8')
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


def test_model_info_endpoint():
    res = client.get("/api/v1/deepfake/model/info")
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "DeepQShield"
    assert data["backbone"] == "ResNeXt50_32x4d"
    assert "pqc_security" in data
    assert data["pqc_security"]["key_exchange"] == "Kyber1024 KEM"


def test_analyze_image_endpoint():
    b64_frame = _make_dummy_base64_frame()
    res = client.post("/api/v1/deepfake/image", json={"image_base64": b64_frame})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "payload" in data and "integrity" in data
    assert data["integrity"]["hash_algorithm"] == "SHA3-256"


def test_analyze_video_endpoint():
    frames = [_make_dummy_base64_frame() for _ in range(3)]
    res = client.post("/api/v1/deepfake/video", json={"frames_base64": frames})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["payload"]["input"]["frames_analyzed"] == 3


def test_live_session_flow():
    # 1. Start live session
    start_res = client.post("/api/v1/deepfake/live/start", json={"user_id": "user_test_101"})
    assert start_res.status_code == 200
    session_id = start_res.json()["session_id"]
    
    # 2. Ingest frames
    for i in range(2):
        frame_res = client.post("/api/v1/deepfake/live/frame", json={
            "session_id": session_id,
            "frame_base64": _make_dummy_base64_frame(),
            "frame_timestamp": i * 0.5
        })
        assert frame_res.status_code == 200
        
    # 3. End live session & verify result record
    end_res = client.post("/api/v1/deepfake/live/end", json={"session_id": session_id})
    assert end_res.status_code == 200
    result_data = end_res.json()
    assert result_data["status"] == "SUCCESS"
    
    # 4. Fetch result by analysis ID & verify signature
    analysis_id = result_data["payload"]["analysis_id"]
    fetch_res = client.get(f"/api/v1/deepfake/result/{analysis_id}")
    assert fetch_res.status_code == 200
    fetch_data = fetch_res.json()
    assert fetch_data["integrity_verification"]["valid"] is True
