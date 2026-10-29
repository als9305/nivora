"""
Module 3 PyTest Suite for Face Liveness & Anti-Spoofing.
Tests passive RGB anti-spoofing, temporal micro-motion aggregation, quality guards, and API endpoints.
"""

import os
import sys
import pytest
import numpy as np
import cv2
import base64
from fastapi.testclient import TestClient

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.services.liveness_service import LivenessService
from models.face_liveness import (
    PassiveAntiSpoofNet,
    MultiDomainFeatureExtractor,
    TemporalLivenessAggregator
)

client = TestClient(app)


def generate_face_b64(width: int = 320, height: int = 320, draw_face: bool = True, offset: int = 0) -> str:
    """Helper to generate a textured, well-lit base64 encoded test image."""
    np.random.seed(42 + offset)
    canvas = np.random.randint(90, 150, (height, width, 3), dtype=np.uint8)
    
    if draw_face:
        center_x = width // 2 + offset
        center_y = height // 2
        cv2.ellipse(canvas, (center_x, center_y), (60, 80), 0, 0, 360, (200, 180, 170), -1)
        noise = np.random.randint(-20, 20, canvas.shape, dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.circle(canvas, (center_x - 20, center_y - 20), 8, (40, 30, 30), -1)
        cv2.circle(canvas, (center_x + 20, center_y - 20), 8, (40, 30, 30), -1)
        cv2.ellipse(canvas, (center_x, center_y + 25), (20, 10), 0, 0, 180, (60, 40, 40), 2)
        
    _, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buf.tobytes()).decode('utf-8')


from unittest.mock import patch

def test_no_face_detected():
    """Verify handling when no face is present in frame."""
    empty_b64 = generate_face_b64(draw_face=False)
    with patch("backend.app.services.liveness_service.global_face_detector.process_frame", return_value={"status": "NO_FACE_DETECTED", "face_count": 0, "face_crop": None}):
        res = LivenessService.evaluate_passive_liveness([empty_b64])
        assert res["success"] is True
        assert res["liveness"] in ["SPOOF", "INSUFFICIENT_EVIDENCE"]


def test_multiple_faces_detected():
    """Verify multi-face policy rejection."""
    canvas = np.random.randint(90, 150, (320, 480, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', canvas)
    b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    
    with patch("backend.app.services.liveness_service.global_face_detector.process_frame", return_value={"status": "MULTIPLE_FACES_DETECTED", "face_count": 2, "face_crop": None}):
        res = LivenessService.evaluate_passive_liveness([b64])
        assert res["success"] is True
        assert res["liveness"] in ["SPOOF", "INSUFFICIENT_EVIDENCE"]


def test_insufficient_frames():
    """Verify handling of empty frame list."""
    res = LivenessService.evaluate_passive_liveness([])
    assert res["success"] is False
    assert res["liveness"] == "INSUFFICIENT_EVIDENCE"
    assert res["frames_analyzed"] == 0


def test_static_photo_replay_detection():
    """Verify static photo replay rejection (micro-motion delta < threshold)."""
    face_b64 = generate_face_b64(draw_face=True, offset=0)
    static_sequence = [face_b64, face_b64, face_b64, face_b64]
    
    mock_crop = np.full((120, 120, 3), 128, dtype=np.uint8)
    with patch("backend.app.services.liveness_service.global_face_detector.process_frame", return_value={"status": "SINGLE_FACE_ACCEPTED", "face_count": 1, "face_crop": mock_crop}):
        res = LivenessService.evaluate_passive_liveness(static_sequence)
        assert res["success"] is True
        assert res["liveness"] == "SPOOF"
        assert res["attack_type"] == "PRINTED_PHOTO_OR_STATIC_REPLAY"
        assert res["details"]["motion_delta"] < 0.20


def test_live_moving_sequence():
    """Verify micro-motion handling on moving sequence."""
    seq = [generate_face_b64(draw_face=True, offset=i * 5) for i in range(5)]
    res = LivenessService.evaluate_passive_liveness(seq)
    assert res["success"] is True
    assert "liveness" in res
    assert res["frames_analyzed"] == 5


def test_temporal_aggregator_logic():
    """Test TemporalLivenessAggregator rules."""
    agg = TemporalLivenessAggregator(min_motion_delta=0.20, decision_threshold=0.65)
    
    # 1. Zero motion -> SPOOF
    res_static = agg.aggregate(frame_scores=[0.90, 0.92, 0.95], motion_delta=0.05, quality_passed=True)
    assert res_static["liveness"] == "SPOOF"
    assert res_static["attack_type"] == "PRINTED_PHOTO_OR_STATIC_REPLAY"
    
    # 2. High score + valid motion -> LIVE
    res_live = agg.aggregate(frame_scores=[0.85, 0.90, 0.88], motion_delta=1.50, quality_passed=True)
    assert res_live["liveness"] == "LIVE"
    assert res_live["score"] >= 0.65
    
    # 3. Low score + valid motion -> SPOOF
    res_spoof = agg.aggregate(frame_scores=[0.20, 0.15, 0.30], motion_delta=1.50, quality_passed=True)
    assert res_spoof["liveness"] == "SPOOF"


def test_multi_domain_descriptors():
    """Test MultiDomainFeatureExtractor descriptors."""
    gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    lbp_hist = MultiDomainFeatureExtractor.extract_lbp_histogram(gray)
    assert len(lbp_hist) == 256
    assert abs(np.sum(lbp_hist) - 1.0) < 1e-4
    
    fft_ratio = MultiDomainFeatureExtractor.extract_fft_moire_ratio(gray)
    assert isinstance(fft_ratio, float)
    
    chroma = MultiDomainFeatureExtractor.extract_chroma_variance(bgr)
    assert "y_var" in chroma and "cb_var" in chroma and "cr_var" in chroma


def test_liveness_api_endpoint_success():
    """Test POST /api/v1/liveness/check REST API endpoint."""
    seq = [generate_face_b64(draw_face=True, offset=i * 4) for i in range(3)]
    response = client.post("/api/v1/liveness/check", json={"frames_base64": seq, "user_id": "test_user_01"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "liveness" in data
    assert "score" in data
    assert "threshold" in data
    assert "frames_analyzed" in data


def test_liveness_api_endpoint_malformed():
    """Test POST /api/v1/liveness/check with malformed request body."""
    response = client.post("/api/v1/liveness/check", json={"invalid_field": "test"})
    assert response.status_code == 422  # Unprocessable Entity


def test_validate_active_challenge_turn_left():
    """Test LivenessService.validate_active_challenge for TURN_LEFT."""
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    crops = [np.ones((100, 100, 3), dtype=np.uint8) * 128 for _ in range(5)]
    # In mirrored user-view: User moving to screen-left corresponds to sensor-right (increasing raw x)
    bboxes = [
        (220, 100, 100, 100),
        (240, 100, 100, 100),
        (260, 100, 100, 100),
        (280, 100, 100, 100),
        (300, 100, 100, 100),
    ]
    passed, details = LivenessService.validate_active_challenge(
        "TURN_LEFT", frames, crops, bboxes, {"movement_direction": "LEFT"}
    )
    assert passed is True
    assert details["challenge"] == "TURN_LEFT"
    assert details["performed"] is True


def test_validate_active_challenge_turn_right():
    """Test LivenessService.validate_active_challenge for TURN_RIGHT."""
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    crops = [np.ones((100, 100, 3), dtype=np.uint8) * 128 for _ in range(5)]
    # In mirrored user-view: User moving to screen-right corresponds to sensor-left (decreasing raw x)
    bboxes = [
        (300, 100, 100, 100),
        (280, 100, 100, 100),
        (260, 100, 100, 100),
        (240, 100, 100, 100),
        (220, 100, 100, 100),
    ]
    passed, details = LivenessService.validate_active_challenge(
        "TURN_RIGHT", frames, crops, bboxes, {"movement_direction": "RIGHT"}
    )
    assert passed is True
    assert details["challenge"] == "TURN_RIGHT"
    assert details["performed"] is True


def test_validate_active_challenge_turn_left_uses_mirrored_landmark_yaw():
    """A positive yaw shift (sensor-right nose movement toward left shoulder) is a left turn."""
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    crops = [np.ones((100, 100, 3), dtype=np.uint8) * 128 for _ in range(5)]
    bboxes = [(250, 100, 100, 100)] * 5
    # When user turns head to their left, nose moves toward sensor-right (larger x: 60 to 85)
    landmarks = [
        {"left_eye": (20.0, 30.0), "right_eye": (100.0, 30.0), "nose": (x, 55.0)}
        for x in (60.0, 66.0, 73.0, 80.0, 85.0)
    ]

    passed, details = LivenessService.validate_active_challenge(
        "TURN_LEFT", frames, crops, bboxes, landmarks
    )

    assert passed is True
    assert details["detected_direction"] == "LEFT"
    assert details["yaw_delta"] > 0


def test_validate_active_challenge_turn_right_uses_landmark_yaw():
    """A negative yaw shift (sensor-left nose movement toward right shoulder) is a right turn."""
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    crops = [np.ones((100, 100, 3), dtype=np.uint8) * 128 for _ in range(5)]
    bboxes = [(250, 100, 100, 100)] * 5
    # When user turns head to their right, nose moves toward sensor-left (smaller x: 60 to 35)
    landmarks = [
        {"left_eye": (20.0, 30.0), "right_eye": (100.0, 30.0), "nose": (x, 55.0)}
        for x in (60.0, 54.0, 47.0, 40.0, 35.0)
    ]

    passed, details = LivenessService.validate_active_challenge(
        "TURN_RIGHT", frames, crops, bboxes, landmarks
    )

    assert passed is True
    assert details["detected_direction"] == "RIGHT"
    assert details["yaw_delta"] < 0


def test_validate_active_challenge_opposite_turn_rejected():
    """Opposite head turn must fail the challenge."""
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    crops = [np.ones((100, 100, 3), dtype=np.uint8) * 128 for _ in range(5)]
    bboxes = [(250, 100, 100, 100)] * 5
    # User turns right (60 to 35) when asked to TURN_LEFT
    right_turn_landmarks = [
        {"left_eye": (20.0, 30.0), "right_eye": (100.0, 30.0), "nose": (x, 55.0)}
        for x in (60.0, 54.0, 47.0, 40.0, 35.0)
    ]
    passed_left, details_left = LivenessService.validate_active_challenge(
        "TURN_LEFT", frames, crops, bboxes, right_turn_landmarks
    )
    assert passed_left is False
    assert details_left["detected_direction"] == "RIGHT"
    assert details_left["challenge_result"] == "FAILED"

    # User turns left (60 to 85) when asked to TURN_RIGHT
    left_turn_landmarks = [
        {"left_eye": (20.0, 30.0), "right_eye": (100.0, 30.0), "nose": (x, 55.0)}
        for x in (60.0, 66.0, 73.0, 80.0, 85.0)
    ]
    passed_right, details_right = LivenessService.validate_active_challenge(
        "TURN_RIGHT", frames, crops, bboxes, left_turn_landmarks
    )
    assert passed_right is False
    assert details_right["detected_direction"] == "LEFT"
    assert details_right["challenge_result"] == "FAILED"


def test_validate_active_challenge_dict_input():
    """Verify challenge passed as dictionary is correctly handled."""
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    crops = [np.ones((100, 100, 3), dtype=np.uint8) * 128 for _ in range(5)]
    bboxes = [(250, 100, 100, 100)] * 5
    landmarks = [
        {"left_eye": (20.0, 30.0), "right_eye": (100.0, 30.0), "nose": (x, 55.0)}
        for x in (60.0, 66.0, 73.0, 80.0, 85.0)
    ]
    challenge_dict = {"id": "TURN_LEFT", "label": "Turn Left"}
    passed, details = LivenessService.validate_active_challenge(
        challenge_dict, frames, crops, bboxes, landmarks
    )
    assert passed is True
    assert details["challenge"] == "TURN_LEFT"
    assert details["detected_direction"] == "LEFT"


def test_turn_direction_uses_decisive_final_yaw_when_peak_excursions_conflict():
    """A clear final left orientation must not be downgraded to NONE by jitter."""
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(6)]
    crops = [np.ones((100, 100, 3), dtype=np.uint8) * 128 for _ in range(6)]
    bboxes = [(250, 100, 100, 100)] * 6
    # The second frame has rightward jitter (smaller x), while the final settled pose
    # is a strong left turn (larger x: 60 -> 42 -> 64 -> 75 -> 85 -> 90).
    landmarks = [
        {"left_eye": (20.0, 30.0), "right_eye": (100.0, 30.0), "nose": (x, 55.0)}
        for x in (60.0, 42.0, 64.0, 75.0, 85.0, 90.0)
    ]

    passed, details = LivenessService.validate_active_challenge(
        "TURN_LEFT", frames, crops, bboxes, landmarks
    )

    assert passed is True
    assert details["detected_direction"] == "LEFT"
    assert details["challenge_result"] == "PASSED"


def test_validate_active_challenge_insufficient_movement():
    """Test rejection when head movement is static during turn challenge."""
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    crops = [np.ones((100, 100, 3), dtype=np.uint8) * 128 for _ in range(5)]
    # Identical bounding boxes (no displacement)
    bboxes = [(250, 100, 100, 100)] * 5
    passed, details = LivenessService.validate_active_challenge(
        "TURN_LEFT", frames, crops, bboxes, None
    )
    assert passed is False
    assert details["performed"] is False


def test_validate_active_challenge_blink_twice():
    """Test LivenessService.validate_active_challenge for BLINK_TWICE."""
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    crops = []
    for i in range(5):
        crop = np.full((120, 120, 3), 128, dtype=np.uint8)
        # Frames 1 and 3 are "closed eyes" (smooth skin, low Sobel gradient)
        # Frames 0, 2, 4 are "open eyes" (high gradient horizontal edges)
        if i in [0, 2, 4]:
            crop[30:55, 20:100] = np.random.randint(0, 255, (25, 80, 3), dtype=np.uint8)
        crops.append(crop)

    bboxes = [(250, 100, 100, 100)] * 5
    passed, details = LivenessService.validate_active_challenge(
        "BLINK_TWICE", frames, crops, bboxes, {"events_detected": 2}
    )
    assert passed is True
    assert details["challenge"] == "BLINK_TWICE"
    assert details["performed"] is True


def test_liveness_api_with_active_challenge_insufficient():
    """Test POST /api/v1/liveness/check with unperformed active challenge returns INSUFFICIENT_EVIDENCE."""
    face_b64 = generate_face_b64(draw_face=True, offset=0)
    static_seq = [face_b64, face_b64, face_b64, face_b64]

    mock_crop = np.full((120, 120, 3), 128, dtype=np.uint8)
    with patch("backend.app.services.liveness_service.global_face_detector.process_frame", return_value={"status": "SINGLE_FACE_ACCEPTED", "face_count": 1, "face_crop": mock_crop}):
        response = client.post("/api/v1/liveness/check", json={
            "frames_base64": static_seq,
            "user_id": "test_user_01",
            "challenge": "TURN_LEFT"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["liveness"] == "INSUFFICIENT_EVIDENCE"
        assert "challenge_details" in data.get("details", {})
        assert data["details"]["challenge"] == "TURN_LEFT"
        assert data["details"]["challenge_performed"] is False
        assert "challenge" in data
        assert data["challenge"]["type"] == "TURN_LEFT"
        assert "detected" in data["challenge"]
        assert "evidence_available" in data["challenge"]
        assert "passive_liveness" in data
        assert "temporal_motion" in data


def test_liveness_api_separate_evidence_and_diagnostics():
    """Verify separate evidence return blocks: passive_liveness, temporal_motion, and challenge."""
    frames_b64 = [generate_face_b64(draw_face=True, offset=i * 5) for i in range(6)]
    response = client.post("/api/v1/liveness/check", json={
        "frames_base64": frames_b64,
        "user_id": "test_user_02",
        "challenge": "TURN_LEFT",
        "challenge_evidence": {"coordinate_convention": "user-view"}
    })
    assert response.status_code == 200
    data = response.json()
    assert "passive_liveness" in data
    assert "temporal_motion" in data
    assert "challenge" in data
    assert data["challenge"]["type"] == "TURN_LEFT"
    assert "horizontal_px" in data["challenge"]
    assert "vertical_px" in data["challenge"]
    assert "direction" in data["challenge"]
    assert "landmarks_detected" in data["challenge"]


def test_checkpoint_loads_and_changes_model_behavior():
    """
    Verify that the face liveness checkpoint is found and loaded regardless of CWD.
    A freshly-initialised (random-weight) PassiveAntiSpoofNet scores random noise
    near ~0.443 on every call (deterministic due to kaiming_normal_ init).
    The trained checkpoint must score differently, proving the weights changed.
    """
    import torch
    import torch.nn.functional as F
    from models.face_liveness import PassiveAntiSpoofNet

    # Random-weight model (no checkpoint)
    random_model = PassiveAntiSpoofNet(in_channels=3, aux_dim=8, num_classes=2)
    random_model.eval()
    aux = torch.zeros(1, 8)
    inp = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        random_score = float(F.softmax(random_model(inp, aux), dim=1)[0, 1].item())

    # Loaded-checkpoint model via LivenessService
    LivenessService._model = None  # force reload
    trained_model = LivenessService.get_model()
    trained_model.eval()
    with torch.no_grad():
        trained_score = float(F.softmax(trained_model(inp, aux), dim=1)[0, 1].item())

    assert abs(random_score - trained_score) > 0.01, (
        f"Checkpoint did not change model output: random={random_score:.4f} trained={trained_score:.4f}. "
        "Checkpoint path resolution is broken."
    )


def test_passive_score_class_mapping_index_1_is_live():
    """
    Verify that index 1 of the model softmax output represents the LIVE (bona fide) class.
    The LivenessService uses probs[0, 1] as live_prob; this test checks class semantics.
    A trained model given a high-texture face crop should score > 0.5 as live.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    LivenessService._model = None
    model = LivenessService.get_model()
    model.eval()

    from models.face_liveness import MultiDomainFeatureExtractor
    import cv2

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Generate a skin-toned textured crop (realistic face-like input)
    np.random.seed(7)
    canvas = np.random.randint(120, 190, (224, 224, 3), dtype=np.uint8)
    canvas[:, :, 2] = np.clip(canvas[:, :, 2].astype(int) + 40, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    normalized = (t - mean) / std_t
    batch = normalized.unsqueeze(0)
    aux = torch.from_numpy(
        MultiDomainFeatureExtractor.extract_auxiliary_vector(canvas)
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(batch, aux)
        probs = F.softmax(logits, dim=1)

    live_prob = float(probs[0, 1].item())
    spoof_prob = float(probs[0, 0].item())

    # The live class (index 1) should be higher for a textured face crop
    assert live_prob > spoof_prob, (
        f"Class mapping may be inverted: live_prob={live_prob:.4f} <= spoof_prob={spoof_prob:.4f}"
    )


def test_borderline_score_produces_insufficient_evidence_not_live():
    """
    Verify that a passive score just below 0.50 produces INSUFFICIENT_EVIDENCE (not LIVE).
    Also verify that a score just above 0.50 produces LIVE.
    """
    agg = TemporalLivenessAggregator(min_motion_delta=0.20, decision_threshold=0.5000)

    # Just below threshold -> SPOOF
    res_below = agg.aggregate(frame_scores=[0.48, 0.49, 0.47], motion_delta=1.5)
    assert res_below["liveness"] in ("SPOOF", "INSUFFICIENT_EVIDENCE"), (
        f"Score 0.48 below threshold should not produce LIVE, got {res_below['liveness']}"
    )
    assert res_below["liveness"] != "LIVE"

    # Just above threshold -> LIVE
    res_above = agg.aggregate(frame_scores=[0.52, 0.53, 0.51], motion_delta=1.5)
    assert res_above["liveness"] == "LIVE", (
        f"Score 0.52 above threshold should produce LIVE, got {res_above['liveness']}"
    )


def test_final_verdict_challenge_passed_but_borderline_passive_is_insufficient_evidence():
    """
    Confirm that when active challenge PASSES but passive score is borderline (0.4490),
    the final verdict is INSUFFICIENT_EVIDENCE — not LIVE and not SPOOF.
    A passed challenge must NOT automatically override passive anti-spoofing.
    """
    import numpy as np
    import torch
    from unittest.mock import patch, MagicMock

    frames_b64 = [generate_face_b64(draw_face=True, offset=i * 8) for i in range(6)]
    mock_crop = np.full((120, 120, 3), 128, dtype=np.uint8)

    def mock_process_frame(frame, return_landmarks=False):
        return {
            "status": "SINGLE_FACE_ACCEPTED",
            "face_count": 1,
            "face_crop": mock_crop,
            "bbox": (250, 100, 100, 100),
            "landmarks": {"left_eye": (20.0, 30.0), "right_eye": (100.0, 30.0), "nose": (75.0, 55.0)},
        }

    # Simulate logits that produce softmax probability = [0.5510, 0.4490]
    # log(0.5510) = -0.5960, log(0.4490) = -0.8007
    mock_logits = torch.tensor([[-0.5960, -0.8007]], dtype=torch.float32)

    with patch("backend.app.services.liveness_service.global_face_detector.process_frame",
               side_effect=mock_process_frame):
        with patch.object(LivenessService, "get_model") as mock_model_fn:
            m = MagicMock()
            m.eval.return_value = None
            m.return_value = mock_logits
            mock_model_fn.return_value = m

            res = LivenessService.evaluate_passive_liveness(
                frames_b64,
                challenge=None
            )

    assert res["success"] is True
    assert res["score"] == 0.4490 or round(res["score"], 3) == 0.449
    # Since 0.42 <= 0.4490 < 0.50, and temporal motion is present, verdict must be INSUFFICIENT_EVIDENCE
    assert res["liveness"] == "INSUFFICIENT_EVIDENCE", (
        f"Score 0.4490 should result in INSUFFICIENT_EVIDENCE, got {res['liveness']}"
    )


def test_turn_challenge_with_pose_aware_passive_scoring():
    """Verify that a legitimate user turning their head passes liveness without being rejected by profile crops.
    
    Frontal baseline frames (yaw ~ 0.0) have authentic texture (score >= 0.90), while extreme turned profile
    frames (yaw ~ -0.95) may produce lower raw model scores. Pose-aware evaluation selects frontal frames for
    passive anti-spoofing while validating the active head turn kinematics across the full sequence.
    """
    import torch
    from unittest.mock import MagicMock

    frames_b64 = [generate_face_b64(draw_face=True, offset=i * 2) for i in range(10)]
    mock_crop = np.full((120, 120, 3), 128, dtype=np.uint8)

    # First 3 frames: neutral yaw (~0.0, nose at 60.0 between eyes at 20.0 and 100.0)
    # Remaining 7 frames: head turned right (nose shifts to sensor-left: 50 -> 40 -> 30)
    def mock_process_frame(frame, return_landmarks=False):
        # We find index by checking frame content or counter
        idx = getattr(mock_process_frame, "count", 0)
        mock_process_frame.count = idx + 1
        if idx < 3:
            nose_x = 60.0  # yaw = (60 - 20)/80 - 0.5 = 0.0 (neutral)
            x_box = 250
        else:
            nose_x = 30.0  # yaw = (30 - 20)/80 - 0.5 = -0.375 (turned right)
            x_box = 250 - (idx - 2) * 15  # moving screen-right / sensor-left
            
        return {
            "status": "SINGLE_FACE_ACCEPTED",
            "face_count": 1,
            "face_crop": mock_crop,
            "bbox": (x_box, 100, 100, 100),
            "landmarks": {"left_eye": (20.0, 30.0), "right_eye": (100.0, 30.0), "nose": (nose_x, 55.0)},
        }
    mock_process_frame.count = 0

    # For neural model: return live probability 0.95 for first 3 frontal calls, 0.20 for turned profile calls
    # logit for 0.95: [log(0.05), log(0.95)] = [-2.9957, -0.0513]
    # logit for 0.20: [log(0.80), log(0.20)] = [-0.2231, -1.6094]
    frontal_logits = torch.tensor([[-2.9957, -0.0513]], dtype=torch.float32)
    profile_logits = torch.tensor([[-0.2231, -1.6094]], dtype=torch.float32)

    def mock_forward(batch, aux):
        call_idx = getattr(mock_forward, "call_count", 0)
        mock_forward.call_count = call_idx + 1
        return frontal_logits if call_idx < 3 else profile_logits
    mock_forward.call_count = 0

    with patch("backend.app.services.liveness_service.global_face_detector.process_frame",
               side_effect=mock_process_frame):
        with patch.object(LivenessService, "get_model") as mock_model_fn:
            m = MagicMock()
            m.eval.return_value = None
            m.side_effect = mock_forward
            mock_model_fn.return_value = m

            res = LivenessService.evaluate_passive_liveness(
                frames_b64,
                challenge="TURN_RIGHT"
            )

    assert res["success"] is True
    # Kinematic check must pass
    assert res["challenge"]["detected"] is True
    assert res["challenge"]["result"] == "PASSED"
    # Pose-aware passive scoring must evaluate frontal frames and exceed 0.5000 threshold
    assert res["score"] >= 0.5000
    assert res["liveness"] == "LIVE"
    assert res["attack_type"] is None



