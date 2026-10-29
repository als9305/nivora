import cv2
import numpy as np
import base64
import json
from app.services.face_service import FaceService
from app.services.liveness_service import LivenessService
from app.services.deepfake_service import DeepfakeService
from app.services.voice_service import VoiceService
from app.services.risk_engine import RiskEngine

def generate_synthetic_face_frame(yaw_offset=0, eye_open=True, mouth_open=False, noise=0):
    img = np.ones((480, 640, 3), dtype=np.uint8) * 220
    cx = 320 + yaw_offset
    cy = 240
    cv2.ellipse(img, (cx, cy), (100, 130), 0, 0, 360, (180, 190, 200), -1)
    cv2.ellipse(img, (cx, cy), (100, 130), 0, 0, 360, (50, 50, 50), 2)

    eye_y = cy - 30
    lx = cx - 40
    rx = cx + 40
    if eye_open:
        cv2.circle(img, (lx, eye_y), 15, (255, 255, 255), -1)
        cv2.circle(img, (rx, eye_y), 15, (255, 255, 255), -1)
        cv2.circle(img, (lx, eye_y), 6, (50, 50, 50), -1)
        cv2.circle(img, (rx, eye_y), 6, (50, 50, 50), -1)
    else:
        cv2.line(img, (lx - 15, eye_y), (lx + 15, eye_y), (50, 50, 50), 3)
        cv2.line(img, (rx - 15, eye_y), (rx + 15, eye_y), (50, 50, 50), 3)

    cv2.line(img, (cx, cy - 10), (cx, cy + 20), (50, 50, 50), 3)

    mouth_y = cy + 50
    if mouth_open:
        cv2.ellipse(img, (cx, mouth_y), (30, 20), 0, 0, 360, (50, 50, 150), -1)
    else:
        cv2.line(img, (cx - 25, mouth_y), (cx + 25, mouth_y), (50, 50, 50), 3)

    if noise > 0:
        gauss = np.random.normal(0, noise, img.shape).astype(np.uint8)
        img = cv2.add(img, gauss)

    _, buf = cv2.imencode('.jpg', img)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}"

def run_acceptance_tests():
    print("=========================================================================")
    print("          DEEPGUARD ACCEPTANCE SCENARIOS EVALUATION (A through H)        ")
    print("=========================================================================")

    # SCENARIO A — GENUINE USER
    print("\n--- SCENARIO A: GENUINE USER ---")
    f1 = generate_synthetic_face_frame(yaw_offset=0, eye_open=True)
    f2 = generate_synthetic_face_frame(yaw_offset=0, eye_open=False)
    f3 = generate_synthetic_face_frame(yaw_offset=0, eye_open=True)

    risk_a = RiskEngine.calculate_risk(
        face_score=0.95, liveness_score=0.95,
        deepfake_probability=0.04, voice_similarity=0.94, voice_spoof_probability=0.03
    )
    print(f"Decision: {risk_a['decision']} | Risk: {risk_a['risk_score']}/100 | Confidence: {risk_a['confidence_state']}")
    assert risk_a['decision'] == 'ALLOW', f"Scenario A failed: expected ALLOW, got {risk_a['decision']}"

    # SCENARIO B — DEEPFAKE AI ATTACK
    print("\n--- SCENARIO B: DEEPFAKE AI ATTACK ---")
    risk_b = RiskEngine.calculate_risk(
        face_score=0.88, liveness_score=0.35,
        deepfake_probability=0.94, voice_similarity=0.85, voice_spoof_probability=0.88
    )
    print(f"Decision: {risk_b['decision']} | Risk: {risk_b['risk_score']}/100 | Confidence: {risk_b['confidence_state']}")
    assert risk_b['decision'] == 'BLOCK', f"Scenario B failed: expected BLOCK, got {risk_b['decision']}"

    # SCENARIO C — STATIC PHOTO ATTACK
    print("\n--- SCENARIO C: STATIC PHOTO ATTACK ---")
    f_static = generate_synthetic_face_frame(yaw_offset=0)
    live_c = LivenessService.evaluate_challenge("BLINK", [f_static, f_static, f_static])
    risk_c = RiskEngine.calculate_risk(
        face_score=0.90, liveness_score=live_c["liveness_score"],
        deepfake_probability=0.35, voice_similarity=0.80, voice_spoof_probability=0.10
    )
    print(f"Liveness Status: {live_c['status']} | Message: {live_c['message']}")
    print(f"Decision: {risk_c['decision']} | Risk: {risk_c['risk_score']}/100")
    assert risk_c['decision'] == 'BLOCK', f"Scenario C failed: expected BLOCK, got {risk_c['decision']}"

    # SCENARIO D — SCREEN REPLAY ATTACK
    print("\n--- SCENARIO D: SCREEN REPLAY ATTACK ---")
    risk_d = RiskEngine.calculate_risk(
        face_score=0.92, liveness_score=0.25,
        deepfake_probability=0.68, voice_similarity=0.82, voice_spoof_probability=0.25
    )
    print(f"Decision: {risk_d['decision']} | Risk: {risk_d['risk_score']}/100")
    assert risk_d['decision'] == 'BLOCK', f"Scenario D failed: expected BLOCK, got {risk_d['decision']}"

    # SCENARIO E — SYNTHETIC VOICE SPOOF
    print("\n--- SCENARIO E: SYNTHETIC VOICE SPOOF ---")
    risk_e = RiskEngine.calculate_risk(
        face_score=0.91, liveness_score=0.88,
        deepfake_probability=0.12, voice_similarity=0.42, voice_spoof_probability=0.91
    )
    print(f"Decision: {risk_e['decision']} | Risk: {risk_e['risk_score']}/100")
    assert risk_e['decision'] == 'BLOCK', f"Scenario E failed: expected BLOCK, got {risk_e['decision']}"

    # SCENARIO F — BORDERLINE CASE (INCONCLUSIVE)
    print("\n--- SCENARIO F: BORDERLINE CASE ---")
    risk_f = RiskEngine.calculate_risk(
        face_score=0.78, liveness_score=0.65,
        deepfake_probability=0.45, voice_similarity=0.70, voice_spoof_probability=0.35
    )
    print(f"Decision: {risk_f['decision']} | Risk: {risk_f['risk_score']}/100 | Confidence: {risk_f['confidence_state']}")
    assert risk_f['decision'] in ['STEP-UP OTP', 'STEP-UP'], f"Scenario F failed: expected STEP-UP OTP, got {risk_f['decision']}"

    # SCENARIO G — WRONG CHALLENGE MOVEMENT (TURN_RIGHT vs TURN_LEFT)
    print("\n--- SCENARIO G: WRONG CHALLENGE MOVEMENT ---")
    f_left1 = generate_synthetic_face_frame(yaw_offset=0)
    f_left2 = generate_synthetic_face_frame(yaw_offset=-200) # Turn Left
    live_g = LivenessService.evaluate_challenge("TURN_RIGHT", [f_left1, f_left2])
    print(f"Expected TURN_RIGHT, User performed TURN_LEFT -> Result: {live_g['status']} ({live_g['details']['sequence_state']})")
    assert live_g['status'] == 'FAIL', f"Scenario G failed: expected FAIL, got {live_g['status']}"

    # SCENARIO H — CORRECT CHALLENGE MOVEMENT (TURN_RIGHT vs TURN_RIGHT)
    print("\n--- SCENARIO H: CORRECT CHALLENGE MOVEMENT ---")
    f_right1 = generate_synthetic_face_frame(yaw_offset=0)
    f_right2 = generate_synthetic_face_frame(yaw_offset=200) # Turn Right
    live_h = LivenessService.evaluate_challenge("TURN_RIGHT", [f_right1, f_right2])
    print(f"Details: {live_h['details']}")
    print(f"Expected TURN_RIGHT, User performed TURN_RIGHT -> Result: {live_h['status']} ({live_h['details']['sequence_state']})")
    assert live_h['status'] == 'PASS', f"Scenario H failed: expected PASS, got {live_h['status']}"

    print("\n=========================================================================")
    print(" ALL 8 ACCEPTANCE SCENARIOS (A through H) PASSED 100% SUCCESSFULLY!     ")
    print("=========================================================================")

if __name__ == "__main__":
    run_acceptance_tests()
