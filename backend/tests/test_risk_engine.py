import pytest
from app.services.risk_engine import RiskEngine

def test_risk_engine_low_risk_allow():
    # Perfect biometric & liveness scores
    result = RiskEngine.calculate_risk(
        face_score=0.98,
        liveness_score=0.95,
        deepfake_probability=0.03,
        voice_similarity=0.96,
        voice_spoof_probability=0.02,
        device_risk=10.0,
        session_risk=10.0
    )
    assert result["risk_score"] <= 35
    assert result["risk_level"] == "LOW"
    assert result["decision"] == "ALLOW"

def test_risk_engine_medium_risk_stepup():
    # Moderate anomaly (e.g. slight face discrepancy or borderline deepfake)
    result = RiskEngine.calculate_risk(
        face_score=0.65,
        liveness_score=0.60,
        deepfake_probability=0.48,
        voice_similarity=0.70,
        voice_spoof_probability=0.35,
        device_risk=30.0,
        session_risk=30.0
    )
    assert 36 <= result["risk_score"] <= 70
    assert result["risk_level"] == "MEDIUM"
    assert result["decision"] in ["STEP-UP", "STEP-UP OTP"]

def test_risk_engine_high_risk_block_deepfake():
    # High deepfake probability + low liveness
    result = RiskEngine.calculate_risk(
        face_score=0.95,
        liveness_score=0.20,
        deepfake_probability=0.92,
        voice_similarity=0.85,
        voice_spoof_probability=0.75,
        device_risk=50.0,
        session_risk=50.0
    )
    assert result["risk_score"] >= 71
    assert result["risk_level"] == "HIGH"
    assert result["decision"] == "BLOCK"
    assert any("deepfake" in r.lower() or "liveness" in r.lower() for r in result["reasons"])
