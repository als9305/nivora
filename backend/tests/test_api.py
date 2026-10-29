from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "DeepGuard Security API"

def test_risk_calculation_api():
    payload = {
        "face_score": 0.95,
        "liveness_score": 0.90,
        "deepfake_probability": 0.05,
        "voice_similarity": 0.92,
        "voice_spoof_probability": 0.05,
        "device_risk": 10.0,
        "session_risk": 10.0
    }
    response = client.post("/api/auth/calculate-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "ALLOW"
    assert "risk_score" in data

def test_attack_simulation_api():
    response = client.post("/api/auth/simulate?simulation_type=DEEPFAKE_ATTACK")
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "BLOCK"
    assert data["attack_type"] == "DEEPFAKE_ATTACK"
