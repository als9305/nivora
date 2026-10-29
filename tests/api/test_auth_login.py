"""Credential login regression tests for the production auth route."""

import uuid

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_register_then_login_uses_same_password_hash_and_user_store():
    email = f"login-{uuid.uuid4().hex}@example.test"
    password = "CorrectHorseBatteryStaple!42"

    registered = client.post("/api/auth/register", json={
        "name": "Login Test",
        "email": email,
        "password": password,
    })
    assert registered.status_code == 200
    user_id = registered.json()["user_id"]

    logged_in = client.post("/api/auth/login", json={"email": email, "password": password})
    assert logged_in.status_code == 200
    body = logged_in.json()
    assert body["success"] is True
    assert body["user_id"] == user_id
    assert body["email"] == email


def test_login_rejects_wrong_password_and_unknown_user_without_session():
    email = f"login-{uuid.uuid4().hex}@example.test"
    password = "CorrectHorseBatteryStaple!42"
    client.post("/api/auth/register", json={"email": email, "password": password})

    wrong_password = client.post("/api/auth/login", json={"email": email, "password": "incorrect"})
    assert wrong_password.status_code == 401
    assert wrong_password.json()["detail"]["message"] == "Invalid email or password."

    unknown = client.post("/api/auth/login", json={"email": f"missing-{uuid.uuid4().hex}@example.test", "password": password})
    assert unknown.status_code == 401
    assert unknown.json()["detail"]["message"] == "Invalid email or password."


def test_login_rejects_empty_fields_with_validation_error():
    response = client.post("/api/auth/login", json={"email": "", "password": ""})
    assert response.status_code == 422


def test_registration_requires_real_credentials_and_does_not_activate_duplicates():
    missing = client.post("/api/auth/register", json={})
    assert missing.status_code == 422

    email = f"duplicate-{uuid.uuid4().hex}@example.test"
    payload = {"name": "Duplicate Test", "email": email, "password": "CorrectHorseBatteryStaple!42"}
    assert client.post("/api/auth/register", json=payload).status_code == 200
    duplicate = client.post("/api/auth/register", json=payload)
    assert duplicate.status_code == 409
