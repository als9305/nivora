"""
PyTest Suite for Nivora Module 2: Face Identity Enrollment & Verification.
Tests face embedding extraction, L2 normalization, multi-face policies,
template aggregation, AEAD template encryption/decryption, security error handling,
and REST API endpoints.
"""

import os
import sys
import asyncio
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
backend_dir = os.path.join(root_dir, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.face_recognition import FaceRecognitionEngine, face_engine_instance
from video.face_detector import FaceDetector, global_face_detector
from video.quality_checker import VideoQualityChecker
from security.biometric_crypto import encrypt_template, decrypt_template, derive_256_key
from backend.app.ml.face_model import face_model_instance
from backend.app.services.face_service import FaceService
from backend.app.database.repositories import FaceTemplateRepository
from backend.app.main import app


def create_synthetic_face_image(color=(200, 180, 160)) -> np.ndarray:
    """Generates a synthetic 200x200 BGR image simulating a face crop."""
    img = np.full((200, 200, 3), 30, dtype=np.uint8)
    # Face oval
    cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, color, -1)
    # Eyes
    cv2.circle(img, (75, 85), 8, (20, 20, 20), -1)
    cv2.circle(img, (125, 85), 8, (20, 20, 20), -1)
    # Mouth
    cv2.ellipse(img, (100, 130), (25, 12), 0, 0, 180, (20, 20, 20), 3)
    return img


def cv2_to_base64(img: np.ndarray) -> str:
    import base64
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")


class TestFaceIdentityModule:

    def test_embedding_shape_and_l2_normalization(self):
        img = create_synthetic_face_image()
        emb = face_engine_instance.extract_embedding(img)
        assert emb is not None
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (512,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4

    def test_cosine_similarity_computation(self):
        img1 = create_synthetic_face_image(color=(200, 180, 160))
        img2 = create_synthetic_face_image(color=(190, 175, 155))

        emb1 = face_engine_instance.extract_embedding(img1)
        emb2 = face_engine_instance.extract_embedding(img2)

        # Self-similarity must be 1.0
        self_sim = face_engine_instance.compute_cosine_similarity(emb1, emb1)
        assert abs(self_sim - 1.0) < 1e-4

        # Similar face crops must yield high similarity
        pair_sim = face_engine_instance.compute_cosine_similarity(emb1, emb2)
        assert pair_sim > 0.85

    def test_template_aggregation(self):
        img1 = create_synthetic_face_image()
        img2 = create_synthetic_face_image(color=(210, 190, 170))

        emb1 = face_engine_instance.extract_embedding(img1)
        emb2 = face_engine_instance.extract_embedding(img2)

        template = face_engine_instance.aggregate_template_embeddings([emb1, emb2])
        assert len(template) == 512
        assert abs(np.linalg.norm(template) - 1.0) < 1e-4

    def test_quality_checker_gate(self):
        checker = VideoQualityChecker()

        # Valid face frame
        valid_img = create_synthetic_face_image()
        q_valid = checker.check_frame_quality(valid_img)
        assert q_valid["pass"] is True

        # Empty/Blurs check
        empty_img = np.zeros((200, 200, 3), dtype=np.uint8)
        q_empty = checker.check_frame_quality(empty_img)
        assert q_empty["pass"] is False
        assert q_empty["reason"] in ["BLURRY_FRAME", "EXTREME_LIGHTING"]

    def test_template_encryption_and_decryption_roundtrip(self):
        raw_emb = [0.002 * i for i in range(512)]
        
        # Test encryption
        encrypted_doc = encrypt_template(raw_emb)
        assert "encrypted_template" in encrypted_doc
        assert "nonce" in encrypted_doc
        assert "authentication_tag" in encrypted_doc
        assert "embedding" not in encrypted_doc  # Plaintext embedding must NOT be in document

        # Test successful decryption round-trip equality
        decrypted_emb = decrypt_template(encrypted_doc)
        assert len(decrypted_emb) == 512
        for orig, dec in zip(raw_emb, decrypted_emb):
            assert abs(orig - dec) < 1e-6

    def test_corrupted_ciphertext_decryption_failure(self):
        raw_emb = [0.001 * i for i in range(512)]
        encrypted_doc = encrypt_template(raw_emb)
        
        # Corrupt authentication tag
        corrupted_doc = dict(encrypted_doc)
        corrupted_doc["authentication_tag"] = "AAAA" + encrypted_doc["authentication_tag"][4:]

        with pytest.raises(ValueError, match="AEAD Authentication Tag Verification Failed"):
            decrypt_template(corrupted_doc)

    def test_wrong_encryption_key_failure(self):
        raw_emb = [0.001 * i for i in range(512)]
        key1 = b"secret_key_1_32_bytes_long_12345"
        key2 = b"secret_key_2_32_bytes_long_54321"

        encrypted_doc = encrypt_template(raw_emb, key=key1)

        with pytest.raises(ValueError, match="AEAD Authentication Tag Verification Failed"):
            decrypt_template(encrypted_doc, key=key2)

    def test_missing_encryption_key_handling(self):
        # Testing decryption when doc is invalid or missing required fields
        invalid_doc = {"user_id": "test_missing"}
        with pytest.raises(ValueError, match="missing required AEAD fields"):
            decrypt_template(invalid_doc)

    def test_face_enrollment_service_encrypted_storage(self):
        async def run_test():
            user_id = "test_user_crypto_001"
            img = create_synthetic_face_image()
            b64 = cv2_to_base64(img)

            # Enroll user with 2 samples
            res = await FaceService.enroll_face(user_id, [b64, b64])
            assert res["success"] is True

            # Verify template info API response does NOT return raw 512-D embedding
            info = await FaceService.get_template_info(user_id)
            assert info["enrolled"] is True
            assert info["template"]["encrypted"] is True
            assert "embedding" not in info["template"]

            # Delete enrolled template
            await FaceService.remove_template(user_id)

        asyncio.run(run_test())

    def test_face_verification_using_encrypted_template(self):
        async def run_test():
            user_id = "test_user_crypto_002"
            img = create_synthetic_face_image()
            b64 = cv2_to_base64(img)

            # 1. Enroll (AEAD Encrypted)
            await FaceService.enroll_face(user_id, [b64])

            # 2. Verify with probe image (decrypted safely in memory)
            verif = await FaceService.verify_face(user_id, b64)
            assert verif["success"] is True
            assert verif["matched"] is True
            assert verif["similarity_score"] >= 0.65
            assert verif["status"] == "PASS"

            # Clean up
            await FaceService.remove_template(user_id)

        asyncio.run(run_test())

    def test_face_verification_unenrolled_user(self):
        async def run_test():
            user_id = "non_existent_user_99999"
            img = create_synthetic_face_image()
            b64 = cv2_to_base64(img)

            verif = await FaceService.verify_face(user_id, b64)
            assert verif["success"] is False
            assert verif["matched"] is False
            assert verif["error_code"] == "USER_NOT_ENROLLED"

        asyncio.run(run_test())

    def test_api_face_endpoints(self):
        with TestClient(app) as client:
            user_id = "api_test_user_crypto"
            img = create_synthetic_face_image()
            b64 = cv2_to_base64(img)

            # Test POST /api/v1/face/enroll
            enroll_resp = client.post("/api/v1/face/enroll", json={
                "user_id": user_id,
                "images_base64": [b64]
            })
            assert enroll_resp.status_code == 200
            data = enroll_resp.json()
            assert data["success"] is True
            assert data["template_id"]

            # Test POST /api/v1/face/verify
            verify_resp = client.post("/api/v1/face/verify", json={
                "user_id": user_id,
                "image_base64": b64
            })
            assert verify_resp.status_code == 200
            vdata = verify_resp.json()
            assert vdata["success"] is True
            assert vdata["matched"] is True

            # Verify raw 512-D embedding is NOT in response payload
            assert "embedding" not in vdata
            assert "decrypted_template" not in vdata

            # Test GET /api/v1/face/template/{user_id}
            temp_resp = client.get(f"/api/v1/face/template/{user_id}")
            assert temp_resp.status_code == 200
            tdata = temp_resp.json()
            assert tdata["enrolled"] is True
            assert tdata["template"]["template_id"] == data["template_id"]
            assert tdata["template"]["encrypted"] is True
            assert "embedding" not in tdata["template"]

            # Test DELETE /api/v1/face/template/{user_id}
            del_resp = client.delete(f"/api/v1/face/template/{user_id}")
            assert del_resp.status_code == 200
            assert del_resp.json()["success"] is True
