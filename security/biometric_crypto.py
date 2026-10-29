"""
Application-Level Authenticated Encryption (AEAD AES-256-GCM) for Biometric Templates.
Provides cryptographic confidentiality and tag-verified integrity for stored 512-d face embeddings.
"""

import os
import json
import base64
import hashlib
from typing import List, Dict, Any, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag


def _get_default_key() -> bytes:
    """Derives a 256-bit AES key from system configuration secret."""
    try:
        from app.config import settings
        secret_str = getattr(settings, "BIOMETRIC_ENCRYPTION_KEY", "default_nivora_fallback_key_2026")
    except Exception:
        secret_str = os.environ.get("NIVORA_BIOMETRIC_KEY", "default_nivora_fallback_key_2026")

    return hashlib.sha256(secret_str.encode("utf-8")).digest()


def derive_256_key(key_input: str | bytes) -> bytes:
    """Derives a deterministic 32-byte AES-256 key using SHA-256."""
    if isinstance(key_input, str):
        key_bytes = key_input.encode("utf-8")
    else:
        key_bytes = key_input

    return hashlib.sha256(key_bytes).digest()


def encrypt_template(
    embedding: List[float],
    key: Optional[bytes] = None,
    key_ref: str = "ENV_BIOMETRIC_KEY_V1"
) -> Dict[str, Any]:
    """
    Encrypts a biometric embedding (e.g. 512-d face, 128-d voice) using AES-256-GCM AEAD.
    Returns encrypted template document structure (no plaintext embedding).
    """
    if embedding is None or not isinstance(embedding, list) or len(embedding) == 0:
        raise ValueError(f"Invalid embedding for encryption: expected non-empty list of floats, got {len(embedding) if embedding else 0}.")

    if key is None:
        aes_key = _get_default_key()
    else:
        aes_key = derive_256_key(key)

    # Convert float list to compact JSON bytes
    payload_bytes = json.dumps(embedding).encode("utf-8")

    # Generate random 12-byte nonce
    nonce = os.urandom(12)

    # Encrypt payload with associated data tag
    aesgcm = AESGCM(aes_key)
    full_ct = aesgcm.encrypt(nonce, payload_bytes, associated_data=b"nivora_face_template_v1")

    # AESGCM appends 16-byte auth tag at the end of ciphertext
    ciphertext_body = full_ct[:-16]
    auth_tag = full_ct[-16:]

    return {
        "encrypted_template": base64.b64encode(ciphertext_body).decode("utf-8"),
        "nonce": base64.b64encode(nonce).decode("utf-8"),
        "authentication_tag": base64.b64encode(auth_tag).decode("utf-8"),
        "encryption_key_reference": key_ref,
        "template_version": "2.0",
        "model_version": "SpeechBrain_ECAPA_VoxCeleb_v2",
        "embedding_dim": len(embedding)
    }


def decrypt_template(
    encrypted_doc: Dict[str, Any],
    key: Optional[bytes] = None
) -> List[float]:
    """
    Decrypts an encrypted template document using AES-256-GCM.
    Verifies authentication tag and returns plaintext float list.
    """
    if not encrypted_doc or not isinstance(encrypted_doc, dict):
        raise ValueError("Invalid template document provided for decryption.")

    enc_b64 = encrypted_doc.get("encrypted_template")
    nonce_b64 = encrypted_doc.get("nonce")
    tag_b64 = encrypted_doc.get("authentication_tag")

    if not enc_b64 or not nonce_b64 or not tag_b64:
        raise ValueError("Template document missing required AEAD fields (encrypted_template, nonce, authentication_tag).")

    if key is None:
        aes_key = _get_default_key()
    else:
        aes_key = derive_256_key(key)

    try:
        ct_body = base64.b64decode(enc_b64)
        nonce = base64.b64decode(nonce_b64)
        tag = base64.b64decode(tag_b64)
    except Exception as e:
        raise ValueError(f"Base64 decoding failed for template fields: {str(e)}")

    # Reassemble full ciphertext + tag for AESGCM
    full_ct = ct_body + tag

    try:
        aesgcm = AESGCM(aes_key)
        decrypted_bytes = aesgcm.decrypt(nonce, full_ct, associated_data=b"nivora_face_template_v1")
        embedding = json.loads(decrypted_bytes.decode("utf-8"))
        if not isinstance(embedding, list) or len(embedding) == 0:
            raise ValueError("Decrypted payload format invalid: expected non-empty list of floats.")
        return embedding
    except InvalidTag:
        raise ValueError("AEAD Authentication Tag Verification Failed: Corrupted ciphertext or wrong decryption key.")
    except Exception as e:
        raise ValueError(f"Failed to decrypt biometric template: {str(e)}")
