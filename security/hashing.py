"""
Cryptographic Hashing & Canonicalization Module.
Provides SHA3-256 hashing for tamper-proof evidence integrity verification.
"""

import hashlib
import json
from typing import Any, Dict


def canonicalize_payload(payload: Dict[str, Any]) -> bytes:
    """
    Produce deterministic UTF-8 byte representation of a dictionary payload
    by sorting keys and eliminating whitespace formatting differences.
    """
    canonical_json = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return canonical_json.encode('utf-8')


def compute_sha3_256(payload: Dict[str, Any]) -> str:
    """
    Calculate the SHA3-256 hash hex string of a canonical payload.
    """
    data_bytes = canonicalize_payload(payload)
    hasher = hashlib.sha3_256()
    hasher.update(data_bytes)
    return hasher.hexdigest()


def verify_sha3_256(payload: Dict[str, Any], expected_hash: str) -> bool:
    """
    Verify if a payload matches an expected SHA3-256 hash digest.
    """
    computed = compute_sha3_256(payload)
    return computed.lower() == expected_hash.lower()
