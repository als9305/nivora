"""
Unit test for Post-Quantum Cryptography (PQC) & SHA3-256 signing engine.
"""

import sys
sys.path.insert(0, "d:/projects/DeepGuard")

from security.hashing import compute_sha3_256, verify_sha3_256
from security.pqc.kyber import KyberKEMProvider
from security.pqc.dilithium import DilithiumSignatureProvider
from security.pqc.provider import PQCProvider
from security.signing import create_signed_deepfake_record, verify_deepfake_record


def test_pqc_pipeline():
    print("Testing PQC Security Engine...")
    
    # 1. Test SHA3-256 Hashing
    data = {"test": 123, "user": "alice"}
    h = compute_sha3_256(data)
    assert verify_sha3_256(data, h), "SHA3-256 hash verification failed"
    print(f"[PASS] SHA3-256 Hash Digest: {h[:16]}...")
    
    # 2. Test Kyber KEM
    kyber = KyberKEMProvider(algorithm="Kyber1024")
    pk, sk = kyber.generate_keypair()
    ct, ss1 = kyber.encapsulate(pk)
    ss2 = kyber.decapsulate(ct, sk)
    assert len(ss1) == 32 and len(ss2) == 32, "Kyber shared secret length invalid"
    print("[PASS] Kyber-1024 KEM Key Exchange")
    
    # 3. Test Dilithium Signatures
    dilithium = DilithiumSignatureProvider(algorithm="Dilithium5")
    d_pk, d_sk = dilithium.generate_keypair()
    msg = b"DeepGuard Authentication Record #1001"
    sig = dilithium.sign(msg, d_sk)
    assert dilithium.verify(msg, sig, d_pk), "Dilithium signature verification failed"
    
    # Test signature failure on tampered message
    assert not dilithium.verify(b"Tampered Message", sig, d_pk), "Dilithium failed to reject tampered message"
    print("[PASS] Dilithium-5 Digital Signature Signing & Verification")
    
    # 4. Test Full Signed Analysis Record Creation & Verification
    analysis_res = {
        "fake_probability": 0.04,
        "confidence": 0.95,
        "classification": "REAL"
    }
    input_meta = {"type": "live_video", "frames": 20}
    model_info = {"name": "DeepQShield", "version": "1.0.0"}
    
    record = create_signed_deepfake_record(analysis_res, input_meta, model_info)
    is_valid, msg = verify_deepfake_record(record)
    assert is_valid, f"Record verification failed: {msg}"
    print(f"[PASS] Signed Record Verification: {msg}")
    
    # Test Tamper Detection
    record_tampered = dict(record)
    record_tampered["payload"] = dict(record["payload"])
    record_tampered["payload"]["result"]["fake_probability"] = 0.99 # Tamper score!
    
    is_valid_tampered, msg_tampered = verify_deepfake_record(record_tampered)
    assert not is_valid_tampered, "Failed to catch tampered security record!"
    print(f"[PASS] Caught Tampered Record: {msg_tampered}")
    
    print("\nALL PQC SECURITY UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_pqc_pipeline()
