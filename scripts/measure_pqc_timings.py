"""
PQC Operation Timing Measurement Script.
Measures latency statistics for SHA3-256, Kyber-1024, and Dilithium-5 operations.
"""

import sys
import time
import numpy as np

sys.path.insert(0, "d:/projects/DeepGuard")

from security.hashing import compute_sha3_256
from security.pqc.kyber import KyberKEMProvider
from security.pqc.dilithium import DilithiumSignatureProvider
from security.pqc.provider import PQCProvider


def measure_pqc_latencies(n_iters: int = 100):
    print(f"Measuring PQC Latency over {n_iters} iterations...")
    
    sha3_times = []
    kyber_keygen_times = []
    kyber_encap_times = []
    kyber_decap_times = []
    dilithium_keygen_times = []
    dilithium_sign_times = []
    dilithium_verify_times = []
    
    kyber = KyberKEMProvider(algorithm="Kyber1024")
    dilithium = DilithiumSignatureProvider(algorithm="Dilithium5")
    pqc = PQCProvider()
    
    dummy_payload = {
        "analysis_id": "test-id-12345",
        "result": {"video_fake_probability": 0.88, "classification": "DEEPFAKE"}
    }
    
    for _ in range(n_iters):
        # 1. SHA3-256
        t0 = time.perf_counter()
        _ = compute_sha3_256(dummy_payload)
        t1 = time.perf_counter()
        sha3_times.append((t1 - t0) * 1000.0)
        
        # 2. Kyber-1024 Keygen
        t0 = time.perf_counter()
        pk_k, sk_k = kyber.generate_keypair()
        t1 = time.perf_counter()
        kyber_keygen_times.append((t1 - t0) * 1000.0)
        
        # 3. Kyber-1024 Encapsulate
        t0 = time.perf_counter()
        ct, ss1 = kyber.encapsulate(pk_k)
        t1 = time.perf_counter()
        kyber_encap_times.append((t1 - t0) * 1000.0)
        
        # 4. Kyber-1024 Decapsulate
        t0 = time.perf_counter()
        _ = kyber.decapsulate(ct, sk_k)
        t1 = time.perf_counter()
        kyber_decap_times.append((t1 - t0) * 1000.0)
        
        # 5. Dilithium-5 Keygen
        t0 = time.perf_counter()
        pk_d, sk_d = dilithium.generate_keypair()
        t1 = time.perf_counter()
        dilithium_keygen_times.append((t1 - t0) * 1000.0)
        
        # 6. Dilithium-5 Sign
        msg = b"Canonical Security Record Test Bytes"
        t0 = time.perf_counter()
        sig = dilithium.sign(msg, sk_d)
        t1 = time.perf_counter()
        dilithium_sign_times.append((t1 - t0) * 1000.0)
        
        # 7. Dilithium-5 Verify
        t0 = time.perf_counter()
        _ = dilithium.verify(msg, sig, pk_d)
        t1 = time.perf_counter()
        dilithium_verify_times.append((t1 - t0) * 1000.0)
        
    print("\n--- PQC OPERATION LATENCY STATISTICS ---")
    print(f"SHA3-256 Hashing Mean Latency:           {np.mean(sha3_times):.4f} ms")
    print(f"Kyber-1024 Key Generation Mean Latency:  {np.mean(kyber_keygen_times):.4f} ms")
    print(f"Kyber-1024 Encapsulation Mean Latency:   {np.mean(kyber_encap_times):.4f} ms")
    print(f"Kyber-1024 Decapsulation Mean Latency:   {np.mean(kyber_decap_times):.4f} ms")
    print(f"Dilithium-5 Key Generation Mean Latency: {np.mean(dilithium_keygen_times):.4f} ms")
    print(f"Dilithium-5 Signing Mean Latency:        {np.mean(dilithium_sign_times):.4f} ms")
    print(f"Dilithium-5 Verification Mean Latency:   {np.mean(dilithium_verify_times):.4f} ms")


if __name__ == "__main__":
    measure_pqc_latencies()
