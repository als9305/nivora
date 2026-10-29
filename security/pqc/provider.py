"""
Unified Post-Quantum Cryptography (PQC) Security Provider.
Orchestrates Kyber1024 KEM and Dilithium5 Digital Signatures with unified key management.
Matches exact DeepQShield research paper specification.
"""

import hashlib
from typing import Dict, Any, Tuple, Optional
from security.pqc.kyber import KyberKEMProvider
from security.pqc.dilithium import DilithiumSignatureProvider
from security.hashing import canonicalize_payload, compute_sha3_256, verify_sha3_256


class PQCProvider:
    """
    Unified PQC Security Engine (Kyber-1024 & Dilithium-5).
    """
    def __init__(self, kem_algo: str = "Kyber1024", sig_algo: str = "Dilithium5"):
        self.kem_provider = KyberKEMProvider(algorithm=kem_algo)
        self.sig_provider = DilithiumSignatureProvider(algorithm=sig_algo)
        
        # System service keypair initialization
        self.sig_pk, self.sig_sk = self.sig_provider.generate_keypair()
        self.kem_pk, self.kem_sk = self.kem_provider.generate_keypair()

    def get_public_keys(self) -> Dict[str, str]:
        """Expose system public keys in hex format."""
        return {
            "dilithium_public_key_hex": self.sig_pk.hex(),
            "kyber_public_key_hex": self.kem_pk.hex(),
            "algorithms": {
                "kem": self.kem_provider.algorithm,
                "signature": self.sig_provider.algorithm,
                "hashing": "SHA3-256"
            }
        }

    def sign_analysis_record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a signed audit record containing SHA3-256 digest, Dilithium-5 signature,
        and Kyber-1024 session key encapsulation.
        """
        canonical_bytes = canonicalize_payload(payload)
        digest = compute_sha3_256(payload)
        
        sig_bytes = self.sig_provider.sign(canonical_bytes, self.sig_sk)
        
        # Execute genuine Kyber-1024 Key Encapsulation (ML-KEM-1024)
        ct, ss = self.kem_provider.encapsulate(self.kem_pk)
        ss_sha3 = hashlib.sha3_256(ss).hexdigest()
        
        kyber_kem = {
            "algorithm": self.kem_provider.algorithm,
            "public_key_hex": self.kem_pk.hex(),
            "ciphertext_hex": ct.hex(),
            "ciphertext_length": len(ct),
            "shared_secret_sha3": ss_sha3,
            "status": "KEM_VERIFIED"
        }
        
        return {
            "payload": payload,
            "kyber_kem": kyber_kem,
            "dilithium_signature": sig_bytes.hex(),
            "sha3_256_digest": digest,
            "integrity": {
                "hash_algorithm": "SHA3-256",
                "sha3_256_digest": digest,
                "signature_algorithm": self.sig_provider.algorithm,
                "signature_hex": sig_bytes.hex(),
                "public_key_hex": self.sig_pk.hex(),
                "kyber_kem": kyber_kem
            }
        }

    def verify_analysis_record(self, signed_record: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verify the integrity and authenticity of a signed security record.
        """
        if "payload" not in signed_record or "integrity" not in signed_record:
            return False, "INVALID_RECORD_STRUCTURE"
            
        payload = signed_record["payload"]
        integrity = signed_record["integrity"]
        
        expected_digest = integrity.get("sha3_256_digest", "")
        sig_hex = integrity.get("signature_hex", "")
        pk_hex = integrity.get("public_key_hex", "")
        
        if not verify_sha3_256(payload, expected_digest):
            return False, "SHA3_DIGEST_MISMATCH_TAMPERED_PAYLOAD"
            
        try:
            canonical_bytes = canonicalize_payload(payload)
            sig_bytes = bytes.fromhex(sig_hex)
            pk_bytes = bytes.fromhex(pk_hex)
            
            is_valid = self.sig_provider.verify(canonical_bytes, sig_bytes, pk_bytes)
            if not is_valid:
                return False, "DILITHIUM5_SIGNATURE_INVALID"
        except Exception as e:
            return False, f"PQC_VERIFICATION_EXCEPTION: {str(e)}"

        # Verify Kyber-1024 decapsulation if KEM evidence is present
        kyber_data = signed_record.get("kyber_kem") or integrity.get("kyber_kem")
        if kyber_data and isinstance(kyber_data, dict) and "ciphertext_hex" in kyber_data:
            try:
                ct_bytes = bytes.fromhex(kyber_data["ciphertext_hex"])
                ss_rec = self.kem_provider.decapsulate(ct_bytes, self.kem_sk)
                ss_expected = kyber_data.get("shared_secret_sha3")
                if ss_expected and hashlib.sha3_256(ss_rec).hexdigest() != ss_expected:
                    return False, "KYBER_KEM_SHARED_SECRET_MISMATCH"
            except Exception as e:
                return False, f"KYBER_KEM_VERIFICATION_EXCEPTION: {str(e)}"

        return True, "VERIFICATION_SUCCESSFUL"


# Singleton PQC provider instance
pqc_provider_instance = PQCProvider()
