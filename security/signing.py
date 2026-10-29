"""
Authentication Result Signing & Security Record Generation.
Integrates PQC provider into the result signature workflow.
"""

from datetime import datetime, timezone
import uuid
from typing import Dict, Any, Tuple
from security.pqc.provider import pqc_provider_instance


def create_signed_deepfake_record(
    analysis_result: Dict[str, Any],
    input_metadata: Dict[str, Any],
    model_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Construct a canonical authentication security record and attach PQC Dilithium signature.
    """
    analysis_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    
    payload = {
        "analysis_id": analysis_id,
        "timestamp": now_iso,
        "model": model_info,
        "input": input_metadata,
        "result": analysis_result,
        "quality": {
            "status": analysis_result.get("quality_status", "PASS")
        }
    }
    
    signed_record = pqc_provider_instance.sign_analysis_record(payload)
    return signed_record


def verify_deepfake_record(signed_record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Verify signature and hash integrity of a deepfake security record.
    """
    return pqc_provider_instance.verify_analysis_record(signed_record)
