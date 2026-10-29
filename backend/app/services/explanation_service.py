from typing import Dict, Any, List
from app.config import settings
from app.utils.logging import logger

class ExplanationService:
    @staticmethod
    async def generate_explanation(risk_result: Dict[str, Any], verification_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates security decision report & evidence explanation.
        Uses Gemini LLM if GEMINI_API_KEY is provided, otherwise generates structured deterministic report.
        """
        decision = risk_result.get("decision", "BLOCK")
        risk_score = risk_result.get("risk_score", 0)
        risk_level = risk_result.get("risk_level", "HIGH")
        reasons = risk_result.get("reasons", [])

        face_score = verification_data.get("face_score", 0.0)
        liveness_score = verification_data.get("liveness_score", 0.0)
        deepfake_prob = verification_data.get("deepfake_probability", 0.0)
        voice_sim = verification_data.get("voice_similarity", 0.0)
        voice_spoof = verification_data.get("voice_spoof_probability", 0.0)

        # Attempt Gemini LLM call if API key present
        if settings.GEMINI_API_KEY:
            try:
                import google.genai as genai
                client = genai.Client(api_key=settings.GEMINI_API_KEY)
                
                prompt = f"""
You are an expert Cybersecurity Security Decision Agent.
Analyze this authentication attempt telemetry:
- Overall Risk Score: {risk_score}/100 ({risk_level})
- Decision Made by Engine: {decision}
- Face Match Score: {int(face_score * 100)}%
- Liveness Score: {int(liveness_score * 100)}%
- Deepfake Probability: {int(deepfake_prob * 100)}%
- Voice Match: {int(voice_sim * 100)}%
- Voice Spoof Risk: {int(voice_spoof * 100)}%
- Risk Reasons: {", ".join(reasons)}

Provide a concise cybersecurity analyst response formatted as:
THREAT SUMMARY: <1 sentence threat classification>
EVIDENCE:
- <bullet 1>
- <bullet 2>
WHY DECISION WAS MADE: <1-2 sentences explaining why engine chose {decision}>
RECOMMENDED ACTION: <1 sentence recommended security team response>
"""
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                if response and response.text:
                    return {
                        "threat_summary": response.text,
                        "evidence": reasons,
                        "why_decision_made": f"Deterministic risk score of {risk_score}/100 triggered {decision} policy.",
                        "recommendation": "Review security audit logs if recurring attack vector.",
                        "is_llm_generated": True
                    }
            except Exception as e:
                logger.warning(f"Gemini LLM call failed or key invalid: {e}. Utilizing deterministic fallback explanation.")

        # Deterministic structured explanation fallback
        if decision == "ALLOW":
            summary = "AUTHENTIC USER TRANSACTION CONFIRMED"
            why = f"Overall composite risk score is low ({risk_score}/100). All identity, liveness, and deepfake verification checks met strict security thresholds."
            rec = "Grant standard session privileges and issue access token."
        elif decision == "STEP-UP":
            summary = "SUSPICIOUS AUTHENTICATION - ELEVATED BIOMETRIC ANOMALY"
            why = f"Overall risk score ({risk_score}/100) falls in MEDIUM risk band. Secondary verification required due to minor biometric variance."
            rec = "Enforce One-Time Password (OTP) multi-factor step-up verification before granting session."
        else: # BLOCK
            if deepfake_prob > 0.60:
                summary = "CRITICAL THREAT: PROBABLE SYNTHETIC MEDIA / DEEPFAKE REPLAY ATTACK"
            elif liveness_score < 0.30:
                summary = "CRITICAL THREAT: PASSIVE PHOTO REPLAY / LIVENESS CHALLENGE FAILURE"
            else:
                summary = "HIGH SECURITY RISK: MULTI-MODAL BIOMETRIC MISMATCH"

            why = f"Authentication attempt generated high risk score ({risk_score}/100). " + " ".join(reasons)
            rec = "Block session immediately. Log security incident for SOC review."

        evidence_list = [
            f"Face Match Similarity: {int(face_score * 100)}%",
            f"Active Liveness Score: {int(liveness_score * 100)}%",
            f"Deepfake Probability: {int(deepfake_prob * 100)}%",
            f"Voice Similarity: {int(voice_sim * 100)}%",
            f"Voice Anti-Spoof Risk: {int(voice_spoof * 100)}%"
        ] + reasons

        return {
            "threat_summary": summary,
            "evidence": evidence_list,
            "why_decision_made": why,
            "recommendation": rec,
            "is_llm_generated": False
        }
