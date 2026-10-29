"""
Speech Recognition & Challenge Phrase Verification Service for Nivora Module 4.
Uses faster-whisper (tiny.en quantized int8) for low-latency (<250ms), offline-capable
automatic speech recognition with phonetic and fuzzy matching for pronunciation and accent tolerance.
"""

import re
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from app.utils.logging import logger

_asr_model = None

def get_asr_model():
    """Singleton provider for WhisperModel (tiny.en int8 on CPU)."""
    global _asr_model
    if _asr_model is None:
        try:
            from faster_whisper import WhisperModel
            logger.info("Initializing faster-whisper tiny.en model on CPU (int8)...")
            _asr_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            logger.info("faster-whisper model initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize faster-whisper ASR model: {e}")
            _asr_model = None
    return _asr_model


class SpeechRecognitionService:
    """
    Evaluates spoken challenge phrases against server-issued one-time challenge tokens.
    Provides honest, independent phrase verification with tolerance for Indian English accents.
    """

    @staticmethod
    def transcribe_audio_array(audio_float32: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
        """
        Transcribes a 16kHz mono float32 audio array directly in memory.
        Bypasses container decoding for maximum speed and compatibility.
        """
        model = get_asr_model()
        if model is None:
            return "", 0.0

        if audio_float32 is None or len(audio_float32) < int(sample_rate * 0.3):
            return "", 0.0

        try:
            # Resample if not 16000
            if sample_rate != 16000 and sample_rate > 0:
                import scipy.signal
                from math import gcd
                g = gcd(int(sample_rate), 16000)
                audio_float32 = scipy.signal.resample_poly(
                    audio_float32, 16000 // g, int(sample_rate) // g
                ).astype(np.float32)

            segments, info = model.transcribe(audio_float32, beam_size=1, language="en")
            transcription = " ".join(s.text for s in segments).strip()
            confidence = float(info.language_probability) if hasattr(info, "language_probability") else 0.95
            return transcription, confidence
        except Exception as e:
            logger.error(f"Error during ASR transcription: {e}")
            return "", 0.0

    @staticmethod
    def evaluate_phrase(transcribed_text: str, challenge_phrase: str) -> Dict[str, Any]:
        """
        Evaluates whether transcribed speech matches the challenge phrase with tolerance
        for Indian English accents, minor phoneme variations, and ASR compounding.
        
        Statuses:
          - MATCHED: Phrase verified against challenge words.
          - MISMATCH: Spoken words clearly contradict the challenge phrase.
          - INCONCLUSIVE: Partial match, muffled, or ambiguous utterance.
          - UNVALIDATED: ASR not available or no challenge issued.
        """
        if not challenge_phrase or not challenge_phrase.strip():
            return {
                "phrase_match": "UNVALIDATED",
                "phrase_match_score": 0.0,
                "transcribed_text": transcribed_text or "",
                "challenge_phrase": "",
                "matched_words": "0/0",
                "reason": "No challenge phrase issued for this session."
            }

        if not transcribed_text or not transcribed_text.strip():
            return {
                "phrase_match": "INCONCLUSIVE",
                "phrase_match_score": 0.0,
                "transcribed_text": "",
                "challenge_phrase": challenge_phrase,
                "matched_words": f"0/{len(challenge_phrase.split())}",
                "reason": "No intelligible speech detected in recording."
            }

        try:
            from rapidfuzz import fuzz
        except ImportError:
            # Fallback to difflib if rapidfuzz is unavailable
            import difflib
            class _FuzzFallback:
                @staticmethod
                def ratio(s1, s2):
                    return int(difflib.SequenceMatcher(None, s1, s2).ratio() * 100)
                @staticmethod
                def token_set_ratio(s1, s2):
                    return int(difflib.SequenceMatcher(None, s1, s2).ratio() * 100)
            fuzz = _FuzzFallback()

        # Normalize strings
        clean_target = re.sub(r"[^\w\s]", "", challenge_phrase.lower()).strip()
        clean_spoken = re.sub(r"[^\w\s]", "", transcribed_text.lower()).strip()

        target_words = clean_target.split()
        spoken_words = clean_spoken.split()

        if not target_words or not spoken_words:
            return {
                "phrase_match": "INCONCLUSIVE",
                "phrase_match_score": 0.0,
                "transcribed_text": transcribed_text,
                "challenge_phrase": challenge_phrase,
                "matched_words": f"0/{len(target_words)}",
                "reason": "Insufficient speech tokens to validate challenge."
            }

        # Word-level fuzzy matching with pronunciation tolerance
        matched_words = 0
        word_scores = []
        for tw in target_words:
            best_w_score = max((fuzz.ratio(tw, sw) for sw in spoken_words), default=0)
            word_scores.append(best_w_score)
            # 70% allows for minor phonetic mishearings (e.g. 'mangoe' vs 'mango', 'colud' vs 'cloud', 'flowr' vs 'flower')
            if best_w_score >= 70:
                matched_words += 1

        # Full phrase token metrics
        set_ratio = fuzz.token_set_ratio(clean_target, clean_spoken)
        avg_word_score = sum(word_scores) / len(word_scores) if word_scores else 0.0
        combined_score = round(float(0.5 * avg_word_score + 0.5 * set_ratio) / 100.0, 4)

        total_target_words = len(target_words)
        matched_ratio = matched_words / total_target_words

        # Decision: MATCHED if challenge words are confirmed; MISMATCHED if spoken words do not match
        is_matched = (
            (matched_words == total_target_words and combined_score >= 0.55) or
            (matched_words >= 2 and combined_score >= 0.60) or
            (total_target_words <= 2 and matched_words >= 1 and combined_score >= 0.65) or
            (combined_score >= 0.70)
        )

        if is_matched:
            verdict = "MATCHED"
            reason = f"Challenge phrase confirmed ({matched_words}/{total_target_words} words matched, score: {combined_score:.2f})."
        else:
            verdict = "MISMATCHED"
            reason = f"Spoken words '{transcribed_text}' do not match challenge phrase (score: {combined_score:.2f})."

        return {
            "phrase_match": verdict,
            "phrase_match_score": combined_score,
            "transcribed_text": transcribed_text,
            "challenge_phrase": challenge_phrase,
            "matched_words": f"{matched_words}/{total_target_words}",
            "reason": reason
        }

    @classmethod
    def verify_challenge(cls, audio_float32: np.ndarray, challenge_phrase: Optional[str], sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Complete pipeline: transcribe audio and evaluate against challenge phrase.
        If ASR is unavailable, retains INCONCLUSIVE without falsely claiming match.
        """
        if not challenge_phrase:
            return {
                "phrase_match": "UNVALIDATED",
                "phrase_match_score": 0.0,
                "transcribed_text": "",
                "challenge_phrase": "",
                "matched_words": "0/0",
                "reason": "No challenge phrase specified."
            }

        model = get_asr_model()
        if model is None:
            return {
                "phrase_match": "INCONCLUSIVE",
                "phrase_match_score": 0.0,
                "transcribed_text": "",
                "challenge_phrase": challenge_phrase,
                "matched_words": f"0/{len(challenge_phrase.split()) if challenge_phrase else 0}",
                "reason": "ASR engine unavailable for phrase verification."
            }

        transcribed_text, _ = cls.transcribe_audio_array(audio_float32, sample_rate)
        return cls.evaluate_phrase(transcribed_text, challenge_phrase)
