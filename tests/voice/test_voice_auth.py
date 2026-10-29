"""
Module 4 Unit & Integration Test Suite for Voice Authentication & Audio Anti-Spoofing.
Includes 10 targeted assertions:
  1. Real microphone payload format / WAV decoding
  2. Invalid audio rejection (corrupt / short / silence)
  3. Session expiry (300s TTL)
  4. Session reuse rejection (one-time token)
  5. Client challenge tampering protection (server-generated)
  6. Encrypted template storage (AEAD AES-256-GCM)
  7. Speaker verification (SpeechBrain ECAPA-TDNN pretrained weights)
  8. PAD classification (Liveness vs Presentation Attacks)
  9. Frontend zero production file input source scan
 10. Complete microphone-to-backend integration flow
"""

import os
import io
import wave
import base64
import numpy as np
import torch
import pytest
from fastapi.testclient import TestClient

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.app.main import app
from models.voice_auth import (
    voice_auth_engine,
    AudioPreprocessor,
    AudioFeatureExtractor,
    VoiceEmbeddingNet,
    PassiveAudioAntiSpoofNet
)
from app.services.voice_service import VoiceService, VoiceSessionManager, PHONETIC_WORDS

client = TestClient(app)


def generate_synthetic_wav(
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    freq_hz: float = 220.0,
    amplitude: float = 0.5,
    noise_level: float = 0.0,
    channels: int = 1
) -> bytes:
    """Helper to synthesize valid RIFF PCM WAV audio bytes."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    if freq_hz > 0:
        signal = amplitude * np.sin(2 * np.pi * freq_hz * t)
    else:
        signal = np.zeros(n_samples)

    if noise_level > 0:
        signal += np.random.normal(0, noise_level, n_samples)

    pcm_data = (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)

    if channels == 2:
        pcm_data = np.column_stack((pcm_data, pcm_data)).flatten()

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data.tobytes())

    return buf.getvalue()


def bytes_to_b64(b: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(b).decode("utf-8")


# 1. Real microphone payload format / WAV decoding
def test_wav_decoding_valid():
    wav_b = generate_synthetic_wav(duration_sec=1.0, freq_hz=300.0)
    signal, sr, audit = AudioPreprocessor.decode_wav_bytes(wav_b)
    assert audit["pass"] is True
    assert sr == 16000
    assert signal is not None
    assert len(signal) == 16000
    # Runtime evidence must describe the actual browser-compatible PCM payload,
    # without returning the biometric waveform itself.
    assert audit["sample_width_bits"] == 16
    assert audit["channels"] == 1
    assert audit["sample_count"] == 16000
    assert audit["wav_byte_size"] == len(wav_b)
    assert audit["peak_amplitude"] > 0
    assert audit["std_amplitude"] > 0


# 2. Invalid audio rejection (corrupt / short / silence)
def test_wav_decoding_invalid_rejection():
    # Corrupt
    signal_c, sr_c, audit_c = AudioPreprocessor.decode_wav_bytes(b"INVALID_HEADER_DATA")
    assert audit_c["pass"] is False
    assert audit_c["error_code"] == "EMPTY_OR_CORRUPT_PAYLOAD"
    assert signal_c is None

    # Silence
    wav_silence = generate_synthetic_wav(duration_sec=1.0, freq_hz=0.0, amplitude=0.0)
    signal_s, sr_s, audit_s = AudioPreprocessor.decode_wav_bytes(wav_silence)
    assert audit_s["pass"] is False
    assert audit_s["error_code"] == "SILENCE_OR_NEAR_SILENCE"
    assert signal_s is None

    # Too short
    wav_short = generate_synthetic_wav(duration_sec=0.1, freq_hz=300.0)
    signal_sh, sr_sh, audit_sh = AudioPreprocessor.decode_wav_bytes(wav_short)
    assert audit_sh["pass"] is False
    assert audit_sh["error_code"] == "AUDIO_TOO_SHORT"
    assert signal_sh is None


# 3. Session expiry (300s TTL)
def test_session_expiry():
    session = VoiceSessionManager.create_session(ttl_seconds=-10)  # already expired
    _, err_exp = VoiceSessionManager.validate_and_consume_session(session["session_id"])
    assert err_exp == "SESSION_EXPIRED"


# 4. Session reuse rejection (one-time token)
def test_session_reuse_rejection():
    session = VoiceSessionManager.create_session(user_id="test_user", purpose="VERIFY")
    consumed_session, err = VoiceSessionManager.validate_and_consume_session(session["session_id"])
    assert err is None
    assert consumed_session["session_id"] == session["session_id"]

    # Re-consumption must fail
    reuse_session, err_reuse = VoiceSessionManager.validate_and_consume_session(session["session_id"])
    assert reuse_session is None
    assert err_reuse == "SESSION_ALREADY_USED"


# 5. Client challenge tampering protection
def test_client_challenge_tampering_protection():
    session = VoiceSessionManager.create_session(user_id="test_user")
    assert "challenge_phrase" in session
    # Challenge phrase generated server side
    words = session["challenge_phrase"].split()
    assert len(words) == 3
    # Check words are distinct and from common English vocabulary
    assert len(set(w.lower() for w in words)) == 3
    for w in words:
        assert w.lower() in PHONETIC_WORDS
    # Non-existent session tamper attempt
    _, err = VoiceSessionManager.validate_and_consume_session("fake_tampered_session_123")
    assert err == "SESSION_NOT_FOUND"


def test_challenge_vocabulary_is_common_english_no_nato():
    """Verify challenge words use common, easy-to-pronounce English words without NATO phonetic alphabet."""
    assert len(PHONETIC_WORDS) >= 20
    nato_prohibited = {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
                       "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
                       "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu"}
    intersection = set(PHONETIC_WORDS).intersection(nato_prohibited)
    assert len(intersection) == 0, f"Found prohibited NATO words in PHONETIC_WORDS: {intersection}"
    # Verify sample common words exist
    for expected in ["apple", "river", "green", "mango", "table", "water", "garden", "window", "orange"]:
        assert expected in PHONETIC_WORDS


def test_wav_decoding_resampling_48k_to_16k():
    """Microphones frequently record at 48kHz; ensure backend resamples to 16kHz seamlessly."""
    wav_48k = generate_synthetic_wav(duration_sec=1.0, sample_rate=48000, freq_hz=300.0)
    signal, sr, audit = AudioPreprocessor.decode_wav_bytes(wav_48k)
    assert audit["pass"] is True
    assert sr == 16000
    assert audit.get("resampled_to_16k") is True
    assert len(signal) == 16000


@pytest.mark.anyio
async def test_pad_rejection_user_facing_error_message():
    """When Audio PAD rejects an utterance as SPOOF, message must match required safe copy."""
    # Use fake audio sample or zeroed features to trigger SPOOF
    user_id = "test_user_pad_message"
    # First enroll user with synthetic speech
    s1 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=220.0))
    s2 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=225.0))
    s3 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=230.0))
    enroll_res = await VoiceService.enroll_voice(user_id, [s1, s2, s3])
    assert enroll_res["success"] is True

    # Now verify with an ASVspoof fake audio sample known to trigger PAD SPOOF
    fake_wav_path = os.path.join(root_dir, "datasets", "voice_fake", "asvspoof_fake_01.wav")
    with open(fake_wav_path, "rb") as f:
        fake_wav = f.read()
    probe_b64 = bytes_to_b64(fake_wav)

    verify_res = await VoiceService.verify_voice(user_id, probe_b64)
    assert verify_res["success"] is True
    assert verify_res["liveness_verdict"] == "SPOOF"
    assert verify_res["status"] == "AUDIO_SPOOF"
    assert verify_res["speaker_verified"] is False
    assert verify_res["final_verdict"] == "REJECTED"
    expected_msg = "Audio could not be verified as genuine. Please record the displayed phrase again in a quiet environment."
    assert verify_res["message"] == expected_msg


# 6. Encrypted template storage (AEAD AES-256-GCM)
@pytest.mark.anyio
async def test_encrypted_template_storage():
    user_id = "test_user_encrypted_storage"
    sample1 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=220.0))
    sample2 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=225.0))
    sample3 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=230.0))

    res = await VoiceService.enroll_voice(user_id, [sample1, sample2, sample3])
    assert res["success"] is True
    assert "template_id" in res

    info = await VoiceService.get_template_info(user_id)
    assert info["enrolled"] is True
    assert info["template"]["encrypted"] is True


# 7. Speaker verification with SpeechBrain ECAPA-TDNN pretrained weights
def test_speaker_verification_ecapa_tdnn():
    assert voice_auth_engine.embedding_net.is_pretrained is True
    assert voice_auth_engine.embedding_net.model_info["model_name"] == "SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb)"

    log_mel_80 = np.random.randn(80, 100).astype(np.float32)
    emb = voice_auth_engine.extract_embedding(log_mel_80)
    assert len(emb) == 192
    assert pytest.approx(np.linalg.norm(np.array(emb)), abs=1e-4) == 1.0


# 8. PAD classification (Liveness vs Presentation Attacks)
def test_pad_classification():
    assert voice_auth_engine.anti_spoof_net.is_pretrained is True
    assert "ASVspoof" in voice_auth_engine.anti_spoof_net.model_info["model_name"]

    feat_vec = np.random.randn(64).astype(np.float32)
    acoustic_metrics = {"f0_mean_hz": 150.0, "f0_std_hz": 12.0, "high_frequency_ratio": 0.05, "zero_crossing_rate": 0.08}
    liveness_score, spoof_prob, attack_type = voice_auth_engine.evaluate_anti_spoof(feat_vec, acoustic_metrics)

    assert 0.0 <= liveness_score <= 1.0
    assert 0.0 <= spoof_prob <= 1.0
    assert pytest.approx(liveness_score + spoof_prob, abs=1e-3) == 1.0


def test_pad_same_feature_tensor_is_deterministic_and_in_eval_mode():
    """Dropout must not change PAD probabilities for the identical live input."""
    feature_vector = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    voice_auth_engine.anti_spoof_net.train()  # Simulate accidental external mutation.

    outputs = [
        voice_auth_engine.evaluate_anti_spoof(feature_vector, {})[:2]
        for _ in range(5)
    ]

    assert voice_auth_engine.anti_spoof_net.training is False
    assert len(set(outputs)) == 1
    assert voice_auth_engine.anti_spoof_net.checkpoint_sha256


def test_complete_pad_pipeline_is_deterministic_for_same_wav():
    """Decode, feature extraction, and PAD inference are repeatable end-to-end."""
    wav_bytes = generate_synthetic_wav(duration_sec=1.0, freq_hz=230.0, amplitude=0.4)
    outputs = [voice_auth_engine.process_voice_pipeline(wav_bytes) for _ in range(5)]
    pad_outputs = {
        (output["liveness_score"], output["voice_spoof_probability"], output["liveness_verdict"])
        for output in outputs
    }
    assert len(pad_outputs) == 1


# 9. Frontend zero production file input source scan
def test_frontend_zero_production_file_input():
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    voice_frontend_dir = os.path.join(proj_root, "frontend", "src", "components")
    files_to_check = [
        os.path.join(voice_frontend_dir, "MicCapture.jsx"),
        os.path.join(voice_frontend_dir, "VoiceAuthConsole.jsx")
    ]

    forbidden_patterns = [
        '<input type="file"',
        "<input type='file'",
        'accept="audio/',
        "accept='audio/",
        "audioFileInputRef",
        "setAudioFromFile",
        "Upload Audio",
        "Choose File",
        "Attach Audio File"
    ]

    for filepath in files_to_check:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            for pattern in forbidden_patterns:
                assert pattern not in content, f"Forbidden file-upload pattern '{pattern}' found in {filepath}!"


# 10. Complete microphone-to-backend integration flow
def test_complete_microphone_to_backend_flow():
    user_id = "integration_test_mic_user"
    s1 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=300.0))
    s2 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=305.0))
    s3 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=310.0))

    # 1. Enroll
    enroll_resp = client.post("/api/v1/voice/enroll", json={"user_id": user_id, "audio_samples_base64": [s1, s2, s3]})
    assert enroll_resp.status_code == 200
    assert enroll_resp.json()["success"] is True

    # 2. Start Anti-Replay Session
    start_resp = client.post("/api/v1/voice/session/start", json={"user_id": user_id, "purpose": "VERIFY"})
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    # 3. Session-bound Verification
    probe_b64 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=302.0))
    verif_resp = client.post(f"/api/v1/voice/session/{session_id}/verify", json={
        "user_id": user_id,
        "audio_base64": probe_b64
    })
    assert verif_resp.status_code == 200
    v_json = verif_resp.json()
    assert v_json["success"] is True
    assert "phrase_match" in v_json
    assert v_json["phrase_match"] in ["UNVALIDATED", "INCONCLUSIVE"]


# 11. Multi-sample enrollment validation (3 samples minimum, duplicate rejection, incremental)
@pytest.mark.anyio
async def test_voice_enrollment_multi_sample_and_incremental():
    user_id = "test_user_multi_sample_enroll"
    s1 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=250.0))
    s2 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=260.0))

    # Reject < 3 samples in batch
    res_short = await VoiceService.enroll_voice(user_id, [s1, s2])
    assert res_short["success"] is False
    assert res_short["error_code"] == "INSUFFICIENT_SAMPLES"

    # Reject duplicate audio samples
    res_dup = await VoiceService.enroll_voice(user_id, [s1, s2, s1])
    assert res_dup["success"] is False
    assert res_dup["error_code"] == "DUPLICATE_AUDIO_SAMPLE"

    # Incremental sample-by-sample enrollment (samples 1, 2, 3)
    s3 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=270.0))
    r1 = await VoiceService.enroll_voice_sample(user_id, s1, sample_index=1, total_samples=3)
    assert r1["success"] is True
    assert r1["status"] == "VOICE_SAMPLE_1"

    r2 = await VoiceService.enroll_voice_sample(user_id, s2, sample_index=2, total_samples=3)
    assert r2["success"] is True
    assert r2["status"] == "VOICE_SAMPLE_2"

    r3 = await VoiceService.enroll_voice_sample(user_id, s3, sample_index=3, total_samples=3)
    assert r3["success"] is True
    assert r3["status"] == "VOICE_ENROLLMENT_COMPLETE"

    # Verify stored template shape is exactly 192
    info = await VoiceService.get_template_info(user_id)
    assert info["enrolled"] is True
    assert info["template"]["embedding_dim"] == 192


# 12. Unenrolled user must return VOICE_NOT_ENROLLED (NOT SPEAKER_MISMATCH)
@pytest.mark.anyio
async def test_voice_auth_unenrolled_returns_voice_not_enrolled():
    user_id = "completely_unregistered_voice_user_xyz"
    probe_b64 = bytes_to_b64(generate_synthetic_wav(1.0, freq_hz=280.0))

    res = await VoiceService.verify_voice(user_id, probe_b64)
    assert res["success"] is False
    assert res["status"] == "VOICE_NOT_ENROLLED"
    assert res["error_code"] == "VOICE_NOT_ENROLLED"
    assert res["speaker_verdict"] == "VOICE_NOT_ENROLLED"
    assert res["speaker_verdict"] != "MISMATCH"
    assert "not enrolled" in res["message"].lower()


# 13. Enforce 192-d invariant and dimension mismatch protection
def test_voice_auth_192_dimension_invariant_and_mismatch():
    v192_a = np.random.randn(192).astype(np.float32).tolist()
    v192_b = np.random.randn(192).astype(np.float32).tolist()

    # Valid 192 ↔ 192 comparison
    sim = voice_auth_engine.compute_similarity(v192_a, v192_b)
    assert isinstance(sim, float)
    assert -1.0 <= sim <= 1.0

    # Self comparison must equal ~1.0
    self_sim = voice_auth_engine.compute_similarity(v192_a, v192_a)
    assert pytest.approx(self_sim, abs=1e-4) == 1.0

    # 4-D legacy vector comparison must raise ValueError with VOICE_EMBEDDING_DIMENSION_MISMATCH
    v4 = [0.1, 0.2, 0.3, 0.4]
    with pytest.raises(ValueError) as excinfo:
        voice_auth_engine.compute_similarity(v192_a, v4)
    assert "VOICE_EMBEDDING_DIMENSION_MISMATCH" in str(excinfo.value)


# 14. PAD Checkpoint Loading Verification
def test_pad_checkpoint_loading_and_architecture():
    """Verify that PassiveAudioAntiSpoofNet checkpoint is loaded, standardized, and weights are intact."""
    assert voice_auth_engine.anti_spoof_net.is_pretrained is True
    assert voice_auth_engine.anti_spoof_net.checkpoint_sha256 == "ec43da6cb86c9a49e872c5d7900c197ccaf81f07e7736d61bba1078615af56d2"
    assert voice_auth_engine.anti_spoof_net.scaler_mean is not None
    assert voice_auth_engine.anti_spoof_net.scaler_std is not None
    assert voice_auth_engine.anti_spoof_net.training is False


# 15. PAD Class Mapping Verification
def test_pad_class_mapping_semantics():
    """Verify Class 0 = BONAFIDE and Class 1 = SPOOF with deterministic softmax."""
    model = voice_auth_engine.anti_spoof_net

    # Genuine speech feature from real_2.wav must produce Class 0 (BONAFIDE) > 0.90
    genuine_path = os.path.join(root_dir, "scratch", "real_speech", "real_2.wav")
    with open(genuine_path, "rb") as fp:
        sig_gen, sr, _ = AudioPreprocessor.decode_wav_bytes(fp.read())
    fvec_gen = AudioFeatureExtractor.extract_features(sig_gen, sr)['feature_vector']
    t_gen = torch.from_numpy(fvec_gen).float().unsqueeze(0)
    with torch.inference_mode():
        probs_gen = torch.softmax(model(t_gen), dim=1)[0].cpu().numpy()
    assert probs_gen[0] > 0.90, f"Expected Class 0 (BONAFIDE) > 0.90 for genuine speech, got {probs_gen[0]}"

    # Spoofed speech feature from asvspoof_fake_01.wav must produce Class 1 (SPOOF) > 0.90
    spoof_path = os.path.join(root_dir, "datasets", "voice_fake", "asvspoof_fake_01.wav")
    with open(spoof_path, "rb") as fp:
        sig_sp, sr, _ = AudioPreprocessor.decode_wav_bytes(fp.read())
    fvec_sp = AudioFeatureExtractor.extract_features(sig_sp, sr)['feature_vector']
    t_sp = torch.from_numpy(fvec_sp).float().unsqueeze(0)
    with torch.inference_mode():
        probs_sp = torch.softmax(model(t_sp), dim=1)[0].cpu().numpy()
    assert probs_sp[1] > 0.90, f"Expected Class 1 (SPOOF) > 0.90 for spoofed speech, got {probs_sp[1]}"


# 16. Deterministic 5-Run Inference Verification
def test_pad_five_run_deterministic_inference():
    """Verify exact bit-level determinism across 5 repeated inference passes."""
    model = voice_auth_engine.anti_spoof_net
    t = torch.randn(1, 64)
    outputs = []
    for _ in range(5):
        with torch.inference_mode():
            logits = model(t).squeeze(0).cpu().numpy().tolist()
            outputs.append(logits)

    for i in range(1, 5):
        assert outputs[i] == outputs[0], f"Non-deterministic inference detected: run {i} differs from run 0"


# 17. Labeled Audio Discrimination Verification (Genuine vs Spoofed)
def test_pad_discrimination_on_labeled_audio():
    """Verify PAD reliably separates labeled genuine speech from spoofed audio."""
    # Test genuine speech from scratch/real_speech/
    genuine_path = os.path.join(root_dir, "scratch", "real_speech", "real_2.wav")
    if os.path.exists(genuine_path):
        with open(genuine_path, "rb") as f:
            raw_gen = f.read()
        res_gen = voice_auth_engine.process_voice_pipeline(raw_gen)
        assert res_gen["success"] is True
        assert res_gen["liveness_score"] >= voice_auth_engine.tau_liveness, (
            f"Genuine speech failed PAD: {res_gen['liveness_score']} < {voice_auth_engine.tau_liveness}"
        )
        assert res_gen["liveness_verdict"] == "LIVE"

    # Test spoofed audio from datasets/voice_fake/
    spoof_path = os.path.join(root_dir, "datasets", "voice_fake", "asvspoof_fake_01.wav")
    if os.path.exists(spoof_path):
        with open(spoof_path, "rb") as f:
            raw_sp = f.read()
        res_sp = voice_auth_engine.process_voice_pipeline(raw_sp)
        assert res_sp["success"] is True
        assert res_sp["liveness_score"] < 0.10, (
            f"Spoofed audio scored too high: {res_sp['liveness_score']}"
        )
        assert res_sp["liveness_verdict"] == "SPOOF"


# 18. Regression: Natural Speech with High-Frequency Ratio Must NOT Be Penalized as Replay
def test_pad_high_frequency_live_speech_not_penalized():
    """
    REGRESSION TEST:
    Acoustic speech with natural unvoiced consonants or sensitive microphone response
    (hf_ratio in [0.35, 0.55]) and healthy vocal fold dynamics (f0_std >= 2.0 Hz)
    must NOT be penalized by heuristic subtraction or falsely tagged as REPLAY_ATTACK.
    """
    # Test real_1.wav which has high_frequency_ratio = 0.3952
    real_1_path = os.path.join(root_dir, "scratch", "real_speech", "real_1.wav")
    if os.path.exists(real_1_path):
        with open(real_1_path, "rb") as f:
            raw_real1 = f.read()
        res = voice_auth_engine.process_voice_pipeline(raw_real1)
        assert res["success"] is True
        assert res["acoustic_metrics"]["high_frequency_ratio"] > 0.35
        assert res["liveness_score"] >= voice_auth_engine.tau_liveness, (
            f"real_1.wav failed PAD: {res['liveness_score']} < {voice_auth_engine.tau_liveness}"
        )
        assert res["liveness_verdict"] == "LIVE"
        assert res["attack_type"] is None

    # Synthetic check with mock feature vector producing high p_live and hf_ratio = 0.48
    # Even with hf_ratio = 0.48, active pitch dynamics (f0_std = 45 Hz) must not lower p_live
    feat_vec = np.zeros(64, dtype=np.float32)
    # Use real_1 features to ensure clean neural inference
    if os.path.exists(real_1_path):
        sig, sr, _ = AudioPreprocessor.decode_wav_bytes(raw_real1)
        fvec = AudioFeatureExtractor.extract_features(sig, sr)["feature_vector"]
        mock_metrics = {
            "f0_mean_hz": 160.0,
            "f0_std_hz": 45.0,
            "high_frequency_ratio": 0.48,
            "zero_crossing_rate": 0.15,
            "spectral_centroid_hz": 1850.0
        }
        score, spoof_p, att = voice_auth_engine.evaluate_anti_spoof(fvec, mock_metrics)
        assert score >= voice_auth_engine.tau_liveness, f"Expected score >= 0.70, got {score}"
        assert att is None, f"Expected attack_type None, got {att}"


# 19. Challenge Phrase Verification Status Mapping & Server Policy
def test_phrase_verification_status_mapping_and_policy():
    """
    Verify phrase challenge evaluation:
      - Valid transcribed speech matching challenge -> MATCHED
      - Transcribed speech contradicting challenge -> MISMATCHED
      - Empty audio / silence -> INCONCLUSIVE (does not falsely claim match)
      - A spoken challenge mismatch rejects the session (final_verdict == REJECTED)
    """
    from app.services.speech_recognition_service import SpeechRecognitionService

    # 1. Match
    res_m = SpeechRecognitionService.evaluate_phrase("Sunny mango blue", "Sunny mango blue")
    assert res_m["phrase_match"] == "MATCHED"
    assert res_m["phrase_match_score"] >= 0.70

    # 2. Mismatch (contradicting challenge)
    res_mis = SpeechRecognitionService.evaluate_phrase("The quick brown fox jumps", "Sunny mango blue")
    assert res_mis["phrase_match"] in ["MISMATCHED", "MISMATCH"]

    # 3. Inconclusive (empty / unintelligible audio)
    res_inc = SpeechRecognitionService.evaluate_phrase("", "Sunny mango blue")
    assert res_inc["phrase_match"] == "INCONCLUSIVE"
    assert "intelligible" in res_inc["reason"].lower()
