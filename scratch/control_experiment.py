import os
import sys
import numpy as np
import scipy.signal
import io
import wave
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor

def synthesize_formant_speech(duration_sec=1.5, sample_rate=16000, base_f0=120.0, f1=500.0, f2=1500.0, f3=2500.0):
    """
    Synthesizes realistic human speech acoustic waveform using glottal pulse train 
    filtered through 3 vocal tract formant resonators (F1, F2, F3).
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Glottal pulse train with pitch variation
    f0_contour = base_f0 + 10.0 * np.sin(2 * np.pi * 2.5 * t) + np.random.normal(0, 1.5, n_samples)
    phase = 2 * np.pi * np.cumsum(f0_contour) / sample_rate
    excitation = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)

    # Apply 2nd order IIR formant resonators (bandpass filters)
    b1, a1 = scipy.signal.iirpeak(f1 / (sample_rate / 2.0), Q=5.0)
    b2, a2 = scipy.signal.iirpeak(f2 / (sample_rate / 2.0), Q=7.0)
    b3, a3 = scipy.signal.iirpeak(f3 / (sample_rate / 2.0), Q=9.0)

    formant1 = scipy.signal.lfilter(b1, a1, excitation)
    formant2 = scipy.signal.lfilter(b2, a2, excitation)
    formant3 = scipy.signal.lfilter(b3, a3, excitation)

    # Combined vocal tract acoustic signal
    signal = 0.5 * formant1 + 0.3 * formant2 + 0.2 * formant3
    # Envelope modulation (syllables)
    envelope = np.sin(np.pi * 3.0 * t) ** 2
    signal = signal * envelope

    # Normalize int16 PCM
    pcm_data = (np.clip(signal / (np.max(np.abs(signal)) + 1e-6), -1.0, 1.0) * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data.tobytes())

    return buf.getvalue()

def run_control_experiment():
    print("=" * 65)
    print("CONTROL EXPERIMENT — ECAPA-TDNN ON ACOUSTIC FORMANT SPEECH")
    print("=" * 65)

    # Speaker A: Male vocal tract parameters (F0=120 Hz, F1=500 Hz, F2=1500 Hz, F3=2500 Hz)
    spkA_utt1 = synthesize_formant_speech(1.5, base_f0=120.0, f1=500.0, f2=1500.0, f3=2500.0)
    spkA_utt2 = synthesize_formant_speech(1.5, base_f0=122.0, f1=510.0, f2=1490.0, f3=2510.0)

    # Speaker B: Female vocal tract parameters (F0=210 Hz, F1=850 Hz, F2=1950 Hz, F3=2950 Hz)
    spkB_utt1 = synthesize_formant_speech(1.5, base_f0=210.0, f1=850.0, f2=1950.0, f3=2950.0)
    spkB_utt2 = synthesize_formant_speech(1.5, base_f0=208.0, f1=840.0, f2=1960.0, f3=2940.0)

    # Decode & Extract embeddings
    sigA_1, srA1, _ = AudioPreprocessor.decode_wav_bytes(spkA_utt1)
    sigA_2, srA2, _ = AudioPreprocessor.decode_wav_bytes(spkA_utt2)
    sigB_1, srB1, _ = AudioPreprocessor.decode_wav_bytes(spkB_utt1)
    sigB_2, srB2, _ = AudioPreprocessor.decode_wav_bytes(spkB_utt2)

    featsA_1 = AudioFeatureExtractor.extract_features(sigA_1, srA1)
    featsA_2 = AudioFeatureExtractor.extract_features(sigA_2, srA2)
    featsB_1 = AudioFeatureExtractor.extract_features(sigB_1, srB1)
    featsB_2 = AudioFeatureExtractor.extract_features(sigB_2, srB2)

    # Apply CMVN
    log_melA_1 = featsA_1["log_mel_80"] - np.mean(featsA_1["log_mel_80"], axis=1, keepdims=True)
    log_melA_2 = featsA_2["log_mel_80"] - np.mean(featsA_2["log_mel_80"], axis=1, keepdims=True)
    log_melB_1 = featsB_1["log_mel_80"] - np.mean(featsB_1["log_mel_80"], axis=1, keepdims=True)
    log_melB_2 = featsB_2["log_mel_80"] - np.mean(featsB_2["log_mel_80"], axis=1, keepdims=True)

    embA_1 = voice_auth_engine.extract_embedding(log_melA_1)
    embA_2 = voice_auth_engine.extract_embedding(log_melA_2)
    embB_1 = voice_auth_engine.extract_embedding(log_melB_1)
    embB_2 = voice_auth_engine.extract_embedding(log_melB_2)

    sim_same_A = voice_auth_engine.compute_similarity(embA_1, embA_2)
    sim_same_B = voice_auth_engine.compute_similarity(embB_1, embB_2)
    sim_diff_AB = voice_auth_engine.compute_similarity(embA_1, embB_1)

    print("\nCONTROL EXPERIMENT RESULTS (Multi-Formant Vocal Tract Speech):")
    print(f"  Same Speaker A Similarity (Utterance 1 vs Utterance 2): {sim_same_A:.4f}")
    print(f"  Same Speaker B Similarity (Utterance 1 vs Utterance 2): {sim_same_B:.4f}")
    print(f"  Different Speaker Similarity (Speaker A vs Speaker B):  {sim_diff_AB:.4f}")
    print(f"  Score Gap (Same A - Different AB):                       {sim_same_A - sim_diff_AB:.4f}")

    if sim_same_A > sim_diff_AB:
        print("\nCONCLUSION: Pretrained ECAPA-TDNN model produces clear speaker score separation when acoustic formant structures are present!")
        print("Same-speaker similarity is higher than different-speaker similarity.")
    else:
        print("\nCONCLUSION: Model anomaly detected.")

if __name__ == "__main__":
    run_control_experiment()
