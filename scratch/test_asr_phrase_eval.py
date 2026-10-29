import os
import sys
import wave
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.app.services.speech_recognition_service import SpeechRecognitionService

def test():
    real_speech_dir = os.path.join(root_dir, "scratch", "real_speech")
    sample_files = sorted([os.path.join(real_speech_dir, f) for f in os.listdir(real_speech_dir) if f.endswith(".wav")])
    
    print(f"Testing ASR transcription on {len(sample_files)} genuine speech samples...")
    for sf in sample_files[:4]:
        with wave.open(sf, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        text, dur = SpeechRecognitionService.transcribe_audio_array(audio, sr)
        print(f"  [{os.path.basename(sf)}] Spoken: \"{text}\"")
        
        # Test positive phrase match (simulating user speaking the challenge phrase)
        res = SpeechRecognitionService.evaluate_phrase(text, text)
        print(f"    Self-Match: status={res['phrase_match']}, score={res['phrase_match_score']:.2f}, words={res['matched_words']}")
        assert res["phrase_match"] == "MATCHED"

    # 2. Test Mismatch: user says sentence A, challenge was sentence B
    res_mis = SpeechRecognitionService.evaluate_phrase("The quick brown fox jumps", "Bright paper lemon honest")
    print(f"  Mismatch Test: status={res_mis['phrase_match']}, score={res_mis['phrase_match_score']:.2f}, words={res_mis['matched_words']}")
    assert res_mis["phrase_match"] in ["MISMATCH", "MISMATCHED"]

    # 3. Test Inconclusive: empty audio / silence
    res_inc = SpeechRecognitionService.evaluate_phrase("", "Bright paper lemon")
    print(f"  Inconclusive Test: status={res_inc['phrase_match']}, score={res_inc['phrase_match_score']:.2f}")
    assert res_inc["phrase_match"] == "INCONCLUSIVE"

    # 4. Test Unvalidated: no challenge phrase
    res_unval = SpeechRecognitionService.evaluate_phrase("Hello world", None)
    print(f"  Unvalidated Test: status={res_unval['phrase_match']}")
    assert res_unval["phrase_match"] == "UNVALIDATED"

    # 5. Test Indian English / accent phonetic tolerance (fuzzy match)
    res_fuzzy = SpeechRecognitionService.evaluate_phrase("bright paper lemons honest", "bright paper lemon honest")
    print(f"  Fuzzy Plural/Accent Test: status={res_fuzzy['phrase_match']}, score={res_fuzzy['phrase_match_score']:.2f}, words={res_fuzzy['matched_words']}")
    assert res_fuzzy["phrase_match"] == "MATCHED"

    print("\n==================================================")
    print("ALL ASR PHRASE MATCHING TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    test()
