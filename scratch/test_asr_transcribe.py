import time
import os
import wave
import numpy as np
from faster_whisper import WhisperModel

def test():
    t0 = time.perf_counter()
    print("Loading tiny.en...")
    model = WhisperModel('tiny.en', device='cpu', compute_type='int8')
    t_load = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {t_load:.1f}ms")

    wav_path = "scratch/real_speech/real_1.wav"
    if os.path.exists(wav_path):
        with wave.open(wav_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if n_channels > 1:
                audio = audio[::n_channels]

        print(f"Loaded audio array: shape={audio.shape}, sr={framerate}")
        t0 = time.perf_counter()
        segments, info = model.transcribe(audio, beam_size=1)
        text = " ".join(s.text for s in segments).strip()
        t_trans = (time.perf_counter() - t0) * 1000
        print(f"Transcribed in {t_trans:.1f}ms: '{text}'")

if __name__ == "__main__":
    test()
