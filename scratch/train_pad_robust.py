import os
import sys
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy.signal
import pandas as pd
import huggingface_hub

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import AudioPreprocessor, AudioFeatureExtractor, PassiveAudioAntiSpoofNet
from scripts.benchmark_voice_auth import fetch_librispeech_samples

print("=" * 70)
print("EXPERIMENT: ROBUST PAD RETRAINING WITH GAIN & CHANNEL AUGMENTATION")
print("=" * 70)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 1. Load Data
libri_dict = fetch_librispeech_samples()
spk_ids = sorted(list(libri_dict.keys()))

# Split LibriSpeech by speaker
train_spks = spk_ids[:18]
test_spks = spk_ids[18:]

train_bonafide_signals = []
test_bonafide_signals = []

for spk in train_spks:
    for b in libri_dict[spk][:8]:
        sig, sr, audit = AudioPreprocessor.decode_wav_bytes(b)
        if sig is not None and audit["pass"]:
            train_bonafide_signals.append((sig, sr))

for spk in test_spks:
    for b in libri_dict[spk][:8]:
        sig, sr, audit = AudioPreprocessor.decode_wav_bytes(b)
        if sig is not None and audit["pass"]:
            test_bonafide_signals.append((sig, sr))

# Add real speech from scratch/real_speech
real_wavs = sorted(glob.glob(os.path.join(root_dir, "scratch", "real_speech", "*.wav")))
for i, f in enumerate(real_wavs):
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(open(f, "rb").read())
    if sig is not None and audit["pass"]:
        if i < 4:
            train_bonafide_signals.append((sig, sr))
        else:
            test_bonafide_signals.append((sig, sr))

# Download ASVspoof 2017 & 2015
p17 = huggingface_hub.hf_hub_download(
    repo_id="DynamicSuperb/SpoofDetection_ASVspoof2017",
    filename="data/test-00000-of-00001-e7ed034c56ededdd.parquet",
    repo_type="dataset"
)
df17 = pd.read_parquet(p17)
auth_17 = df17[df17['label'] == 'authentic']['audio'].tolist()
spoof_17 = df17[df17['label'] == 'spoofed']['audio'].tolist()

p15 = huggingface_hub.hf_hub_download(
    repo_id="DynamicSuperb/SpoofDetection_ASVspoof2015",
    filename="data/test-00000-of-00001-4a252b1e92dc86af.parquet",
    repo_type="dataset"
)
df15 = pd.read_parquet(p15)
auth_15 = df15[df15['label'] == 'authentic']['audio'].tolist()
spoof_15 = df15[df15['label'] == 'spoofed']['audio'].tolist()

# Add ASV authentic
for r in auth_17[:10]:
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(r['bytes'])
    if sig is not None and audit["pass"]:
        train_bonafide_signals.append((sig, sr))

for r in auth_17[10:]:
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(r['bytes'])
    if sig is not None and audit["pass"]:
        test_bonafide_signals.append((sig, sr))

for r in auth_15[:6]:
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(r['bytes'])
    if sig is not None and audit["pass"]:
        train_bonafide_signals.append((sig, sr))

for r in auth_15[6:]:
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(r['bytes'])
    if sig is not None and audit["pass"]:
        test_bonafide_signals.append((sig, sr))

# Spoofs
train_spoof_signals = []
test_spoof_signals = []

for r in spoof_17[:80]:
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(r['bytes'])
    if sig is not None and audit["pass"]:
        train_spoof_signals.append((sig, sr))

for r in spoof_17[80:120]:
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(r['bytes'])
    if sig is not None and audit["pass"]:
        test_spoof_signals.append((sig, sr))

for r in spoof_15[:80]:
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(r['bytes'])
    if sig is not None and audit["pass"]:
        train_spoof_signals.append((sig, sr))

for r in spoof_15[80:120]:
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(r['bytes'])
    if sig is not None and audit["pass"]:
        test_spoof_signals.append((sig, sr))

# Add datasets/voice_fake
for f in glob.glob(os.path.join(root_dir, "datasets", "voice_fake", "*.wav")):
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(open(f, "rb").read())
    if sig is not None and audit["pass"]:
        test_spoof_signals.append((sig, sr))

print(f"Bona-Fide Signals: {len(train_bonafide_signals)} train, {len(test_bonafide_signals)} test")
print(f"Spoof Signals:     {len(train_spoof_signals)} train, {len(test_spoof_signals)} test")

# Acoustic Augmentation function
def augment_signal(sig: np.ndarray, sr: int = 16000) -> np.ndarray:
    out = sig.copy()
    
    # 1. Random Gain / Volume scaling (from 0.15 to 1.8)
    gain = random.uniform(0.15, 1.8)
    out = out * gain
    
    # 2. Random Mild Filtering / Spectral Tilt (simulating consumer mic frequency response)
    if random.random() < 0.6:
        # 1st order IIR tilt filter: y[n] = x[n] + alpha * y[n-1]
        alpha = random.uniform(-0.35, 0.35)
        out = scipy.signal.lfilter([1.0], [1.0, -alpha], out).astype(np.float32)
    
    # 3. Random Mild Ambient Room Noise (25 - 45 dB SNR)
    if random.random() < 0.5:
        noise = np.random.randn(len(out)).astype(np.float32)
        sig_power = np.mean(out ** 2) + 1e-8
        noise_power = np.mean(noise ** 2) + 1e-8
        snr_db = random.uniform(25.0, 45.0)
        target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = noise * np.sqrt(target_noise_power / noise_power)
        out = out + noise

    # Ensure peak amplitude in [-1.0, 1.0]
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak

    return out

# 2. Extract Features with Augmentations for Training
print("\nExtracting Features with Data Augmentations...")
X_train_list = []
y_train_list = []

# For bona-fide: extract original + augmented versions
for sig, sr in train_bonafide_signals:
    # original
    f0 = AudioFeatureExtractor.extract_features(sig, sr)["feature_vector"]
    X_train_list.append(f0)
    y_train_list.append(0)
    
    # 3 augmented variations per bona-fide sample
    for _ in range(3):
        aug_sig = augment_signal(sig, sr)
        fa = AudioFeatureExtractor.extract_features(aug_sig, sr)["feature_vector"]
        X_train_list.append(fa)
        y_train_list.append(0)

# For spoof: extract original + augmented versions
for sig, sr in train_spoof_signals:
    f0 = AudioFeatureExtractor.extract_features(sig, sr)["feature_vector"]
    X_train_list.append(f0)
    y_train_list.append(1)
    
    # 1 augmented variation per spoof sample
    aug_sig = augment_signal(sig, sr)
    fa = AudioFeatureExtractor.extract_features(aug_sig, sr)["feature_vector"]
    X_train_list.append(fa)
    y_train_list.append(1)

X_train_raw = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.int64)

# Test set: NO augmentation (evaluate clean held-out)
X_test_list = []
y_test_list = []
for sig, sr in test_bonafide_signals:
    f = AudioFeatureExtractor.extract_features(sig, sr)["feature_vector"]
    X_test_list.append(f)
    y_test_list.append(0)

for sig, sr in test_spoof_signals:
    f = AudioFeatureExtractor.extract_features(sig, sr)["feature_vector"]
    X_test_list.append(f)
    y_test_list.append(1)

X_test_raw = np.array(X_test_list, dtype=np.float32)
y_test = np.array(y_test_list, dtype=np.int64)

print(f"Training features shape: {X_train_raw.shape}, Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)}")
print(f"Test features shape:     {X_test_raw.shape}, Class 0: {sum(y_test==0)}, Class 1: {sum(y_test==1)}")

# 3. Fit StandardScaler strictly on Training Data
scaler_mean = np.mean(X_train_raw, axis=0)
scaler_std = np.std(X_train_raw, axis=0)
scaler_std[scaler_std < 1e-4] = 1.0

X_train_norm = (X_train_raw - scaler_mean) / scaler_std
X_test_norm = (X_test_raw - scaler_mean) / scaler_std

# 4. Train Model with regularized setup
dataset = TensorDataset(torch.from_numpy(X_train_norm), torch.from_numpy(y_train))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PassiveAudioAntiSpoofNet(in_features=64)
# Make sure model doesn't use embedded scaler during training since inputs are pre-normalized
model.scaler_mean = None
model.scaler_std = None

optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
# Balanced loss with label smoothing (0.05) to prevent overconfident saturated logits
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

print("\nTraining PassiveAudioAntiSpoofNet for 30 epochs...")
model.train()
best_loss = float('inf')
best_weights = None

for epoch in range(1, 31):
    total_loss = 0.0
    correct = 0
    total = 0
    for Xb, yb in loader:
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(yb)
        preds = torch.argmax(logits, dim=1)
        correct += int(torch.sum(preds == yb))
        total += len(yb)
    
    acc = correct / total
    avg_loss = total_loss / total
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:02d}/30: loss={avg_loss:.4f}, acc={acc*100:.2f}%")

model.eval()
model.load_state_dict(best_weights)

# 5. Held-Out Evaluation
with torch.inference_mode():
    test_logits = model(torch.from_numpy(X_test_norm).float())
    test_probs = F.softmax(test_logits, dim=1).numpy()
    p_live_test = test_probs[:, 0]

tau = 0.70
bf_mask = (y_test == 0)
sp_mask = (y_test == 1)

bpcer = float(np.mean(p_live_test[bf_mask] < tau))
apcer = float(np.mean(p_live_test[sp_mask] >= tau))
acer = (bpcer + apcer) / 2.0

print("\n" + "=" * 50)
print(f"HELD-OUT EVALUATION METRICS (Threshold = {tau}):")
print("=" * 50)
print(f"  Total Held-Out Samples: {len(y_test)}")
print(f"    - Genuine (Bona-Fide): {sum(bf_mask)}")
print(f"    - Spoof (Replay + TTS/VC): {sum(sp_mask)}")
print(f"  BPCER (False Rejection Rate): {bpcer * 100:.2f}%")
print(f"  APCER (False Acceptance Rate): {apcer * 100:.2f}%")
print(f"  ACER: {acer * 100:.2f}%")

# 6. Test on ALL 8 scratch/real_speech samples
print("\n" + "=" * 50)
print("TEST ON REAL DESKTOP MICROPHONE SAMPLES (scratch/real_speech):")
print("=" * 50)
for f in real_wavs:
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(open(f, "rb").read())
    fvec = AudioFeatureExtractor.extract_features(sig, sr)["feature_vector"]
    norm_f = (fvec - scaler_mean) / scaler_std
    with torch.inference_mode():
        logits = model(torch.from_numpy(norm_f[None, :]).float())
        pl = F.softmax(logits, dim=1)[0, 0].item()
    status = "PASS" if pl >= tau else "FAIL"
    print(f"  {os.path.basename(f):<15}: p_live = {pl:.4f} [{status}] logits=[{logits[0,0]:.3f}, {logits[0,1]:.3f}]")

# 7. Test on Volume Scaled real_1 (Gain Invariance Test)
print("\nGAIN INVARIANCE TEST on real_1.wav:")
sig1, sr1, _ = AudioPreprocessor.decode_wav_bytes(open(real_wavs[0], "rb").read())
for scale in [1.5, 1.0, 0.7, 0.5, 0.3, 0.2]:
    sig_sc = sig1 * scale
    fvec = AudioFeatureExtractor.extract_features(sig_sc, sr1)["feature_vector"]
    norm_f = (fvec - scaler_mean) / scaler_std
    with torch.inference_mode():
        pl = F.softmax(model(torch.from_numpy(norm_f[None, :]).float()), dim=1)[0, 0].item()
    print(f"  Scale {scale:4.1f}: p_live = {pl:.4f} {'[PASS]' if pl >= tau else '[FAIL]'}")

# 8. Test on 10 voice fakes from datasets/voice_fake
print("\nDATASET VOICE FAKES TEST (datasets/voice_fake):")
fake_wavs = sorted(glob.glob(os.path.join(root_dir, "datasets", "voice_fake", "*.wav")))
fakes_passed = 0
for f in fake_wavs:
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(open(f, "rb").read())
    fvec = AudioFeatureExtractor.extract_features(sig, sr)["feature_vector"]
    norm_f = (fvec - scaler_mean) / scaler_std
    with torch.inference_mode():
        pl = F.softmax(model(torch.from_numpy(norm_f[None, :]).float()), dim=1)[0, 0].item()
    status = "REJECTED (CORRECT)" if pl < tau else "ACCEPTED (FAIL)"
    print(f"  {os.path.basename(f):<25}: p_live = {pl:.4f} [{status}]")
    if pl >= tau:
        fakes_passed += 1
print(f"Fakes correctly rejected: {len(fake_wavs) - fakes_passed}/{len(fake_wavs)}")
