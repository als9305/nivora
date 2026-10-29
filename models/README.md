# DeepGuard Models Directory

This directory contains pretrained model adapters and documentation for computer vision and audio verification models.

## Pretrained Models & Adapters

1. **Face Embedding Model**:
   - Model Adapter: `app.ml.face_model.FaceModel`
   - Description: OpenCV HOG / Spatial Gradient descriptor producing 128-dimensional L2-normalized embeddings.
   - Replaceable with InsightFace / ArcFace / dlib / ResNet-50 face embedding weights.

2. **Deepfake Detector**:
   - Model Adapter: `app.ml.deepfake_model.LightweightMLAdapter` & `DemoModelAdapter`
   - Description: Multi-frame spatial-temporal frequency analysis examining FFT high-frequency energy ratio, color gamut channel variance, and frame-to-frame temporal residual variance.
   - Replaceable with PyTorch/Timm EfficientNet-B4 or Xception deepfake weights.

3. **Voice Verification Model**:
   - Model Adapter: `app.ml.voice_model.VoiceModel`
   - Description: Audio spectral band energy extraction producing 16-dimensional speaker embeddings and pitch stability anti-spoof checks.
   - Replaceable with ECAPA-TDNN or SpeechBrain speaker embeddings.
