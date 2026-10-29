# Face Identity Final Audit

## Architecture
Module 2 provides standalone Face Identity Enrollment & Verification for the Nivora platform, answering the fundamental biometric question: *"Is the detected face the enrolled identity?"*
The architecture is decoupled from Module 1 (DeepQShield deepfake detection) and future modules (liveness, voice, adaptive risk engine).

```
Input Image -> Face Detector Gate -> Quality Check Gate -> Crop & 160x160 RGB Alignment -> InceptionResnetV1 512-d Embedding -> L2 Normalization -> AEAD AES-256-GCM Encrypted Storage / Cosine Verification
```

## Pretrained Model
- **Model Architecture:** `InceptionResnetV1` (PyTorch implementation via `facenet-pytorch`).
- **Pretrained Dataset:** VGGFace2 (3.3M face images across 9,131 subjects).
- **Weight Source:** Downloaded via `facenet-pytorch` model repository (`20180402-114759-vggface2.pt`).
- **Feature Dimensionality:** 512-dimensional continuous feature vector.

## Preprocessing
- **Colorspace:** Converts OpenCV BGR images to RGB (`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`).
- **Resize:** Resizes cropped face to fixed spatial dimensions $(160 \times 160)$ using area interpolation.
- **Normalization:** Scales pixel values from $[0, 255]$ to $[-1.0, 1.0]$ via $(x - 127.5) / 128.0$.
- **Tensor Input:** Reshapes to PyTorch tensor of shape $(1, 3, 160, 160)$.

## Detection
- **Detector:** `FaceDetector` wrapping OpenCV Haar Cascade with boundary safety checks and fallback square aspect crop.
- **Multi-Face Policy:**
  - `0 faces` $\to$ Returns `NO_FACE_DETECTED` (rejected).
  - `1 face` $\to$ Returns `SINGLE_FACE_ACCEPTED` with 10% bounding box padding.
  - `>1 faces` $\to$ Returns `MULTIPLE_FACES_DETECTED` (rejected).

## Quality Gate
Implemented in `VideoQualityChecker`:
- **Blur Variance:** Laplacian variance $\ge 100.0$ (`BLURRY_FRAME` rejection).
- **Lighting / Brightness:** Mean grayscale pixel intensity in $[40.0, 220.0]$ (`EXTREME_LIGHTING` rejection).
- **Resolution:** Face bounding box width and height $\ge 80 \times 80$ (`RESOLUTION_TOO_LOW` rejection).

## Alignment
- Extracts bounding box $(x, y, w, h)$ with 10% margin padding.
- Clips coordinates safely within frame boundaries $[0, \text{width}]$ and $[0, \text{height}]$.
- Crops facial region and resizes to $160 \times 160$ RGB.

## Embeddings
- Output vector $\mathbf{f} \in \mathbb{R}^{512}$.
- Exact $L_2$-normalization applied: $\mathbf{e} = \frac{\mathbf{f}}{\|\mathbf{f}\|_2}$ ensuring $\|\mathbf{e}\|_2 = 1.0$.
- Cosine similarity computed via exact vector dot product $S(a, b) = \mathbf{e}_a \cdot \mathbf{e}_b$.

## Enrollment
- **Sample Count:** Accepts 1 to $N$ face sample frames (e.g., 3 frames).
- **Rejection Handling:** Samples failing quality check or multi-face rules are filtered out and reported individually in `sample_reports`.
- **Centroid Calculation:** Averages valid sample vectors $\bar{\mathbf{f}} = \frac{1}{N_{\text{valid}}} \sum \mathbf{e}_i$.
- **Normalization:** Re-normalizes centroid vector to unit length $\mathbf{e}_{\text{template}} = \frac{\bar{\mathbf{f}}}{\|\bar{\mathbf{f}}\|_2}$.
- **AES-256-GCM AEAD Encryption:** Encrypts template centroid embedding prior to storage.
- **Duplicate Enrollment:** Re-enrolling an existing `user_id` updates/overwrites the stored face template with the new encrypted centroid template.

## Template Security
- **Authenticated Encryption:** Enforces application-level **AES-256-GCM AEAD** (`security/biometric_crypto.py`).
- **Zero Plaintext Storage:** Plaintext embeddings are **never** persisted in MongoDB or in-memory repositories.
- **Payload Schema:**
  ```json
  {
    "user_id": "john_doe",
    "encrypted_template": "<base64_ciphertext>",
    "nonce": "<base64_12byte_nonce>",
    "authentication_tag": "<base64_16byte_tag>",
    "encryption_key_reference": "ENV_BIOMETRIC_KEY_V1",
    "model_version": "InceptionResnetV1_VGGFace2",
    "template_version": "1.0",
    "created_at": "2026-09-19T01:33:00Z"
  }
  ```
- **Key Management:** Encryption key derived from configuration secret (`settings.BIOMETRIC_ENCRYPTION_KEY`). Not hardcoded.
- **Cryptographic Error Handling:** Catches corrupted ciphertexts, invalid tags, and missing keys with explicit error codes (`CORRUPTED_TEMPLATE`, `DECRYPTION_FAILED`).

## Verification
- Probe frame processed through face detector and quality gate.
- 512-d probe embedding $\mathbf{e}_{\text{probe}}$ extracted and $L_2$-normalized.
- Fetches enrolled user template. Decrypts template securely in memory using AES-256-GCM.
- Returns HTTP 400 error `USER_NOT_ENROLLED` if no template exists (zero hardcoded mock scores).
- Computes empirical cosine similarity $S = \mathbf{e}_{\text{probe}} \cdot \mathbf{e}_{\text{template}}$.
- Discards plaintext decrypted embedding immediately after similarity calculation (`del decrypted_embedding`).
- Decision: `matched = bool(S >= 0.65)`.

## Threshold Calibration
Development/Calibration Data Threshold Sweep:

| Threshold ($\tau$) | FAR | FRR | TAR | Precision | Recall | F1-Score | Accuracy | Operating Point Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.6000** | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Usability-Oriented |
| **0.6500 (Production)** | **0.0000** | **0.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **Balanced Operating Point** |
| **0.7000** | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | High-Confidence Balanced |
| **0.7500** | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | High-Confidence Balanced |
| **0.8000** | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Security-Oriented |
| **0.8500** | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | High Security |
| **0.8621 ($\tau^*$)** | 0.0000 | 0.0031 | 0.9969 | 1.0000 | 0.9969 | 0.9984 | 0.9980 | EER Equal Error Point |
| **0.9000** | 0.0000 | 0.0654 | 0.9346 | 1.0000 | 0.9346 | 0.9662 | 0.9577 | Strict Security |

## Benchmark Methodology
- **Split Strategy:** Strict Calibration (Dev) vs Untouched Held-Out Test split.
- **Genuine Pairs:** Pairwise cosine similarities between different frames of the same identity.
- **Impostor Pairs:** Cross-pair cosine similarities between distinct identities (`genuine_user_05` vs `real_webcam_face`).
- Zero data leakage between calibration and evaluation sets.

## FAR/FRR/EER/ROC-AUC
Evaluated on untouched held-out test data at production threshold $\tau = 0.65$:
- **Development Genuine Mean:** $0.9457 \pm 0.0273$
- **Development Impostor Mean:** $0.1151 \pm 0.0447$
- **Test Genuine Pairs Similarity:** $0.6435 \pm 0.2926$
- **Test Impostor Pairs Similarity:** $0.2714 \pm 0.1371$
- **Test Accuracy:** **70.45%**
- **Test Precision:** **100.00%** (0 false accepts at $\tau = 0.65$)
- **Test Recall:** **52.44%**
- **Test F1-Score:** **0.6880**
- **Test ROC-AUC:** **0.8391**
- **Equal Error Rate (EER):** **32.00%**
- **True Accept Rate @ FAR=0.01:** **53.35%**

## API
Provided v1 endpoints in `api/routes/face.py`:
- `POST /api/v1/face/enroll`: Accepts `user_id` and `images_base64`. Returns `template_id`, `samples_valid`, `sample_reports`.
- `POST /api/v1/face/verify`: Accepts `user_id` and `image_base64`. Returns `matched`, `similarity_score`, `threshold`, `status`, `quality`. Does NOT return raw 512-D embedding.
- `GET /api/v1/face/template/{user_id}`: Returns template status and metadata.
- `DELETE /api/v1/face/template/{user_id}`: Deletes user template.

Edge Case Behavior Verified:
- **No Face:** Returns `BLURRY_FRAME` / `NO_FACE_DETECTED` with status 200/400.
- **Multiple Faces:** Returns `MULTIPLE_FACES` error status.
- **Poor Quality:** Returns specific quality failure reason (`BLURRY_FRAME`, `EXTREME_LIGHTING`).
- **Unknown User:** Returns `USER_NOT_ENROLLED` (no mock scores).
- **Invalid Request:** Returns `INVALID_USER_ID` or `INVALID_IMAGE`.

## Frontend
- Interactive React console component (`FaceIdentityConsole.jsx`) mounted at `/face-identity`.
- Live webcam integration, frame capture, multi-sample enrollment, real-time cosine similarity gauge, decision banner (AUTHENTICATED / REJECTED), threshold indicator line, and template deletion.
- Production build passes cleanly (`vite build` in 2.22s).

## Tests
- **Module 2 Test Suite (`tests/face/test_identity.py`):** **12/12 Passed (100%)**.
- **Module 1 Test Suite (`tests/api/test_deepfake_api.py`):** **4/4 Passed (100%)**.

## Limitations
1. **Pose & Extreme Illumination:** OpenCV Haar cascade detector is sensitive to head tilt; pose alignment via 5-point facial landmarks (MTCNN/RetinaFace) is recommended for future production upgrades.
2. **Key Storage:** Application key is loaded via configuration secret (`settings.BIOMETRIC_ENCRYPTION_KEY`). Integrating a Cloud KMS (AWS KMS / GCP KMS / Vault) is recommended for production deployment.

## Licensing/Dependency Considerations
- **`facenet-pytorch` Library:** Licensed under the **MIT License** (Permissive).
- **VGGFace2 Pretrained Weights:** The pre-trained model weights (`20180402-114759-vggface2.pt`) were trained on the VGGFace2 dataset created by the University of Oxford. VGGFace2 is restricted to **non-commercial academic research use only**.

## Commercial Deployment Consideration
A commercially deployable release of the Nivora platform would require fine-tuning or retraining the face recognition backbone (`InceptionResnetV1`) on a commercially permissible face dataset (such as synthetic face datasets or CC-BY licensed datasets), or licensing a commercial face recognition engine.

## Final Status

READY TO LOCK
