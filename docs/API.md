# DeepGuard API Specification (v1 REST API)

DeepGuard exposes FastAPI REST endpoints for multi-modal security verification and DeepQShield deepfake detection.

---

## DeepQShield Deepfake Detection Endpoints (`/api/v1/deepfake`)

### 1. `GET /api/v1/deepfake/model/info`
Returns active model architecture, version, and PQC cryptographic algorithms.

### 2. `POST /api/v1/deepfake/image`
Analyzes a single Base64 encoded image frame.
- **Request:** `{"image_base64": "..."}`
- **Response:** Signed PQC security record with `video_fake_probability`, `classification`, SHA3-256 digest, and Dilithium3 signature.

### 3. `POST /api/v1/deepfake/video`
Analyzes a sequence of Base64 encoded video frames.
- **Request:** `{"frames_base64": ["...", "..."]}`

### 4. `POST /api/v1/deepfake/live/start`
Initializes a transient live video analysis session. Returns `session_id`.

### 5. `POST /api/v1/deepfake/live/frame`
Ingests an incremental video frame into an active session.

### 6. `POST /api/v1/deepfake/live/end`
Concludes live session, computes temporal score aggregation, and returns signed PQC result.

### 7. `GET /api/v1/deepfake/result/{analysis_id}`
Retrieves a signed analysis record and verifies SHA3-256 digest & Dilithium3 digital signature integrity.
