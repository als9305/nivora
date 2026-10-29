# DeepGuard Privacy & Data Minimization Policy

DeepGuard processes sensitive biometric video streams during authentication. This document details our privacy controls and data minimization guarantees.

---

## 1. Transient Processing Principle

- **No Storage of Raw Media:** By default, raw video frames and audio clips are processed **in-memory** and immediately discarded after inference completes.
- **Derived Metadata Storage:** Only non-biometric derived security scores (`fake_probability`, `confidence`, `timestamp`, `analysis_id`, `PQC signature`) are stored for auditing purposes.

---

## 2. Configurable Data Retention

Retention settings are controlled via `configs/security.yaml`:

```yaml
privacy:
  transient_media_processing: true
  store_raw_video: false
  store_raw_audio: false
  retention_days: 30
```

---

## 3. Log Hygiene

- Log files **never** contain raw Base64 image payload strings, raw video streams, passwords, or private key materials.
- All log messages use structured JSON formatting with sanitized transaction identifiers.
