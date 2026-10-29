# DeepGuard Security Architecture & Privacy Policy

## 1. Biometric Template Security
- Raw camera frames and recorded audio files are processed in-memory or saved in ephemeral temporary files (`scratch/temp/`).
- Ephemeral files are automatically unlinked and cleaned up immediately after inference completes.
- Only non-reversible mathematical embedding vectors (128-d face arrays and 16-d audio band energy vectors) are stored in user profiles.
- Raw biometric images are **never** exposed through public REST APIs.

## 2. Password & Token Hashing
- User passwords are salted and hashed using `bcrypt`.
- Authentication tokens are signed using JWT (HS256 algorithm) with configurable expiration (`ACCESS_TOKEN_EXPIRE_MINUTES`).

## 3. Strict Boundary Control Flow
- The LLM Explanation Agent acts strictly in an advisory / explainability role.
- The deterministic Risk Engine owns final security decisions (`ALLOW`, `STEP-UP`, `BLOCK`).
- The LLM is never permitted to override the Risk Engine's decision.
