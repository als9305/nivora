import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60-second timeout for multi-modal processing
});

/**
  * Check backend service status.
  */
export const healthCheck = async () => {
  const res = await api.get('/health');
  return res.data;
};

/**
  * Fetch DeepQShield model info and PQC security configuration.
  */
export const getModelInfo = async () => {
  const res = await api.get('/api/v1/deepfake/model/info');
  return res.data;
};

/**
  * Register new user account.
  */
export const registerUser = async (userData) => {
  const res = await api.post('/api/auth/register', userData);
  return res.data;
};

/**
  * Login existing user with password.
  */
export const loginUser = async (credentials) => {
  const res = await api.post('/api/auth/login', credentials);
  return res.data;
};

/**
  * Module 1: Check enrolled face template status for a user.
  */
export const getFaceTemplateStatus = async (userId) => {
  try {
    const res = await api.get(`/api/v1/face/template/${userId}`);
    return res.data;
  } catch (err) {
    return { enrolled: false, exists: false };
  }
};

/**
  * Module 1: Enroll face identity template using live webcam frames.
  */
export const enrollFaceApi = async (userId, imagesBase64) => {
  const res = await api.post('/api/v1/face/enroll', {
    user_id: userId,
    images_base64: Array.isArray(imagesBase64) ? imagesBase64 : [imagesBase64],
  });
  return res.data;
};

/**
  * Module 1: Verify probe face image against enrolled user face template.
  */
export const verifyFaceApi = async (userId, imageBase64, framesBase64 = null, sessionId = null) => {
  const res = await api.post('/api/v1/face/verify', {
    user_id: userId,
    image_base64: imageBase64,
    frames_base64: framesBase64,
    session_id: sessionId,
  });
  return res.data;
};

/**
  * Module 2: Check Face Liveness & Anti-Spoofing (Passive + Active Challenge-Response).
  */
export const checkFaceLiveness = async (framesBase64, userId = null, challenge = null, challengeEvidence = null, sessionId = null) => {
  const res = await api.post('/api/v1/liveness/check', {
    frames_base64: Array.isArray(framesBase64) ? framesBase64 : [framesBase64],
    user_id: userId,
    challenge: challenge,
    challenge_evidence: challengeEvidence,
    session_id: sessionId,
  });
  if (res.data && typeof res.data === 'object') {
    res.data._httpStatus = res.status;
  }
  return res.data;
};

/**
  * Module 3: Start short-lived anti-replay voice authentication session.
  */
export const startVoiceSession = async (userId = null, purpose = 'VERIFY') => {
  const res = await api.post('/api/v1/voice/session/start', {
    user_id: userId,
    purpose: purpose,
  });
  return res.data;
};

/**
  * Module 3: Verify live microphone audio recording against active session.
  */
export const verifyVoiceSession = async (sessionId, userId, audioBase64) => {
  const res = await api.post(`/api/v1/voice/session/${sessionId}/verify`, {
    user_id: userId,
    audio_base64: audioBase64,
  });
  return res.data;
};

/**
  * Module 3: Check enrolled voice template status for a user.
  */
export const getVoiceTemplateStatus = async (userId) => {
  try {
    const res = await api.get(`/api/v1/voice/template/${userId}`);
    return res.data;
  } catch (err) {
    return { enrolled: false, exists: false };
  }
};

/**
  * Module 3: Enroll speaker voice profile from batch audio samples (>= 3).
  */
export const enrollVoiceApi = async (userId, samplesBase64) => {
  const res = await api.post('/api/v1/voice/enroll', {
    user_id: userId,
    audio_samples_base64: samplesBase64,
  });
  return res.data;
};

/**
  * Module 3: Enroll incremental single speaker voice sample (1, 2, or 3).
  */
export const enrollVoiceSampleApi = async (userId, audioB64, sampleIndex, totalSamples = 3, sessionId = null) => {
  const res = await api.post('/api/v1/voice/enroll', {
    user_id: userId,
    audio_b64: audioB64,
    sample_index: sampleIndex,
    total_samples: totalSamples,
    session_id: sessionId,
  });
  return res.data;
};

/**
  * Module 3: Direct voice verification endpoint.
  */
export const verifyVoiceApi = async (userId, audioBase64, sessionId = null) => {
  const res = await api.post('/api/v1/voice/verify', {
    user_id: userId,
    audio_base64: audioBase64,
    session_id: sessionId,
  });
  return res.data;
};

/**
  * Module 4: Deepfake Detection (DeepQShield Video Frame Analysis).
  */
export const analyzeVideoFrames = async (framesB64List, userId = null, sessionId = null) => {
  const res = await api.post('/api/v1/deepfake/video', {
    frames_base64: framesB64List,
    user_id: userId,
    session_id: sessionId,
  });
  return res.data;
};

/**
  * Module 4: Deepfake Detection (DeepQShield Base64 image frame analysis).
  */
export const analyzeImageBase64 = async (imageB64) => {
  const res = await api.post('/api/v1/deepfake/image', {
    image_base64: imageB64,
  });
  return res.data;
};

/**
  * Module 5: Final Verification (Unified Multimodal Fusion & Authentication Orchestrator Endpoint).
  */
export const authenticateMultimodalApi = async (payload) => {
  const res = await api.post('/api/v1/authenticate/multimodal', payload);
  return res.data;
};

/**
  * Retrieve authentication log history.
  */
export const getAuthHistory = async () => {
  const res = await api.get('/api/dashboard/history');
  return res.data;
};

/**
  * Retrieve completed DeepQShield analysis history.
  */
export const getDeepfakeHistory = async () => {
  try {
    const res = await api.get('/api/v1/deepfake/history');
    return res.data;
  } catch (err) {
    return { history: [] };
  }
};

/**
  * Retrieve dashboard statistics.
  */
export const getAuthStats = async () => {
  const res = await api.get('/api/dashboard/stats');
  return res.data;
};

/**
  * Retrieve analysis result by ID.
  */
export const getAnalysisResultById = async (analysisId) => {
  const res = await api.get(`/api/v1/deepfake/result/${analysisId}`);
  return res.data;
};

/**
  * Retrieve incident log by ID.
  */
export const getIncidentById = async (id) => {
  try {
    const res = await api.get(`/api/dashboard/log/${id}`);
    return res.data;
  } catch (err) {
    return null;
  }
};

/**
  * Execute attack simulation API.
  */
export const runSimulationApi = async (type = 'DEEPFAKE_ATTACK', email = 'demo@deepguard.sec') => {
  const res = await api.post(`/api/auth/simulate?simulation_type=${type}&email=${email}`);
  return res.data;
};

/**
  * Delete face template by user ID.
  */
export const deleteFaceTemplate = async (userId) => {
  const res = await api.delete(`/api/v1/face/template/${encodeURIComponent(userId)}`);
  return res.data;
};

/**
  * Delete voice template by user ID.
  */
export const deleteVoiceTemplate = async (userId) => {
  const res = await api.delete(`/api/v1/voice/template/${encodeURIComponent(userId)}`);
  return res.data;
};

export default api;
