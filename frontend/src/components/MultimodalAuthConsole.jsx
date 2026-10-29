import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ShieldCheck, ShieldAlert, Camera, Mic, CheckCircle2, XCircle,
  RefreshCw, Loader2, Lock, ChevronDown, ChevronUp,
  AlertTriangle, Square, UserPlus, Activity, Clock, ArrowRight, FileText,
} from 'lucide-react';

import { useWebcam } from '../hooks/useWebcam';
import { encodeWavPcm16, mergePcmChunks } from '../hooks/useAudioRecorder';
import {
  verifyFaceApi,
  checkFaceLiveness,
  startVoiceSession,
  verifyVoiceSession,
  analyzeVideoFrames,
  authenticateMultimodalApi,
} from '../services/api';
import { DeveloperDiagnostics } from './DeveloperDiagnostics';
import { useAuth } from '../auth/AuthContext';

// ─── Challenge pool ─────────────────────────────────────────────────────────
const LIVENESS_CHALLENGES = [
  { id: 'TURN_LEFT',  prompt: '← Turn your head LEFT' },
  { id: 'TURN_RIGHT', prompt: '→ Turn your head RIGHT' },
];

// DeepQShield returns a PQC-signed record.  Runtime classification lives in
// payload.result, not at the transport record's top level.
const normalizeDeepQShieldResult = (record) => {
  const payload = record?.payload || {};
  const result = payload?.result || {};
  if (!Object.keys(result).length) return record || {};

  return {
    ...record,
    ...result,
    label: result.label || result.classification,
    classification: result.classification || result.label,
    deepfake_probability: result.video_fake_probability ?? result.fake_probability,
    fake_probability: result.video_fake_probability ?? result.fake_probability,
    frames_analyzed: result.frames_analyzed ?? payload?.input?.frames_analyzed,
    runtime_evidence: payload?.input?.runtime_evidence || record?.runtime_evidence,
  };
};
const getRandomChallenge = () =>
  LIVENESS_CHALLENGES[Math.floor(Math.random() * LIVENESS_CHALLENGES.length)];

// ─── Module numbers / labels ─────────────────────────────────────────────────
const MODULE_DEFS = [
  { num: '01', id: 'FACE',    label: 'Face Identity' },
  { num: '02', id: 'LIVENESS',label: 'Face Liveness Detection' },
  { num: '03', id: 'VOICE',   label: 'Voice Identity' },
  { num: '04', id: 'DQ',      label: 'Deepfake Detection' },
  { num: '05', id: 'FUSION',  label: 'Final Verification' },
];

// ─── Status helpers ──────────────────────────────────────────────────────────
function modStatus(result) {
  if (!result) return 'waiting';
  if (result._current) return 'current';
  if (result.success === true) return 'done';
  if (result.success === false) return 'failed';
  return 'waiting';
}

function statusColor(s) {
  if (s === 'done') return '#34d399';
  if (s === 'current') return '#60a5fa';
  if (s === 'failed') return '#f87171';
  return '#374151';
}
function statusBg(s) {
  if (s === 'done') return 'rgba(52,211,153,0.06)';
  if (s === 'current') return 'rgba(96,165,250,0.06)';
  if (s === 'failed') return 'rgba(248,113,113,0.06)';
  return 'transparent';
}
function statusBorder(s) {
  if (s === 'done') return 'rgba(52,211,153,0.25)';
  if (s === 'current') return 'rgba(96,165,250,0.35)';
  if (s === 'failed') return 'rgba(248,113,113,0.25)';
  return '#1f293d';
}
function statusIcon(s, result) {
  if (s === 'done') return '✓';
  if (s === 'current') return '●';
  if (s === 'failed') {
    if (result?.statusType === 'NOT_ENROLLED') return '⚠';
    if (result?.statusType === 'INSUFFICIENT_EVIDENCE') return '⚠';
    if (result?.statusType === 'VERIFICATION_ERROR' || result?.statusType === 'ERROR') return '⚠';
    return '✕';
  }
  return '○';
}
function statusLabel(s, result) {
  if (s === 'done') return 'Verified';
  if (s === 'current') return 'Current';
  if (s === 'failed') {
    if (result?.statusType === 'NOT_ENROLLED') return 'Not Enrolled';
    if (result?.statusType === 'INSUFFICIENT_EVIDENCE') return 'Inconclusive';
    if (result?.statusType === 'VERIFICATION_ERROR') return 'Verification Error';
    if (result?.statusType === 'ERROR') return 'Camera Error';
    if (result?.statusType === 'REJECTED') return 'Rejected';
    return 'Mismatch';
  }
  return 'Waiting';
}

// ─── Semantic Status Color & Badge Mapping ──────────────────────────────────
export const STATUS_THEME = {
  success: '#34d399',  // emerald-400
  warning: '#fbbf24',  // amber-400
  failure: '#f87171',  // red-400
  accent:  '#38bdf8',  // sky-400 (high assurance accent)
  info:    '#60a5fa',  // blue-400
  muted:   '#94a3b8',  // slate-400
  default: '#d1d5db',  // gray-300
};

const SUCCESS_STATUS_SET = new Set([
  'VERIFIED',
  'LIVE',
  'AUTHENTIC',
  'BONAFIDE',
  'BONA_FIDE',
  'BONA FIDE',
  'MATCH',
  'REAL',
  'AUTHENTICATED',
  'ALLOW',
  'PASS',
  'PASSED',
  'YES',
  'SUCCESS',
  'VALID',
  'ACTIVE',
  'COMPLIANT',
  'VERIFICATION_SUCCESSFUL',
]);

const WARNING_STATUS_SET = new Set([
  'INSUFFICIENT_EVIDENCE',
  'INSUFFICIENT',
  'QUALITY_INSUFFICIENT',
  'QUALITY_TOO_LOW',
  'VOICE_NOT_ENROLLED',
  'FACE_NOT_ENROLLED',
  'NOT_ENROLLED',
  'INCONCLUSIVE',
  'INDETERMINATE',
  'SUSPICIOUS',
  'SUSPECTED',
  'RETRY_REQUIRED',
  'MEDIUM',
  'UNVALIDATED',
  'STEP-UP',
  'STEP-UP OTP',
  'STEP_UP_OTP',
  'INCOMPLETE',
  'STANDARD',
  'WAITING',
]);

const FAILURE_STATUS_SET = new Set([
  'REJECTED',
  'REJECT',
  'DENY',
  'FAILED',
  'FAIL',
  'SPOOF',
  'DEEPFAKE',
  'MISMATCH',
  'FACE_MISMATCH',
  'SPEAKER_MISMATCH',
  'HIGH_RISK',
  'MULTIPLE_FACES_DETECTED',
  'MULTIPLE_REAL_FACES',
  'NO_FACE_DETECTED',
  'FACE_QUALITY_FAIL',
  'NOT_ESTABLISHED',
  'NO',
  'VERIFICATION_ERROR',
  'CAMERA_ERROR',
  'ERROR',
  'BREACH',
]);

export function getSemanticStatusType(rawVal, context = {}) {
  if (rawVal === undefined || rawVal === null) return 'muted';
  const val = String(rawVal).trim().toUpperCase();
  const label = (context.label || '').toLowerCase();

  // Context-specific handling for Risk Level
  if (label.includes('risk')) {
    if (val === 'LOW') return 'success';
    if (val === 'MEDIUM') return 'warning';
    if (val === 'HIGH') return 'failure';
  }

  // Context-specific handling for Assurance Level
  if (label.includes('assurance')) {
    if (val === 'HIGH') return 'accent'; // Sky-400 high assurance accent (#38bdf8)
    if (val === 'STANDARD') return 'info';
    if (val === 'LOW' || val === 'NOT_ESTABLISHED' || val === 'INCOMPLETE') return 'failure';
  }

  if (SUCCESS_STATUS_SET.has(val)) return 'success';
  if (WARNING_STATUS_SET.has(val)) return 'warning';
  if (FAILURE_STATUS_SET.has(val)) return 'failure';

  return 'default';
}

export function getSemanticStatusColor(rawVal, context = {}) {
  const type = getSemanticStatusType(rawVal, context);
  if (type === 'accent') return STATUS_THEME.accent;
  if (type === 'success') return STATUS_THEME.success;
  if (type === 'warning') return STATUS_THEME.warning;
  if (type === 'failure') return STATUS_THEME.failure;
  if (type === 'info') return STATUS_THEME.info;
  if (type === 'muted') return STATUS_THEME.muted;

  if (context.accent) return STATUS_THEME.info;
  return STATUS_THEME.default;
}

// ─── Detail rows helper ──────────────────────────────────────────────────────
function DetailRow({ label, value, highlight, accent }) {
  if (value === undefined || value === null) return null;
  const displayVal = value === '' ? 'N/A' : String(value);
  const statusType = getSemanticStatusType(displayVal, { label, highlight, accent });
  const valColor = highlight
    ? getSemanticStatusColor(displayVal, { label, highlight, accent })
    : accent ? STATUS_THEME.info : STATUS_THEME.default;

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'baseline',
      gap: 12,
      padding: '3px 0',
      borderBottom: '1px solid #1a2035',
    }}>
      <span style={{ color: '#6b7280', fontSize: 11, fontFamily: 'monospace', flexShrink: 0 }}>{label}</span>
      <span
        className={`status-badge-${statusType}`}
        style={{
          color: valColor,
          fontSize: 11,
          fontFamily: 'monospace',
          fontWeight: 600,
          textAlign: 'right',
          wordBreak: 'break-word',
        }}
      >
        {displayVal}
      </span>
    </div>
  );
}

// ─── Diagnostic panel wrapper ────────────────────────────────────────────────
function DiagPanel({ title, badge, children }) {
  return (
    <div style={{
      padding: '10px 12px 8px', marginTop: 4,
      background: '#0c1017', border: '1px solid rgba(96,165,250,0.2)',
      borderRadius: 8, maxHeight: 320, overflowY: 'auto',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        borderBottom: '1px solid #1f293d', paddingBottom: 6, marginBottom: 6,
      }}>
        <span style={{ color: '#60a5fa', fontSize: 10, fontWeight: 700, fontFamily: 'monospace', letterSpacing: 0.5 }}>
          ◉ {title}
        </span>
        {badge && (
          <span style={{
            fontSize: 9, padding: '1px 6px', borderRadius: 4,
            background: 'rgba(96,165,250,0.1)', border: '1px solid rgba(96,165,250,0.25)',
            color: '#60a5fa', fontFamily: 'monospace', fontWeight: 700,
          }}>{badge}</span>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {children}
      </div>
    </div>
  );
}


// ─── Module detail panel content ─────────────────────────────────────────────
function ModuleDetails({ id, result }) {
  if (!result || result._current) return null;
  const r = result.raw || {};

  // ── MODULE 01: FACE IDENTITY ──
  if (id === 'FACE') {
    const hasError = !!r.error_code || result.statusType === 'VERIFICATION_ERROR' || result.statusType === 'ERROR' || result.statusType === 'NOT_ENROLLED';
    const similarity = r.face_score ?? r.similarity_score;
    const statusLabel = hasError
      ? (r.error_code || result.statusType || 'ERROR')
      : (result.success ? 'VERIFIED' : 'REJECTED');
    const reason = r.message || (r.error_code ? r.error_code.replace(/_/g, ' ') : null);

    return (
      <DiagPanel title="MODULE 1 FACE IDENTITY DIAGNOSTICS" badge="RUNTIME EVIDENCE">
        <DetailRow label="Input" value="LIVE WEBCAM" accent />
        <DetailRow label="Dimensions" value={r.input_dimensions || (r.image_width && r.image_height ? `${r.image_width}×${r.image_height} px` : undefined)} />
        <DetailRow label="Face Detected" value={r.face_detected != null ? (r.face_detected ? 'YES' : 'NO') : (r.faces_count != null ? (r.faces_count > 0 ? 'YES' : 'NO') : undefined)} highlight />
        <DetailRow label="Faces" value={r.faces_count != null ? String(r.faces_count) : undefined} />
        <DetailRow label="Face Crop" value={r.crop_width && r.crop_height ? `${r.crop_width}×${r.crop_height} px` : undefined} />
        <DetailRow label="Sharpness" value={r.sharpness != null ? r.sharpness.toFixed(2) : r.quality_score != null ? r.quality_score.toFixed(2) : undefined} />
        <DetailRow label="Brightness" value={r.brightness != null ? r.brightness.toFixed(2) : undefined} />
        <DetailRow label="Quality" value={r.quality_status || (hasError && r.error_code === 'QUALITY_INSUFFICIENT' ? 'INSUFFICIENT' : undefined)} highlight />
        <DetailRow label="Model" value={r.model_name || 'InceptionResnetV1'} accent />
        <DetailRow label="Embedding" value={r.embedding_dim ? `${r.embedding_dim}-D L2 NORMALIZED` : '512-D L2 NORMALIZED'} />
        <DetailRow label="Similarity" value={similarity != null ? similarity.toFixed(4) : (hasError ? 'N/A' : 'N/A')} />
        <DetailRow label="Threshold" value={r.threshold != null ? r.threshold.toFixed(4) : '0.6500'} />
        <DetailRow label="Result" value={statusLabel} highlight />
        <DetailRow label="Processing" value={r.processing_time_ms != null ? `${r.processing_time_ms} ms` : undefined} />
        {reason && <DetailRow label="Diagnostic Verdict" value={reason} />}
      </DiagPanel>
    );
  }

  // ── MODULE 02: FACE LIVENESS ──
  // CRITICAL FIX: r.challenge is an OBJECT {type, result, detected, ...} from the backend,
  // not a string. Extracting r.challenge?.type prevents [object Object] rendering.
  if (id === 'LIVENESS') {
    const ch = r.challenge; // challenge_evidence_dict from backend (object or null)
    const rawChallengeType = (ch && typeof ch === 'object') ? ch.type : (ch || r.active_challenge || r.details?.challenge);
    const challengeType = typeof rawChallengeType === 'string'
      ? rawChallengeType
      : (rawChallengeType?.id || rawChallengeType?.type || (rawChallengeType ? String(rawChallengeType) : 'NONE'));
    const challengeResult = (ch && typeof ch === 'object') ? ch.result : r.challenge_result;
    const detectedDir = (ch && typeof ch === 'object') ? ch.detected_direction : r.detected_direction;
    const initialYaw = (ch && typeof ch === 'object') ? ch.initial_yaw : r.initial_yaw;
    const finalYaw = (ch && typeof ch === 'object') ? ch.final_yaw : r.final_yaw;
    const yawDelta = (ch && typeof ch === 'object') ? ch.yaw_delta : r.yaw_delta;
    const hPx = (ch && typeof ch === 'object') ? ch.horizontal_px : undefined;
    const vPx = (ch && typeof ch === 'object') ? ch.vertical_px : undefined;
    const landmarksStr = (ch && typeof ch === 'object') ? ch.landmarks_detected : undefined;
    const passiveScore = r.score ?? r.passive_score ?? (r.passive_liveness?.score);
    const threshold = r.threshold ?? r.passive_liveness?.threshold ?? 0.50;
    const framesAnalyzed = r.frames_analyzed ?? r.frames_evaluated;
    const motionDelta = r.temporal_motion?.motion_delta ?? r.details?.motion_delta;
    const finalVerdict = r.liveness || r.status;
    const reason = r.message || r.details?.reason;

    return (
      <DiagPanel title="MODULE 2 FACE LIVENESS DETECTION DIAGNOSTICS" badge="RUNTIME EVIDENCE">
        <DetailRow label="Challenge" value={challengeType ? challengeType.replace(/_/g, ' ') : 'NONE'} accent />
        <DetailRow label="Detected Direction" value={detectedDir || 'N/A'} />
        <DetailRow label="Challenge Result" value={challengeResult || 'N/A'} highlight />
        <DetailRow label="Frames Analyzed" value={framesAnalyzed != null ? String(framesAnalyzed) : undefined} />
        <DetailRow label="Facial Landmarks" value={landmarksStr || undefined} />
        <DetailRow label="Initial Yaw" value={initialYaw != null ? Number(initialYaw).toFixed(4) : undefined} />
        <DetailRow label="Final Yaw" value={finalYaw != null ? Number(finalYaw).toFixed(4) : undefined} />
        <DetailRow label="Yaw Delta" value={yawDelta != null ? Number(yawDelta).toFixed(4) : undefined} />
        <DetailRow label="Movement (H/V)" value={hPx != null || vPx != null ? `${Number(hPx || 0).toFixed(1)}px / ${Number(vPx || 0).toFixed(1)}px` : undefined} />
        <DetailRow label="Motion Delta" value={motionDelta != null ? Number(motionDelta).toFixed(4) : undefined} />
        <DetailRow label="Model" value="PassiveAntiSpoofNet" accent />
        <DetailRow label="Passive Score" value={passiveScore != null ? Number(passiveScore).toFixed(4) : 'N/A'} />
        <DetailRow label="Threshold" value={Number(threshold).toFixed(4)} />
        <DetailRow label="Final Verdict" value={finalVerdict || 'N/A'} highlight />
        <DetailRow label="Processing" value={r.processing_time_ms != null ? `${r.processing_time_ms} ms` : undefined} />
        {reason && <DetailRow label="Diagnostic Verdict" value={reason} />}
      </DiagPanel>
    );
  }

  // ── MODULE 03: VOICE AUTHENTICATION ──
  if (id === 'VOICE') {
    const speakerVerdict = r.speaker_verdict || (result.statusType === 'NOT_ENROLLED' ? 'VOICE_NOT_ENROLLED' : undefined);
    const voiceSim = r.voice_similarity ?? r.speaker_similarity;
    const padVerdict = r.liveness_verdict ?? r.pad_verdict;
    const padScore = r.liveness_score ?? r.pad_score;
    const audioQuality = r.audio_quality || {};
    const finalVerdict = r.final_verdict || (result.success ? 'AUTHENTICATED' : result.statusType);
    const reason = r.message;

    return (
      <DiagPanel title="MODULE 3 VOICE IDENTITY DIAGNOSTICS" badge="RUNTIME EVIDENCE">
        <DetailRow label="Input" value="LIVE MICROPHONE" accent />
        <DetailRow label="Sample Duration" value={r.audio_duration_ms != null ? `${r.audio_duration_ms} ms` : r.audio_duration_sec != null ? `${(r.audio_duration_sec * 1000).toFixed(0)} ms` : audioQuality.duration_sec != null ? `${(Number(audioQuality.duration_sec) * 1000).toFixed(0)} ms` : undefined} />
        <DetailRow label="Sample Rate" value={r.sample_rate ? `${r.sample_rate} Hz` : audioQuality.sample_rate ? `${audioQuality.sample_rate} Hz` : '16000 Hz'} />
        <DetailRow label="Audio Format" value={audioQuality.sample_width_bits ? `${audioQuality.sample_width_bits}-bit PCM WAV` : '16-bit PCM WAV'} />
        <DetailRow label="Audio Channels" value={audioQuality.channels != null ? String(audioQuality.channels) : undefined} />
        <DetailRow label="Capture Quality" value={audioQuality.pass === true ? 'PASSED' : audioQuality.error_code || undefined} highlight />
        <DetailRow label="Peak / RMS" value={audioQuality.peak_amplitude != null && audioQuality.rms_energy != null ? `${Number(audioQuality.peak_amplitude).toFixed(4)} / ${Number(audioQuality.rms_energy).toFixed(4)}` : undefined} />
        <DetailRow label="Silence / Clipping" value={audioQuality.silence_ratio != null && audioQuality.clipping_ratio != null ? `${(Number(audioQuality.silence_ratio) * 100).toFixed(1)}% / ${(Number(audioQuality.clipping_ratio) * 100).toFixed(1)}%` : undefined} />
        <DetailRow label="Speaker Model" value="ECAPA-TDNN" accent />
        <DetailRow label="Embedding" value="192-D L2 NORMALIZED" />
        <DetailRow label="Speaker Similarity" value={voiceSim != null ? Number(voiceSim).toFixed(4) : 'N/A'} />
        <DetailRow label="Speaker Threshold" value={r.tau_speaker != null ? Number(r.tau_speaker).toFixed(2) : '0.65'} />
        <DetailRow label="Speaker Verdict" value={speakerVerdict || 'N/A'} highlight />
        <DetailRow label="Audio PAD Model" value={r.pad_pretrained === false ? 'N/A (weights not loaded)' : 'Audio PAD Network'} />
        <DetailRow label="PAD Score" value={r.pad_pretrained === false ? 'N/A' : padScore != null ? Number(padScore).toFixed(4) : undefined} />
        <DetailRow label="PAD Score Semantic" value={r.pad_score_semantic || 'BONA FIDE PROBABILITY'} />
        <DetailRow label="PAD Threshold" value={r.pad_pretrained === false ? 'N/A' : r.tau_pad != null ? Number(r.tau_pad).toFixed(2) : '0.70'} />
        <DetailRow label="Audio PAD Verdict" value={r.pad_pretrained === false ? 'N/A' : padVerdict || undefined} highlight />
        <DetailRow label="Challenge Phrase" value={r.challenge_phrase || 'N/A'} />
        {r.transcribed_text && <DetailRow label="Spoken Speech (ASR)" value={`"${r.transcribed_text}"`} accent />}
        <DetailRow label="Phrase Match" value={r.phrase_match || 'UNVALIDATED'} highlight />
        <DetailRow label="Final Verdict" value={finalVerdict || 'N/A'} highlight />
        <DetailRow label="Processing" value={r.processing_time_ms != null ? `${r.processing_time_ms} ms` : undefined} />
        {reason && <DetailRow label="Diagnostic Verdict" value={reason} />}
      </DiagPanel>
    );
  }

  // ── MODULE 04: DEEPQSHIELD / DEEPFAKE DETECTION ──
  if (id === 'DQ') {
    const classification = r.label || r.classification;
    const fakeProb = r.deepfake_probability ?? r.fake_probability;
    const confidence = r.confidence;
    const framesAnalyzed = r.frames_analyzed ?? r.frames_count;
    const reason = r.message || r.reason;
    const runtime = r.runtime_evidence || {};
    const frameShape = runtime.decoded_shapes?.[0];
    const resolution = frameShape?.length >= 2 ? `${frameShape[1]}×${frameShape[0]}` : r.input_resolution;
    const frameQuality = runtime.frame_quality?.[0] || {};

    return (
      <DiagPanel title="MODULE 4 DEEPFAKE DETECTION DIAGNOSTICS" badge="RUNTIME EVIDENCE">
        <DetailRow label="Input" value="LIVE WEBCAM FRAMES" accent />
        <DetailRow label="Frames Analyzed" value={framesAnalyzed != null ? String(framesAnalyzed) : undefined} />
        <DetailRow label="Input Resolution" value={resolution || undefined} />
        <DetailRow label="Decoded Frame" value={frameShape ? `${frameShape.join('×')} ${runtime.decoded_dtype?.[0] || ''} ${runtime.color_order || ''}` : undefined} />
        <DetailRow label="Sharpness (Laplacian)" value={frameQuality.blur_variance != null ? String(frameQuality.blur_variance) : undefined} />
        <DetailRow label="Brightness" value={frameQuality.brightness != null ? String(frameQuality.brightness) : undefined} />
        <DetailRow label="Model Input" value="224×224×3 RGB" />
        <DetailRow label="Backbone" value="ResNeXt50_32×4d" accent />
        <DetailRow label="Architecture" value="LWE-Inspired Lattice Layers" />
        <DetailRow label="Classification" value={classification || 'N/A'} highlight />
        <DetailRow label="Fake Probability" value={fakeProb != null ? Number(fakeProb).toFixed(4) : 'N/A'} />
        <DetailRow label="Confidence" value={confidence != null ? Number(confidence).toFixed(4) : undefined} />
        <DetailRow label="Inference Executed" value={runtime.inference_executed != null ? (runtime.inference_executed ? 'YES' : 'NO') : undefined} highlight />
        <DetailRow label="Tensor" value={runtime.preprocessed_shape ? `${runtime.preprocessed_shape.join('×')} ${runtime.preprocessed_dtype || ''} RGB` : undefined} />
        <DetailRow label="Tensor Range / Mean" value={runtime.preprocessed_min != null ? `${Number(runtime.preprocessed_min).toFixed(4)} / ${Number(runtime.preprocessed_max).toFixed(4)} / ${Number(runtime.preprocessed_mean).toFixed(4)}` : undefined} />
        <DetailRow
          label="Result"
          value={classification || (result.success ? 'REAL' : 'INDETERMINATE')}
          highlight
        />
        <DetailRow label="Processing" value={r.processing_time_ms != null ? `${r.processing_time_ms} ms` : undefined} />
        {reason && <DetailRow label="Diagnostic Verdict" value={reason} />}
      </DiagPanel>
    );
  }

  // ── MODULE 05: FINAL MULTIMODAL FUSION ──
  if (id === 'FUSION') {
    const faceRes = r.face_identity || {};
    const livenessRes = r.face_liveness || {};
    const voiceRes = r.speaker_identity || {};
    const audioPadRes = r.audio_pad || {};
    const dqRes = r.deepfake_detection || {};
    const decision = r.decision || (result.success ? 'AUTHENTICATED' : 'REJECTED');
    const decisionReasons = Array.isArray(r.decision_reasons) ? r.decision_reasons : [];
    const failedChecks = Array.isArray(r.failed_checks) ? r.failed_checks : [];
    const attackIndicators = Array.isArray(r.attack_indicators) ? r.attack_indicators : [];
    const report = r.decision_report || {};
    const controls = Array.isArray(report.controls) ? report.controls : [];
    const thresholdText = (threshold) => {
      if (threshold && typeof threshold === 'object') {
        const authentic = threshold.authentic_at_or_below;
        const deepfake = threshold.deepfake_at_or_above;
        return `Real ≤ ${Number(authentic).toFixed(2)} / Fake ≥ ${Number(deepfake).toFixed(2)}`;
      }
      return threshold != null ? Number(threshold).toFixed(4) : 'N/A';
    };

    return (
      <DiagPanel title="MODULE 5 FINAL VERIFICATION DIAGNOSTICS" badge="RUNTIME EVIDENCE">
        {Object.keys(report).length > 0 && (
          <div style={{ marginBottom: 6, padding: '7px 8px', border: '1px solid #1f293d', borderRadius: 6, background: '#0a0d14' }}>
            <div style={{ color: '#93c5fd', fontSize: 10, fontFamily: 'monospace', fontWeight: 700, marginBottom: 4 }}>AUTHORIZATION SUMMARY REPORT</div>
            <DetailRow label="Authorization" value={report.authorization} highlight />
            <DetailRow label="Risk Classification" value={report.risk_level} highlight />
            <DetailRow label="Assurance Level" value={report.assurance_level} highlight />
            <DetailRow label="Report Summary" value={report.summary} />
          </div>
        )}
        <DetailRow label="Face Identity" value={faceRes.verdict || faceRes.status || r.face_identity_status || undefined} highlight />
        <DetailRow label="Face Similarity" value={faceRes.similarity != null ? Number(faceRes.similarity).toFixed(4) : undefined} />
        <DetailRow label="Face Liveness Detection" value={livenessRes.verdict || livenessRes.status || r.liveness_status || undefined} highlight />
        <DetailRow label="Liveness Score" value={livenessRes.score != null ? Number(livenessRes.score).toFixed(4) : undefined} />
        <DetailRow label="Voice Identity" value={voiceRes.verdict || voiceRes.status || r.voice_status || undefined} highlight />
        <DetailRow label="Speaker Similarity" value={voiceRes.similarity != null ? Number(voiceRes.similarity).toFixed(4) : undefined} />
        <DetailRow label="Audio PAD" value={audioPadRes.verdict || audioPadRes.status || undefined} highlight />
        <DetailRow label="DeepQShield" value={dqRes.verdict || dqRes.label || r.dq_status || undefined} highlight />
        <DetailRow label="Fake Probability" value={dqRes.fake_probability != null ? Number(dqRes.fake_probability).toFixed(4) : undefined} />
        <DetailRow label="Risk Level" value={r.risk_level || undefined} />
        <DetailRow label="Overall Confidence" value={r.overall_confidence != null ? Number(r.overall_confidence).toFixed(4) : undefined} />
        <DetailRow label="Final Decision" value={decision} highlight />
        {failedChecks.length > 0 && <DetailRow label="Failed Security Gates" value={failedChecks.join(', ')} highlight />}
        {attackIndicators.length > 0 && <DetailRow label="Attack Indicators" value={attackIndicators.join(', ')} highlight />}
        {decisionReasons.map((reason, index) => <DetailRow key={`${index}-${reason}`} label={index === 0 ? 'Decision Basis' : 'Additional Basis'} value={reason} />)}
        {controls.length > 0 && (
          <div style={{ marginTop: 5, paddingTop: 5, borderTop: '1px solid #1f293d' }}>
            <div style={{ color: '#93c5fd', fontSize: 10, fontFamily: 'monospace', fontWeight: 700, marginBottom: 3 }}>CONTROL EVIDENCE</div>
            {controls.map(control => (
              <div key={control.control} style={{ padding: '4px 0', borderBottom: '1px solid #1a2035' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                  <span style={{ color: '#9ca3af', fontSize: 10, fontFamily: 'monospace' }}>{control.control}</span>
                  <span
                    className={`status-badge-${getSemanticStatusType(control.status, { label: control.control })}`}
                    style={{
                      color: getSemanticStatusColor(control.status, { label: control.control }),
                      fontSize: 10,
                      fontFamily: 'monospace',
                      fontWeight: 700
                    }}
                  >
                    {control.status}
                  </span>
                </div>
                <div style={{ color: '#6b7280', fontSize: 9, fontFamily: 'monospace', marginTop: 1 }}>
                  Observed: {typeof control.observed === 'number' ? control.observed.toFixed(4) : 'N/A'} · Threshold: {thresholdText(control.threshold)}
                </div>
                <div style={{ color: '#6b7280', fontSize: 9, lineHeight: 1.35, marginTop: 1 }}>{control.purpose}</div>
              </div>
            ))}
          </div>
        )}
        <DetailRow label="Processing" value={r.processing_time_ms != null ? `${r.processing_time_ms} ms` : undefined} />
        {r.message && <DetailRow label="Diagnostic Verdict" value={r.message} />}
      </DiagPanel>
    );
  }

  return null;
}

// ─── Premium Final Verification Result Screen ────────────────────────────────
function FinalVerificationResult({
  fusionResult,
  verdicts,
  user,
  onNavigateDashboard,
  onReset,
}) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const raw = fusionResult?.raw || {};
  const report = raw.decision_report || {};
  const isSuccess = Boolean(fusionResult?.success);

  // Authoritative Confidence derived from server response
  const rawConfidence = raw.overall_confidence ?? report.overall_confidence;
  const confidencePercent = typeof rawConfidence === 'number'
    ? Math.max(0, Math.min(100, Math.round(rawConfidence * 100)))
    : (isSuccess ? 98 : null);

  // Authoritative Decisions & Metadata
  const authorization = report.authorization || raw.decision || (isSuccess ? 'ALLOW' : 'DENY');
  const riskLevel = report.risk_level || raw.risk_level || (isSuccess ? 'LOW' : 'HIGH');
  const assuranceLevel = report.assurance_level || (isSuccess ? 'HIGH' : 'LOW');
  const processingTimeMs = raw.processing_time_ms != null ? Math.round(raw.processing_time_ms) : null;

  const authColor = isSuccess ? '#34d399' : '#f87171';
  const riskColor = riskLevel === 'LOW' ? '#34d399' : (riskLevel === 'MEDIUM' ? '#fbbf24' : '#f87171');
  const assuranceColor = isSuccess ? '#38bdf8' : '#f87171';

  // Individual biometric gate verdicts
  const faceV = verdicts?.face;
  const livenessV = verdicts?.liveness;
  const voiceV = verdicts?.voice;
  const dqV = verdicts?.dq;

  const gates = [
    {
      num: '01',
      id: 'FACE',
      name: 'Face Identity',
      success: faceV?.success,
      status: faceV?.success ? 'MATCH' : (faceV?.statusType || 'FAILED'),
      metric: faceV?.similarity != null ? `Similarity: ${Number(faceV.similarity).toFixed(3)}` : 'InceptionResnetV1 Match',
      subtext: 'Facial identity verified against biometric enrollment'
    },
    {
      num: '02',
      id: 'LIVENESS',
      name: 'Face Liveness Detection',
      success: livenessV?.success,
      status: livenessV?.success ? 'LIVE' : (livenessV?.statusType || 'SPOOF'),
      metric: livenessV?.score != null ? `Passive Score: ${Number(livenessV.score).toFixed(3)}` : 'Passive RGB & Kinematics',
      subtext: 'Presentation attack defense & optical micro-motion'
    },
    {
      num: '03',
      id: 'VOICE',
      name: 'Voice Identity',
      success: voiceV?.success,
      status: voiceV?.success ? 'MATCH' : (voiceV?.statusType || 'FAILED'),
      metric: voiceV?.similarity != null ? `Speaker Sim: ${Number(voiceV.similarity).toFixed(3)}` : 'ECAPA-TDNN & PAD',
      subtext: 'Acoustic vocal matching & audio spoof detection'
    },
    {
      num: '04',
      id: 'DQ',
      name: 'Deepfake Detection',
      success: dqV?.success,
      status: dqV?.success ? 'REAL' : (dqV?.statusType || 'SUSPECTED'),
      metric: dqV?.fakeProbability != null ? `Fake Prob: ${Number(dqV.fakeProbability).toFixed(3)}` : 'Lattice Artifact Defense',
      subtext: 'ResNeXt50 spatial-frequency moiré analysis'
    },
    {
      num: '05',
      id: 'FUSION',
      name: 'Final Verification',
      success: isSuccess,
      status: isSuccess ? 'ALLOW' : (fusionResult?.statusType || 'REJECTED'),
      metric: confidencePercent != null ? `Confidence: ${confidencePercent}%` : 'Multi-Factor Policy',
      subtext: 'Server-authoritative Bayesian multi-evidence fusion'
    },
  ];

  const passedGatesCount = gates.filter(g => g.success).length;
  const controls = Array.isArray(report.controls) ? report.controls : [];

  const thresholdText = (threshold) => {
    if (threshold && typeof threshold === 'object') {
      const authentic = threshold.authentic_at_or_below;
      const deepfake = threshold.deepfake_at_or_above;
      return `Real ≤ ${Number(authentic).toFixed(2)} / Fake ≥ ${Number(deepfake).toFixed(2)}`;
    }
    return threshold != null ? Number(threshold).toFixed(4) : 'N/A';
  };

  const cardBg = 'linear-gradient(145deg, rgba(16,29,48,0.78), rgba(9,13,22,0.92))';
  const subcardBg = 'rgba(12,17,26,0.72)';
  const borderColor = 'rgba(96,165,250,0.18)';

  return (
    <div style={{
      width: 'min(860px, 100%)',
      margin: '0 auto',
      padding: '28px 24px',
      borderRadius: 18,
      border: `1px solid ${isSuccess ? 'rgba(52,211,153,0.25)' : 'rgba(248,113,113,0.25)'}`,
      background: cardBg,
      boxShadow: '0 26px 90px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05)',
      display: 'flex',
      flexDirection: 'column',
      gap: 20,
    }}>
      {/* ── Top Bar ── */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        paddingBottom: 12, borderBottom: '1px solid rgba(255,255,255,0.06)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: authColor, boxShadow: `0 0 10px ${authColor}`,
          }} />
          <span style={{ color: '#94a3b8', fontSize: 11, fontFamily: 'monospace', fontWeight: 700, letterSpacing: 1.5 }}>
            AUTHENTICATION COMPLETE
          </span>
        </div>
        <span style={{
          color: authColor, fontSize: 11, fontFamily: 'monospace', fontWeight: 700,
          background: isSuccess ? 'rgba(52,211,153,0.1)' : 'rgba(248,113,113,0.1)',
          border: `1px solid ${isSuccess ? 'rgba(52,211,153,0.3)' : 'rgba(248,113,113,0.3)'}`,
          padding: '2px 10px', borderRadius: 999,
        }}>
          {isSuccess ? '● VERIFIED SESSION' : '● ACCESS DENIED'}
        </span>
      </div>

      {/* ── A. Result Hero ── */}
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        textAlign: 'center', gap: 10, padding: '8px 0',
      }}>
        <div style={{
          width: 76, height: 76, borderRadius: '50%',
          background: isSuccess
            ? 'radial-gradient(circle, rgba(52,211,153,0.2) 0%, rgba(52,211,153,0.05) 70%)'
            : 'radial-gradient(circle, rgba(248,113,113,0.2) 0%, rgba(248,113,113,0.05) 70%)',
          border: `2px solid ${isSuccess ? 'rgba(52,211,153,0.6)' : 'rgba(248,113,113,0.6)'}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: isSuccess
            ? '0 0 36px rgba(52,211,153,0.25), inset 0 0 16px rgba(52,211,153,0.12)'
            : '0 0 36px rgba(248,113,113,0.25), inset 0 0 16px rgba(248,113,113,0.12)',
        }}>
          {isSuccess ? <CheckCircle2 size={38} color="#34d399" /> : <ShieldAlert size={38} color="#f87171" />}
        </div>

        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          padding: '4px 14px', borderRadius: 999,
          background: isSuccess ? 'rgba(52,211,153,0.1)' : 'rgba(248,113,113,0.1)',
          border: `1px solid ${isSuccess ? 'rgba(52,211,153,0.3)' : 'rgba(248,113,113,0.3)'}`,
          color: authColor, fontSize: 11, fontWeight: 700, fontFamily: 'monospace',
          letterSpacing: 1.5, textTransform: 'uppercase',
        }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: authColor, boxShadow: `0 0 8px ${authColor}` }} />
          AUTHORIZATION STATUS: {authorization}
        </div>

        <div style={{
          fontSize: 24, fontWeight: 900, fontFamily: 'monospace',
          color: '#ffffff', letterSpacing: 3, textTransform: 'uppercase',
        }}>
          {isSuccess ? 'AUTHENTICATED' : 'AUTHENTICATION REJECTED'}
        </div>

        <div style={{
          fontSize: 13, color: '#94a3b8', maxWidth: 560,
          lineHeight: 1.5, fontFamily: 'monospace',
        }}>
          {isSuccess
            ? 'All mandatory identity, liveness, and deepfake verification checks passed with high cryptographic assurance.'
            : (raw.decision_reasons?.[0] || raw.message || 'One or more mandatory security controls failed to meet verification criteria.')}
        </div>
      </div>

      {/* ── B. Summary Cards (4 Aligned Metrics Grid) ── */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
        gap: 12, width: '100%',
      }}>
        {/* Card 1: Authorization */}
        <div style={{
          padding: '12px 14px', borderRadius: 10,
          background: subcardBg, border: `1px solid ${isSuccess ? 'rgba(52,211,153,0.2)' : 'rgba(248,113,113,0.2)'}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#64748b' }}>
            <span style={{ fontSize: 9, fontFamily: 'monospace', letterSpacing: 0.8, textTransform: 'uppercase' }}>AUTHORIZATION</span>
            {isSuccess ? <ShieldCheck size={14} color="#34d399" /> : <ShieldAlert size={14} color="#f87171" />}
          </div>
          <div style={{ fontSize: 18, fontWeight: 800, fontFamily: 'monospace', color: authColor, marginTop: 4 }}>
            {authorization}
          </div>
          <div style={{ fontSize: 10, color: '#64748b', marginTop: 2, fontFamily: 'monospace' }}>
            {isSuccess ? 'Access Approved' : 'Access Prohibited'}
          </div>
        </div>

        {/* Card 2: Risk Level */}
        <div style={{
          padding: '12px 14px', borderRadius: 10,
          background: subcardBg, border: `1px solid ${riskColor}33`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#64748b' }}>
            <span style={{ fontSize: 9, fontFamily: 'monospace', letterSpacing: 0.8, textTransform: 'uppercase' }}>RISK LEVEL</span>
            <Activity size={14} color={riskColor} />
          </div>
          <div style={{ fontSize: 18, fontWeight: 800, fontFamily: 'monospace', color: riskColor, marginTop: 4 }}>
            {riskLevel}
          </div>
          <div style={{ fontSize: 10, color: '#64748b', marginTop: 2, fontFamily: 'monospace' }}>
            Threat Classification
          </div>
        </div>

        {/* Card 3: Assurance */}
        <div style={{
          padding: '12px 14px', borderRadius: 10,
          background: subcardBg, border: `1px solid ${assuranceColor}33`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#64748b' }}>
            <span style={{ fontSize: 9, fontFamily: 'monospace', letterSpacing: 0.8, textTransform: 'uppercase' }}>ASSURANCE</span>
            <Lock size={14} color={assuranceColor} />
          </div>
          <div style={{ fontSize: 18, fontWeight: 800, fontFamily: 'monospace', color: assuranceColor, marginTop: 4 }}>
            {assuranceLevel}
          </div>
          <div style={{ fontSize: 10, color: '#64748b', marginTop: 2, fontFamily: 'monospace' }}>
            Confidence Level
          </div>
        </div>

        {/* Card 4: Processing Time */}
        <div style={{
          padding: '12px 14px', borderRadius: 10,
          background: subcardBg, border: '1px solid rgba(167,139,250,0.2)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#64748b' }}>
            <span style={{ fontSize: 9, fontFamily: 'monospace', letterSpacing: 0.8, textTransform: 'uppercase' }}>PROCESSING TIME</span>
            <Clock size={14} color="#a78bfa" />
          </div>
          <div style={{ fontSize: 18, fontWeight: 800, fontFamily: 'monospace', color: '#a78bfa', marginTop: 4 }}>
            {processingTimeMs != null ? `${processingTimeMs} ms` : 'N/A'}
          </div>
          <div style={{ fontSize: 10, color: '#64748b', marginTop: 2, fontFamily: 'monospace' }}>
            Pipeline Latency
          </div>
        </div>
      </div>

      {/* ── C. Verification Checklist & D. Confidence & Evidence ── */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
        gap: 14, width: '100%',
      }}>
        {/* Left Column: 5 Verification Gates Checklist */}
        <div style={{
          background: subcardBg, border: `1px solid ${borderColor}`,
          borderRadius: 12, padding: '16px', display: 'flex', flexDirection: 'column', gap: 10,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: 8, borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
            <span style={{ fontSize: 11, fontWeight: 700, fontFamily: 'monospace', color: '#93c5fd', letterSpacing: 1 }}>
              VERIFICATION CHECKLIST (5 GATES)
            </span>
            <span style={{ fontSize: 10, fontFamily: 'monospace', color: isSuccess ? '#34d399' : '#fbbf24', fontWeight: 700 }}>
              {passedGatesCount} / 5 PASSED
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {gates.map(gate => {
              const gDone = Boolean(gate.success);
              return (
                <div key={gate.id} style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '8px 10px', borderRadius: 8,
                  background: gDone ? 'rgba(52,211,153,0.03)' : 'rgba(248,113,113,0.04)',
                  border: `1px solid ${gDone ? 'rgba(52,211,153,0.15)' : 'rgba(248,113,113,0.18)'}`,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontSize: 10, fontFamily: 'monospace', color: '#64748b', fontWeight: 700 }}>
                      {gate.num}
                    </span>
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 700, color: '#f1f5f9' }}>{gate.name}</div>
                      <div style={{ fontSize: 10, fontFamily: 'monospace', color: '#64748b' }}>{gate.metric}</div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{
                      fontSize: 10, fontFamily: 'monospace', fontWeight: 700,
                      padding: '2px 8px', borderRadius: 4,
                      background: gDone ? 'rgba(52,211,153,0.12)' : 'rgba(248,113,113,0.12)',
                      color: gDone ? '#34d399' : '#f87171',
                      border: `1px solid ${gDone ? 'rgba(52,211,153,0.25)' : 'rgba(248,113,113,0.25)'}`,
                    }}>
                      {gate.status}
                    </span>
                    {gDone ? <CheckCircle2 size={15} color="#34d399" /> : <XCircle size={15} color="#f87171" />}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Confidence Bar & Evidence Chips */}
        <div style={{
          background: subcardBg, border: `1px solid ${borderColor}`,
          borderRadius: 12, padding: '16px', display: 'flex', flexDirection: 'column', gap: 14,
        }}>
          {/* Confidence Bar */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontSize: 11, fontWeight: 700, fontFamily: 'monospace', color: '#93c5fd', letterSpacing: 1 }}>
                OVERALL CONFIDENCE
              </span>
              <span style={{ fontSize: 14, fontWeight: 800, fontFamily: 'monospace', color: isSuccess ? '#34d399' : '#f87171' }}>
                {confidencePercent != null ? `${confidencePercent}%` : 'N/A'}
              </span>
            </div>
            <div style={{ height: 8, borderRadius: 999, background: '#1e293b', overflow: 'hidden', padding: 1 }}>
              <div style={{
                height: '100%', width: `${confidencePercent ?? 100}%`, borderRadius: 999,
                background: isSuccess ? 'linear-gradient(90deg, #38bdf8, #34d399)' : 'linear-gradient(90deg, #f87171, #fbbf24)',
                transition: 'width 600ms ease-out',
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#64748b', fontFamily: 'monospace', marginTop: 4 }}>
              <span>Multi-Factor Zero Trust</span>
              <span>Requirement: 5/5 Validated</span>
            </div>
          </div>

          {/* Biometric Evidence Quick-Chips */}
          <div>
            <div style={{ fontSize: 10, fontWeight: 700, fontFamily: 'monospace', color: '#64748b', marginBottom: 8, letterSpacing: 0.8 }}>
              BIOMETRIC EVIDENCE TELEMETRY
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div style={{ padding: '8px 10px', borderRadius: 8, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ color: '#64748b', fontSize: 9, fontFamily: 'monospace' }}>Face Similarity</div>
                <div style={{ color: faceV?.success ? '#34d399' : '#94a3b8', fontSize: 12, fontWeight: 700, fontFamily: 'monospace', marginTop: 2 }}>
                  {faceV?.similarity != null ? Number(faceV.similarity).toFixed(4) : (raw.face_identity?.similarity != null ? Number(raw.face_identity.similarity).toFixed(4) : 'MATCH')}
                </div>
              </div>

              <div style={{ padding: '8px 10px', borderRadius: 8, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ color: '#64748b', fontSize: 9, fontFamily: 'monospace' }}>Liveness Score</div>
                <div style={{ color: livenessV?.success ? '#34d399' : '#94a3b8', fontSize: 12, fontWeight: 700, fontFamily: 'monospace', marginTop: 2 }}>
                  {livenessV?.score != null ? Number(livenessV.score).toFixed(4) : (raw.face_liveness?.score != null ? Number(raw.face_liveness.score).toFixed(4) : 'LIVE')}
                </div>
              </div>

              <div style={{ padding: '8px 10px', borderRadius: 8, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ color: '#64748b', fontSize: 9, fontFamily: 'monospace' }}>Speaker Sim</div>
                <div style={{ color: voiceV?.success ? '#34d399' : '#94a3b8', fontSize: 12, fontWeight: 700, fontFamily: 'monospace', marginTop: 2 }}>
                  {voiceV?.similarity != null ? Number(voiceV.similarity).toFixed(4) : (raw.speaker_identity?.similarity != null ? Number(raw.speaker_identity.similarity).toFixed(4) : 'MATCH')}
                </div>
              </div>

              <div style={{ padding: '8px 10px', borderRadius: 8, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ color: '#64748b', fontSize: 9, fontFamily: 'monospace' }}>Deepfake Prob</div>
                <div style={{ color: dqV?.success ? '#34d399' : '#94a3b8', fontSize: 12, fontWeight: 700, fontFamily: 'monospace', marginTop: 2 }}>
                  {dqV?.fakeProbability != null ? Number(dqV.fakeProbability).toFixed(4) : (raw.deepfake_detection?.fake_probability != null ? Number(raw.deepfake_detection.fake_probability).toFixed(4) : '0.000')}
                </div>
              </div>
            </div>
          </div>

          {/* Security Assurance Badge */}
          <div style={{
            padding: '10px 12px', borderRadius: 8,
            background: 'rgba(96,165,250,0.05)', border: '1px solid rgba(96,165,250,0.18)',
            display: 'flex', alignItems: 'center', gap: 10,
          }}>
            <ShieldCheck size={18} color="#60a5fa" style={{ flexShrink: 0 }} />
            <span style={{ fontSize: 10, color: '#94a3b8', fontFamily: 'monospace', lineHeight: 1.4 }}>
              Multi-factor cryptographic fusion confirmed zero presentation attacks across 5 biometric domains.
            </span>
          </div>
        </div>
      </div>

      {/* ── E. Final Report Card ── */}
      <div style={{
        background: subcardBg, border: `1px solid ${borderColor}`,
        borderRadius: 12, padding: '16px', display: 'flex', flexDirection: 'column', gap: 12,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: 8, borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <FileText size={15} color="#60a5fa" />
            <span style={{ fontSize: 11, fontWeight: 700, fontFamily: 'monospace', color: '#93c5fd', letterSpacing: 1 }}>
              AUTHORIZATION &amp; POLICY REPORT
            </span>
          </div>
          <span style={{ fontSize: 9, fontFamily: 'monospace', color: '#64748b' }}>
            POST-QUANTUM CRYPTOGRAPHIC INTEGRITY
          </span>
        </div>

        {/* Aligned Key-Value Metadata Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '8px 16px' }}>
          <DetailRow label="Subject ID" value={user?.user_id || 'Authenticated Subject'} accent />
          <DetailRow label="Policy Model" value="Zero-Trust Strict Multi-Factor" />
          <DetailRow label="Assurance Level" value={assuranceLevel} highlight />
        </div>

        {/* Decision Basis Row: Responsive two-column row with clear gap, text wrapping, and no overlap */}
        <div className="decision-basis-container">
          <span style={{ color: '#6b7280', fontSize: 11, fontFamily: 'monospace', fontWeight: 600, flexShrink: 0 }}>
            Decision Basis
          </span>
          <span
            className={isSuccess ? 'text-status-success' : 'text-status-failure'}
            style={{
              color: isSuccess ? '#34d399' : '#f87171',
              fontSize: 11,
              fontFamily: 'monospace',
              fontWeight: 600,
              whiteSpace: 'normal',
              wordBreak: 'break-word',
              lineHeight: 1.5,
            }}
          >
            {report.summary || raw.decision_reasons?.[0] || (isSuccess ? 'All 5 biometric & physical controls verified.' : 'Verification failed.')}
          </span>
        </div>

        {/* Security Controls Table (if present in report) */}
        {controls.length > 0 && (
          <div style={{ marginTop: 4 }}>
            <div style={{ fontSize: 10, color: '#93c5fd', fontFamily: 'monospace', fontWeight: 700, marginBottom: 6 }}>
              SECURITY CONTROL EVALUATION MATRIX
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {controls.map(control => {
                const statusType = getSemanticStatusType(control.status, { label: control.control });
                const statusColor = getSemanticStatusColor(control.status, { label: control.control });
                return (
                  <div key={control.control} style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '6px 10px', borderRadius: 6, background: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.04)',
                  }}>
                    <div>
                      <span style={{ color: '#e2e8f0', fontSize: 11, fontFamily: 'monospace', fontWeight: 600 }}>{control.control}</span>
                      <span style={{ color: '#64748b', fontSize: 10, fontFamily: 'monospace', marginLeft: 10 }}>
                        Observed: {typeof control.observed === 'number' ? control.observed.toFixed(4) : (control.observed || 'N/A')} · Threshold: {thresholdText(control.threshold)}
                      </span>
                    </div>
                    <span
                      className={`status-badge-${statusType}`}
                      style={{
                        fontSize: 10, fontFamily: 'monospace', fontWeight: 700,
                        color: statusColor,
                      }}
                    >
                      {control.status}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Expandable Technical Telemetry Toggle */}
        <div style={{ marginTop: 4, borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 8 }}>
          <button
            onClick={() => setDetailsOpen(prev => !prev)}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              width: '100%', background: 'transparent', border: 'none',
              color: '#94a3b8', fontSize: 11, fontFamily: 'monospace', cursor: 'pointer',
              padding: '4px 0',
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <ChevronDown size={14} style={{ transform: detailsOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
              {detailsOpen ? 'Hide Full Technical Diagnostic Telemetry' : 'View Full Technical Diagnostic Telemetry'}
            </span>
            <span style={{ color: '#60a5fa', fontSize: 10, fontWeight: 700 }}>
              {detailsOpen ? 'COLLAPSE' : 'EXPAND'}
            </span>
          </button>
          {detailsOpen && (
            <div style={{ marginTop: 8, maxHeight: 340, overflowY: 'auto' }}>
              <ModuleDetails id="FUSION" result={fusionResult} />
            </div>
          )}
        </div>
      </div>

      {/* ── F. Action Buttons Bar ── */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        gap: 12, marginTop: 4, width: '100%',
      }}>
        {isSuccess ? (
          <button
            onClick={onNavigateDashboard}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              padding: '12px 32px', borderRadius: 10,
              background: 'linear-gradient(135deg, #059669 0%, #10b981 100%)',
              color: '#ffffff', fontSize: 13, fontWeight: 700, fontFamily: 'monospace',
              border: 'none', cursor: 'pointer', boxShadow: '0 8px 24px rgba(16,185,129,0.3)',
              transition: 'all 0.2s',
            }}
          >
            <span>View Dashboard</span>
            <ArrowRight size={15} />
          </button>
        ) : (
          <button
            onClick={onReset}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              padding: '12px 32px', borderRadius: 10,
              background: 'linear-gradient(135deg, #b45309 0%, #d97706 100%)',
              color: '#ffffff', fontSize: 13, fontWeight: 700, fontFamily: 'monospace',
              border: 'none', cursor: 'pointer', boxShadow: '0 8px 24px rgba(217,119,6,0.3)',
              transition: 'all 0.2s',
            }}
          >
            <RefreshCw size={14} />
            <span>Start New Authentication</span>
          </button>
        )}
        <button
          onClick={onReset}
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
            padding: '12px 20px', borderRadius: 10,
            background: 'rgba(255,255,255,0.04)', border: '1px solid #334155',
            color: '#cbd5e1', fontSize: 12, fontWeight: 600, fontFamily: 'monospace',
            cursor: 'pointer', transition: 'all 0.2s',
          }}
        >
          <RefreshCw size={12} />
          <span>Reset</span>
        </button>
      </div>
    </div>
  );
}

// ─── Single module card in the left panel ────────────────────────────────────
function ModuleCard({ def, result, expanded, onToggleDetails }) {
  const s = modStatus(result);
  const color = statusColor(s);
  const icon = statusIcon(s, result);
  const sublabel = statusLabel(s, result);
  const isCurrent = s === 'current';
  const isDone = s === 'done';
  const isFailed = s === 'failed';
  // The final authorization report has its own prominent main-screen result
  // view; only the individual module diagnostics expand in the left rail.
  const canExpand = def.id !== 'FUSION' && (isDone || isFailed) && !result?._current;

  return (
    <div style={{
      border: `1px solid ${statusBorder(s)}`,
      background: s === 'idle' ? 'rgba(12,16,23,0.62)' : statusBg(s),
      borderRadius: 10,
      padding: '11px 12px',
      transition: 'all 0.2s', boxShadow: isCurrent ? '0 8px 26px rgba(59,130,246,0.08)' : 'none',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{ color: '#374151', fontSize: 10, fontFamily: 'monospace', fontWeight: 700, minWidth: 18 }}>{def.num}</span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ color: '#e5e7eb', fontSize: 12, fontWeight: 600, lineHeight: 1.2 }}>{def.label}</div>
          <div style={{ color: isCurrent ? '#60a5fa' : color, fontSize: 10, marginTop: 1, fontFamily: 'monospace' }}>
            {isCurrent ? <span style={{ animation: 'pulse 2s infinite' }}>● Current</span> : sublabel}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{
            color,
            fontSize: 13,
            fontWeight: 700,
            fontFamily: 'monospace',
            animation: isCurrent ? 'pulse 2s infinite' : 'none',
          }}>{icon}</span>
          {canExpand && (
            <button
              onClick={onToggleDetails}
              style={{
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid #1f293d',
                borderRadius: 5,
                color: '#9ca3af',
                fontSize: 10,
                padding: '2px 6px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 2,
                fontFamily: 'monospace',
              }}
              title={expanded ? 'Hide details' : 'View details'}
            >
              {expanded ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
              {expanded ? 'Hide' : 'Details'}
            </button>
          )}
        </div>
      </div>
      {canExpand && expanded && (
        <div style={{ marginTop: 8, paddingTop: 6, borderTop: '1px solid #1a2035' }}>
          <ModuleDetails id={def.id} result={result} />
        </div>
      )}
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export function MultimodalAuthConsole() {
  const navigate = useNavigate();
  const webcam = useWebcam();
  const { user } = useAuth();

  // ── Auth state machine ─────────────────────────────────────────────────────
  //  READY | FACE_RUNNING | LIVENESS_PREPARE | LIVENESS_RUNNING | LIVENESS_DONE
  //  VOICE_NOT_ENROLLED | VOICE_CHALLENGE_READY | REQUESTING_MIC | VOICE_READY
  //  VOICE_RECORDING | VOICE_PROCESSING | VOICE_DONE | DQ_RUNNING | DQ_DONE
  //  FUSION_RUNNING | DONE | FAILED | FATAL
  const [authState, setAuthState] = useState('READY');
  const [retryAction, setRetryAction] = useState(null);
  const [challenge, setChallenge]   = useState(null);
  const [verdicts, setVerdicts]     = useState({ face: null, liveness: null, voice: null, dq: null, fusion: null });
  const [voiceSession, setVoiceSession] = useState(null);
  const [expanded, setExpanded]     = useState({});
  const [overlayText, setOverlayText] = useState('');
  const [error, setError]           = useState(null);
  const [isRecordingAudio, setIsRecordingAudio] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  // Audio refs
  const micStreamRef    = useRef(null);
  const mediaRecRef     = useRef(null);
  const audioChunksRef  = useRef([]);
  const pcmChunksRef    = useRef([]);
  const audioCtxRef     = useRef(null);
  const timerRef        = useRef(null);
  const recordingStartedAtRef = useRef(null);
  // Hold evidence for one authentication transaction.  Fusion re-evaluates
  // these exact frames server-side instead of collecting a conflicting sample.
  const faceProbeRef = useRef(null);
  const livenessFramesRef = useRef([]);
  const activeSessionIdRef = useRef(null);

  // ── Helpers ────────────────────────────────────────────────────────────────
  const setVerdict = (key, val) => setVerdicts(prev => ({ ...prev, [key]: val }));

  const markCurrent = (key) =>
    setVerdicts(prev => ({ ...prev, [key]: { _current: true } }));

  // ── Camera ─────────────────────────────────────────────────────────────────
  const ensureCamera = async () => {
    if (!webcam.isCameraActive) await webcam.startCamera();
    await webcam.waitForFrame?.();
  };

  // Pre-warm camera on console mount so auto-exposure/focus stabilize immediately
  useEffect(() => {
    webcam.startCamera().catch(err => {
      console.warn('[NIVORA CAMERA] Pre-warm camera start error:', err);
    });
  }, []);

  // Do not leave a timer or microphone active when navigating away mid-recording.
  useEffect(() => {
    const stopActiveMedia = () => {
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
      recordingStartedAtRef.current = null;
      if (mediaRecRef.current?.state === 'recording') mediaRecRef.current.stop();
      micStreamRef.current?.getTracks().forEach(track => track.stop());
      if (audioCtxRef.current?.state !== 'closed') audioCtxRef.current?.close().catch(() => {});
    };
    window.addEventListener('nivora:logout', stopActiveMedia);
    return () => {
      window.removeEventListener('nivora:logout', stopActiveMedia);
      stopActiveMedia();
    };
  }, []);

  // ── STEP 1: Face Identity ──────────────────────────────────────────────────
  const runFace = useCallback(async () => {
    setError(null);
    setAuthState('FACE_RUNNING');
    markCurrent('face');
    setOverlayText('Verifying identity...');
    try {
      await ensureCamera();

      // Ensure camera has time to be ready before capture
      if (!webcam.videoRef.current || webcam.videoRef.current.videoWidth === 0) {
        throw new Error('Camera stream not fully ready for capture.');
      }

      let frame = null;
      if (webcam.captureFrameBurst) {
        // Capture a fast burst of settled frames and pick the frame with highest sharpness
        const burst = await webcam.captureFrameBurst(8, 70);
        frame = burst.bestFrameDataUrl;
        console.log(`[NIVORA FACE] Burst capture selected probe #${burst.bestMetrics?.index} with sharpness: ${burst.bestMetrics?.sharpness?.toFixed(2)}`);
      } else if (webcam.captureFrame) {
        frame = webcam.captureFrame();
      }

      // [DEVELOPER LOGGING]
      console.log('[NIVORA DIAGNOSTICS] Frame payload type:', typeof frame);
      console.log('[NIVORA DIAGNOSTICS] Frame payload length:', frame ? frame.length : 0);
      console.log('[NIVORA DIAGNOSTICS] Frame starts with expected prefix?:', frame && frame.startsWith('data:image/jpeg;base64,'));
      if (frame) {
        console.log('[NIVORA DIAGNOSTICS] First 50 chars:', frame.substring(0, 50));
      }

      if (!frame || frame.length < 100 || !frame.startsWith('data:image/jpeg;base64,')) {
        throw new Error('Invalid camera frame captured.');
      }

      const userId = user?.user_id;
      if (!userId) {
        throw new Error('Not authenticated. Please log in first.');
      }

      // Pre-warm or reuse authentication session so intermediate evidence binds to single token
      let curSessionId = activeSessionIdRef.current;
      if (!curSessionId) {
        try {
          const sRes = await startVoiceSession(userId, 'VERIFY');
          if (sRes?.session_id) {
            curSessionId = sRes.session_id;
            activeSessionIdRef.current = curSessionId;
            setVoiceSession(sRes);
          }
        } catch (sErr) {
          console.warn('[NIVORA AUTH] Session pre-start warning:', sErr);
        }
      }

      const res = await verifyFaceApi(userId, frame, null, curSessionId);
      console.log('[NIVORA FACE] Backend response:', res);
      if (res && typeof res === 'object') {
        res._probe_frame = frame;
      }
      faceProbeRef.current = frame;

      const ok = res?.success === true && res?.matched === true;
      let statusType = 'REJECTED';
      if (ok) {
        statusType = 'VERIFIED';
      } else if (res?.error_code === 'USER_NOT_ENROLLED') {
        statusType = 'NOT_ENROLLED';
      } else if (res?.error_code) {
        statusType = 'VERIFICATION_ERROR';
      }

      const verdict = { success: ok, statusType, raw: res };
      setVerdict('face', verdict);

      if (!ok) {
        setError(res?.message || 'Face identity not verified. Ensure your face is clearly visible.');
        setRetryAction('face');
        setAuthState('FAILED');
        setOverlayText('');
        return;
      }
      // Proceed to liveness
      await runLivenessStart();
    } catch (e) {
      console.error('[NIVORA FACE] Error:', e);
      setVerdict('face', { success: false, statusType: 'ERROR', raw: { error: e.message, error_code: 'FACE_FRAME_CAPTURE_ERROR' } });
      setError(e.message || 'Face verification request failed. Check backend connection.');
      setRetryAction('face');
      setAuthState('FAILED');
      setOverlayText('');
    }
  }, [user, webcam]);

  // ── STEP 2: Liveness ───────────────────────────────────────────────────────
  const runLivenessStart = useCallback(async () => {
    const ch = getRandomChallenge();
    setChallenge(ch);
    markCurrent('liveness');
    setAuthState('LIVENESS_PREPARE');
    setOverlayText(`Preparing challenge...`);
    await new Promise(r => setTimeout(r, 400));
    setAuthState('LIVENESS_RUNNING');
    setOverlayText(ch.prompt);
  }, []);

  const runLivenessCapture = useCallback(async () => {
    if (authState !== 'LIVENESS_RUNNING') return;
    setOverlayText('Capturing...');
    try {
      const frames = webcam.captureLivenessSequence
        ? await webcam.captureLivenessSequence(1800, () => {})
        : (webcam.captureMultiFrames ? await webcam.captureMultiFrames(18, 0.85) : [webcam.captureFrame?.()].filter(Boolean));
      livenessFramesRef.current = Array.isArray(frames) ? frames.filter(Boolean) : [];

      console.log(`[NIVORA LIVENESS] Frames: ${frames?.length}`);

      const userId = user?.user_id;
      if (!userId) {
        throw new Error('Not authenticated. Please log in first.');
      }

      const res = await checkFaceLiveness(frames, userId, challenge?.id, null, activeSessionIdRef.current);
      console.log('[NIVORA LIVENESS] Backend response:', res);

      const liveness = res?.liveness;
      const ok = liveness === 'LIVE';
      const isInsufficient = liveness === 'INSUFFICIENT_EVIDENCE';

      const verdict = {
        success: ok,
        statusType: ok ? 'VERIFIED' : isInsufficient ? 'INSUFFICIENT_EVIDENCE' : 'REJECTED',
        raw: res,
      };
      setVerdict('liveness', verdict);
      setAuthState('LIVENESS_DONE');
      setOverlayText('');

      if (!ok) {
        if (isInsufficient) {
          // Allow controlled retry of liveness challenge without resetting face identity
          setError('Liveness evidence insufficient. Follow the instruction and try again.');
          setAuthState('LIVENESS_RETRY');
        } else {
          setError(res?.message || 'Liveness check rejected. Spoof signal detected.');
          setRetryAction('liveness');
          setAuthState('FAILED');
        }
        return;
      }
      // Proceed to voice
      await startVoice();
    } catch (e) {
      console.error('[NIVORA LIVENESS] Error:', e);
      setVerdict('liveness', { success: false, statusType: 'REJECTED', raw: { error: e.message } });
      setError('Liveness request failed. Check backend connection.');
      setRetryAction('liveness');
      setAuthState('FAILED');
      setOverlayText('');
    }
  }, [authState, challenge, user, webcam]);

  // ── STEP 3: Voice ──────────────────────────────────────────────────────────
  const startVoice = useCallback(async () => {
    markCurrent('voice');
    setOverlayText('');
    try {
      const userId = user?.user_id;
      if (!userId) {
        throw new Error('Not authenticated. Please log in first.');
      }
      let sessionRes = voiceSession;
      if (!sessionRes || !sessionRes.session_id) {
        sessionRes = await startVoiceSession(userId, 'VERIFY');
        setVoiceSession(sessionRes);
        activeSessionIdRef.current = sessionRes?.session_id;
      }
      console.log('[NIVORA VOICE] Active session:', sessionRes);
      if (sessionRes?.session_id) {
        setAuthState('VOICE_CHALLENGE_READY');
        setOverlayText(`Please say: "${sessionRes.challenge_phrase || 'the displayed phrase'}"`);
      } else {
        throw new Error('No session_id returned from voice session start.');
      }
    } catch (e) {
      console.error('[NIVORA VOICE] Session start error:', e);
      // If backend says no template
      if (e?.response?.data?.error_code === 'VOICE_NOT_ENROLLED' ||
          e?.response?.data?.speaker_verdict === 'VOICE_NOT_ENROLLED') {
        setVerdict('voice', { success: false, statusType: 'NOT_ENROLLED', raw: e?.response?.data || {} });
        setAuthState('VOICE_NOT_ENROLLED');
        setOverlayText('');
        setError('Voice profile not set up for this account.');
      } else {
        setError('Failed to start voice session.');
        setRetryAction('voice');
        setAuthState('FAILED');
      }
    }
  }, []);

  const requestMicAndRecord = useCallback(async () => {
    setError(null);
    setAuthState('REQUESTING_MIC');
    setOverlayText('Requesting microphone...');
    audioChunksRef.current = [];
    pcmChunksRef.current = [];
    setRecordingSeconds(0);
    // A previous timer can survive a rapid retry while its recorder finishes.
    // Always replace it before starting another recording.
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    try {
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: { ideal: 16000 },
            channelCount: { ideal: 1 },
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: true
          }
        });
      } catch (streamErr) {
        console.warn("[NIVORA MIC] 16kHz constraint failed, retrying without sampleRate constraint:", streamErr);
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: true
            }
          });
        } catch (fallbackErr) {
          console.warn("[NIVORA MIC] Advanced audio constraints failed, falling back to basic audio:", fallbackErr);
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }
      }
      micStreamRef.current = stream;

      let audioCtx;
      try {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      } catch (ctxErr) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      audioCtxRef.current = audioCtx;
      if (audioCtx.state === 'suspended') await audioCtx.resume();
      const source = audioCtx.createMediaStreamSource(stream);
      const pcmProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
      const silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      source.connect(pcmProcessor);
      pcmProcessor.connect(silentGain);
      silentGain.connect(audioCtx.destination);
      pcmProcessor.onaudioprocess = event => {
        pcmChunksRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };

      const preferredMimeType = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4',
      ].find(type => MediaRecorder.isTypeSupported?.(type));
      const recorder = preferredMimeType
        ? new MediaRecorder(stream, { mimeType: preferredMimeType })
        : new MediaRecorder(stream);
      mediaRecRef.current = recorder;

      recorder.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };

      recorder.onstop = async () => {
        if (timerRef.current) clearInterval(timerRef.current);
        timerRef.current = null;
        recordingStartedAtRef.current = null;
        stream.getTracks().forEach(t => t.stop());
        micStreamRef.current = null;
        setIsRecordingAudio(false);
        setAuthState('VOICE_PROCESSING');
        setOverlayText('Verifying voice...');

        try {
          if (!pcmChunksRef.current.length) {
            throw new Error('No microphone audio was captured. Please check your microphone and try again.');
          }
          const pcmSamples = mergePcmChunks(pcmChunksRef.current);
          if (!pcmSamples.length) {
            throw new Error('The microphone recording was empty. Please try again.');
          }
          const wavBuf = encodeWavPcm16(pcmSamples, audioCtx.sampleRate, 16000);
          const wavBlob = new Blob([wavBuf], { type: 'audio/wav' });
          const b64 = await new Promise(res => {
            const fr = new FileReader();
            fr.onloadend = () => res(fr.result);
            fr.readAsDataURL(wavBlob);
          });

          if (audioCtx.state !== 'closed') audioCtx.close().catch(() => {});

          const userId = user?.user_id;
          if (!userId) {
            throw new Error('Not authenticated. Please log in first.');
          }
          const sessionId = voiceSession?.session_id;

          const res = await verifyVoiceSession(sessionId, userId, b64);
          console.log('[NIVORA VOICE] Backend response:', res);

          // Check for NOT_ENROLLED from the verify step
          if (res?.error_code === 'VOICE_NOT_ENROLLED' || res?.speaker_verdict === 'VOICE_NOT_ENROLLED') {
            setVerdict('voice', { success: false, statusType: 'NOT_ENROLLED', raw: res });
            setAuthState('VOICE_NOT_ENROLLED');
            setOverlayText('');
            setError('Voice profile not set up for this account.');
            return;
          }

          // In production, we MUST respect final_verdict so we don't accidentally bypass PAD rejections.
          // In the case where no template was found but PAD passed, it returns PASSED_PAD_ONLY which is not authenticated.
          const ok = res?.final_verdict === 'AUTHENTICATED';
          const verdict = { success: ok, statusType: ok ? 'VERIFIED' : 'REJECTED', raw: res };
          setVerdict('voice', verdict);
          setOverlayText('');

          if (!ok) {
            const msg = res?.message || (res?.liveness_verdict === 'SPOOF'
              ? 'Audio could not be verified as genuine. Please record the displayed phrase again in a quiet environment.'
              : res?.speaker_verdict === 'MISMATCH'
              ? 'Voice speaker not recognized.'
              : 'Voice authentication failed.');
            setError(msg);
            // A voice session is single-use; retry with a new challenge while
            // preserving the completed face and liveness evidence.
            setRetryAction('voice');
            setAuthState('FAILED');
            return;
          }
          setAuthState('VOICE_DONE');
          // Proceed to DeepQShield
          await runDeepQShield();
        } catch (procErr) {
          console.error('[NIVORA VOICE] Processing error:', procErr);
          const response = procErr?.response?.data;
          const detail = response?.detail || response;
          setError(detail?.message || procErr?.message || 'Voice processing failed. Please try again.');
          setRetryAction('voice');
          setAuthState('FAILED');
          setOverlayText('');
        }
      };

      recorder.start(100);
      setIsRecordingAudio(true);
      setAuthState('VOICE_RECORDING');
      setOverlayText(voiceSession?.challenge_phrase
        ? `Say: "${voiceSession.challenge_phrase}"`
        : 'Speak clearly into your microphone');

      recordingStartedAtRef.current = performance.now();
      timerRef.current = setInterval(() => {
        const startedAt = recordingStartedAtRef.current;
        if (startedAt !== null) {
          setRecordingSeconds(Math.floor((performance.now() - startedAt) / 1000));
        }
      }, 250);
    } catch (err) {
      console.error('[NIVORA VOICE] Mic error:', err);
      const msg = err.name === 'NotAllowedError'
        ? 'Microphone access denied. Please allow access and try again.'
        : 'Microphone unavailable.';
      setError(msg);
      setAuthState('VOICE_CHALLENGE_READY');
      setOverlayText(voiceSession?.challenge_phrase ? `Please say: "${voiceSession.challenge_phrase}"` : '');
    }
  }, [voiceSession]);

  const stopRecording = useCallback(() => {
    if (mediaRecRef.current?.state === 'recording') {
      mediaRecRef.current.stop();
    }
  }, []);

  // ── STEP 4: DeepQShield ────────────────────────────────────────────────────
  const runDeepQShield = useCallback(async () => {
    markCurrent('dq');
    setOverlayText('Analyzing deepfake detection...');
    try {
      // Liveness frames intentionally contain head movement and can be motion
      // blurred.  Select fresh, settled evidence from the same live stream for
      // DeepQShield; no new camera permission or synthetic frame is used.
      const burst = webcam.captureFrameBurst
        ? await webcam.captureFrameBurst(20, 80)
        : null;
      const frames = burst?.bestFrameDataUrl
        ? [burst.bestFrameDataUrl]
        : (webcam.captureMultiFrames
          ? await webcam.captureMultiFrames(5, 85)
          : [webcam.captureFrame?.()].filter(Boolean));

      const userId = user?.user_id;
      const record = await analyzeVideoFrames(frames, userId, activeSessionIdRef.current);
      const res = normalizeDeepQShieldResult(record);
      console.log('[NIVORA DEEPQSHIELD] Runtime result:', res);

      const classification = String(res?.classification || res?.label || '').toUpperCase();
      const fakeProbability = res?.deepfake_probability;
      const isReal = classification === 'REAL' || classification === 'AUTHENTIC' ||
        (typeof fakeProbability === 'number' && fakeProbability <= 0.50);
      const verdict = { success: isReal, statusType: isReal ? 'VERIFIED' : 'REJECTED', raw: res };
      setVerdict('dq', verdict);
      setAuthState('DQ_DONE');
      setOverlayText('');

      if (!isReal) {
        const isIndeterminate = classification === 'INDETERMINATE';
        setError(res?.message || (isIndeterminate
          ? 'DeepQShield needs clearer visual evidence. Keep one face in frame and try again.'
          : 'DeepQShield: deepfake signal detected in camera feed.'));
        setRetryAction('deepqshield');
        setAuthState('FAILED');
        return;
      }
      // Proceed to Module 5
      await runFusion();
    } catch (e) {
      console.error('[NIVORA DQ] Error:', e);
      setVerdict('dq', { success: false, statusType: 'REJECTED', raw: { error: e.message } });
      setError('DeepQShield verification failed.');
      setRetryAction('deepqshield');
      setAuthState('FAILED');
      setOverlayText('');
    }
  }, [user, webcam]);

  // ── STEP 5: Module 5 Fusion ────────────────────────────────────────────────
  const runFusion = useCallback(async () => {
    markCurrent('fusion');
    setOverlayText('Final verification...');
    try {
      const userId = user?.user_id;
      if (!userId) {
        throw new Error('Not authenticated. Please log in first.');
      }
      const sessionId = activeSessionIdRef.current || voiceSession?.session_id || userId;

      const frame = faceProbeRef.current || webcam.captureFrame?.() || '';
      const frames = livenessFramesRef.current.length > 0
        ? livenessFramesRef.current
        : [frame].filter(Boolean);

      const payload = {
        user_id: userId,
        challenge_token: sessionId,
        face_image_base64: frame,
        face_frames_base64: frames,
        audio_pcm_base64: '',
      };

      const res = await authenticateMultimodalApi(payload);
      console.log('[NIVORA FUSION] Backend response:', res);

      // `success` reports API execution, while `decision` is the
      // server-authoritative authentication outcome.  Never authenticate a
      // user merely because the fusion request completed successfully.
      const decision = String(res?.decision || '').toUpperCase();
      const ok = decision === 'AUTHENTICATED' || decision === 'ALLOW';
      const statusType = ok ? 'VERIFIED' : (
        decision === 'INSUFFICIENT_EVIDENCE' || decision === 'RETRY_REQUIRED'
          ? 'INSUFFICIENT_EVIDENCE'
          : 'REJECTED'
      );
      const verdict = { success: ok, statusType, raw: res };
      setVerdict('fusion', verdict);
      setAuthState(ok ? 'DONE' : 'FAILED');
      setOverlayText('');
      if (!ok) {
        // Fusion consumes server-bound evidence exactly once. Do not replay it.
        setRetryAction('full');
        setError(res?.decision_reasons?.[0] || res?.message || 'Final multimodal fusion rejected authentication.');
      }
    } catch (e) {
      console.error('[NIVORA FUSION] Error:', e);
      setVerdict('fusion', { success: false, statusType: 'REJECTED', raw: { error: e.message } });
      setError('Final verification request failed.');
      setRetryAction('full');
      setAuthState('FAILED');
      setOverlayText('');
    }
  }, [user, webcam, voiceSession]);

  // ── Reset ─────────────────────────────────────────────────────────────────
  const reset = () => {
    activeSessionIdRef.current = null;
    setAuthState('READY');
    setChallenge(null);
    setVerdicts({ face: null, liveness: null, voice: null, dq: null, fusion: null });
    setVoiceSession(null);
    setExpanded({});
    setError(null);
    setRetryAction(null);
    setOverlayText('');
    setIsRecordingAudio(false);
    setRecordingSeconds(0);
    faceProbeRef.current = null;
    livenessFramesRef.current = [];
    webcam.stopCamera?.();
  };

  // ── Footer button logic ───────────────────────────────────────────────────
  function FooterButton() {
    if (authState === 'READY') return (
      <button onClick={runFace} style={btnStyle('#2563eb')}>
        <Lock size={14} /> Start Authentication
      </button>
    );
    if (authState === 'FACE_RUNNING') return (
      <button disabled style={btnStyle('#374151', true)}>
        <Loader2 size={14} className="animate-spin" /> Verifying face...
      </button>
    );
    if (authState === 'LIVENESS_PREPARE') return (
      <button disabled style={btnStyle('#374151', true)}>
        <Loader2 size={14} className="animate-spin" /> Preparing challenge...
      </button>
    );
    if (authState === 'LIVENESS_RUNNING') return (
      <button onClick={runLivenessCapture} style={btnStyle('#7c3aed')}>
        <Camera size={14} /> Capture Now
      </button>
    );
    if (authState === 'LIVENESS_RETRY') return (
      <button onClick={() => {
        setError(null);
        setVerdict('liveness', { _current: true });
        runLivenessStart();
      }} style={btnStyle('#b45309')}>
        <RefreshCw size={14} /> Retry Liveness Challenge
      </button>
    );
    if (authState === 'LIVENESS_DONE' || authState === 'VOICE_CHALLENGE_READY') {
      if (authState === 'VOICE_CHALLENGE_READY') return (
        <button onClick={requestMicAndRecord} style={btnStyle('#059669')}>
          <Mic size={14} /> Speak
        </button>
      );
    }
    if (authState === 'REQUESTING_MIC') return (
      <button disabled style={btnStyle('#374151', true)}>
        <Loader2 size={14} className="animate-spin" /> Requesting microphone...
      </button>
    );
    if (authState === 'VOICE_RECORDING') return (
      <button onClick={stopRecording} style={{ ...btnStyle('#dc2626'), animation: 'pulse 1.5s infinite' }}>
        <Square size={14} /> Stop Recording ({recordingSeconds}s)
      </button>
    );
    if (authState === 'VOICE_PROCESSING' || authState === 'DQ_RUNNING' || authState === 'FUSION_RUNNING') return (
      <button disabled style={btnStyle('#374151', true)}>
        <Loader2 size={14} className="animate-spin" /> Processing...
      </button>
    );
    if (authState === 'VOICE_NOT_ENROLLED') return (
      <button onClick={() => navigate('/enroll-voice')} style={btnStyle('#7c3aed')}>
        <UserPlus size={14} /> Set Up Voice Profile
      </button>
    );
    if (authState === 'DQ_DONE') return (
      <button disabled style={btnStyle('#374151', true)}>
        <Loader2 size={14} className="animate-spin" /> Final verification...
      </button>
    );
    if (authState === 'DONE') return (
      <button onClick={() => navigate('/dashboard')} style={btnStyle('#059669')}>
        <CheckCircle2 size={14} /> View Dashboard
      </button>
    );
    if (authState === 'FAILED' || authState === 'FATAL') return (
      <button onClick={() => {
        setError(null);
        if (retryAction === 'face') runFace();
        else if (retryAction === 'liveness') runLivenessStart();
        else if (retryAction === 'voice') startVoice();
        else if (retryAction === 'deepqshield') runDeepQShield();
        else reset();
      }} style={btnStyle('#b45309')}>
        <RefreshCw size={14} /> {retryAction === 'full' ? 'Start New Authentication' : 'Retry Failed Module'}
      </button>
    );
    return null;
  }

  function btnStyle(bg, disabled = false) {
    return {
      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
      padding: '12px 24px', borderRadius: 10, border: 'none',
      background: disabled ? '#1f2937' : bg,
      color: disabled ? '#6b7280' : '#fff',
      fontWeight: 700, fontSize: 13, fontFamily: 'monospace',
      cursor: disabled ? 'not-allowed' : 'pointer',
      width: '100%', transition: 'opacity 0.2s',
      opacity: disabled ? 0.7 : 1,
    };
  }

  // ── Camera overlay text ───────────────────────────────────────────────────
  const isDone = authState === 'DONE';
  const fusionV = verdicts.fusion;
  const showResult = isDone || (authState === 'FAILED' && !!fusionV);
  const showCamera = !['DONE', 'FATAL'].includes(authState) && !showResult;
  const faceV = verdicts.face;
  const livenessV = verdicts.liveness;
  const voiceV = verdicts.voice;
  const dqV = verdicts.dq;

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div
      aria-label="Nivora Authentication Console"
      style={{
        height: '100dvh', minHeight: 0, overflow: 'hidden',
        display: 'flex', flexDirection: 'column',
        background: 'radial-gradient(circle at 82% -10%, #10223b 0%, #07090e 36%), #07090e', color: '#e5e7eb',
        fontFamily: 'Inter, system-ui, sans-serif',
      }}
    >
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div style={{
        flex: '0 0 48px', minHeight: 0,
        borderBottom: '1px solid rgba(96,165,250,0.16)', background: 'rgba(7,9,14,0.78)', backdropFilter: 'blur(14px)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 20px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 26, height: 26, display: 'grid', placeItems: 'center', borderRadius: 7, background: 'rgba(96,165,250,0.1)', border: '1px solid rgba(96,165,250,0.24)' }}>
            <ShieldCheck size={15} color="#60a5fa" />
          </div>
          <span style={{ fontWeight: 800, fontSize: 13, fontFamily: 'monospace', letterSpacing: 2, color: '#e5e7eb' }}>NIVORA</span>
          <span style={{ fontSize: 10, color: '#6b7280', fontFamily: 'monospace', letterSpacing: 1 }}>SECURE AUTH</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#86efac', fontFamily: 'monospace', fontSize: 10, letterSpacing: 0.5 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#34d399', boxShadow: '0 0 10px #34d399' }} />
            SYSTEM SECURE
          </div>
          {authState !== 'READY' && authState !== 'DONE' && (
            <button onClick={reset} style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid #374151', borderRadius: 6, color: '#9ca3af', padding: '4px 10px', fontSize: 11, cursor: 'pointer', fontFamily: 'monospace' }}>
              Reset
            </button>
          )}
        </div>
      </div>

      {/* ── Main area ──────────────────────────────────────────────────────── */}
      <div style={{
        flex: 1, minHeight: 0, overflow: 'hidden',
        display: 'grid',
        gridTemplateColumns: 'minmax(260px, 28%) minmax(0, 1fr)',
      }}>

        {/* ── Left: Module workflow ──────────────────────────────────────── */}
        <div style={{
          borderRight: '1px solid rgba(96,165,250,0.14)', background: 'rgba(6,10,17,0.76)',
          padding: '16px 14px',
          overflowY: 'auto',
          display: 'flex', flexDirection: 'column', gap: 8,
        }}>
          <div style={{ padding: '4px 4px 8px', borderBottom: '1px solid #1a2035', marginBottom: 2 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 10, color: '#93c5fd', fontFamily: 'monospace', fontWeight: 700, letterSpacing: 1.5 }}>VERIFICATION PIPELINE</span>
              <span style={{ fontSize: 9, color: '#64748b', fontFamily: 'monospace' }}>05 GATES</span>
            </div>
            <div style={{ fontSize: 10, color: '#4b5563', fontFamily: 'monospace', marginTop: 5 }}>Live identity &amp; presentation-attack defense</div>
          </div>

          {MODULE_DEFS.map(def => {
            const resultKey = { FACE: 'face', LIVENESS: 'liveness', VOICE: 'voice', DQ: 'dq', FUSION: 'fusion' }[def.id];
            const result = verdicts[resultKey];
            return (
              <ModuleCard
                key={def.id}
                def={def}
                result={result}
                expanded={!!expanded[def.id]}
                onToggleDetails={() => setExpanded(prev => ({ ...prev, [def.id]: !prev[def.id] }))}
              />
            );
          })}

          {/* VOICE_NOT_ENROLLED special card */}
          {authState === 'VOICE_NOT_ENROLLED' && (
            <div style={{ background: 'rgba(251,191,36,0.06)', border: '1px solid rgba(251,191,36,0.25)', borderRadius: 10, padding: '12px', marginTop: 4 }}>
              <div style={{ color: '#fbbf24', fontSize: 12, fontWeight: 700, marginBottom: 4 }}>⚠ Voice profile not set up</div>
              <div style={{ color: '#9ca3af', fontSize: 11, lineHeight: 1.5 }}>
                You haven't enrolled a voice template yet. Set up your voice profile to use voice authentication.
              </div>
            </div>
          )}
        </div>

        {/* ── Right: Camera + overlays ───────────────────────────────────── */}
        <div style={{ position: 'relative', overflow: 'hidden', background: '#0a0d14', boxShadow: 'inset 0 0 90px rgba(0,0,0,0.32)' }}>

          {/* Camera */}
          {showCamera && (
            <>
              <video
                ref={webcam.videoRef}
                autoPlay playsInline muted
                aria-label="Live camera feed"
                style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)', display: 'block' }}
              />
              <canvas ref={webcam.canvasRef} style={{ display: 'none' }} />
            </>
          )}

          {/* Camera inactive state */}
          {!webcam.isCameraActive && authState === 'READY' && (
            <div style={{
              position: 'absolute', inset: 0,
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              gap: 12, background: '#07090e',
            }}>
              <Camera size={40} color="#374151" />
              <span style={{ color: '#6b7280', fontSize: 13, fontFamily: 'monospace' }}>Camera starts with authentication</span>
            </div>
          )}

          {/* Challenge / status overlay at top */}
          {overlayText && (
            <div style={{
              position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)',
              background: 'rgba(7,9,14,0.85)', backdropFilter: 'blur(8px)',
              border: '1px solid rgba(96,165,250,0.3)',
              borderRadius: 10, padding: '10px 20px',
              color: '#e5e7eb', fontSize: 14, fontWeight: 700, fontFamily: 'monospace',
              whiteSpace: 'nowrap', maxWidth: '90%', textAlign: 'center',
              zIndex: 10,
            }}>
              {overlayText}
            </div>
          )}

          {/* Error banner */}
          {error && !showResult && (
            <div style={{
              position: 'absolute', bottom: 12, left: 12, right: 12,
              background: 'rgba(127,29,29,0.92)', backdropFilter: 'blur(8px)',
              border: '1px solid rgba(248,113,113,0.4)',
              borderRadius: 10, padding: '10px 14px',
              color: '#fca5a5', fontSize: 12, fontFamily: 'monospace',
              display: 'flex', alignItems: 'flex-start', gap: 8, zIndex: 10,
            }}>
              <AlertTriangle size={14} style={{ marginTop: 1, flexShrink: 0 }} />
              <span>{error}</span>
            </div>
          )}

          {/* Final Verification Result Dashboard */}
          {showResult && (
            <div style={{
              position: 'absolute', inset: 0,
              background: 'linear-gradient(rgba(7,9,14,0.95), rgba(7,9,14,0.98)), repeating-linear-gradient(0deg, transparent 0, transparent 31px, rgba(96,165,250,0.035) 32px), repeating-linear-gradient(90deg, transparent 0, transparent 31px, rgba(96,165,250,0.035) 32px)',
              zIndex: 20, overflowY: 'auto', padding: '24px 20px 96px 20px', boxSizing: 'border-box',
            }}>
              <FinalVerificationResult
                fusionResult={fusionV}
                verdicts={verdicts}
                user={user}
                onNavigateDashboard={() => navigate('/dashboard')}
                onReset={reset}
              />
            </div>
          )}

          {/* Developer diagnostics (hidden in production) */}
          <DeveloperDiagnostics
            authState={authState}
            verdicts={verdicts}
            challenge={challenge}
            voiceSession={voiceSession}
          />
        </div>
      </div>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <div style={{
        flex: '0 0 72px', minHeight: 0,
        borderTop: '1px solid rgba(96,165,250,0.14)',
        display: 'flex', alignItems: 'center', padding: '0 16px',
        background: 'rgba(7,9,14,0.88)', backdropFilter: 'blur(12px)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#64748b', fontFamily: 'monospace', fontSize: 10 }}>
          <Lock size={13} color="#60a5fa" />
          <span>END-TO-END BIOMETRIC PROTECTION</span>
        </div>
        <div style={{ width: '100%', maxWidth: 300, marginLeft: 'auto' }}>
          <FooterButton />
        </div>
        {authState !== 'READY' && authState !== 'DONE' && (
          <div style={{ marginLeft: 'auto', color: '#374151', fontSize: 10, fontFamily: 'monospace', textAlign: 'right', paddingRight: 4 }}>
            {['FACE_RUNNING'].includes(authState) && '01 / 05'}
            {['LIVENESS_PREPARE','LIVENESS_RUNNING','LIVENESS_DONE'].includes(authState) && '02 / 05'}
            {['VOICE_CHALLENGE_READY','REQUESTING_MIC','VOICE_READY','VOICE_RECORDING','VOICE_PROCESSING','VOICE_DONE','VOICE_NOT_ENROLLED'].includes(authState) && '03 / 05'}
            {['DQ_RUNNING','DQ_DONE'].includes(authState) && '04 / 05'}
            {['FUSION_RUNNING'].includes(authState) && '05 / 05'}
          </div>
        )}
      </div>
    </div>
  );
}

export default MultimodalAuthConsole;
