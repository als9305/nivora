import React from 'react';
import { Eye, Activity } from 'lucide-react';

/**
 * DeveloperDiagnostics
 *
 * Renders all diagnostic debug panels.
 * Only mounted when VITE_DEVELOPER_MODE=true (set in .env / .env.local).
 *
 * NOT accessible via URL params or runtime toggles — must be set at build time.
 */

const IS_DEV_MODE = import.meta.env.VITE_DEVELOPER_MODE === 'true';

export function DeveloperDiagnostics({
  submittedProbeFrame,
  probeMetrics,
  livenessDiagnostics,
  multimodalResponse,
  activityLogs,
  verdicts,
  authState,
  challenge,
}) {
  if (!IS_DEV_MODE) return null;

  // Seamlessly adapt if passed verdicts dictionary
  const faceRaw = verdicts?.face?.raw;
  const livenessRaw = verdicts?.liveness?.raw;
  const fusionRaw = verdicts?.fusion?.raw;

  const probe = submittedProbeFrame || faceRaw?._probe_frame;
  const metrics = probeMetrics || (faceRaw ? {
    width: faceRaw.image_width || 640,
    height: faceRaw.image_height || 480,
    sharpness: faceRaw.sharpness ?? faceRaw.quality_score,
    brightness: faceRaw.brightness,
  } : null);

  const chObj = livenessRaw?.challenge;
  const rawCh = (chObj && typeof chObj === 'object' ? chObj.type : chObj) || challenge;
  const challengeStr = typeof rawCh === 'string' ? rawCh : (rawCh?.id || rawCh?.type || rawCh?.challenge || (rawCh ? String(rawCh) : 'TURN_HEAD'));
  const livenessDiag = livenessDiagnostics || (livenessRaw ? {
    challenge: challengeStr,
    framesReceived: livenessRaw.frames_analyzed || livenessRaw.frames_evaluated || 0,
    captureDurationMs: livenessRaw.capture_duration_ms || 1800,
    landmarksDetectedFrames: (chObj && typeof chObj === 'object' ? chObj.landmarks_detected : undefined) || 'TRACKED',
    initialYaw: (chObj && typeof chObj === 'object' ? chObj.initial_yaw : livenessRaw.initial_yaw) ?? 0,
    finalYaw: (chObj && typeof chObj === 'object' ? chObj.final_yaw : livenessRaw.final_yaw) ?? 0,
    yawDelta: (chObj && typeof chObj === 'object' ? chObj.yaw_delta : livenessRaw.yaw_delta) ?? 0,
    detectedDirection: (chObj && typeof chObj === 'object' ? chObj.detected_direction : livenessRaw.detected_direction) || 'NONE',
    challengeDetectorResult: (chObj && typeof chObj === 'object' ? chObj.result : livenessRaw.challenge_result) || 'N/A',
    passiveScore: livenessRaw.score ?? livenessRaw.passive_score ?? 0,
    passiveThreshold: livenessRaw.threshold ?? 0.50,
    finalVerdict: livenessRaw.liveness || livenessRaw.status || 'N/A',
    horizontalMovement: (chObj && typeof chObj === 'object' ? chObj.horizontal_px : undefined) ?? 0,
    verticalMovement: (chObj && typeof chObj === 'object' ? chObj.vertical_px : undefined) ?? 0,
    reason: livenessRaw.message || (chObj && typeof chObj === 'object' ? chObj.details : '') || '',
    timestamp: new Date().toLocaleTimeString(),
  } : null);

  const fusionResp = multimodalResponse || fusionRaw;

  return (
    <div className="space-y-4 font-mono text-xs border-t border-yellow-500/30 pt-4 mt-4">
      {/* DEV MODE BADGE */}
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 text-[10px] font-bold">
        <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
        DEVELOPER MODE — diagnostics visible
      </div>

      {/* MODULE 1 FACE IDENTITY PROBE FRAME */}
      {probe && (
        <div className="p-3.5 rounded-xl bg-[#07090e] border border-blue-500/30 space-y-2">
          <div className="flex items-center justify-between text-blue-400 font-bold border-b border-[#1f293d] pb-2 text-[11px]">
            <span className="flex items-center gap-1.5">
              <Eye className="w-3.5 h-3.5" />
              MODULE 1 FACE IDENTITY PROBE FRAME (EXACT SUBMITTED)
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/30">DEV DEBUG</span>
          </div>
          <div className="flex flex-col sm:flex-row items-start gap-4 pt-1">
            <img
              src={probe}
              alt="Module 1 Face Identity Probe Frame"
              className="w-44 h-32 object-cover rounded-lg border border-blue-500/40 shadow-lg shrink-0"
            />
            <div className="space-y-1.5 text-[11px] text-gray-300">
              <p><span className="text-gray-500">Dimensions:</span> {metrics?.width || 640}×{metrics?.height || 480} px</p>
              <p><span className="text-gray-500">Base64 Size:</span> {Math.round((probe.length * 0.75) / 1024)} KB ({probe.length} chars)</p>
              <p><span className="text-gray-500">Sharpness Score:</span> <span className="text-emerald-400 font-bold">{metrics?.sharpness?.toFixed(2) || 'N/A'}</span></p>
              <p><span className="text-gray-500">Mean Brightness:</span> <span className="text-blue-300 font-bold">{metrics?.brightness?.toFixed(2) || 'N/A'}</span></p>
              <p><span className="text-gray-500">MIME Format:</span> image/jpeg</p>
            </div>
          </div>
        </div>
      )}

      {/* MODULE 2 FACE LIVENESS DETECTION DIAGNOSTICS */}
      {livenessDiag && (
        <div className="p-3.5 rounded-xl bg-[#07090e] border border-cyan-500/30 space-y-2.5">
          <div className="flex items-center justify-between text-cyan-400 font-bold border-b border-[#1f293d] pb-2 text-[11px]">
            <span className="flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5" />
              MODULE 2 FACE LIVENESS DETECTION DIAGNOSTICS
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/30">DEV DEBUG</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-[11px] pt-1">
            <div className="space-y-1.5">
              <p><span className="text-gray-500">Challenge:</span> <span className="text-white font-semibold">{livenessDiag.challenge}</span></p>
              <p><span className="text-gray-500">Capture:</span> <span className="text-cyan-300 font-semibold">{livenessDiag.framesReceived} frames / {livenessDiag.captureDurationMs} ms</span></p>
              <p><span className="text-gray-500">Facial Landmarks:</span> <span className="text-emerald-400 font-bold">{livenessDiag.landmarksDetectedFrames}</span></p>
              <p><span className="text-gray-500">Initial Yaw:</span> <span className="text-purple-300">{livenessDiag.initialYaw !== undefined ? livenessDiag.initialYaw.toFixed(4) : '0.0000'}</span></p>
              <p><span className="text-gray-500">Final Yaw:</span> <span className="text-purple-300">{livenessDiag.finalYaw !== undefined ? livenessDiag.finalYaw.toFixed(4) : '0.0000'}</span></p>
              <p><span className="text-gray-500">Yaw Delta:</span> <span className="text-cyan-400 font-bold">{livenessDiag.yawDelta !== undefined ? livenessDiag.yawDelta.toFixed(4) : '0.0000'}</span></p>
            </div>
            <div className="space-y-1.5">
              <p><span className="text-gray-500">Detected Direction:</span> <span className="text-blue-300 font-semibold">{livenessDiag.detectedDirection || 'NONE'}</span></p>
              <p>
                <span className="text-gray-500">Challenge Result:</span>{' '}
                <span className={`font-bold px-1.5 py-0.5 rounded text-[10px] ${
                  livenessDiag.challengeDetectorResult === 'PASSED'
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                }`}>
                  {livenessDiag.challengeDetectorResult}
                </span>
              </p>
              <p>
                <span className="text-gray-500">Passive Score:</span>{' '}
                <span className="text-amber-300 font-bold">{livenessDiag.passiveScore !== undefined ? livenessDiag.passiveScore.toFixed(4) : 'N/A'}</span>
                {' '}<span className="text-gray-500">(Thr: {livenessDiag.passiveThreshold?.toFixed(4)})</span>
              </p>
              <p>
                <span className="text-gray-500">Final Verdict:</span>{' '}
                <span className={`font-bold px-1.5 py-0.5 rounded text-[10px] ${
                  livenessDiag.finalVerdict === 'LIVE'
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : livenessDiag.finalVerdict === 'INSUFFICIENT_EVIDENCE'
                    ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                    : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                }`}>
                  {livenessDiag.finalVerdict}
                </span>
              </p>
              <p><span className="text-gray-500">Movement (H/V):</span> <span className="text-gray-300">{livenessDiag.horizontalMovement !== undefined ? `${livenessDiag.horizontalMovement.toFixed(1)}px` : '0.0px'} / {livenessDiag.verticalMovement !== undefined ? `${livenessDiag.verticalMovement.toFixed(1)}px` : '0.0px'}</span></p>
            </div>
          </div>
          {livenessDiag.reason && (
            <div className="pt-1.5 border-t border-[#1f293d] text-[10px] text-gray-400 flex items-center justify-between">
              <span><span className="text-gray-500">Diagnostic Verdict:</span> {livenessDiag.reason}</span>
              <span className="text-gray-600">{livenessDiag.timestamp}</span>
            </div>
          )}
        </div>
      )}

      {/* MODULE 5 FINAL VERIFICATION RAW RESPONSE */}
      {fusionResp && (
        <div className="p-3.5 rounded-xl bg-[#07090e] border border-emerald-500/20 space-y-2">
          <div className="text-emerald-400 font-bold text-[11px] border-b border-[#1f293d] pb-2">MODULE 5 FINAL VERIFICATION RAW RESPONSE</div>
          <pre className="text-[10px] text-gray-400 overflow-x-auto whitespace-pre-wrap break-all">
            {JSON.stringify(fusionResp, null, 2)}
          </pre>
        </div>
      )}

      {/* ACTIVITY LOG */}
      {activityLogs && activityLogs.length > 0 && (
        <div className="p-3.5 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-2">
          <div className="text-gray-400 font-bold text-[11px] border-b border-[#1f293d] pb-2 uppercase">Authentication Activity Log</div>
          <div className="max-h-48 overflow-y-auto space-y-1 pr-1">
            {activityLogs.map((log, i) => (
              <div key={i} className="flex items-start gap-2 text-[10px]">
                <span className="text-gray-600 shrink-0">[{log.time}]</span>
                <span className={log.type === 'success' ? 'text-emerald-400' : log.type === 'error' ? 'text-rose-400' : log.type === 'warn' ? 'text-amber-400' : 'text-gray-300'}>
                  {log.msg}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default DeveloperDiagnostics;
