import React, { useState, useRef } from 'react';
import { 
  Scan, 
  ShieldCheck, 
  ShieldAlert, 
  AlertTriangle, 
  Camera, 
  Play, 
  Layers, 
  Activity, 
  CheckCircle2, 
  FileText
} from 'lucide-react';
import { checkFaceLiveness } from '../services/api';
import { useAuth } from '../auth/AuthContext';

export function FaceLivenessConsole() {
  const { user } = useAuth();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Challenge Response state
  const [challengeType, setChallengeType] = useState('BLINK'); // 'BLINK' | 'TURN_LEFT' | 'TURN_RIGHT' | 'SMILE'
  const [recordedFrames, setRecordedFrames] = useState([]);

  // Live Camera state
  const videoRef = useRef(null);
  const [webcamActive, setWebcamActive] = useState(false);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setWebcamActive(true);
      }
    } catch (err) {
      console.error("Webcam access error:", err);
      setError("Unable to access live webcam. Please ensure camera permissions are granted.");
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      setWebcamActive(false);
    }
  };

  const captureFrameBase64 = () => {
    if (!videoRef.current) return null;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
  };

  const handleRunLiveness = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const frames = [];
      if (webcamActive) {
        // Capture 5 frames over 1.5s
        for (let i = 0; i < 5; i++) {
          const f = captureFrameBase64();
          if (f) frames.push(f);
          await new Promise(res => setTimeout(res, 300));
        }
      }

      const res = await checkFaceLiveness(frames, user?.user_id || null);
      setResult(res);
    } catch (err) {
      console.error("Liveness check error:", err);
      setError(err?.response?.data?.detail?.message || err?.message || "Failed to execute liveness check.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Module Banner Header */}
      <div className="bg-[#0c1017] border border-[#1f293d] rounded-2xl p-6 shadow-2xl">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="px-2.5 py-0.5 rounded text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                MODULE 2 ACTIVE
              </span>
              <span className="text-xs text-gray-400 font-mono">ISO/IEC 30107-3 PAD Standard</span>
            </div>
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              <Scan className="w-6 h-6 text-blue-400" />
              Face Liveness Detection Console
            </h2>
            <p className="text-gray-400 text-sm mt-1">
              Evaluates passive RGB spatial texture, 2D FFT frequency moiré patterns, and temporal micro-motion to reject photo prints, screen replays, and video presentation attacks.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {!webcamActive ? (
              <button
                onClick={startWebcam}
                className="px-4 py-2.5 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 text-sm font-semibold rounded-xl border border-blue-500/30 transition flex items-center gap-2 cursor-pointer shadow-md"
              >
                <Camera className="w-4 h-4 text-blue-400" />
                Enable Camera
              </button>
            ) : (
              <button
                onClick={stopWebcam}
                className="px-4 py-2.5 bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 text-sm font-semibold rounded-xl border border-rose-500/30 transition flex items-center gap-2 cursor-pointer shadow-md"
              >
                <Camera className="w-4 h-4" />
                Stop Camera
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Console Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Camera Viewport & Capture */}
        <div className="lg:col-span-6 space-y-4">
          <div className="bg-[#0c1017] border border-[#1f293d] rounded-2xl p-6 shadow-2xl relative overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-bold text-white uppercase tracking-wider font-mono">Live Video Stream</h3>
              <span className={`w-2.5 h-2.5 rounded-full ${webcamActive ? 'bg-emerald-500 animate-ping' : 'bg-gray-600'}`} />
            </div>

            <div className="relative aspect-video rounded-xl bg-[#07090e] border border-[#1f293d] overflow-hidden flex items-center justify-center">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover transform -scale-x-100"
              />
              {!webcamActive && (
                <div className="absolute inset-0 bg-[#07090e]/90 flex flex-col items-center justify-center text-gray-400 space-y-2 p-4">
                  <Camera className="w-8 h-8 text-gray-500" />
                  <p className="text-xs font-mono">Camera Disconnected</p>
                </div>
              )}
            </div>

            <button
              onClick={handleRunLiveness}
              disabled={loading || !webcamActive}
              className="mt-4 w-full py-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-xs shadow-lg shadow-blue-500/25 disabled:opacity-50 transition cursor-pointer flex items-center justify-center gap-2"
            >
              <Play className="w-4 h-4" />
              <span>{loading ? "Analyzing Liveness & Micro-motion..." : "Execute Liveness Check"}</span>
            </button>
          </div>
        </div>

        {/* Right Column: Structured Decision & Metrics */}
        <div className="lg:col-span-6 space-y-4">
          <div className="bg-[#0c1017] border border-[#1f293d] rounded-2xl p-6 shadow-2xl h-full flex flex-col justify-between">
            <div>
              <h3 className="text-xs font-bold text-gray-300 uppercase tracking-wider mb-4 flex items-center gap-2 font-mono">
                <Activity className="w-4 h-4 text-blue-400" />
                Anti-Spoofing Verdict & Evidence
              </h3>

              {error && (
                <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400 text-sm flex items-start gap-3 mb-4 shadow-sm">
                  <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-bold text-rose-300">Liveness Error</p>
                    <p className="text-xs mt-0.5 text-rose-400">{error}</p>
                  </div>
                </div>
              )}

              {!result && !error && (
                <div className="py-16 text-center border border-dashed border-[#1f293d] rounded-xl bg-[#07090e]">
                  <Layers className="w-10 h-10 text-gray-500 mx-auto mb-2" />
                  <p className="text-sm text-white font-bold">Awaiting Evaluation</p>
                  <p className="text-xs text-gray-400 max-w-xs mx-auto mt-1">
                    Execute a liveness check to view structured verdict, confidence scores, and physical feature descriptors.
                  </p>
                </div>
              )}

              {result && (
                <div className="space-y-5">
                  {/* Status Banner Card */}
                  <div className={`p-4 rounded-xl border flex items-center justify-between shadow-sm ${
                    result.liveness === 'LIVE' 
                      ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                      : result.liveness === 'SPOOF'
                      ? 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                      : 'bg-amber-500/10 border-amber-500/30 text-amber-400'
                  }`}>
                    <div className="flex items-center gap-3">
                      {result.liveness === 'LIVE' ? (
                        <ShieldCheck className="w-8 h-8 text-emerald-400 shrink-0" />
                      ) : result.liveness === 'SPOOF' ? (
                        <ShieldAlert className="w-8 h-8 text-rose-400 shrink-0" />
                      ) : (
                        <AlertTriangle className="w-8 h-8 text-amber-400 shrink-0" />
                      )}
                      <div>
                        <div className="text-xs font-mono tracking-wider uppercase font-bold text-gray-400">
                          VERDICT STATUS
                        </div>
                        <div className="text-xl font-extrabold tracking-tight mt-0.5">
                          {result.liveness}
                        </div>
                      </div>
                    </div>

                    <div className="text-right">
                      <span className="text-xs font-mono text-gray-400 block font-semibold">SCORE</span>
                      <span className="text-xl font-extrabold font-mono text-white">
                        {typeof result.score === 'number' ? result.score.toFixed(4) : result.score}
                      </span>
                    </div>
                  </div>

                  {/* Attack Type & Details if Spoof */}
                  {result.attack_type && (
                    <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-xl text-xs text-rose-300 flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                      <span><strong>ATTACK TYPE:</strong> {result.attack_type}</span>
                    </div>
                  )}

                  {/* Metrics Table */}
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="bg-[#07090e] p-3.5 rounded-xl border border-[#1f293d]">
                      <span className="text-xs text-gray-400 block font-mono font-semibold">THRESHOLD (τ*)</span>
                      <span className="text-base font-bold text-white font-mono">
                        {result.threshold ? result.threshold.toFixed(4) : '0.6500'}
                      </span>
                    </div>

                    <div className="bg-[#07090e] p-3.5 rounded-xl border border-[#1f293d]">
                      <span className="text-xs text-gray-400 block font-mono font-semibold">CONFIDENCE</span>
                      <span className="text-base font-bold text-white font-mono">
                        {result.confidence ? (result.confidence * 100).toFixed(1) + '%' : 'N/A'}
                      </span>
                    </div>

                    <div className="bg-[#07090e] p-3.5 rounded-xl border border-[#1f293d]">
                      <span className="text-xs text-gray-400 block font-mono font-semibold">FRAMES ANALYZED</span>
                      <span className="text-base font-bold text-white font-mono">
                        {result.frames_analyzed || 0} frames
                      </span>
                    </div>

                    <div className="bg-[#07090e] p-3.5 rounded-xl border border-[#1f293d]">
                      <span className="text-xs text-gray-400 block font-mono font-semibold">MOTION DELTA</span>
                      <span className="text-base font-bold text-white font-mono">
                        {result.details?.motion_delta !== undefined ? result.details.motion_delta : 'N/A'}
                      </span>
                    </div>
                  </div>

                  {/* Message Detail Box */}
                  <div className="p-4 bg-[#07090e] rounded-xl border border-[#1f293d] text-xs text-gray-300">
                    <span className="text-gray-400 font-bold block mb-1">EVIDENCE RATIONALE:</span>
                    <p className="leading-relaxed font-medium">{result.message}</p>
                  </div>
                </div>
              )}
            </div>

            <div className="mt-4 pt-3 border-t border-[#1f293d] text-[11px] text-gray-500 flex items-center justify-between font-mono">
              <span>NIVORA ANTI-SPOOF ENGINE</span>
              <span>INDEPENDENT MODULE 2</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default FaceLivenessConsole;
