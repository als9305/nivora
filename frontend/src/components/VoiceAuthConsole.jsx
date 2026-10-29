import React, { useState, useEffect } from 'react';
import { Volume2, ShieldCheck, Mic, RefreshCw, AlertCircle, CheckCircle2 } from 'lucide-react';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import { MicCapture } from './MicCapture';
import { startVoiceSession, verifyVoiceSession, verifyVoiceApi } from '../services/api';
import { useAuth } from '../auth/AuthContext';

export function VoiceAuthConsole() {
  const [mode, setMode] = useState('verify'); // 'verify' | 'enroll'
  const { user } = useAuth();
  const [userId, setUserId] = useState(user?.user_id || '');
  const [sessionData, setSessionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const audioRecorder = useAudioRecorder();

  // Start Voice Session on mount or when mode/userId changes
  useEffect(() => {
    initSession();
  }, [mode, userId]);

  const initSession = async () => {
    try {
      setErrorMsg(null);
      const res = await startVoiceSession(userId, mode === 'enroll' ? 'ENROLLMENT' : 'AUTHENTICATION');
      if (res?.success || res?.session_id) {
        setSessionData(res);
      } else {
        setSessionData(null);
        setErrorMsg("Failed to start voice authentication session.");
      }
    } catch (err) {
      console.warn("Could not start voice session:", err);
      setSessionData(null);
      setErrorMsg(err?.response?.data?.message || err?.response?.data?.detail || "Could not start voice session. Ensure the backend is reachable.");
    }
  };

  const handleVerifyVoice = async () => {
    if (!audioRecorder.audioBlob) {
      setErrorMsg("Please record microphone audio first.");
      return;
    }

    setLoading(true);
    setErrorMsg(null);
    setResult(null);

    try {
      // Convert audio blob to base64
      const reader = new FileReader();
      reader.readAsDataURL(audioRecorder.audioBlob);
      reader.onloadend = async () => {
        const base64Audio = reader.result.split(',')[1];
        try {
          let res;
          if (sessionData?.session_id) {
            res = await verifyVoiceSession(sessionData.session_id, userId, base64Audio);
          } else {
            res = await verifyVoiceApi(userId, base64Audio);
          }
          setResult(res);
        } catch (err) {
          console.error("Voice verification error:", err);
          setErrorMsg(err?.response?.data?.detail?.message || err?.message || "Voice authentication failed.");
        } finally {
          setLoading(false);
        }
      };
    } catch (err) {
      setErrorMsg("Failed to read recorded audio buffer.");
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto text-gray-200 font-sans">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between p-6 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-3 rounded-xl bg-blue-500/10 border border-blue-500/30 text-blue-400">
            <Volume2 className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">Module 3 — Voice Identity</h1>
            <p className="text-xs text-gray-400">
              Live Microphone Biometrics Only (WebRTC MediaRecorder Stream • Zero File Upload)
            </p>
          </div>
        </div>

        {/* Mode Selector */}
        <div className="flex items-center bg-[#07090e] p-1 rounded-xl border border-[#1f293d]">
          <button
            onClick={() => { setMode('verify'); setResult(null); setErrorMsg(null); }}
            className={`px-4 py-2 text-xs font-semibold rounded-lg transition cursor-pointer ${
              mode === 'verify' ? 'bg-blue-600 text-white shadow-md font-bold' : 'text-gray-400 hover:text-white'
            }`}
          >
            Verification
          </button>
          <button
            onClick={() => { setMode('enroll'); setResult(null); setErrorMsg(null); }}
            className={`px-4 py-2 text-xs font-semibold rounded-lg transition cursor-pointer ${
              mode === 'enroll' ? 'bg-blue-600 text-white shadow-md font-bold' : 'text-gray-400 hover:text-white'
            }`}
          >
            Enrollment
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: User Input & Microphone Capture */}
        <div className="lg:col-span-6 space-y-5">
          <div className="p-6 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-gray-400 font-mono mb-1">
                  Target User ID
                </label>
                <input
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  placeholder="Authenticated user ID"
                  className="w-full px-3.5 py-2 rounded-xl bg-[#07090e] border border-[#1f293d] text-white text-sm focus:outline-none focus:border-blue-500 font-mono"
                />
              </div>

              <button
                onClick={initSession}
                className="px-3 py-1.5 rounded-lg bg-[#07090e] border border-[#1f293d] text-gray-400 hover:text-white text-xs font-mono flex items-center gap-1 cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>New Session</span>
              </button>
            </div>

            {/* Embedded Mic Capture */}
            <MicCapture
              isRecording={audioRecorder.isRecording}
              audioBlob={audioRecorder.audioBlob}
              micError={audioRecorder.micError}
              recordingTimeSec={audioRecorder.recordingTimeSec}
              startRecording={audioRecorder.startRecording}
              stopRecording={audioRecorder.stopRecording}
              challengePhrase={sessionData?.challenge_phrase || "Apple river green"}
            />

            <button
              onClick={handleVerifyVoice}
              disabled={loading || !audioRecorder.audioBlob}
              className="w-full py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-xs shadow-lg shadow-blue-500/25 disabled:opacity-50 transition cursor-pointer flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Processing ECAPA-TDNN Acoustic Analysis...</span>
                </>
              ) : (
                <span>Submit Voice Verification</span>
              )}
            </button>

            {errorMsg && (
              <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center space-x-2 font-medium">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Verification Results */}
        <div className="lg:col-span-6 space-y-5">
          {result ? (
            <div className="p-6 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl space-y-4">
              {/* Final Verdict Header */}
              <div className="flex items-center justify-between p-4 rounded-xl bg-[#07090e] border border-[#1f293d]">
                <div>
                  <span className="text-[10px] uppercase tracking-wider text-gray-400 font-bold block font-mono">
                    Final Authentication Verdict
                  </span>
                  <p className={`text-lg font-extrabold font-mono ${
                    result.final_verdict === "AUTHENTICATED" || result.success ? "text-emerald-400" : "text-rose-400"
                  }`}>
                    {result.final_verdict || (result.success ? "ENROLLED_SUCCESS" : "REJECTED")}
                  </p>
                </div>
                <div className={`p-2.5 rounded-xl border ${
                  result.final_verdict === "AUTHENTICATED" || result.success
                    ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                    : "bg-rose-500/10 border-rose-500/30 text-rose-400"
                }`}>
                  <ShieldCheck className="w-6 h-6" />
                </div>
              </div>

              {/* Speaker Match, Liveness & Phrase Verdict Badges */}
              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="p-3 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-1">
                  <span className="text-[10px] text-gray-400 font-bold block uppercase font-mono">Speaker</span>
                  <span className={`text-xs font-bold font-mono block ${
                    result.speaker_verdict === "MATCH" ? "text-emerald-400" : "text-amber-400"
                  }`}>
                    {result.speaker_verdict || "N/A"}
                  </span>
                  <span className="text-[10px] text-gray-500 block font-mono">
                    {((result.voice_similarity || 0) * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="p-3 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-1">
                  <span className="text-[10px] text-gray-400 font-bold block uppercase font-mono">Liveness</span>
                  <span className={`text-xs font-bold font-mono block ${
                    result.liveness_verdict === "LIVE" ? "text-emerald-400" : "text-rose-400"
                  }`}>
                    {result.liveness_verdict || "N/A"}
                  </span>
                  <span className="text-[10px] text-gray-500 block font-mono">
                    Spoof: {((result.voice_spoof_probability || 0) * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="p-3 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-1">
                  <span className="text-[10px] text-gray-400 font-bold block uppercase font-mono">Phrase</span>
                  <span className={`text-xs font-bold font-mono block ${
                    result.phrase_match === 'MATCHED' ? 'text-emerald-400' :
                    (result.phrase_match === 'MISMATCH' || result.phrase_match === 'MISMATCHED') ? 'text-rose-400' : 'text-amber-400'
                  }`}>
                    {result.phrase_match || "UNVALIDATED"}
                  </span>
                  <span className="text-[10px] text-gray-500 block font-mono">
                    {result.phrase_match_score != null ? `${(result.phrase_match_score * 100).toFixed(0)}% Match` : 'Anti-Replay'}
                  </span>
                </div>
              </div>

              {/* Acoustic Metrics */}
              {result.acoustic_metrics && (
                <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-2">
                  <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block font-mono">
                    Acoustic Spectral Features
                  </span>
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Centroid:</span>
                      <span className="text-white font-semibold">{result.acoustic_metrics.spectral_centroid_hz} Hz</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Rolloff:</span>
                      <span className="text-white font-semibold">{result.acoustic_metrics.spectral_rolloff_hz} Hz</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Pitch (F0):</span>
                      <span className="text-white font-semibold">{result.acoustic_metrics.f0_mean_hz} Hz</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Pitch Std:</span>
                      <span className="text-white font-semibold">{result.acoustic_metrics.f0_std_hz} Hz</span>
                    </div>
                  </div>
                </div>
              )}

              <p className="text-xs text-gray-300 italic">
                "{result.message}"
              </p>
            </div>
          ) : (
            <div className="p-8 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl text-center space-y-3">
              <Mic className="w-12 h-12 text-gray-500 mx-auto" />
              <h3 className="text-sm font-bold text-white">Live Microphone Biometric Status</h3>
              <p className="text-xs text-gray-400 max-w-sm mx-auto">
                Press Start Recording, speak the challenge phrase into your device microphone, and submit verification to evaluate live speaker match and audio liveness.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default VoiceAuthConsole;
