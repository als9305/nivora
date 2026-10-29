import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldCheck, Mic, CheckCircle2, AlertCircle, Loader2, Square, Volume2 } from 'lucide-react';
import { encodeWavPcm16, mergePcmChunks } from '../hooks/useAudioRecorder';
import { enrollVoiceSampleApi } from '../services/api';
import { useAuth } from '../auth/AuthContext';

const ENROLLMENT_PHRASES = [
  { index: 1, text: "Blue Quebec Yankee", phonetic: "bloo keh-bek yang-kee" },
  { index: 2, text: "Green Delta Seven", phonetic: "green del-tah sev-en" },
  { index: 3, text: "Silver River Nine", phonetic: "sil-ver riv-er nyne" },
];

export const VoiceEnrollmentPage = () => {
  const navigate = useNavigate();
  const { user, setOnboardingStep } = useAuth();

  const [currentSampleIndex, setCurrentSampleIndex] = useState(1);
  const [completedSamples, setCompletedSamples] = useState([]);
  const [step, setStep] = useState('IDLE'); // 'IDLE' | 'RECORDING' | 'PROCESSING' | 'COMPLETE' | 'FAILED'
  const [error, setError] = useState(null);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [audioFrequencies, setAudioFrequencies] = useState([15, 20, 15, 30, 25, 40, 20, 35, 15, 25, 15, 20, 15]);
  const [sessionId, setSessionId] = useState(null);

  // Audio recording refs
  const micStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const pcmChunksRef = useRef([]);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const animFrameRef = useRef(null);
  const timerIntervalRef = useRef(null);
  const recordingStartedAtRef = useRef(null);

  // Clean up media streams and audio context on unmount
  useEffect(() => {
    return () => {
      stopAllMedia();
    };
  }, []);

  const stopAllMedia = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop();
      } catch (e) {}
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(t => t.stop());
      micStreamRef.current = null;
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      try {
        audioCtxRef.current.close();
      } catch (e) {}
    }
  };

  const handleStartRecording = async () => {
    setError(null);
    setRecordingSeconds(0);
    audioChunksRef.current = [];
    pcmChunksRef.current = [];
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }

    const activePhrase = ENROLLMENT_PHRASES[currentSampleIndex - 1];

    try {
      console.log('[NIVORA VOICE ENROLL] Requesting microphone...');
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
        console.warn("[NIVORA VOICE ENROLL] 16kHz constraint failed, retrying without sampleRate constraint:", streamErr);
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: true
            }
          });
        } catch (fallbackErr) {
          console.warn("[NIVORA VOICE ENROLL] Advanced audio constraints failed, falling back to basic audio:", fallbackErr);
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }
      }
      micStreamRef.current = stream;

      const audioTracks = stream.getAudioTracks();
      console.log('[NIVORA VOICE ENROLL] Microphone granted.');
      console.log(`[NIVORA VOICE ENROLL] Audio tracks: ${audioTracks.length}`);

      // Web Audio API context for live spectrum visualizer and PCM conversion
      let audioCtx;
      try {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      } catch (ctxErr) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      audioCtxRef.current = audioCtx;
      if (audioCtx.state === 'suspended') await audioCtx.resume();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 32;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Use raw Web Audio PCM instead of decoding a browser-specific
      // MediaRecorder WebM/MP4 container after recording.
      const pcmProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
      const silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      source.connect(pcmProcessor);
      pcmProcessor.connect(silentGain);
      silentGain.connect(audioCtx.destination);
      pcmProcessor.onaudioprocess = event => {
        pcmChunksRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const updateFrequencies = () => {
        if (analyserRef.current) {
          analyserRef.current.getByteFrequencyData(dataArray);
          const freqs = Array.from(dataArray.slice(0, 13)).map(val =>
            Math.max(15, Math.min(100, Math.round((val / 255) * 100)))
          );
          setAudioFrequencies(freqs);
          animFrameRef.current = requestAnimationFrame(updateFrequencies);
        }
      };

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
        recordingStartedAtRef.current = null;
        const chunkCount = pcmChunksRef.current.length;
        console.log(`[NIVORA VOICE ENROLL] PCM blocks received: ${chunkCount}`);
        console.log('[NIVORA VOICE ENROLL] Recording stopped.');
        setStep('PROCESSING');

        if (animFrameRef.current) {
          cancelAnimationFrame(animFrameRef.current);
          animFrameRef.current = null;
        }

        try {
          const pcmSamples = mergePcmChunks(pcmChunksRef.current);
          if (!pcmSamples.length) throw new Error('No microphone samples were captured.');
          const durationSec = pcmSamples.length / audioCtx.sampleRate;

          // Encode direct PCM to standard 16-bit 16kHz mono RIFF PCM WAV.
          const wavArrayBuffer = encodeWavPcm16(pcmSamples, audioCtx.sampleRate, 16000);
          const wavBytes = wavArrayBuffer.byteLength;
          console.log(`[NIVORA VOICE ENROLL] PCM WAV generated: sampleRate=16000 channels=1 bits=16 duration=${durationSec.toFixed(2)} bytes=${wavBytes}`);

          const wavBlob = new Blob([wavArrayBuffer], { type: 'audio/wav' });
          const reader = new FileReader();

          reader.onloadend = async () => {
            const base64Audio = reader.result;
            await submitEnrollmentSample(base64Audio);
          };
          reader.readAsDataURL(wavBlob);

        } catch (procErr) {
          console.error('[NIVORA VOICE ENROLL] Audio processing error:', procErr);
          setError("Failed to process microphone audio recording. Please try again.");
          setStep('IDLE');
        } finally {
          stream.getTracks().forEach(t => t.stop());
          micStreamRef.current = null;
          if (audioCtx.state !== 'closed') {
            audioCtx.close().catch(() => {});
          }
        }
      };

      recorder.start(100);
      setStep('RECORDING');
      console.log(`[NIVORA VOICE ENROLL] Recording started. Speaking: "${activePhrase.text}"`);
      updateFrequencies();

      // Start recording duration timer
      recordingStartedAtRef.current = performance.now();
      timerIntervalRef.current = setInterval(() => {
        const startedAt = recordingStartedAtRef.current;
        if (startedAt !== null) {
          setRecordingSeconds(Math.floor((performance.now() - startedAt) / 1000));
        }
      }, 250);

      // Automatically stop after 3.5 seconds
      setTimeout(() => {
        if (recorder.state === 'recording') {
          recorder.stop();
        }
      }, 3500);

    } catch (err) {
      console.error('[NIVORA VOICE ENROLL] Microphone access error:', err);
      let msg = "Microphone access denied or unavailable.";
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = "Microphone permission was denied. Please allow microphone access in your browser.";
      }
      setError(msg);
      setStep('FAILED');
    }
  };

  const handleStopRecording = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    recordingStartedAtRef.current = null;
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
  };

  const submitEnrollmentSample = async (audioB64) => {
    const userId = user?.user_id;
    if (!userId) {
      setError('Not authenticated. Please log in or register first.');
      return;
    }

    try {
      const res = await enrollVoiceSampleApi(
        userId,
        audioB64,
        currentSampleIndex,
        3,
        sessionId
      );

      console.log(`[NIVORA VOICE ENROLL] Sample ${currentSampleIndex} response:`, res);

      if (res?.session_id) {
        setSessionId(res.session_id);
      }

      if (res?.success) {
        const nextCompleted = [...completedSamples, currentSampleIndex];
        setCompletedSamples(nextCompleted);

        if (res.status === 'VOICE_ENROLLMENT_COMPLETE' || currentSampleIndex >= 3) {
          setStep('COMPLETE');
        } else {
          setCurrentSampleIndex(prev => prev + 1);
          setStep('IDLE');
        }
      } else {
        throw new Error(res?.message || `Voice sample ${currentSampleIndex} validation failed.`);
      }
    } catch (err) {
      console.error('[NIVORA VOICE ENROLL] Enrollment API error:', err);
      const errMsg = err?.response?.data?.detail?.message || err?.message || "Failed to enroll voice sample.";
      setError(errMsg);
      setStep('IDLE');
    }
  };

  const currentPhrase = ENROLLMENT_PHRASES[currentSampleIndex - 1] || ENROLLMENT_PHRASES[0];

  return (
    <div className="min-h-screen bg-[#07090e] text-gray-200 flex flex-col justify-center items-center p-6 relative font-sans">
      {/* Background Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-600/10 blur-[120px] rounded-full pointer-events-none" />

      <div className="w-full max-w-lg bg-[#0c1017] border border-[#1f293d] rounded-2xl p-6 sm:p-8 space-y-6 shadow-2xl relative z-10">

        {/* Header */}
        <div className="text-center space-y-2">
          <div className="inline-flex p-3 rounded-2xl bg-purple-500/10 border border-purple-500/20 text-purple-400 mb-1">
            <Mic className="w-8 h-8" />
          </div>
          <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight">
            Speaker Voice Profile Setup
          </h1>
          <p className="text-xs text-gray-400 font-medium max-w-sm mx-auto">
            Provide 3 distinct voice samples using your live microphone to build your encrypted 192-d biometric voiceprint.
          </p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start gap-3">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <div className="flex-1">
              <span className="font-bold">Enrollment Notice: </span>
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* Step Indicator */}
        <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
          {ENROLLMENT_PHRASES.map((phrase) => {
            const isDone = completedSamples.includes(phrase.index);
            const isCurrent = currentSampleIndex === phrase.index && step !== 'COMPLETE';
            return (
              <div
                key={phrase.index}
                className={`p-2.5 rounded-xl border transition-all ${
                  isDone
                    ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-400 font-bold'
                    : isCurrent
                    ? 'bg-purple-500/15 border-purple-500/50 text-purple-300 font-bold ring-1 ring-purple-500/40'
                    : 'bg-[#07090e] border-[#1f293d] text-gray-500'
                }`}
              >
                <div className="text-[10px] text-gray-400 mb-0.5">Sample {phrase.index}/3</div>
                <div className="truncate text-xs font-bold">
                  {isDone ? '✓ Enrolled' : isCurrent ? '● Active' : 'Waiting'}
                </div>
              </div>
            );
          })}
        </div>

        {/* Prompt Card */}
        {step !== 'COMPLETE' && (
          <div className="p-5 rounded-2xl bg-[#07090e] border border-[#1f293d] space-y-3 text-center">
            <div className="flex items-center justify-center gap-1.5 text-purple-400 text-xs font-mono font-bold uppercase tracking-wider">
              <Volume2 className="w-3.5 h-3.5" />
              <span>Prompt Phrase {currentPhrase.index} of 3</span>
            </div>
            <div className="text-xl sm:text-2xl font-black text-white font-mono tracking-wide py-1">
              "{currentPhrase.text}"
            </div>
            <p className="text-[11px] text-gray-400 font-mono">
              Pronunciation: <span className="text-gray-300 italic">/{currentPhrase.phonetic}/</span>
            </p>

            {/* Spectrum Visualizer */}
            <div className="flex items-end justify-center gap-1.5 h-12 pt-2 px-4">
              {audioFrequencies.map((freq, i) => (
                <div
                  key={i}
                  className={`w-2 rounded-full transition-all duration-75 ${
                    step === 'RECORDING'
                      ? 'bg-gradient-to-t from-purple-500 to-indigo-400 shadow-sm shadow-purple-500/50'
                      : 'bg-[#1f293d]'
                  }`}
                  style={{ height: step === 'RECORDING' ? `${Math.max(15, freq)}%` : '15%' }}
                />
              ))}
            </div>

            <p className="text-[11px] text-gray-500 font-mono">
              {step === 'RECORDING'
                ? `Recording live speech... (${recordingSeconds}s)`
                : step === 'PROCESSING'
                ? 'Processing ECAPA-TDNN 192-d speaker embedding...'
                : 'Click [Start Recording] and read the phrase clearly.'}
            </p>
          </div>
        )}

        {/* Action Controls */}
        {step !== 'COMPLETE' ? (
          <div className="space-y-3">
            {step === 'RECORDING' ? (
              <button
                onClick={handleStopRecording}
                className="w-full py-4 rounded-xl bg-gradient-to-r from-rose-600 to-red-600 hover:from-rose-500 hover:to-red-500 text-white font-bold text-xs uppercase tracking-wider font-mono shadow-lg shadow-rose-500/25 transition animate-pulse flex items-center justify-center gap-2 cursor-pointer"
              >
                <Square className="w-4 h-4 fill-white" />
                <span>Stop Recording ({recordingSeconds}s)</span>
              </button>
            ) : (
              <button
                onClick={handleStartRecording}
                disabled={step === 'PROCESSING'}
                className="w-full py-4 rounded-xl bg-gradient-to-r from-purple-600 via-indigo-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white font-bold text-xs uppercase tracking-wider font-mono shadow-lg shadow-purple-500/25 disabled:opacity-50 transition flex items-center justify-center gap-2 cursor-pointer"
              >
                {step === 'PROCESSING' ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Extracting 192-d Voice Embedding...</span>
                  </>
                ) : (
                  <>
                    <Mic className="w-4 h-4" />
                    <span>Start Recording Sample {currentSampleIndex}/3</span>
                  </>
                )}
              </button>
            )}
          </div>
        ) : (
          /* Completion State */
          <div className="space-y-4">
            <div className="p-5 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 space-y-2 text-center">
              <div className="inline-flex p-2.5 rounded-full bg-emerald-500/20 mb-1">
                <CheckCircle2 className="w-7 h-7 text-emerald-400" />
              </div>
              <h2 className="text-base font-bold text-white">Voice Profile Enrolled Successfully</h2>
              <p className="text-xs text-gray-300 leading-relaxed max-w-sm mx-auto">
                3 distinct vocal samples captured. 192-dimensional ECAPA-TDNN centroid template generated and securely encrypted with AES-256-GCM.
              </p>
            </div>

            <button
              onClick={() => {
                setOnboardingStep(null);
                navigate('/dashboard', { replace: true });
              }}
              className="w-full py-4 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-xs uppercase tracking-wider font-mono shadow-lg shadow-emerald-500/25 transition cursor-pointer flex items-center justify-center gap-2"
            >
              <span>Continue to Dashboard →</span>
            </button>
          </div>
        )}

        {/* Security Badge */}
        <div className="pt-2 border-t border-[#1f293d] flex items-center justify-center gap-2 text-[10px] text-gray-500 font-mono">
          <ShieldCheck className="w-3.5 h-3.5 text-purple-400" />
          <span>SpeechBrain ECAPA-TDNN 192-d Centroid • AES-256-GCM Encrypted</span>
        </div>

      </div>
    </div>
  );
};

export default VoiceEnrollmentPage;
