import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldCheck, Camera, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { useWebcam } from '../hooks/useWebcam';
import { enrollFaceApi, getFaceTemplateStatus } from '../services/api';
import { useAuth } from '../auth/AuthContext';

export const FaceEnrollmentPage = () => {
  const navigate = useNavigate();
  const { user, setOnboardingStep } = useAuth();
  const webcam = useWebcam();

  const [step, setStep] = useState('IDLE'); // 'IDLE' | 'DETECTING' | 'CAPTURING' | 'CAPTURED' | 'SECURING' | 'COMPLETE' | 'FAILED'
  const [sampleCount, setSampleCount] = useState(0);
  const totalSamples = 4;
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [capturedFrames, setCapturedFrames] = useState([]);

  useEffect(() => {
    webcam.startCamera();
    return () => {
      webcam.stopCamera();
    };
  }, []);

  const handleCapture = async () => {
    setError(null);
    setLoading(true);
    setStep('DETECTING');

    try {
      await webcam.waitForFrame?.();
      // Step 1: Detect face & check camera readiness
      await new Promise(res => setTimeout(res, 800));
      setStep('CAPTURING');

      // Step 2: Capture N valid live frames
      const capturedFrames = [];
      for (let i = 1; i <= totalSamples; i++) {
        setSampleCount(i);
        const frame = webcam.captureFrame();
        if (frame) {
          capturedFrames.push(frame);
        }
        await new Promise(res => setTimeout(res, 300));
      }

      if (capturedFrames.length === 0) {
        throw new Error("Unable to capture clear face frames from live camera.");
      }

      setCapturedFrames(capturedFrames);
      setStep('CAPTURED');
    } catch (err) {
      console.error("Enrollment error:", err);
      setError(err?.response?.data?.detail?.message || err?.message || "Face enrollment failed. Ensure camera is working and face is clearly visible.");
      setStep('FAILED');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setError(null);
    if (!user?.user_id || capturedFrames.length === 0) {
      setError('Capture live face samples before saving enrollment.');
      return;
    }
    setLoading(true);
    setStep('SECURING');
    try {
      const res = await enrollFaceApi(user.user_id, capturedFrames);
      if (!res?.success || !res?.template_id) throw new Error(res?.message || 'Face template could not be saved.');
      const persisted = await getFaceTemplateStatus(user.user_id);
      if (!persisted?.enrolled || persisted?.template?.template_id !== res.template_id || !persisted?.template?.encrypted) {
        throw new Error('Face template storage could not be confirmed. Please capture fresh samples and try again.');
      }
      setStep('COMPLETE');
    } catch (err) {
      setError(err?.response?.data?.detail?.message || err?.message || 'Face enrollment failed. Please capture fresh live samples and try again.');
      setStep('FAILED');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#07090e] text-gray-200 flex items-center justify-center p-4 font-sans selection:bg-blue-600 selection:text-white relative overflow-hidden">
      {/* Background ambient lighting */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl pointer-events-none" />

      <div className="w-full max-w-lg bg-[#0c1017] rounded-2xl border border-[#1f293d] shadow-2xl p-8 space-y-6 text-center relative z-10 glow-box">
        
        {/* Brand Header */}
        <div className="space-y-2">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 mx-auto flex items-center justify-center shadow-lg shadow-blue-500/30">
            <ShieldCheck className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-2xl font-extrabold tracking-tight bg-gradient-to-r from-white via-gray-200 to-blue-400 bg-clip-text text-transparent font-mono">NIVORA</h1>
          <p className="text-xs text-gray-400 font-medium">Secure Your Account</p>
        </div>

        <div className="space-y-1">
          <h2 className="text-lg font-bold text-white">Set Up Identity Verification</h2>
          <p className="text-xs text-gray-400">Look at the camera to register your encrypted facial template.</p>
        </div>

        {error && (
          <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center space-x-2 font-medium">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Live Webcam Viewport - Zero Upload Controls */}
        <div className="relative aspect-video rounded-xl bg-[#07090e] overflow-hidden border border-[#1f293d] shadow-inner flex items-center justify-center">
          <video
            ref={webcam.videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover transform -scale-x-100"
          />
          <canvas ref={webcam.canvasRef} className="hidden" />

          {/* Camera Status Overlay */}
          {!webcam.isCameraActive && (
            <div className="absolute inset-0 bg-[#07090e]/90 flex flex-col items-center justify-center text-white space-y-2 p-4">
              <Camera className="w-8 h-8 text-gray-400 animate-pulse" />
              <p className="text-xs font-mono text-gray-300">Initializing Live Webcam...</p>
            </div>
          )}
        </div>

        {/* Setup Progress Indicator */}
        <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-3">
          <div className="text-xs font-mono font-bold text-gray-400 uppercase tracking-wider">
            SETUP PROGRESS
          </div>

          <div className="grid grid-cols-4 gap-2 text-[11px] font-mono">
            <div className={`p-2 rounded-lg border transition-all ${
              step === 'DETECTING' ? 'bg-blue-500/10 border-blue-500/30 text-blue-400 font-bold' :
              step === 'CAPTURING' || step === 'CAPTURED' || step === 'SECURING' || step === 'COMPLETE' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 font-semibold' :
              'bg-[#0c1017] border-[#1f293d] text-gray-500'
            }`}>
              1. Detect Face
            </div>

            <div className={`p-2 rounded-lg border transition-all ${
              step === 'CAPTURING' ? 'bg-blue-500/10 border-blue-500/30 text-blue-400 font-bold' :
              step === 'CAPTURED' || step === 'SECURING' || step === 'COMPLETE' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 font-semibold' :
              'bg-[#0c1017] border-[#1f293d] text-gray-500'
            }`}>
              2. Samples ({sampleCount}/{totalSamples})
            </div>

            <div className={`p-2 rounded-lg border transition-all ${
              step === 'SECURING' ? 'bg-blue-500/10 border-blue-500/30 text-blue-400 font-bold' :
              step === 'COMPLETE' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 font-semibold' :
              'bg-[#0c1017] border-[#1f293d] text-gray-500'
            }`}>
              3. Secure Template
            </div>

            <div className={`p-2 rounded-lg border transition-all ${
              step === 'COMPLETE' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 font-bold' :
              'bg-[#0c1017] border-[#1f293d] text-gray-500'
            }`}>
              4. Complete
            </div>
          </div>
        </div>

        {/* Action Button */}
        {step !== 'COMPLETE' && step !== 'CAPTURED' ? (
          <button
            onClick={handleCapture}
            disabled={loading || !webcam.isCameraActive}
            className="w-full py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-xs shadow-lg shadow-blue-500/25 disabled:opacity-50 transition flex items-center justify-center gap-2 cursor-pointer"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Processing Biometric Setup...</span>
              </>
            ) : (
              <span>Start Identity Setup</span>
            )}
          </button>
        ) : step === 'CAPTURED' ? (
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/30 text-blue-200 text-xs font-semibold">
              Face captured successfully. Review complete — save the encrypted biometric template to continue.
            </div>
            <button onClick={handleSave} disabled={loading} className="w-full py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold text-xs disabled:opacity-50">
              {loading ? 'Saving Face Enrollment...' : 'SAVE FACE ENROLLMENT'}
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold flex items-center justify-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <span>Face enrollment saved successfully. Your biometric identity has been securely enrolled.</span>
            </div>

            <button
              onClick={() => {
                setOnboardingStep('VOICE');
                navigate('/enroll-voice');
              }}
              className="w-full py-3.5 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-bold text-xs shadow-lg shadow-purple-500/25 transition cursor-pointer flex items-center justify-center gap-2"
            >
              <span>Continue to Voice Setup →</span>
            </button>
          </div>
        )}

      </div>
    </div>
  );
};

export default FaceEnrollmentPage;
