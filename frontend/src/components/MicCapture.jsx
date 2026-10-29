import React from 'react';
import { Mic, Square, Volume2, CheckCircle2, AlertCircle } from 'lucide-react';

export const MicCapture = ({
  isRecording,
  audioBlob,
  micError,
  recordingTimeSec = 0,
  audioFrequencies = [15, 20, 15, 30, 25, 40, 20, 35, 15, 25, 15, 20, 15],
  startRecording,
  stopRecording,
  challengePhrase = "Apple river green"
}) => {
  const audioPreviewUrl = audioBlob ? URL.createObjectURL(audioBlob) : null;

  const formatTimer = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="w-full max-w-md mx-auto p-5 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl space-y-4">
      {/* Challenge Phrase Prompt Banner */}
      <div className="p-4 rounded-xl bg-[#07090e] border border-blue-500/30 text-center space-y-1">
        <span className="text-[10px] uppercase font-mono tracking-wider text-blue-400 font-bold block">
          Live Challenge Phrase (Speak into Mic)
        </span>
        <p className="text-xl font-bold font-mono text-white tracking-wide">
          "{challengePhrase}"
        </p>
      </div>

      {/* Microphone Recording Visualizer Container */}
      <div className="relative flex flex-col items-center justify-center p-5 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-3 text-center text-white">
        {isRecording ? (
          <div className="space-y-2 flex flex-col items-center">
            {/* Real Audio Visualizer Spectrum Bars */}
            <div className="flex items-center space-x-1.5 h-10">
              {audioFrequencies.map((freq, i) => (
                <div
                  key={i}
                  style={{ height: `${freq}%` }}
                  className="w-1.5 bg-gradient-to-t from-blue-500 to-cyan-400 rounded-full transition-all duration-75"
                />
              ))}
            </div>
            {/* Recording Live Indicator & Timer */}
            <div className="flex items-center space-x-2 text-rose-400 font-mono text-xs font-bold">
              <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping" />
              <span>RECORDING {formatTimer(recordingTimeSec)}</span>
            </div>
          </div>
        ) : audioBlob ? (
          <div className="space-y-3 w-full flex flex-col items-center">
            <div className="flex items-center space-x-2 text-emerald-400 font-semibold text-xs">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Microphone audio recorded cleanly</span>
            </div>
            {/* Embedded Audio Preview */}
            {audioPreviewUrl && (
              <audio controls src={audioPreviewUrl} className="h-8 w-full max-w-xs rounded-lg shadow-sm" />
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center space-y-1 text-gray-400 text-xs">
            <Volume2 className="w-6 h-6 text-blue-400 mb-1" />
            <span className="text-white font-semibold text-xs">Live Device Microphone Only</span>
            <span className="text-gray-400">Click Start Recording and speak the challenge phrase.</span>
          </div>
        )}

        {micError && (
          <div className="flex items-center space-x-1.5 text-rose-400 text-xs font-medium">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{micError}</span>
          </div>
        )}
      </div>

      {/* Control Buttons (Strictly Microphone Capture Only) */}
      <div className="flex justify-center gap-2">
        {!isRecording ? (
          <button
            onClick={startRecording}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold text-xs shadow-lg shadow-blue-500/25 transition cursor-pointer"
          >
            <Mic className="w-4 h-4" />
            <span>{audioBlob ? "Re-record Microphone" : "Start Recording"}</span>
          </button>
        ) : (
          <button
            onClick={stopRecording}
            className="flex items-center space-x-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-rose-600 to-red-600 hover:from-rose-500 hover:to-red-500 text-white font-semibold text-xs shadow-lg shadow-rose-500/25 transition animate-pulse cursor-pointer"
          >
            <Square className="w-4 h-4 fill-white" />
            <span>Stop Recording</span>
          </button>
        )}
      </div>
    </div>
  );
};
