import { useState, useRef, useCallback, useEffect } from 'react';

// Helper function to encode raw float32 PCM audio samples into a 16-bit 16kHz RIFF/WAVE ArrayBuffer
export function encodeWavPcm16(float32Samples, inputSampleRate = 16000, targetSampleRate = 16000) {
  let pcmSamples = float32Samples;
  if (inputSampleRate !== targetSampleRate) {
    const ratio = inputSampleRate / targetSampleRate;
    const newLength = Math.floor(float32Samples.length / ratio);
    pcmSamples = new Float32Array(newLength);
    if (ratio > 1) {
      // The microphone commonly falls back to 48 kHz.  Linear decimation
      // aliases frequencies above the 16 kHz Nyquist limit into PAD features;
      // apply a windowed-sinc low-pass filter before decimation instead.
      const radius = 32;
      const cutoff = 0.475 / ratio;
      for (let i = 0; i < newLength; i++) {
        const srcPos = i * ratio;
        const center = Math.floor(srcPos);
        let weightedSum = 0;
        let weightSum = 0;
        for (let k = center - radius; k <= center + radius; k++) {
          if (k < 0 || k >= float32Samples.length) continue;
          const distance = k - srcPos;
          const sinc = Math.abs(distance) < 1e-8
            ? (2 * cutoff)
            : Math.sin(2 * Math.PI * cutoff * distance) / (Math.PI * distance);
          const window = 0.5 + (0.5 * Math.cos(Math.PI * distance / radius));
          const weight = sinc * window;
          weightedSum += float32Samples[k] * weight;
          weightSum += weight;
        }
        pcmSamples[i] = weightSum ? weightedSum / weightSum : 0;
      }
    } else {
      for (let i = 0; i < newLength; i++) {
        const srcPos = i * ratio;
        const idx = Math.floor(srcPos);
        const frac = srcPos - idx;
        const s0 = float32Samples[idx] || 0;
        const s1 = (idx + 1 < float32Samples.length) ? float32Samples[idx + 1] : s0;
        pcmSamples[i] = s0 + frac * (s1 - s0);
      }
    }
  }

  const numChannels = 1;
  const bytesPerSample = 2; // 16-bit
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = targetSampleRate * blockAlign;
  const dataSize = pcmSamples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF Chunk Descriptor
  view.setUint8(0, 0x52); view.setUint8(1, 0x49); view.setUint8(2, 0x46); view.setUint8(3, 0x46); // "RIFF"
  view.setUint32(4, 36 + dataSize, true);
  view.setUint8(8, 0x57); view.setUint8(9, 0x41); view.setUint8(10, 0x56); view.setUint8(11, 0x45); // "WAVE"

  // fmt Subchunk
  view.setUint8(12, 0x66); view.setUint8(13, 0x6d); view.setUint8(14, 0x74); view.setUint8(15, 0x20); // "fmt "
  view.setUint32(16, 16, true);       // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true);        // AudioFormat (1 = PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, targetSampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);       // BitsPerSample (16 bits)

  // data Subchunk
  view.setUint8(36, 0x64); view.setUint8(37, 0x61); view.setUint8(38, 0x74); view.setUint8(39, 0x61); // "data"
  view.setUint32(40, dataSize, true);

  // Write PCM 16-bit integer audio samples
  let offset = 44;
  for (let i = 0; i < pcmSamples.length; i++) {
    const s = Math.max(-1, Math.min(1, pcmSamples[i]));
    const val = s < 0 ? s * 0x8000 : s * 0x7FFF;
    view.setInt16(offset, val, true);
    offset += 2;
  }

  return buffer;
}

// Join microphone PCM blocks captured by Web Audio. This avoids relying on a
// browser-specific MediaRecorder container decoder (WebM/MP4/Opus).
export function mergePcmChunks(chunks) {
  const totalLength = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const samples = new Float32Array(totalLength);
  let offset = 0;
  chunks.forEach(chunk => {
    samples.set(chunk, offset);
    offset += chunk.length;
  });
  return samples;
}

export const useAudioRecorder = () => {
  const [status, setStatus] = useState('IDLE'); // 'IDLE' | 'REQUESTING_PERMISSION' | 'READY' | 'RECORDING' | 'PROCESSING' | 'SUCCESS' | 'FAILED'
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioBase64, setAudioBase64] = useState(null);
  const [micError, setMicError] = useState(null);
  const [recordingTimeSec, setRecordingTimeSec] = useState(0);
  const [audioFrequencies, setAudioFrequencies] = useState([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const pcmChunksRef = useRef([]);
  const timerIntervalRef = useRef(null);
  const recordingStartedAtRef = useRef(null);
  const animFrameRef = useRef(null);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);

  const startRecording = useCallback(async () => {
    console.log('[NIVORA MIC] Requesting microphone stream...');
    setMicError(null);
    setAudioBlob(null);
    setAudioBase64(null);
    setRecordingTimeSec(0);
    audioChunksRef.current = [];
    pcmChunksRef.current = [];
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    setStatus('REQUESTING_PERMISSION');

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Microphone API not supported in this browser environment.");
      }

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
      console.log('[NIVORA MIC] Microphone stream acquired.');
      setStatus('READY');

      mediaRecorderRef.current = new MediaRecorder(stream);

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      // Set up AudioContext & AnalyserNode for REAL amplitude measurement
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

      // Capture PCM directly while recording. Browser MediaRecorder blobs are
      // not guaranteed to be decodable by AudioContext on every platform.
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
          const freqs = Array.from(dataArray.slice(0, 13)).map(val => Math.max(15, Math.min(100, Math.round((val / 255) * 100))));
          setAudioFrequencies(freqs);
          animFrameRef.current = requestAnimationFrame(updateFrequencies);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        console.log('[NIVORA MIC] Recording stopped. Processing WAV buffer...');
        setStatus('PROCESSING');
        recordingStartedAtRef.current = null;
        if (animFrameRef.current) {
          cancelAnimationFrame(animFrameRef.current);
          animFrameRef.current = null;
        }

        try {
          const pcmSamples = mergePcmChunks(pcmChunksRef.current);
          if (!pcmSamples.length) throw new Error('No microphone samples were captured.');
          const wavArrayBuffer = encodeWavPcm16(pcmSamples, audioCtx.sampleRate, 16000);
          const wavBlob = new Blob([wavArrayBuffer], { type: 'audio/wav' });
          console.info('[NIVORA MIC] Capture diagnostics:', {
            browserSampleRateHz: audioCtx.sampleRate,
            exportedSampleRateHz: 16000,
            exportedFormat: '16-bit PCM WAV',
            inputSamples: pcmSamples.length,
            exportedSamples: (wavArrayBuffer.byteLength - 44) / 2,
            durationSec: Number((pcmSamples.length / audioCtx.sampleRate).toFixed(3)),
            antiAliasFilterApplied: audioCtx.sampleRate > 16000,
          });
          
          setAudioBlob(wavBlob);

          const reader = new FileReader();
          reader.onloadend = () => {
            setAudioBase64(reader.result);
            setStatus('SUCCESS');
            console.log('[NIVORA MIC] Audio encoding complete.');
          };
          reader.readAsDataURL(wavBlob);
        } catch (err) {
          console.warn("[NIVORA MIC] PCM WAV encoding failed:", err);
          setMicError('Unable to capture microphone audio. Please check your microphone and try again.');
          setStatus('FAILED');
        } finally {
          stream.getTracks().forEach(track => track.stop());
          if (audioCtx.state !== 'closed') {
            audioCtx.close();
          }
        }
      };

      mediaRecorderRef.current.start(100);
      setIsRecording(true);
      setStatus('RECORDING');
      updateFrequencies();

      // Start recording timer
      recordingStartedAtRef.current = performance.now();
      timerIntervalRef.current = setInterval(() => {
        const startedAt = recordingStartedAtRef.current;
        if (startedAt !== null) {
          setRecordingTimeSec(Math.floor((performance.now() - startedAt) / 1000));
        }
      }, 250);

    } catch (err) {
      console.warn("[NIVORA MIC] Microphone access error:", err);
      let msg = "Microphone access denied or unavailable.";
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = "Microphone permission was denied by user.";
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = "No microphone device detected on this system.";
      } else if (err.message) {
        msg = err.message;
      }

      setMicError(msg);
      setStatus('FAILED');
      setIsRecording(false);
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    recordingStartedAtRef.current = null;
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, []);

  const getAudioBase64 = useCallback(() => {
    return new Promise((resolve) => {
      if (audioBase64) {
        resolve(audioBase64);
        return;
      }
      if (!audioBlob) {
        resolve(null);
        return;
      }
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.readAsDataURL(audioBlob);
    });
  }, [audioBase64, audioBlob]);

  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
      recordingStartedAtRef.current = null;
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, []);

  return {
    status,
    isRecording,
    audioBlob,
    audioBase64,
    micError,
    recordingTimeSec,
    audioFrequencies,
    startRecording,
    stopRecording,
    getAudioBase64
  };
};
