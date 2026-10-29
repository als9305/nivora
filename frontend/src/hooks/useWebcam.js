import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * Computes gradient energy sharpness score for a canvas frame.
 * Samples central 60% region of the frame for high performance.
 */
export const computeImageSharpness = (ctx, width, height) => {
  try {
    const startX = Math.floor(width * 0.2);
    const startY = Math.floor(height * 0.2);
    const sampleWidth = Math.floor(width * 0.6);
    const sampleHeight = Math.floor(height * 0.6);

    const imgData = ctx.getImageData(startX, startY, sampleWidth, sampleHeight);
    const data = imgData.data;

    let totalVariance = 0;
    let totalBrightness = 0;
    const pixelCount = sampleWidth * sampleHeight;

    for (let y = 0; y < sampleHeight - 1; y++) {
      for (let x = 0; x < sampleWidth - 1; x++) {
        const idx = (y * sampleWidth + x) * 4;
        const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];

        const idxRight = (y * sampleWidth + (x + 1)) * 4;
        const grayRight = 0.299 * data[idxRight] + 0.587 * data[idxRight + 1] + 0.114 * data[idxRight + 2];

        const idxDown = ((y + 1) * sampleWidth + x) * 4;
        const grayDown = 0.299 * data[idxDown] + 0.587 * data[idxDown + 1] + 0.114 * data[idxDown + 2];

        const diffX = gray - grayRight;
        const diffY = gray - grayDown;

        totalVariance += (diffX * diffX) + (diffY * diffY);
        totalBrightness += gray;
      }
    }

    const sharpness = totalVariance / pixelCount;
    const avgBrightness = totalBrightness / pixelCount;
    return { sharpness, brightness: avgBrightness };
  } catch (e) {
    console.warn('[NIVORA CAMERA] Sharpness calculation warning:', e);
    return { sharpness: 100, brightness: 120 };
  }
};

// Approximates the backend Laplacian-variance measure only for selecting a
// fresh live frame. The server remains authoritative for the quality decision.
export const computeLaplacianVariance = (ctx, width, height) => {
  try {
    const pixels = ctx.getImageData(0, 0, width, height).data;
    const grayAt = (x, y) => {
      const offset = (y * width + x) * 4;
      return (0.299 * pixels[offset]) + (0.587 * pixels[offset + 1]) + (0.114 * pixels[offset + 2]);
    };
    let count = 0;
    let sum = 0;
    let sumSquares = 0;
    // Subsampling retains focus discrimination while keeping burst capture responsive.
    for (let y = 1; y < height - 1; y += 2) {
      for (let x = 1; x < width - 1; x += 2) {
        const lap = (4 * grayAt(x, y)) - grayAt(x - 1, y) - grayAt(x + 1, y) - grayAt(x, y - 1) - grayAt(x, y + 1);
        sum += lap;
        sumSquares += lap * lap;
        count += 1;
      }
    }
    return count ? Math.max(0, (sumSquares / count) - ((sum / count) ** 2)) : 0;
  } catch (error) {
    console.warn('[NIVORA CAMERA] Laplacian quality calculation warning:', error);
    return 0;
  }
};

/**
 * Authoritative Single-Owner Webcam Hook for Nivora Authentication.
 */
export const useWebcam = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const startPromiseRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraState, setCameraState] = useState('IDLE'); // 'IDLE' | 'CAMERA_REQUESTING' | 'CAMERA_INITIALIZING' | 'CAMERA_STABILIZING' | 'CAMERA_READY' | 'CAMERA_FAILED'
  const [cameraError, setCameraError] = useState(null);
  const [cameraErrorCode, setCameraErrorCode] = useState(null);

  const stopCamera = useCallback(() => {
    console.log('[NIVORA CAMERA] Stopping camera stream...');
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setStream(null);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
    setCameraState('IDLE');
  }, []);

  // A live MediaStream can exist before the browser has decoded a drawable
  // video frame. Wait for advancing video time so callers never send a blank
  // startup frame to face detection.
  const waitForFrame = useCallback(async (timeoutMs = 4000) => {
    const video = videoRef.current;
    if (!video || !video.srcObject) throw new Error('CAMERA_STREAM_NOT_RENDERING');

    const initialTime = video.currentTime;
    const startedAt = performance.now();
    return new Promise((resolve, reject) => {
      const checkFrame = () => {
        const drawable = video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0;
        const advanced = video.currentTime > initialTime;
        if (drawable && advanced) {
          resolve();
          return;
        }
        if (performance.now() - startedAt >= timeoutMs) {
          reject(new Error('CAMERA_FRAME_NOT_READY'));
          return;
        }
        requestAnimationFrame(checkFrame);
      };
      requestAnimationFrame(checkFrame);
    });
  }, []);

  const startCamera = useCallback(() => {
    // React Strict Mode and rapid clicks can both request startup twice. Share
    // one in-flight operation so a second stream cannot replace the first.
    if (startPromiseRef.current) return startPromiseRef.current;

    const startPromise = (async () => {
    console.log('[NIVORA CAMERA] Initializing single authoritative camera stream...');
    setCameraState('CAMERA_REQUESTING');
    setCameraError(null);
    setCameraErrorCode(null);

    // Reuse existing live stream if active and rendering
    if (streamRef.current && streamRef.current.getVideoTracks().some(t => t.readyState === 'live')) {
      const video = videoRef.current;
      if (video && video.srcObject === streamRef.current && video.readyState >= 2 && video.videoWidth > 0) {
        console.log('[NIVORA CAMERA] Reusing active verified camera stream.');
        await waitForFrame();
        setIsCameraActive(true);
        setCameraState('CAMERA_READY');
        return streamRef.current;
      }
    }

    try {
      let mediaStream = null;
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false
        });
      } catch (errConstraints) {
        console.log('[NIVORA CAMERA] Retrying getUserMedia with basic video constraints...');
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });
      }

      console.log(`[NIVORA CAMERA] MediaStream acquired. Video tracks: ${mediaStream.getVideoTracks().length}`);
      streamRef.current = mediaStream;
      setStream(mediaStream);
      setCameraState('CAMERA_INITIALIZING');

      const video = videoRef.current;
      if (!video) throw new Error('CAMERA_INITIALIZATION_FAILED');

      video.srcObject = mediaStream;

      // Wait for loadedmetadata event with timeout
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('CAMERA_STREAM_NOT_RENDERING')), 5000);

        const onMetadata = () => {
          clearTimeout(timeout);
          video.removeEventListener('loadedmetadata', onMetadata);
          resolve();
        };

        if (video.readyState >= 1) {
          clearTimeout(timeout);
          resolve();
        } else {
          video.addEventListener('loadedmetadata', onMetadata);
        }
      });

      console.log(`[NIVORA CAMERA] Metadata loaded. Stream dimensions: ${video.videoWidth}x${video.videoHeight}`);
      await video.play();
      console.log('[NIVORA CAMERA] Video play() resolved.');

      setCameraState('CAMERA_STABILIZING');

      // Camera Warm-Up: Wait for 12 actual rendered animation frames where readyState >= 2 and currentTime is advancing
      let renderedFrames = 0;
      let lastTime = video.currentTime;

      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          if (renderedFrames >= 3) {
            resolve(); // Accept partial frame stabilization if stream is active
          } else {
            reject(new Error('CAMERA_STREAM_NOT_RENDERING'));
          }
        }, 3000);

        const checkFrame = () => {
          if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
            if (video.currentTime > lastTime) {
              renderedFrames++;
              lastTime = video.currentTime;
            }
          }

          if (renderedFrames >= 12) {
            clearTimeout(timeout);
            resolve();
          } else {
            requestAnimationFrame(checkFrame);
          }
        };

        requestAnimationFrame(checkFrame);
      });

      console.log(`[NIVORA CAMERA] Camera stabilization complete (${renderedFrames} rendered frames verified). Dims: ${video.videoWidth}x${video.videoHeight}`);

      if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
        setIsCameraActive(true);
        setCameraState('CAMERA_READY');
        return mediaStream;
      } else {
        throw new Error('CAMERA_STREAM_NOT_RENDERING');
      }

    } catch (err) {
      console.warn('[NIVORA CAMERA] Camera failed:', err);
      let code = 'CAMERA_INITIALIZATION_FAILED';
      let msg = 'Camera initialization failed. Please ensure your webcam is connected.';

      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        code = 'CAMERA_PERMISSION_DENIED';
        msg = 'Camera permission was denied. Camera access is required for secure authentication.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError' || err.name === 'NotReadableError') {
        code = 'CAMERA_UNAVAILABLE';
        msg = 'Camera device is unavailable or being used by another application.';
      } else if (err.message === 'CAMERA_STREAM_NOT_RENDERING') {
        code = 'CAMERA_STREAM_NOT_RENDERING';
        msg = 'Camera stream connected but video failed to render pixel data.';
      }

      setCameraErrorCode(code);
      setCameraError(msg);
      setCameraState('CAMERA_FAILED');
      setIsCameraActive(false);

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
      setStream(null);
      throw new Error(msg);
    }
    })();

    startPromiseRef.current = startPromise;
    return startPromise.finally(() => {
      if (startPromiseRef.current === startPromise) startPromiseRef.current = null;
    });
  }, [waitForFrame]);

  // Frame Capture Function for Single Frame
  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || !video.srcObject || video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      console.warn(`[NIVORA CAMERA] Frame capture failed: Invalid stream state (readyState=${video?.readyState}, dims=${video?.videoWidth}x${video?.videoHeight}).`);
      return null;
    }

    const canvas = canvasRef.current || document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
    return dataUrl;
  }, []);

  // Frame Burst Capture & Selection of Best Quality Frame
  const captureFrameBurst = useCallback(async (frameCount = 10, intervalMs = 80) => {
    await waitForFrame();
    const video = videoRef.current;
    if (!video || !video.srcObject || video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      throw new Error('FRAME_CAPTURE_FAILED');
    }

    const canvas = canvasRef.current || document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');

    const frames = [];

    for (let i = 1; i <= frameCount; i++) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
      const { sharpness, brightness } = computeImageSharpness(ctx, canvas.width, canvas.height);
      const laplacianVariance = computeLaplacianVariance(ctx, canvas.width, canvas.height);

      console.log(`[NIVORA FACE CAPTURE] Frame ${i}/${frameCount} — Dims: ${canvas.width}x${canvas.height}, ReadyState: ${video.readyState}, CurrentTime: ${video.currentTime.toFixed(2)}, Laplacian: ${laplacianVariance.toFixed(2)}, Brightness: ${brightness.toFixed(2)}`);

      frames.push({
        dataUrl,
        sharpness,
        laplacianVariance,
        brightness,
        index: i,
        width: canvas.width,
        height: canvas.height
      });

      if (i < frameCount) {
        await new Promise(res => setTimeout(res, intervalMs));
      }
    }

    // Select with a Laplacian-style measure, aligned with the server quality gate.
    const validFrames = frames.filter(f => f.brightness >= 15);
    const candidateList = validFrames.length > 0 ? validFrames : frames;

    candidateList.sort((a, b) => b.laplacianVariance - a.laplacianVariance);
    const bestFrame = candidateList[0];

    console.log(`[NIVORA FACE CAPTURE] Selected Best Probe Frame #${bestFrame.index} (Laplacian: ${bestFrame.laplacianVariance.toFixed(2)}, Brightness: ${bestFrame.brightness.toFixed(2)}, Dims: ${bestFrame.width}x${bestFrame.height}) for verification.`);

    return {
      bestFrameDataUrl: bestFrame.dataUrl,
      allFramesB64: frames.map(f => f.dataUrl),
      bestMetrics: bestFrame
    };
  }, [waitForFrame]);

  const captureMultiFrames = useCallback(async (frameCount = 6, intervalMs = 150) => {
    await waitForFrame();
    const video = videoRef.current;
    if (!video || !video.srcObject || video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      throw new Error('FRAME_CAPTURE_FAILED');
    }

    const canvas = canvasRef.current || document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');

    const frames = [];
    for (let i = 0; i < frameCount; i++) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
      frames.push(dataUrl);

      if (i < frameCount - 1) {
        await new Promise(res => setTimeout(res, intervalMs));
      }
    }
    return frames;
  }, [waitForFrame]);

  // Real Temporal Liveness Sequence Capture driven by requestAnimationFrame & advancing video.currentTime
  const captureLivenessSequence = useCallback(async (optionsOrDuration = 1800, onProgressCallback = null) => {
    await waitForFrame();
    const video = videoRef.current;
    if (!video || !video.srcObject || video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      throw new Error('FRAME_CAPTURE_FAILED');
    }

    let targetDurationMs = 1800;
    let targetFrames = 25;
    let onProgress = onProgressCallback;

    if (typeof optionsOrDuration === 'number') {
      if (optionsOrDuration >= 500) {
        targetDurationMs = optionsOrDuration;
      } else {
        targetFrames = optionsOrDuration;
      }
    } else if (typeof optionsOrDuration === 'object' && optionsOrDuration !== null) {
      targetDurationMs = optionsOrDuration.durationMs || 1800;
      targetFrames = optionsOrDuration.targetFrames || 25;
      if (optionsOrDuration.onProgress) onProgress = optionsOrDuration.onProgress;
    }

    const minIntervalMs = 55; // 55ms interval achieves 25-30 temporal frames over 1800ms

    const canvas = canvasRef.current || document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');

    const startTime = performance.now();
    console.log(`[NIVORA LIVENESS CAPTURE] Temporal capture window initialized. Target duration: ${targetDurationMs}ms, Target frames: ${targetFrames}`);

    const frames = [];
    const timestampsMs = [];
    const currentTimesSec = [];
    const sampleHashes = [];
    let lastVideoTime = -1;
    let lastSampleWallTime = -1;
    let lastSampleHash = null;

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        if (frames.length >= 6) {
          const durationMs = Math.round(performance.now() - startTime);
          console.warn(`[NIVORA LIVENESS CAPTURE] Timeout reached. Returning sequence of ${frames.length} frames.`);
          console.log(`[NIVORA LIVENESS] Frame 1 timestamp: ${timestampsMs[0] || 0} ms`);
          console.log(`[NIVORA LIVENESS] Frame N timestamp: ${timestampsMs[timestampsMs.length - 1] || durationMs} ms`);
          console.log(`[NIVORA LIVENESS] Total duration: ${durationMs} ms`);
          console.log(`[NIVORA LIVENESS] Number of unique frames: ${frames.length}`);
          console.log(`[NIVORA LIVENESS] Frames captured: ${frames.length}`);
          frames.metadata = {
            durationMs,
            timestampsMs,
            currentTimesSec,
            sampleHashes,
            framesCount: frames.length,
            uniqueFramesCount: frames.length,
            previewMirrored: true,
            coordinateConvention: 'user-view'
          };
          resolve(frames);
        } else {
          reject(new Error('LIVENESS_SEQUENCE_TIMEOUT'));
        }
      }, targetDurationMs + 4000);

      const sampleFrame = () => {
        const v = videoRef.current;
        const now = performance.now();
        const elapsed = now - startTime;

        if (v && v.readyState >= 2 && v.currentTime > lastVideoTime) {
          const timeSinceLastSample = lastSampleWallTime < 0 ? minIntervalMs : (now - lastSampleWallTime);

          if (timeSinceLastSample >= minIntervalMs) {
            ctx.drawImage(v, 0, 0, canvas.width, canvas.height);

            // Sample center 40% region for lightweight pixel hash
            const sampleW = Math.floor(canvas.width * 0.4);
            const sampleH = Math.floor(canvas.height * 0.4);
            const imgData = ctx.getImageData(Math.floor(canvas.width * 0.3), Math.floor(canvas.height * 0.3), sampleW, sampleH);
            const data = imgData.data;
            let sampleSum = 0;
            for (let k = 0; k < data.length; k += 32) {
              sampleSum += data[k];
            }

            if (sampleSum !== lastSampleHash) {
              lastVideoTime = v.currentTime;
              lastSampleWallTime = now;
              lastSampleHash = sampleSum;

              const dataUrl = canvas.toDataURL('image/jpeg', 0.88);
              frames.push(dataUrl);
              timestampsMs.push(Math.round(elapsed));
              currentTimesSec.push(Number(v.currentTime.toFixed(3)));
              sampleHashes.push(sampleSum);

              const currentTimeSec = v.currentTime.toFixed(3);
              console.log(`[NIVORA LIVENESS CAPTURE] Frame ${frames.length}/${targetFrames}: video.currentTime = ${currentTimeSec}s, elapsed = ${Math.round(elapsed)}ms, sampleHash = ${sampleSum}`);

              if (typeof onProgress === 'function') {
                const timePct = Math.min(100, Math.round((elapsed / targetDurationMs) * 100));
                const framePct = Math.min(100, Math.round((frames.length / targetFrames) * 100));
                onProgress({
                  current: frames.length,
                  total: targetFrames,
                  pct: Math.min(100, Math.max(timePct, framePct))
                });
              }
            }
          }
        }

        // Complete when target duration has elapsed AND sufficient frames are collected
        if (elapsed >= targetDurationMs && frames.length >= Math.min(18, targetFrames)) {
          clearTimeout(timeout);
          const durationMs = Math.round(performance.now() - startTime);
          console.log(`[NIVORA LIVENESS] Frame 1 timestamp: ${timestampsMs[0] || 0} ms`);
          console.log(`[NIVORA LIVENESS] Frame N timestamp: ${timestampsMs[timestampsMs.length - 1] || durationMs} ms`);
          console.log(`[NIVORA LIVENESS] Total duration: ${durationMs} ms`);
          console.log(`[NIVORA LIVENESS] Number of unique frames: ${frames.length}`);
          console.log(`[NIVORA LIVENESS] Frames captured: ${frames.length}`);
          frames.metadata = {
            durationMs,
            timestampsMs,
            currentTimesSec,
            sampleHashes,
            framesCount: frames.length,
            uniqueFramesCount: frames.length,
            previewMirrored: true,
            coordinateConvention: 'user-view'
          };
          resolve(frames);
        } else {
          requestAnimationFrame(sampleFrame);
        }
      };

      requestAnimationFrame(sampleFrame);
    });
  }, [waitForFrame]);

  // Cleanup on unmount only
  useEffect(() => {
    window.addEventListener('nivora:logout', stopCamera);
    return () => {
      window.removeEventListener('nivora:logout', stopCamera);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  return {
    videoRef,
    canvasRef,
    stream,
    isCameraActive,
    cameraState,
    cameraError,
    cameraErrorCode,
    startCamera,
    stopCamera,
    waitForFrame,
    captureFrame,
    captureFrameBurst,
    captureMultiFrames,
    captureLivenessSequence
  };
};
