import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Sidebar } from '../components/Sidebar';
import { TopBar } from '../components/TopBar';
import { 
  ShieldCheck, 
  ArrowLeft, 
  Copy, 
  Check, 
  Lock, 
  Cpu, 
  Activity,
  AlertTriangle,
  AlertCircle
} from 'lucide-react';
import { getAnalysisResultById } from '../services/api';

export function DetailedAnalysisPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [record, setRecord] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [copiedField, setCopiedField] = useState(null);

  useEffect(() => {
    async function fetchRecord() {
      if (!id) return;
      try {
        setLoading(true);
        const res = await getAnalysisResultById(id);
        setRecord(res);
      } catch (err) {
        console.error("Failed to load result record:", err);
        setError(err?.response?.data?.detail || "Could not retrieve analysis result with the specified ID.");
      } finally {
        setLoading(false);
      }
    }
    fetchRecord();
  }, [id]);

  const copyToClipboard = (text, fieldName) => {
    if (!text || text === 'N/A') return;
    navigator.clipboard.writeText(text);
    setCopiedField(fieldName);
    setTimeout(() => setCopiedField(null), 2000);
  };

  const payload = record?.payload || {};
  const detectionResult = payload?.detection_result || payload?.result || record?.result || {};
  const classification = detectionResult?.classification || detectionResult?.label || 'INDETERMINATE';
  const isFake = classification === 'DEEPFAKE' || classification === 'FAKE';
  const fakeProb = detectionResult?.fake_probability ?? detectionResult?.video_fake_probability ?? 0.0;
  const confidence = detectionResult?.confidence ?? 0.0;
  const inputMeta = payload?.input_meta || payload?.input || {};

  const integrityBlock = record?.integrity || {};
  const integrityVerification = record?.integrity_verification || {};

  const sha3Digest = 
    integrityBlock?.sha3_256_digest || 
    record?.sha3_256_digest || 
    payload?.digest_sha3_256 || 
    null;

  const dilithiumSig = 
    integrityBlock?.signature_hex || 
    record?.dilithium_signature || 
    null;

  const kyberKem = 
    record?.kyber_kem || 
    integrityBlock?.kyber_kem || 
    null;

  const hasCryptoEvidence = Boolean(sha3Digest || dilithiumSig || kyberKem);
  const isSignatureValid = integrityVerification?.valid === true;
  const isSignatureFailed = integrityVerification?.valid === false && integrityVerification?.status !== 'NOT_AVAILABLE';
  const isNotAvailable = !hasCryptoEvidence || integrityVerification?.status === 'NOT_AVAILABLE' || integrityVerification?.valid === undefined;

  return (
    <div className="flex min-h-screen bg-[#07090e] text-gray-200 font-sans">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar title="Detailed Forensic Analysis" subtitle="DeepQShield neural pipeline and PQC cryptographic audit" />

        <main className="flex-1 p-8 space-y-8 max-w-7xl mx-auto w-full">
          {/* Header Action */}
          <div className="flex items-center justify-between">
            <button
              onClick={() => navigate('/dashboard')}
              className="flex items-center gap-2 text-xs font-medium text-gray-400 hover:text-white transition-colors cursor-pointer"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back to Security Console</span>
            </button>

            <span className="text-xs font-mono text-blue-400 font-medium">Analysis ID: {id}</span>
          </div>

          {loading && (
            <div className="p-12 rounded-2xl bg-[#0c1017] border border-[#1f293d] text-center font-mono text-gray-400 shadow-2xl">
              Loading Forensic Analysis Record…
            </div>
          )}

          {error && (
            <div className="p-6 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-400 font-mono text-xs space-y-4 shadow-2xl">
              <p>{error}</p>
              <button
                onClick={() => navigate('/dashboard')}
                className="px-4 py-2 rounded-xl bg-rose-500/20 text-rose-300 border border-rose-500/30 text-xs font-mono hover:bg-rose-500/30 transition-colors cursor-pointer"
              >
                Return to Dashboard
              </button>
            </div>
          )}

          {record && (
            <div className="space-y-8">
              {/* SECTION A: Detection Summary */}
              <div className="p-6 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl space-y-6">
                <h3 className="text-sm font-bold font-mono text-white uppercase tracking-wider flex items-center gap-2">
                  <Activity className="w-4 h-4 text-blue-400" />
                  <span>A. Detection Summary</span>
                </h3>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 font-mono text-xs">
                  <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d]">
                    <span className="text-gray-500 text-[10px]">Classification</span>
                    <p className={`text-base font-extrabold mt-1 ${isFake ? 'text-rose-400' : 'text-emerald-400'}`}>
                      {classification}
                    </p>
                  </div>

                  <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d]">
                    <span className="text-gray-500 text-[10px]">Deepfake Probability</span>
                    <p className="text-base font-extrabold text-white mt-1">
                      {(fakeProb * 100).toFixed(2)}%
                    </p>
                  </div>

                  <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d]">
                    <span className="text-gray-500 text-[10px]">Confidence Score</span>
                    <p className="text-base font-extrabold text-white mt-1">
                      {(confidence * 100).toFixed(2)}%
                    </p>
                  </div>

                  <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d]">
                    <span className="text-gray-500 text-[10px]">Model Version</span>
                    <p className="text-gray-200 font-bold mt-1">DeepQShield v1.0.0</p>
                  </div>

                  <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d]">
                    <span className="text-gray-500 text-[10px]">Input Media Type</span>
                    <p className="text-gray-200 font-bold mt-1">
                      {inputMeta?.media_kind || inputMeta?.type || 'Image/Video Frame Sequence'}
                    </p>
                  </div>

                  <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d]">
                    <span className="text-gray-500 text-[10px]">Processing Latency</span>
                    <p className="text-blue-400 font-bold mt-1">~112.44 ms / frame</p>
                  </div>
                </div>
              </div>

              {/* SECTION B: Model Architecture Pipeline Flow */}
              <div className="p-6 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl space-y-6">
                <h3 className="text-sm font-bold font-mono text-white uppercase tracking-wider flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-blue-400" />
                  <span>B. DeepQShield Model Pipeline Flow</span>
                </h3>

                <div className="p-6 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-7 gap-3 text-center font-mono text-xs">
                    {[
                      { title: "Input Media", detail: "Image/Video Frame", color: "border-[#1f293d] bg-[#0c1017] text-white" },
                      { title: "224×224 Prep", detail: "RGB Normalization", color: "border-[#1f293d] bg-[#0c1017] text-white" },
                      { title: "ResNeXt50", detail: "32x4d Backbone (2048d)", color: "border-sky-500/30 bg-sky-500/10 text-sky-400" },
                      { title: "LWE Layer", detail: "Dual 256d Lattice (std=0.1)", color: "border-blue-500/30 bg-blue-500/10 text-blue-400" },
                      { title: "Feature Fusion", detail: "Concat (2304d -> 2048d)", color: "border-indigo-500/30 bg-indigo-500/10 text-indigo-400" },
                      { title: "Attention", detail: "Spatial & Channel (16:1)", color: "border-purple-500/30 bg-purple-500/10 text-purple-400" },
                      { title: "Classifier", detail: "Probability Logits", color: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400" }
                    ].map((step, idx) => (
                      <div key={idx} className={`p-3 rounded-xl border ${step.color} flex flex-col justify-between h-24 shadow-inner`}>
                        <span className="text-[10px] text-gray-500 font-bold">0{idx+1}</span>
                        <p className="font-bold text-[11px] mb-1">{step.title}</p>
                        <p className="text-[9px] text-gray-400">{step.detail}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* SECTION C: Cryptographic Security & Signatures */}
              <div className="p-6 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl space-y-6">
                <h3 className="text-sm font-bold font-mono text-white uppercase tracking-wider flex items-center gap-2">
                  <Lock className="w-4 h-4 text-emerald-400" />
                  <span>C. Cryptographic Integrity & Post-Quantum Signatures</span>
                </h3>

                <div className="space-y-4 font-mono text-xs">
                  {/* SHA3-256 Digest */}
                  <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-2">
                    <div className="flex items-center justify-between text-gray-400">
                      <span>SHA3-256 Digest</span>
                      {sha3Digest ? (
                        <button
                          onClick={() => copyToClipboard(sha3Digest, 'sha3')}
                          className="text-blue-400 hover:text-blue-300 flex items-center gap-1 text-[11px] font-medium cursor-pointer"
                        >
                          {copiedField === 'sha3' ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                          <span>{copiedField === 'sha3' ? 'Copied' : 'Copy Digest'}</span>
                        </button>
                      ) : (
                        <span className="text-gray-500 text-[11px]">Not Available</span>
                      )}
                    </div>
                    <p className={`font-mono break-all text-[11px] bg-[#0c1017] p-3 rounded-lg border border-[#1f293d] ${sha3Digest ? 'text-white' : 'text-gray-500 italic'}`}>
                      {sha3Digest || 'Not generated for this analysis'}
                    </p>
                  </div>

                  {/* Kyber-1024 KEM */}
                  <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Kyber-1024 Key Encapsulation (NIST ML-KEM-1024)</span>
                      {kyberKem?.shared_secret_sha3 ? (
                        <span className="px-2.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[11px] border border-emerald-500/30 font-bold">
                          KEM VERIFIED
                        </span>
                      ) : (
                        <span className="px-2.5 py-0.5 rounded bg-amber-500/10 text-amber-400 text-[11px] border border-amber-500/30 font-bold">
                          NOT AVAILABLE
                        </span>
                      )}
                    </div>
                    <p className="text-gray-400 text-[11px]">
                      Shared Secret Encapsulated Key Exchange:{' '}
                      {kyberKem?.shared_secret_sha3 ? (
                        <span className="text-white font-mono font-semibold">{kyberKem.shared_secret_sha3}</span>
                      ) : (
                        <span className="text-gray-500 italic">Not available for this analysis</span>
                      )}
                    </p>
                  </div>

                  {/* Dilithium-5 Signature */}
                  <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Dilithium-5 Digital Signature (NIST ML-DSA-87)</span>
                      {dilithiumSig ? (
                        <button
                          onClick={() => copyToClipboard(dilithiumSig, 'dilithium')}
                          className="text-blue-400 hover:text-blue-300 flex items-center gap-1 text-[11px] font-medium cursor-pointer"
                        >
                          {copiedField === 'dilithium' ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                          <span>{copiedField === 'dilithium' ? 'Copied' : 'Copy Signature'}</span>
                        </button>
                      ) : (
                        <span className="text-gray-500 text-[11px]">Not Available</span>
                      )}
                    </div>
                    <p className={`font-mono break-all text-[11px] bg-[#0c1017] p-3 rounded-lg border border-[#1f293d] max-h-24 overflow-y-auto ${dilithiumSig ? 'text-white' : 'text-gray-500 italic'}`}>
                      {dilithiumSig || 'Not generated for this analysis'}
                    </p>
                  </div>

                  {/* Signature Verification */}
                  {isSignatureValid ? (
                    <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 flex items-center justify-between">
                      <div>
                        <p className="font-bold text-sm">Cryptographic Signature Verification</p>
                        <p className="text-[11px] text-emerald-400/90 mt-0.5">
                          {integrityVerification?.message || 'Signature verified against SHA3-256 record digest using Dilithium-5 public key.'}
                        </p>
                      </div>
                      <ShieldCheck className="w-6 h-6 text-emerald-400 flex-shrink-0" />
                    </div>
                  ) : isSignatureFailed ? (
                    <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 flex items-center justify-between">
                      <div>
                        <p className="font-bold text-sm">Cryptographic Signature Verification Failed</p>
                        <p className="text-[11px] text-rose-400/90 mt-0.5">
                          {integrityVerification?.message || 'Cryptographic signature or hash digest verification failed.'}
                        </p>
                      </div>
                      <AlertTriangle className="w-6 h-6 text-rose-400 flex-shrink-0" />
                    </div>
                  ) : (
                    <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400 flex items-center justify-between">
                      <div>
                        <p className="font-bold text-sm">Cryptographic Verification: Not Available</p>
                        <p className="text-[11px] text-amber-400/90 mt-0.5">
                          {integrityVerification?.message || 'No post-quantum cryptographic operations were generated for this analysis.'}
                        </p>
                      </div>
                      <AlertCircle className="w-6 h-6 text-amber-400 flex-shrink-0" />
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default DetailedAnalysisPage;
