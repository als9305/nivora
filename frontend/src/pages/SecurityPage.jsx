import React, { useState } from 'react';
import { Sidebar } from '../components/Sidebar';
import { TopBar } from '../components/TopBar';
import { 
  Lock, 
  ShieldCheck, 
  Key, 
  FileText, 
  Cpu, 
  ChevronDown, 
  ChevronUp, 
  CheckCircle2, 
  Layers, 
  Sparkles 
} from 'lucide-react';

export function SecurityPage() {
  const [showTechnicalSpecs, setShowTechnicalSpecs] = useState(false);

  return (
    <div className="flex h-screen bg-[#07090e] text-gray-200 font-sans overflow-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <TopBar title="Security Architecture" subtitle="Enterprise Protection & Post-Quantum Cryptography" />

        <main className="flex-1 p-8 space-y-8 max-w-6xl mx-auto w-full">
          
          {/* Header */}
          <div>
            <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
              <Lock className="w-7 h-7 text-blue-400" />
              <span>Security & Cryptographic Architecture</span>
            </h1>
            <p className="text-xs text-gray-400 font-medium mt-1">
              High-level overview of multi-modal security layers protecting Nivora identity verification.
            </p>
          </div>

          {/* Core Security Protection Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Identity Protection */}
            <div className="bg-[#0c1017] rounded-2xl border border-[#1f293d] p-6 space-y-3 shadow-2xl">
              <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-blue-400">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white">Identity Protection</h3>
              <p className="text-xs text-gray-400 leading-relaxed">
                512-dimensional InceptionResnetV1 facial embedding centroid comparison with L2-normalization.
              </p>
              <div className="pt-2 flex items-center gap-1.5 text-xs font-semibold text-emerald-400 font-mono">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>AES-256-GCM AEAD Encrypted Biometric Templates</span>
              </div>
            </div>

            {/* Liveness Protection */}
            <div className="bg-[#0c1017] rounded-2xl border border-[#1f293d] p-6 space-y-3 shadow-2xl">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                <Cpu className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white">Face Liveness Detection</h3>
              <p className="text-xs text-gray-400 leading-relaxed">
                Passive RGB facial micro-motion tracking and challenge-response movement verification.
              </p>
              <div className="pt-2 flex items-center gap-1.5 text-xs font-semibold text-emerald-400 font-mono">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>Presentation Attack Detection (PAD) Active</span>
              </div>
            </div>

            {/* Voice Protection */}
            <div className="bg-[#0c1017] rounded-2xl border border-[#1f293d] p-6 space-y-3 shadow-2xl">
              <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
                <Layers className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white">Voice Identity & Audio PAD</h3>
              <p className="text-xs text-gray-400 leading-relaxed">
                ECAPA-TDNN acoustic speaker embedding verification paired with LCNN audio presentation attack defense.
              </p>
              <div className="pt-2 flex items-center gap-1.5 text-xs font-semibold text-emerald-400 font-mono">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>300s Single-Use Anti-Replay Session Challenge</span>
              </div>
            </div>

            {/* DeepQShield Deepfake Detection */}
            <div className="bg-[#0c1017] rounded-2xl border border-[#1f293d] p-6 space-y-3 shadow-2xl">
              <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center text-purple-400">
                <Sparkles className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white">DeepQShield Deepfake Detection</h3>
              <p className="text-xs text-gray-400 leading-relaxed">
                PyTorch ResNeXt50 spatial-frequency analysis detecting moiré artifacts and deepfake synthesis.
              </p>
              <div className="pt-2 flex items-center gap-1.5 text-xs font-semibold text-emerald-400 font-mono">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>Real-Time Live Camera Frame Inspection</span>
              </div>
            </div>

          </div>

          {/* Cryptographic Security Summary */}
          <div className="bg-[#0c1017] rounded-2xl border border-[#1f293d] p-6 space-y-4 shadow-2xl">
            <h3 className="text-sm font-bold font-mono text-white uppercase tracking-wider flex items-center gap-2">
              <Key className="w-4 h-4 text-blue-400" />
              <span>Post-Quantum Cryptographic Verification</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
              <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-1">
                <span className="text-gray-500 text-[10px]">CONTENT DIGEST</span>
                <p className="font-bold text-white">SHA3-256 Hashing</p>
                <p className="text-[11px] text-gray-400">FIPS 202 canonical hash digest</p>
              </div>

              <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-1">
                <span className="text-gray-500 text-[10px]">KEY ENCAPSULATION</span>
                <p className="font-bold text-indigo-400">Kyber-1024 KEM</p>
                <p className="text-[11px] text-gray-400">NIST ML-KEM-1024 lattice encryption</p>
              </div>

              <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-1">
                <span className="text-gray-500 text-[10px]">DIGITAL SIGNATURE</span>
                <p className="font-bold text-emerald-400">Dilithium-5 Signatures</p>
                <p className="text-[11px] text-gray-400">NIST ML-DSA-87 quantum signatures</p>
              </div>
            </div>
          </div>

          {/* COLLAPSIBLE ACCORDION: ACADEMIC & TECHNICAL SPECIFICATIONS */}
          <div className="bg-[#0c1017] rounded-2xl border border-[#1f293d] overflow-hidden shadow-2xl">
            <button
              onClick={() => setShowTechnicalSpecs(!showTechnicalSpecs)}
              className="w-full p-6 flex items-center justify-between hover:bg-gray-800/40 transition cursor-pointer text-left"
            >
              <div>
                <h3 className="text-sm font-bold font-mono text-white uppercase tracking-wider flex items-center gap-2">
                  <FileText className="w-4 h-4 text-blue-400" />
                  <span>Academic & Technical Specifications</span>
                </h3>
                <p className="text-xs text-gray-400 font-medium mt-0.5">
                  Detailed research benchmarks, model dimensions, and quantitative metrics for academic review.
                </p>
              </div>
              {showTechnicalSpecs ? (
                <ChevronUp className="w-5 h-5 text-gray-400 shrink-0" />
              ) : (
                <ChevronDown className="w-5 h-5 text-gray-400 shrink-0" />
              )}
            </button>

            {showTechnicalSpecs && (
              <div className="p-6 pt-0 border-t border-[#1f293d] space-y-6 font-mono text-xs">
                
                {/* Model Architecture Specs */}
                <div className="space-y-3 pt-4">
                  <h4 className="text-xs font-bold text-gray-300 uppercase tracking-wider">Model Architectures & Tensor Specs</h4>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-left">
                    <div className="p-3.5 rounded-xl bg-[#07090e] border border-[#1f293d]">
                      <span className="text-[10px] text-gray-500 block">DeepQShield</span>
                      <p className="font-bold text-white">ResNeXt50_32x4d</p>
                      <span className="text-[10px] text-gray-400">224 x 224 input</span>
                    </div>

                    <div className="p-3.5 rounded-xl bg-[#07090e] border border-[#1f293d]">
                      <span className="text-[10px] text-gray-500 block">Face Identity</span>
                      <p className="font-bold text-white">InceptionResnetV1</p>
                      <span className="text-[10px] text-gray-400">512-d L2 vector</span>
                    </div>

                    <div className="p-3.5 rounded-xl bg-[#07090e] border border-[#1f293d]">
                      <span className="text-[10px] text-gray-500 block">Voice Identity</span>
                      <p className="font-bold text-white">ECAPA-TDNN</p>
                      <span className="text-[10px] text-gray-400">192-d speaker vector</span>
                    </div>

                    <div className="p-3.5 rounded-xl bg-[#07090e] border border-[#1f293d]">
                      <span className="text-[10px] text-gray-500 block">Audio PAD</span>
                      <p className="font-bold text-white">LCNN Architecture</p>
                      <span className="text-[10px] text-gray-400">Soft acoustic correction</span>
                    </div>
                  </div>
                </div>

                {/* Quantitative Benchmark Results */}
                <div className="space-y-3">
                  <h4 className="text-xs font-bold text-gray-300 uppercase tracking-wider">Voice Identity & Final Verification Evaluated Benchmarks</h4>
                  
                  <div className="overflow-x-auto">
                    <table className="w-full text-left font-mono text-xs">
                      <thead className="bg-[#07090e] text-gray-400 border-b border-[#1f293d] uppercase text-[10px]">
                        <tr>
                          <th className="py-3 px-4">Evaluation Metric</th>
                          <th className="py-3 px-4">Target Benchmark</th>
                          <th className="py-3 px-4">Measured Result</th>
                          <th className="py-3 px-4">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-800/80">
                        <tr>
                          <td className="py-3 px-4 font-bold text-white">Audio PAD ROC-AUC</td>
                          <td className="py-3 px-4 text-gray-400">&gt;= 0.80</td>
                          <td className="py-3 px-4 font-bold text-blue-400">0.8883</td>
                          <td className="py-3 px-4 font-bold text-emerald-400">PASSED</td>
                        </tr>
                        <tr>
                          <td className="py-3 px-4 font-bold text-white">Audio PAD ACER</td>
                          <td className="py-3 px-4 text-gray-400">&lt;= 20%</td>
                          <td className="py-3 px-4 font-bold text-blue-400">13.92%</td>
                          <td className="py-3 px-4 font-bold text-emerald-400">PASSED</td>
                        </tr>
                        <tr>
                          <td className="py-3 px-4 font-bold text-white">Audio PAD BPCER</td>
                          <td className="py-3 px-4 text-gray-400">&lt;= 20%</td>
                          <td className="py-3 px-4 font-bold text-blue-400">11.76%</td>
                          <td className="py-3 px-4 font-bold text-emerald-400">PASSED</td>
                        </tr>
                        <tr>
                          <td className="py-3 px-4 font-bold text-white">Audio PAD APCER</td>
                          <td className="py-3 px-4 text-gray-400">&lt;= 20%</td>
                          <td className="py-3 px-4 font-bold text-blue-400">16.07%</td>
                          <td className="py-3 px-4 font-bold text-emerald-400">PASSED</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

              </div>
            )}
          </div>

        </main>
      </div>
    </div>
  );
}

export default SecurityPage;
