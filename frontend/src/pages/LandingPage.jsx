import React from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { 
  ShieldAlert, 
  ArrowRight, 
  Cpu, 
  Lock, 
  FileCheck, 
  Layers, 
  Activity,
  ShieldCheck,
  Shield
} from 'lucide-react';

export function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#07090e] text-gray-200 font-sans selection:bg-blue-600 selection:text-white relative overflow-hidden">
      {/* Landing Header Navigation */}
      <header className="sticky top-0 z-50 bg-[#07090e]/80 backdrop-blur-md border-b border-[#1f293d]">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-bold text-xl tracking-tight bg-gradient-to-r from-white via-gray-200 to-blue-400 bg-clip-text text-transparent font-mono">NIVORA</span>
              <span className="text-[10px] ml-2 px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 font-mono border border-blue-500/30 font-semibold">ENTERPRISE</span>
            </div>
          </div>

          <nav className="hidden md:flex items-center gap-8 text-xs font-semibold text-gray-400">
            <a href="#how-it-works" className="hover:text-white transition-colors">How It Works</a>
            <Link to="/security" className="hover:text-white transition-colors">Security Architecture</Link>
            <Link to="/history" className="hover:text-white transition-colors">Audit History</Link>
          </nav>

          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/login')}
              className="px-4 py-2 rounded-xl bg-[#0c1017] hover:bg-gray-800/80 border border-[#1f293d] text-gray-300 hover:text-white font-semibold text-xs transition-all cursor-pointer"
            >
              Login
            </button>
            <button
              onClick={() => navigate('/auth')}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold text-xs transition-all shadow-lg shadow-blue-500/25 flex items-center gap-2 cursor-pointer"
            >
              <span>Start Authentication</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-6 pt-24 pb-16 grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
        <div className="lg:col-span-7 space-y-8">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-mono font-semibold">
            <ShieldCheck className="w-4 h-4 text-blue-400" />
            <span>NIVORA MULTIMODAL SECURITY ENGINE ACTIVE</span>
          </div>

          <div className="space-y-4">
            <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white font-mono leading-tight">
              NIVORA <br />
              <span className="bg-gradient-to-r from-blue-400 via-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                ENTERPRISE IDENTITY & AUTHENTICATION
              </span>
            </h1>
            <p className="text-xl font-medium text-gray-300 leading-snug">
              "Verify real-time biometric authenticity across Face, Voice, and Deepfake threats."
            </p>
          </div>

          <p className="text-base text-gray-400 leading-relaxed max-w-2xl">
            Nivora orchestrates multi-modal security across DeepQShield deepfake analysis, Face Identity verification, Face Liveness anti-spoofing, and Voice Authentication. Powered by post-quantum cryptographic signatures (Dilithium-5, SHA3-256), Nivora ensures zero-trust media authenticity.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <button
              onClick={() => navigate('/auth')}
              className="px-7 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-base shadow-lg shadow-blue-500/25 transition-all flex items-center gap-3 cursor-pointer"
            >
              <span>Start Authentication</span>
              <ArrowRight className="w-5 h-5" />
            </button>

            <button
              onClick={() => navigate('/security')}
              className="px-7 py-3.5 rounded-xl bg-[#0c1017] hover:bg-gray-800/80 border border-[#1f293d] text-gray-200 font-semibold text-base transition-all shadow-md flex items-center gap-2 cursor-pointer"
            >
              <span>View Security Architecture</span>
            </button>
          </div>
        </div>

        {/* Abstract Digital Security Visual */}
        <div className="lg:col-span-5 flex justify-center">
          <div className="relative w-full max-w-md aspect-square rounded-3xl bg-gradient-to-br from-slate-800/90 via-[#0c1017] to-[#080b12] border border-blue-400/20 p-8 flex flex-col justify-between shadow-[0_28px_80px_-32px_rgba(37,99,235,0.72)] overflow-hidden glow-box">
            <div className="absolute inset-0 opacity-30 pointer-events-none" style={{ backgroundImage: 'linear-gradient(rgba(96,165,250,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(96,165,250,0.12) 1px, transparent 1px)', backgroundSize: '28px 28px' }} />
            <div className="relative z-10 flex justify-between items-center border-b border-[#1f293d] pb-4">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping" />
                <span className="text-xs font-mono font-bold text-white">ENGINE STATUS</span>
              </div>
              <span className="text-xs font-mono text-emerald-400 font-bold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/30">MODULES 1–5 ACTIVE</span>
            </div>

            {/* Neural Diagram Box */}
            <div className="relative z-10 my-auto py-4 space-y-3">
              <div className="p-3.5 rounded-xl bg-[#07090e] border border-[#1f293d] flex items-center justify-between text-xs font-mono">
                <span className="text-gray-400">Module 1 Face Identity</span>
                <span className="text-blue-400 font-bold">512-d InceptionResnetV1</span>
              </div>
              <div className="p-3.5 rounded-xl bg-[#07090e] border border-[#1f293d] flex items-center justify-between text-xs font-mono">
                <span className="text-gray-400">Module 2 Face Liveness Detection</span>
                <span className="text-white font-bold">Passive RGB + Micro-motion</span>
              </div>
              <div className="p-3.5 rounded-xl bg-[#07090e] border border-[#1f293d] flex items-center justify-between text-xs font-mono">
                <span className="text-gray-400">Module 3 Voice Identity</span>
                <span className="text-blue-400 font-bold">ECAPA-TDNN + Acoustic PAD</span>
              </div>
              <div className="p-3.5 rounded-xl bg-[#07090e] border border-[#1f293d] flex items-center justify-between text-xs font-mono">
                <span className="text-gray-400">Module 4 Deepfake Detection</span>
                <span className="text-white font-bold">DeepQShield (ResNeXt50)</span>
              </div>
            </div>

            <div className="relative z-10 flex items-center justify-between text-[11px] font-mono text-gray-400 pt-3 border-t border-[#1f293d]">
              <span>SECURITY LEVEL</span>
              <span className="text-emerald-400 font-bold">POST-QUANTUM DILITHIUM-5</span>
            </div>
          </div>
        </div>
      </section>

      <section className="max-w-7xl mx-auto px-6 pb-20">
        <div className="grid grid-cols-1 sm:grid-cols-3 rounded-2xl overflow-hidden border border-slate-700/60 bg-[#0a0e16]/75 shadow-[0_20px_50px_-32px_rgba(0,0,0,0.85)]">
          {[
            ['05', 'Security Gates', 'Face, liveness, voice, visual analysis, fusion'],
            ['LIVE', 'Threat Defense', 'Real-time presentation attack and deepfake detection'],
            ['PQC', 'Audit Integrity', 'Dilithium-5 signed verification evidence'],
          ].map(([value, label, description], index) => (
            <div key={label} className={`p-5 sm:p-6 ${index < 2 ? 'sm:border-r border-slate-700/60' : ''} ${index > 0 ? 'border-t sm:border-t-0 border-slate-700/60' : ''}`}>
              <div className="text-xl font-extrabold font-mono tracking-wide text-blue-300">{value}</div>
              <div className="mt-1 text-sm font-bold text-white">{label}</div>
              <div className="mt-1.5 text-[11px] leading-relaxed text-slate-400">{description}</div>
            </div>
          ))}
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 border-t border-slate-700/60 bg-[#080b12]/65">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center max-w-2xl mx-auto mb-16">
            <h2 className="text-xs font-mono text-blue-400 tracking-widest uppercase mb-3 font-bold">WORKFLOW PIPELINE</h2>
            <p className="text-3xl font-extrabold text-white font-mono">HOW IT WORKS</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 relative">
            {[
              {
                step: "01",
                title: "Live Media Input",
                desc: "Webcam video frames and microphone PCM audio captured in browser memory.",
                icon: FileCheck
              },
              {
                step: "02",
                title: "Biometric Feature Extraction",
                desc: "Feature extraction across Face Identity, Liveness, and Voice models.",
                icon: Cpu
              },
              {
                step: "03",
                title: "Final Verification Engine",
                desc: "Module 5 Final Verification orchestrates composite risk evaluation and threat overrides.",
                icon: Activity
              },
              {
                step: "04",
                title: "PQC Verification",
                desc: "SHA3-256 digest with Dilithium-5 digital signature audit logging.",
                icon: Lock
              }
            ].map((st, i) => {
              const Icon = st.icon;
              return (
                <div key={i} className="p-6 rounded-2xl bg-gradient-to-br from-slate-800/60 to-[#0c1017] border border-slate-700/70 relative group hover:border-blue-400/40 hover:-translate-y-1 transition-all shadow-[0_18px_38px_-24px_rgba(0,0,0,0.9)]">
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-2xl font-extrabold font-mono text-blue-400">{st.step}</span>
                    <div className="w-10 h-10 rounded-xl bg-[#07090e] border border-[#1f293d] flex items-center justify-center text-blue-400 shadow-inner">
                      <Icon className="w-5 h-5 text-blue-400" />
                    </div>
                  </div>
                  <h3 className="text-base font-bold text-white mb-2 font-mono">{st.title}</h3>
                  <p className="text-xs text-gray-400 leading-relaxed">{st.desc}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 border-t border-[#1f293d] text-center text-xs font-mono text-gray-500 bg-[#07090e]">
        Nivora Enterprise Multimodal Authentication Platform — Deepfake & Biometric Security
      </footer>
    </div>
  );
}

export default LandingPage;
