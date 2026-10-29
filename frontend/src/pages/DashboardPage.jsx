import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Sidebar } from '../components/Sidebar';
import { TopBar } from '../components/TopBar';
import { 
  ShieldCheck, 
  ArrowRight, 
  CheckCircle2, 
  XCircle, 
  History,
  Lock,
  Cpu,
  Layers,
  Sparkles
} from 'lucide-react';
import { getAuthHistory } from '../services/api';
import { useAuth } from '../auth/AuthContext';

export function DashboardPage() {
  const navigate = useNavigate();
  const { user: currentUser } = useAuth();
  const [historyList, setHistoryList] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await getAuthHistory();
      if (Array.isArray(res)) {
        setHistoryList(res);
      } else if (res?.history && Array.isArray(res.history)) {
        setHistoryList(res.history);
      }
    } catch (e) {
      console.warn("Could not fetch history:", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#07090e] text-gray-200 font-sans overflow-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <TopBar title="Dashboard" subtitle="Enterprise Multi-Modal Identity & Authentication Platform" />

        <main className="flex-1 p-8 space-y-8 max-w-6xl mx-auto w-full">
          
          {/* Welcome Header */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-extrabold text-white tracking-tight">
                Welcome back, {currentUser?.name}
              </h1>
              <p className="text-xs text-gray-400 font-medium mt-0.5">
                Nivora Multimodal Biometric Authentication Engine active.
              </p>
            </div>

            <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono font-semibold">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>System Operational</span>
            </div>
          </div>

          {/* Primary Action Card */}
          <div className="p-8 rounded-3xl bg-gradient-to-br from-slate-800/85 via-[#0c1017] to-[#090c13] border border-blue-400/20 glow-box space-y-6 relative overflow-hidden shadow-[0_28px_65px_-36px_rgba(37,99,235,0.7)]">
            <div className="absolute inset-0 opacity-25 pointer-events-none" style={{ backgroundImage: 'radial-gradient(circle at 85% 18%, rgba(96,165,250,0.55) 0, transparent 20%), linear-gradient(rgba(96,165,250,0.07) 1px, transparent 1px), linear-gradient(90deg, rgba(96,165,250,0.07) 1px, transparent 1px)', backgroundSize: 'auto, 32px 32px, 32px 32px' }} />
            <div className="max-w-xl space-y-2 relative z-10">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-mono font-semibold">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>NIVORA AUTHENTICATION ENGINE</span>
              </div>
              <h2 className="text-2xl font-extrabold tracking-tight text-white">Ready to verify your identity?</h2>
              <p className="text-sm text-gray-300 leading-relaxed font-medium">
                Perform single-page multimodal verification across Face Identity, Face Liveness Detection, Voice Identity, DeepQShield Deepfake Detection, and Final Verification.
              </p>
            </div>

            <button
              onClick={() => navigate('/auth')}
              className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-extrabold text-xs shadow-lg shadow-blue-500/25 transition-all flex items-center gap-2 cursor-pointer relative z-10"
            >
              <span>Start Authentication</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            {/* Ambient Background Glow */}
            <div className="absolute -right-20 -bottom-20 w-64 h-64 rounded-full bg-blue-600/10 blur-3xl pointer-events-none" />
          </div>

          {/* Security Features Overview */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 font-sans">
            <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-800/70 to-[#0c1017] border border-slate-700/70 shadow-[0_18px_38px_-28px_rgba(0,0,0,0.9)] space-y-3 hover:-translate-y-1 hover:border-blue-400/30 transition-all">
              <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-blue-400 font-bold">
                01
              </div>
              <h3 className="text-sm font-bold text-white">Biometric Protection</h3>
              <p className="text-xs text-gray-400 leading-relaxed">
                512-d facial embedding centroid matching with AES-256-GCM encrypted template storage.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-800/70 to-[#0c1017] border border-slate-700/70 shadow-[0_18px_38px_-28px_rgba(0,0,0,0.9)] space-y-3 hover:-translate-y-1 hover:border-blue-400/30 transition-all">
              <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-blue-400 font-bold">
                02
              </div>
              <h3 className="text-sm font-bold text-white">DeepQShield & Voice PAD</h3>
              <p className="text-xs text-gray-400 leading-relaxed">
                Spatial-frequency deepfake inspection and ECAPA-TDNN acoustic anti-spoofing.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-800/70 to-[#0c1017] border border-slate-700/70 shadow-[0_18px_38px_-28px_rgba(0,0,0,0.9)] space-y-3 hover:-translate-y-1 hover:border-blue-400/30 transition-all">
              <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-blue-400 font-bold">
                03
              </div>
              <h3 className="text-sm font-bold text-white">Final Verification</h3>
              <p className="text-xs text-gray-400 leading-relaxed">
                Server-authoritative Module 5 Final Verification security decision with Dilithium-5 PQC digital signatures.
              </p>
            </div>
          </div>

          {/* Recent Authentication Activity */}
          <div className="bg-gradient-to-br from-slate-800/65 to-[#0c1017] rounded-2xl border border-slate-700/70 p-6 space-y-4 shadow-[0_18px_38px_-28px_rgba(0,0,0,0.9)]">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <History className="w-4 h-4 text-blue-400" />
                <span>Recent Authentication Attempts</span>
              </h3>
              <button
                onClick={() => navigate('/history')}
                className="text-xs font-bold text-blue-400 hover:text-blue-300 hover:underline"
              >
                View All History →
              </button>
            </div>

            {loading ? (
              <div className="p-6 text-center text-xs text-gray-400 font-mono">Loading authentication records...</div>
            ) : historyList.length === 0 ? (
              <div className="p-8 text-center bg-[#07090e] rounded-xl border border-[#1f293d] text-xs text-gray-400">
                No recent authentication attempts recorded yet.
              </div>
            ) : (
              <div className="divide-y divide-gray-800/80">
                {historyList.slice(0, 5).map((log, idx) => (
                  <div key={idx} className="py-3 flex items-center justify-between text-xs">
                    <div className="flex items-center gap-3">
                      {log.decision === 'AUTHENTICATED' || log.decision === 'ALLOW' || log.decision === 'PASS' ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                      ) : (
                        <XCircle className="w-4 h-4 text-rose-400 shrink-0" />
                      )}
                      <div>
                        <p className="font-semibold text-gray-200">
                          {log.decision === 'AUTHENTICATED' || log.decision === 'ALLOW' || log.decision === 'PASS' 
                            ? 'Authentication Successful' 
                            : 'Authentication Failed'}
                        </p>
                        <p className="text-[10px] text-gray-400 font-mono">{log.timestamp || new Date().toLocaleString()}</p>
                      </div>
                    </div>

                    <span className={`px-2.5 py-1 rounded-full font-mono text-[10px] font-bold border ${
                      log.decision === 'AUTHENTICATED' || log.decision === 'ALLOW' || log.decision === 'PASS'
                        ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                        : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                    }`}>
                      {log.decision || 'PROCESSED'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

        </main>
      </div>
    </div>
  );
}

export default DashboardPage;
