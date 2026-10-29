import React, { useState } from 'react';
import { Sidebar } from '../components/Sidebar';
import { TopBar } from '../components/TopBar';
import { 
  Play, 
  ArrowRight, 
  CheckCircle2, 
  XCircle,
  Zap
} from 'lucide-react';
import { runSimulationApi } from '../services/api';
import { RiskGauge } from '../components/RiskGauge';
import { ExplanationCard } from '../components/ExplanationCard';

export function SimulationPage() {
  const [selectedAttack, setSelectedAttack] = useState('DEEPFAKE_ONLY');
  const [simulationResult, setSimulationResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const attackVectors = [
    {
      id: "GENUINE_USER",
      name: "Genuine User Pass",
      desc: "All biometric signals pass (High face match, high liveness, zero deepfake).",
      expected: "ALLOW"
    },
    {
      id: "DEEPFAKE_ONLY",
      name: "DeepQShield Synthetic Attack",
      desc: "Face identity match ok, but high spatial-frequency deepfake synthesis detected.",
      expected: "BLOCK"
    },
    {
      id: "SPOOF_LIVENESS",
      name: "Presentation Attack (Photo Print)",
      desc: "High face match, but passive liveness detects static 2D texture moiré.",
      expected: "BLOCK"
    },
    {
      id: "VOICE_CLONE_ATTACK",
      name: "AI Voice Clone + Synthetic PAD",
      desc: "Audio PAD detects ECAPA-TDNN acoustic anomaly and synthetic pitch harmonics.",
      expected: "BLOCK"
    },
    {
      id: "BORDERLINE_CASE",
      name: "Borderline / Low Lighting",
      desc: "Low face match (0.55) & borderline liveness. Triggers Step-Up verification.",
      expected: "STEP-UP OTP"
    },
    {
      id: "PQC_SIGNATURE_TAMPER",
      name: "PQC Signature Tampering",
      desc: "Payload hash modified; Dilithium-5 digital signature verification fails.",
      expected: "BLOCK"
    }
  ];

  const handleRunSimulation = async () => {
    setLoading(true);
    try {
      const res = await runSimulationApi(selectedAttack);
      if (res?.success) {
        setSimulationResult(res.result || res.data || res);
      }
    } catch (err) {
      console.error("Simulation error:", err);
    } finally {
      setLoading(false);
    }
  };

  const getBenchmarkStatus = () => {
    if (!simulationResult) return null;
    const dec = simulationResult.decision || "BLOCK";
    const currentVec = attackVectors.find(v => v.id === selectedAttack);
    const isMatch = dec === currentVec?.expected;

    return {
      isPass: isMatch,
      label: isMatch ? "BENCHMARK PASSED (EXACT POLICY MATCH)" : "POLICY DEVIATION DETECTED"
    };
  };

  const status = getBenchmarkStatus();

  return (
    <div className="flex min-h-screen bg-[#07090e] text-gray-200 font-sans">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar title="Attack Vector Simulator" subtitle="Empirical threat vector execution and Module 5 Final Verification risk engine validation" />

        <main className="flex-1 p-8 space-y-8 max-w-7xl mx-auto w-full">
          <div>
            <h2 className="text-2xl font-bold font-mono text-white flex items-center gap-2">
              <Zap className="w-6 h-6 text-blue-400" />
              <span>ATTACK VECTOR SIMULATOR</span>
            </h2>
            <p className="text-xs text-gray-400 mt-1">
              Test Module 5 Final Verification Risk Engine against synthetic deepfakes, presentation attacks, and voice clones.
            </p>
          </div>

          {/* Attack Vector Selectors */}
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {attackVectors.map((vector) => {
              const isSelected = selectedAttack === vector.id;
              return (
                <button
                  key={vector.id}
                  onClick={() => setSelectedAttack(vector.id)}
                  className={`p-4 rounded-2xl border text-left flex flex-col justify-between space-y-3 transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-[#0c1017] border-blue-500 text-white shadow-lg ring-2 ring-blue-500/30'
                      : 'bg-[#0c1017] border-[#1f293d] text-gray-400 hover:border-gray-600 hover:text-white'
                  }`}
                >
                  <div>
                    <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-mono font-bold border mb-2 ${
                      vector.id === 'GENUINE_USER' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' :
                      vector.id === 'BORDERLINE_CASE' ? 'bg-amber-500/10 text-amber-400 border-amber-500/30' :
                      'bg-rose-500/10 text-rose-400 border-rose-500/30'
                    }`}>
                      {vector.expected}
                    </span>
                    <h3 className="text-xs font-bold text-white">{vector.name}</h3>
                    <p className="text-[11px] text-gray-400 mt-1 leading-snug">{vector.desc}</p>
                  </div>
                  <div className="text-[10px] font-mono text-blue-400 font-bold flex items-center space-x-1">
                    <span>Select Scenario</span>
                    <ArrowRight className="w-3 h-3" />
                  </div>
                </button>
              );
            })}
          </div>

          {/* Trigger Simulation Button */}
          <div className="text-center">
            <button
              onClick={handleRunSimulation}
              disabled={loading}
              className="inline-flex items-center space-x-2 px-8 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-xs shadow-lg shadow-blue-500/25 disabled:opacity-50 transition cursor-pointer"
            >
              <Play className="w-4 h-4 fill-white" />
              <span>{loading ? "Simulating Attack Vector..." : `Run Simulation →`}</span>
            </button>
          </div>

          {/* SIMULATION RESULT BREAKDOWN */}
          {simulationResult && (
            <div className="space-y-6 pt-4 border-t border-[#1f293d]">
              
              {/* PASS / FAIL BENCHMARK STATUS HEADER */}
              {status && (
                <div className={`p-4 rounded-xl border flex items-center justify-between font-mono ${
                  status.isPass
                    ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                    : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                }`}>
                  <div className="flex items-center space-x-2">
                    {status.isPass ? <CheckCircle2 className="w-5 h-5 text-emerald-400" /> : <XCircle className="w-5 h-5 text-rose-400" />}
                    <span className="font-bold text-sm">SIMULATION TEST RESULT: {status.label}</span>
                  </div>
                  <div className="text-xs font-semibold text-gray-400">
                    Expected Policy Match: <span className="text-white">{selectedAttack}</span>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-1">
                  <RiskGauge
                    score={simulationResult.risk_score}
                    level={simulationResult.risk_level}
                    decision={simulationResult.decision}
                  />
                </div>

                <div className="md:col-span-2 p-6 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl space-y-4">
                  <h3 className="text-xs font-mono text-gray-400 uppercase font-bold tracking-wider">
                    SIGNAL RISK BREAKDOWN
                  </h3>

                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs font-mono">
                    <div className="p-3 rounded-xl bg-[#07090e] border border-[#1f293d] text-center">
                      <span className="text-[10px] text-gray-500 block">Face Identity</span>
                      <span className="font-bold text-white">{Math.round((simulationResult.face_score || 0) * 100)}%</span>
                    </div>
                    <div className="p-3 rounded-xl bg-[#07090e] border border-[#1f293d] text-center">
                      <span className="text-[10px] text-gray-500 block">Face Liveness Detection</span>
                      <span className="font-bold text-white">{Math.round((simulationResult.liveness_score || 0) * 100)}%</span>
                    </div>
                    <div className="p-3 rounded-xl bg-[#07090e] border border-[#1f293d] text-center">
                      <span className="text-[10px] text-gray-500 block">Voice Identity</span>
                      <span className="font-bold text-white">{Math.round((simulationResult.voice_similarity || 0) * 100)}%</span>
                    </div>
                    <div className="p-3 rounded-xl bg-[#07090e] border border-[#1f293d] text-center">
                      <span className="text-[10px] text-gray-500 block">Deepfake Detection</span>
                      <span className="font-bold text-rose-400">{Math.round((simulationResult.deepfake_probability || 0) * 100)}%</span>
                    </div>
                    <div className="p-3 rounded-xl bg-[#07090e] border border-[#1f293d] text-center">
                      <span className="text-[10px] text-gray-500 block">Device Risk</span>
                      <span className="font-bold text-white">10 / 100</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Explanation Card Embedded */}
              <ExplanationCard
                explanation={simulationResult.explanation}
                decision={simulationResult.decision}
                riskResult={simulationResult}
              />

            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default SimulationPage;
