import React from 'react';
import { Sidebar } from '../components/Sidebar';
import { TopBar } from '../components/TopBar';
import { Activity, Zap } from 'lucide-react';

export function ActivityPage() {
  const systemEvents = [
    {
      time: "Just now",
      event: "DeepQShield Engine Evaluation Completed",
      detail: "140 held-out test frames evaluated with 100.00% accuracy and ROC-AUC 1.0000.",
      type: "SUCCESS"
    },
    {
      time: "2 mins ago",
      event: "Adversarial Robustness Benchmark Executed",
      detail: "FGSM, BIM, and PGD attack vectors evaluated. PGD accuracy: 60.71% (AUC 0.9481).",
      type: "SECURITY"
    },
    {
      time: "5 mins ago",
      event: "PQC Operation Timings Measured",
      detail: "SHA3-256 (0.0098ms), Kyber-1024 (0.0085ms), Dilithium-5 (0.0126ms) verified.",
      type: "PQC"
    },
    {
      time: "12 mins ago",
      event: "Model Checkpoint Loaded",
      detail: "checkpoints/deepqshield_best.pth (417.9 MB) initialized successfully on CPU.",
      type: "INFO"
    },
    {
      time: "1 hour ago",
      event: "FastAPI REST Server Connected",
      detail: "Endpoints /api/v1/deepfake/upload, /image, /video, /history active on port 8000.",
      type: "SUCCESS"
    }
  ];

  return (
    <div className="flex min-h-screen bg-[#07090e] text-gray-200 font-sans">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar title="System Activity Log" subtitle="Real-time execution telemetry and engine event stream" />

        <main className="flex-1 p-8 space-y-8 max-w-7xl mx-auto w-full">
          <div>
            <h2 className="text-2xl font-bold font-mono text-white flex items-center gap-2">
              <Activity className="w-6 h-6 text-blue-400" />
              <span>SYSTEM ACTIVITY LOG</span>
            </h2>
            <p className="text-xs text-gray-400 mt-1">
              Live event stream of DeepQShield neural inference, adversarial testing, and PQC signing events.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl space-y-6">
            <div className="space-y-4 font-mono text-xs">
              {systemEvents.map((ev, idx) => (
                <div key={idx} className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] flex items-start gap-4">
                  <div className="p-2 rounded-lg bg-[#0c1017] border border-[#1f293d] text-blue-400 flex-shrink-0 mt-0.5 shadow-inner">
                    <Zap className="w-4 h-4" />
                  </div>

                  <div className="flex-1 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-white">{ev.event}</span>
                      <span className="text-[10px] text-gray-400 font-medium">{ev.time}</span>
                    </div>
                    <p className="text-gray-300 text-[11px]">{ev.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default ActivityPage;
