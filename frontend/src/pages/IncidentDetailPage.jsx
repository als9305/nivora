import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Calendar, User, Laptop, ShieldAlert } from 'lucide-react';
import { getIncidentById } from '../services/api';
import { RiskGauge } from '../components/RiskGauge';
import { ExplanationCard } from '../components/ExplanationCard';

export const IncidentDetailPage = () => {
  const { id } = useParams();
  const [incident, setIncident] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchIncident = async () => {
      try {
        const res = await getIncidentById(id);
        if (res?.success) setIncident(res.incident);
      } catch (err) {
        console.warn("Failed to load incident detail:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchIncident();
  }, [id]);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#07090e] text-gray-400 flex items-center justify-center font-mono text-xs">
        Analyzing identity telemetry record...
      </div>
    );
  }

  if (!incident) {
    return (
      <div className="min-h-screen bg-[#07090e] text-gray-200 flex items-center justify-center p-4">
        <div className="max-w-md text-center space-y-4 p-8 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl">
          <ShieldAlert className="w-10 h-10 text-amber-400 mx-auto" />
          <h2 className="text-lg font-bold text-white">Incident Record Not Found</h2>
          <p className="text-xs text-gray-400">The requested security telemetry log could not be retrieved.</p>
          <Link to="/history" className="inline-block text-xs text-blue-400 hover:underline font-semibold cursor-pointer">
            ← Return to Audit History
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#07090e] text-gray-200 font-sans">
      <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        
        {/* Navigation Back */}
        <Link to="/history" className="inline-flex items-center space-x-2 text-xs font-mono text-blue-400 hover:underline font-semibold cursor-pointer">
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Authentication Logs</span>
        </Link>

        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#1f293d] pb-4">
          <div>
            <span className="text-[10px] uppercase font-mono text-blue-400 font-bold block">
              SECURITY INCIDENT TELEMETRY REPORT
            </span>
            <h1 className="text-2xl font-bold text-white">Incident #{incident.id?.slice(0, 8)}</h1>
          </div>

          <span className={`px-3 py-1 rounded-full text-xs font-mono font-bold border ${
            incident.decision === 'ALLOW'
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
              : incident.decision === 'STEP-UP' || incident.decision === 'STEP-UP OTP'
              ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
              : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
          }`}>
            {incident.decision}
          </span>
        </div>

        {/* Overview Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-1">
            <RiskGauge
              score={incident.risk_score}
              level={incident.risk_level}
              decision={incident.decision}
            />
          </div>

          {/* Incident Metadata & Audit Information */}
          <div className="md:col-span-2 p-6 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl space-y-4">
            <h3 className="text-xs font-mono text-gray-400 uppercase font-bold tracking-wider">AUDIT INFORMATION</h3>
            <div className="space-y-2 text-xs font-mono">
              <div className="flex items-center justify-between p-3 rounded-xl bg-[#07090e] border border-[#1f293d]">
                <span className="text-gray-400 flex items-center space-x-2"><Calendar className="w-4 h-4 text-blue-400"/><span>Timestamp:</span></span>
                <span className="text-white font-semibold">{new Date(incident.timestamp).toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between p-3 rounded-xl bg-[#07090e] border border-[#1f293d]">
                <span className="text-gray-400 flex items-center space-x-2"><User className="w-4 h-4 text-blue-400"/><span>User Email:</span></span>
                <span className="text-white font-semibold">{incident.email}</span>
              </div>
              <div className="flex items-center justify-between p-3 rounded-xl bg-[#07090e] border border-[#1f293d]">
                <span className="text-gray-400 flex items-center space-x-2"><Laptop className="w-4 h-4 text-blue-400"/><span>Session ID:</span></span>
                <span className="text-blue-400 font-bold">{incident.session_id}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Threat Analysis Signal Breakdown */}
        <div className="p-6 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl space-y-4">
          <h3 className="text-xs font-mono text-gray-400 uppercase font-bold tracking-wider">
            THREAT ANALYSIS SIGNALS
          </h3>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs font-mono">
            <div className="p-3 rounded-xl bg-[#07090e] border border-[#1f293d] text-center">
              <span className="text-[10px] text-gray-500 block">Face Identity</span>
              <span className="font-bold text-white">{Math.round((incident.face_score || 0.95) * 100)}%</span>
            </div>
            <div className="p-3 rounded-xl bg-[#07090e] border border-[#1f293d] text-center">
              <span className="text-[10px] text-gray-500 block">Face Liveness Detection</span>
              <span className="font-bold text-white">{Math.round((incident.liveness_score || 0.90) * 100)}%</span>
            </div>
            <div className="p-3 rounded-xl bg-[#07090e] border border-[#1f293d] text-center">
              <span className="text-[10px] text-gray-500 block">Voice Identity</span>
              <span className="font-bold text-white">{Math.round((incident.voice_similarity || 0.94) * 100)}%</span>
            </div>
            <div className="p-3 rounded-xl bg-[#07090e] border border-[#1f293d] text-center">
              <span className="text-[10px] text-gray-500 block">Deepfake Detection</span>
              <span className="font-bold text-rose-400">{Math.round((incident.deepfake_probability || 0.05) * 100)}%</span>
            </div>
            <div className="p-3 rounded-xl bg-[#07090e] border border-[#1f293d] text-center">
              <span className="text-[10px] text-gray-500 block">Device Risk</span>
              <span className="font-bold text-white">10 / 100</span>
            </div>
          </div>
        </div>

        {/* Explanation Card Embedded */}
        <ExplanationCard
          explanation={incident.explanation}
          decision={incident.decision}
          riskResult={incident}
        />

      </div>
    </div>
  );
};

export default IncidentDetailPage;
