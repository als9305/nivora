import React from 'react';
import { ShieldCheck, ShieldAlert, Cpu, Bot, CheckCircle, AlertTriangle, ArrowRight, Calculator } from 'lucide-react';

export const ExplanationCard = ({ explanation, decision = "ALLOW", riskResult = null }) => {
  if (!explanation) return null;

  const isBlocked = decision === "BLOCK";
  const isStepUp = decision === "STEP-UP OTP" || decision === "STEP-UP";

  const mathData = riskResult?.mathematical_breakdown || explanation?.mathematical_breakdown;
  const ruleTriggers = riskResult?.rule_triggers || explanation?.rule_triggers || [];
  const confidenceState = riskResult?.confidence_state || explanation?.confidence_state || (isBlocked ? "HIGH_CONFIDENCE_ATTACK" : "HIGH_CONFIDENCE_GENUINE");

  return (
    <div className="w-full p-6 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl space-y-5">
      
      {/* Header Banner */}
      <div className="flex items-center justify-between border-b border-[#1f293d] pb-4">
        <div className="flex items-center space-x-2.5">
          {isBlocked ? (
            <ShieldAlert className="w-5 h-5 text-rose-400" />
          ) : isStepUp ? (
            <AlertTriangle className="w-5 h-5 text-amber-400" />
          ) : (
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
          )}
          <div>
            <h3 className="text-xs font-extrabold uppercase tracking-wider text-white font-mono">
              WHY THIS DECISION?
            </h3>
            <span className="text-xs font-mono text-gray-400">
              Confidence State: <span className="text-blue-400 font-semibold">{confidenceState}</span>
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-xs font-mono font-semibold text-blue-400">
          {explanation.is_llm_generated ? (
            <>
              <Bot className="w-3.5 h-3.5 text-blue-400" />
              <span>AI Security Analyst</span>
            </>
          ) : (
            <>
              <Cpu className="w-3.5 h-3.5 text-emerald-400" />
              <span>Deterministic Rule Engine</span>
            </>
          )}
        </div>
      </div>

      {/* Threat Summary */}
      <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-1">
        <span className="text-[10px] uppercase font-mono text-blue-400 font-bold block">
          Threat Summary
        </span>
        <p className="text-sm font-medium text-gray-200 leading-relaxed">
          {explanation.threat_summary}
        </p>
      </div>

      {/* Weighted Contribution Table */}
      {mathData && (
        <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-3">
          <div className="flex items-center justify-between border-b border-[#1f293d] pb-2">
            <span className="text-xs font-mono text-gray-400 font-bold flex items-center space-x-1.5">
              <Calculator className="w-4 h-4 text-blue-400" />
              <span>WEIGHTED CONTRIBUTION BREAKDOWN</span>
            </span>
            <span className="text-xs font-mono font-bold text-blue-400">
              Base Score: {mathData.base_risk_score}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs font-mono">
            <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#1f293d] text-center">
              <span className="text-[10px] text-gray-500 block">Face (20%)</span>
              <span className="font-bold text-white">+{mathData.contributions?.face || 1.6}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#1f293d] text-center">
              <span className="text-[10px] text-gray-500 block">Liveness (25%)</span>
              <span className="font-bold text-white">+{mathData.contributions?.liveness || 1.25}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#1f293d] text-center">
              <span className="text-[10px] text-gray-500 block">Deepfake (30%)</span>
              <span className="font-bold text-white">+{mathData.contributions?.deepfake || 1.2}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#1f293d] text-center">
              <span className="text-[10px] text-gray-500 block">Voice (15%)</span>
              <span className="font-bold text-white">+{mathData.contributions?.voice || 0.68}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-[#0c1017] border border-[#1f293d] text-center">
              <span className="text-[10px] text-gray-500 block">Device (10%)</span>
              <span className="font-bold text-white">+{mathData.contributions?.device || 0.5}</span>
            </div>
          </div>

          {mathData.override_delta > 0 && (
            <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-mono font-bold flex justify-between items-center">
              <span>Security Threat Override Escalation:</span>
              <span>+{mathData.override_delta} Risk Points</span>
            </div>
          )}
        </div>
      )}

      {/* Triggered Rule Overrides */}
      {ruleTriggers.length > 0 && (
        <div className="space-y-2">
          <span className="text-[10px] uppercase font-mono text-rose-400 font-bold block">
            Triggered Security Threat Rules
          </span>
          {ruleTriggers.map((rule, idx) => (
            <div key={idx} className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-xs font-mono space-y-1">
              <div className="flex items-center justify-between font-bold text-rose-300">
                <span>{rule.rule_id}</span>
                <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 text-[10px] font-semibold">{rule.action}</span>
              </div>
              <p className="text-gray-300 font-sans text-xs">{rule.evidence}</p>
            </div>
          ))}
        </div>
      )}

      {/* Security Evidence Bullets */}
      <div className="space-y-2">
        <span className="text-[10px] uppercase font-mono text-gray-400 font-bold block">
          SECURITY EVIDENCE
        </span>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {explanation.evidence?.map((item, idx) => (
            <div key={idx} className="flex items-start space-x-2 p-2.5 rounded-xl bg-[#07090e] border border-[#1f293d] text-xs text-gray-300">
              <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
              <span className="font-medium text-gray-200">{item}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Recommended Action */}
      <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/30 space-y-1">
        <span className="text-[10px] uppercase font-mono text-blue-400 font-bold block">
          RECOMMENDED ACTION
        </span>
        <p className="text-xs text-white font-semibold flex items-center space-x-1.5">
          <ArrowRight className="w-4 h-4 text-blue-400 shrink-0" />
          <span>{explanation.recommendation || decision}</span>
        </p>
      </div>
    </div>
  );
};
