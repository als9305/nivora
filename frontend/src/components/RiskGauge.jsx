import React from 'react';

export const RiskGauge = ({ score = 0, level = "LOW", decision = "ALLOW" }) => {
  const clampedScore = Math.max(0, Math.min(100, score));
  const rotationAngle = (clampedScore / 100) * 180 - 90;

  const getColor = () => {
    if (clampedScore <= 35) return { text: 'text-emerald-400', stroke: '#10B981', badge: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' };
    if (clampedScore <= 70) return { text: 'text-amber-400', stroke: '#F59E0B', badge: 'bg-amber-500/10 text-amber-400 border-amber-500/30' };
    return { text: 'text-rose-400', stroke: '#EF4444', badge: 'bg-rose-500/10 text-rose-400 border-rose-500/30' };
  };

  const theme = getColor();

  return (
    <div className="flex flex-col items-center justify-center p-6 rounded-2xl bg-[#0c1017] border border-[#1f293d] shadow-2xl space-y-3">
      <span className="text-[11px] uppercase font-mono tracking-wider text-gray-400 font-bold">
        Composite Risk Score
      </span>

      {/* SVG Semi-Circle Gauge */}
      <div className="relative w-44 h-26 flex items-end justify-center overflow-hidden">
        <svg viewBox="0 0 200 110" className="w-full h-full">
          {/* Background Track */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="#1f293d"
            strokeWidth="18"
            strokeLinecap="round"
          />
          {/* Active Gradient Track */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke={theme.stroke}
            strokeWidth="18"
            strokeLinecap="round"
            strokeDasharray="251.2"
            strokeDashoffset={251.2 - (clampedScore / 100) * 251.2}
            className="transition-all duration-1000 ease-out"
          />
        </svg>

        {/* Needle Indicator */}
        <div
          className="absolute bottom-2 w-1.5 h-18 bg-white origin-bottom rounded-full transition-transform duration-1000 ease-out shadow-md"
          style={{ transform: `rotate(${rotationAngle}deg)` }}
        >
          <div className="w-3 h-3 rounded-full bg-blue-500 absolute -top-1 -left-0.75 shadow-lg"></div>
        </div>

        {/* Pivot Point */}
        <div className="absolute bottom-0 w-5 h-5 rounded-full bg-[#07090e] border-2 border-gray-600"></div>
      </div>

      {/* Numerical Display & Badge */}
      <div className="text-center space-y-1.5">
        <div className="flex items-baseline justify-center space-x-1">
          <span className={`text-3xl font-extrabold font-mono tracking-tight ${theme.text}`}>
            {clampedScore}
          </span>
          <span className="text-xs font-mono text-gray-500">/ 100</span>
        </div>
        
        <div className="flex items-center justify-center space-x-2">
          <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold font-mono border ${theme.badge}`}>
            {level} RISK
          </span>
          <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold font-mono border ${theme.badge}`}>
            {decision}
          </span>
        </div>
      </div>
    </div>
  );
};
