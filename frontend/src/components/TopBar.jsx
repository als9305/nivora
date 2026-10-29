import React from 'react';
import { ShieldCheck, User, Bell } from 'lucide-react';
import { useAuth } from '../auth/AuthContext';

export function TopBar({ title = "Authentication Console", subtitle = "Multimodal Identity Security & Deepfake Defense" }) {
  const { user } = useAuth();
  return (
    <header className="h-[76px] bg-[#0b101a]/78 backdrop-blur-xl border-b border-slate-700/50 px-8 flex items-center justify-between sticky top-0 z-20 shadow-[0_12px_30px_-24px_rgba(0,0,0,0.9)]">
      {/* Title Area */}
      <div>
        <h1 className="text-base font-semibold text-white flex items-center gap-2">
          <span className="font-bold tracking-[0.08em] bg-gradient-to-r from-white via-slate-200 to-blue-300 bg-clip-text text-transparent font-mono">NIVORA</span>
          <span className="text-slate-600">/</span>
          <span className="text-blue-300 font-mono text-sm">{title}</span>
        </h1>
        {subtitle && <p className="text-xs text-slate-400 font-mono">{subtitle}</p>}
      </div>

      {/* Status & User Area */}
      <div className="flex items-center gap-5">
        {/* System Status Indicator */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-400/25 text-xs font-mono font-medium text-emerald-300 shadow-[0_0_18px_rgba(16,185,129,0.08)]">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_9px_rgba(52,211,153,0.85)]" />
          <span>System Operational</span>
        </div>

        {/* Security Notification Bell */}
        <button className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800/80 transition-colors relative border border-transparent hover:border-slate-700">
          <Bell className="w-4 h-4" />
          <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-blue-500" />
        </button>

        {/* User Profile */}
        <div className="flex items-center gap-3 pl-4 border-l border-slate-700/60">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-slate-700 to-slate-900 border border-slate-600/70 flex items-center justify-center text-slate-300 shadow-inner">
            <User className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-left hidden sm:block font-mono">
            <p className="text-xs font-semibold text-slate-200">{user?.name}</p>
            <p className="text-[10px] text-slate-500">{user?.email}</p>
          </div>
        </div>
      </div>
    </header>
  );
}

export default TopBar;
