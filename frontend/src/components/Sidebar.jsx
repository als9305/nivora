import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { 
  ShieldCheck, 
  LayoutDashboard, 
  History, 
  Lock, 
  Settings,
  Sparkles,
  ArrowRight, LogOut
} from 'lucide-react';
import { useAuth } from '../auth/AuthContext';

export function Sidebar() {
  const navigate = useNavigate();
  const { user, signOut } = useAuth();
  const handleLogout = () => {
    signOut();
    navigate('/login', { replace: true });
  };
  const navItems = [
    { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
    { name: 'Start Authentication', path: '/auth', icon: ShieldCheck },
    { name: 'History', path: '/history', icon: History },
    { name: 'Security', path: '/security', icon: Lock },
    { name: 'Settings', path: '/settings', icon: Settings },
  ];

  return (
    <aside className="w-[272px] bg-gradient-to-b from-[#101827]/95 via-[#0c1017]/95 to-[#080b12]/95 border-r border-slate-700/50 flex flex-col h-screen sticky top-0 z-30 select-none shadow-[18px_0_48px_-28px_rgba(0,0,0,0.9)] backdrop-blur-xl">
      {/* Brand Header */}
      <div className="h-[76px] flex items-center px-5 border-b border-slate-700/50">
        <NavLink to="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 via-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/25 group-hover:scale-105 transition-transform ring-1 ring-blue-300/20">
            <ShieldCheck className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-lg tracking-[0.08em] bg-gradient-to-r from-white via-slate-200 to-blue-300 bg-clip-text text-transparent font-mono">NIVORA</span>
              <span className="text-[9px] px-1.5 py-0.5 rounded-md bg-blue-500/10 text-blue-300 font-mono font-semibold border border-blue-400/25">SECURE</span>
            </div>
            <p className="text-[11px] text-slate-400 -mt-0.5">Enterprise Identity Console</p>
          </div>
        </NavLink>
      </div>

      {/* Navigation Links */}
      <div className="flex-1 overflow-y-auto px-3 py-5 space-y-1.5">
        <div className="px-3 text-[10px] font-bold text-slate-500 tracking-[0.16em] font-mono uppercase mb-3 flex items-center gap-2">
          <span className="h-px flex-1 bg-slate-700/60" /> NAVIGATION <span className="h-px flex-1 bg-slate-700/60" />
        </div>
        {navItems.map((item, itemIdx) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={itemIdx}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-blue-500/18 to-indigo-500/10 text-blue-300 font-semibold border border-blue-400/30 shadow-lg shadow-blue-500/10'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60 border border-transparent'
                }`
              }
            >
              <div className="flex items-center gap-3">
                <Icon className="w-4 h-4 text-blue-400" />
                <span>{item.name}</span>
              </div>
              <ArrowRight className="w-3.5 h-3.5 opacity-40 transition-opacity" />
            </NavLink>
          );
        })}
      </div>

      {/* Active Engine Footer Badge */}
      <div className="p-4 border-t border-slate-700/50 bg-[#07090e]/40">
        <div className="p-3.5 rounded-xl bg-gradient-to-br from-slate-800/70 to-[#0a0d14] border border-slate-700/60 shadow-inner">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-blue-400" />
              Nivora Core
            </span>
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_10px_rgba(52,211,153,0.8)]" />
          </div>
          <p className="text-[11px] text-slate-400 font-mono">Multimodal Engine Active</p>
        </div>
        <div className="mt-3 px-1 flex items-center justify-between gap-2">
          <div className="min-w-0">
            <p className="truncate text-[11px] font-semibold text-slate-300">{user?.name}</p>
            <p className="truncate text-[10px] text-slate-500 font-mono">{user?.email}</p>
          </div>
          <button onClick={handleLogout} className="shrink-0 inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[11px] font-semibold text-rose-300 hover:bg-rose-500/10 hover:text-rose-200 transition" aria-label="Log out">
            <LogOut className="w-3.5 h-3.5" /> Logout
          </button>
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
