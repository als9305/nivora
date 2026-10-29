import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Shield, Activity, History, Settings, Lock } from 'lucide-react';

export const Navbar = () => {
  const location = useLocation();

  const navItems = [
    { label: 'Dashboard', path: '/dashboard', icon: Activity },
    { label: 'Start Authentication', path: '/auth', icon: Shield },
    { label: 'History', path: '/history', icon: History },
    { label: 'Security', path: '/security', icon: Lock },
    { label: 'Settings', path: '/settings', icon: Settings },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#07090e]/80 backdrop-blur-md border-b border-[#1f293d] shadow-2xl">
      <div className="max-w-[1500px] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* LEFT: BRANDING */}
          <Link to="/dashboard" className="flex items-center space-x-3 group">
            <div className="relative flex items-center justify-center w-9 h-9 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/20 group-hover:scale-105 transition-transform">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="text-lg font-bold tracking-tight bg-gradient-to-r from-white via-gray-200 to-blue-400 bg-clip-text text-transparent font-mono">
                NIVORA
              </span>
              <span className="block text-[9px] uppercase tracking-wider font-mono text-gray-400 font-semibold">
                ENTERPRISE MULTIMODAL SECURITY
              </span>
            </div>
          </Link>

          {/* CENTER: NAVIGATION LINKS */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-blue-600/15 text-blue-400 border border-blue-500/40 font-semibold shadow-md'
                      : 'text-gray-400 hover:text-white hover:bg-gray-800/80'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-blue-400' : 'text-gray-400'}`} />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* RIGHT: SYSTEM STATUS */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-mono font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>Risk Engine Operational</span>
            </div>
          </div>

        </div>
      </div>
    </header>
  );
};
