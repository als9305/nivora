import React, { useState } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import { ShieldCheck, Eye, EyeOff, AlertCircle } from 'lucide-react';
import { loginUser } from '../services/api';
import { useAuth } from '../auth/AuthContext';

export const LoginPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { signIn } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (!email.trim() || !password) {
      setError('Email and password are required.');
      return;
    }

    setLoading(true);

    try {
      const res = await loginUser({ email: email.trim(), password });
      if (!res?.success || !res?.user_id) {
        throw new Error('Login could not be completed.');
      }
      signIn(res);
      const destination = location.state?.from?.pathname || '/dashboard';
      navigate(destination, { replace: true });
    } catch (err) {
      console.error("Login error:", err);
      const payload = err?.response?.data;
      const detail = payload?.detail;
      const msg = typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? 'Email and password are required.'
          : detail?.message || payload?.message ||
          (err?.request && !err?.response
            ? 'Unable to reach the authentication server. Please try again later.'
            : err?.message || 'Invalid email or password.');
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#07090e] text-gray-200 flex items-center justify-center p-4 font-sans selection:bg-blue-600 selection:text-white relative overflow-hidden">
      {/* Background ambient lighting */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl pointer-events-none" />

      <div className="w-full max-w-md bg-[#0c1017] rounded-2xl border border-[#1f293d] shadow-2xl p-8 space-y-6 relative z-10 glow-box">

        {/* Brand Header */}
        <div className="text-center space-y-2">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 mx-auto flex items-center justify-center shadow-lg shadow-blue-500/30">
            <ShieldCheck className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-2xl font-extrabold tracking-tight bg-gradient-to-r from-white via-gray-200 to-blue-400 bg-clip-text text-transparent font-mono">NIVORA</h1>
          <p className="text-xs text-gray-400 font-medium">Enterprise Multimodal Authentication Platform</p>
        </div>

        <div className="space-y-1">
          <h2 className="text-lg font-bold text-white">Welcome Back</h2>
          <p className="text-xs text-gray-400">Sign in to access secure authentication and dashboard.</p>
        </div>

        {error && (
          <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center space-x-2 font-medium">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1">
            <label className="text-xs font-mono font-bold uppercase text-gray-400">Email Address</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="user@example.com"
              className="w-full px-4 py-2.5 rounded-xl bg-[#07090e] border border-[#1f293d] text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 text-sm font-sans transition"
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-mono font-bold uppercase text-gray-400">Password</label>
            <div className="relative">
              <input
              type={showPassword ? "text" : "password"}
              value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full px-4 py-2.5 pr-10 rounded-xl bg-[#07090e] border border-[#1f293d] text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 text-sm font-sans transition"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-2.5 text-gray-400 hover:text-white"
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-xs shadow-lg shadow-blue-500/25 disabled:opacity-50 transition cursor-pointer"
          >
            {loading ? "Authenticating..." : "Login"}
          </button>
        </form>

        <div className="pt-4 border-t border-[#1f293d] text-center text-xs text-gray-400">
          <span>Don't have an account? </span>
          <Link to="/register" className="text-blue-400 font-bold hover:underline">
            Register
          </Link>
        </div>

      </div>
    </div>
  );
};

export default LoginPage;
