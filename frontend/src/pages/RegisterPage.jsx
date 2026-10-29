import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { ShieldCheck, Eye, EyeOff, AlertCircle } from 'lucide-react';
import { registerUser } from '../services/api';
import { useAuth } from '../auth/AuthContext';

export const RegisterPage = () => {
  const navigate = useNavigate();
  const { signIn } = useAuth();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      const payload = {
        name: formData.name,
        email: formData.email,
        password: formData.password,
      };

      const res = await registerUser(payload);
      if (res?.success) {
        signIn(res, 'FACE');
        navigate('/enroll-face', { replace: true });
      } else {
        setError(res?.message || "Registration failed.");
      }
    } catch (err) {
      console.error("Registration error:", err);
      const msg = err?.response?.data?.detail?.message || err?.response?.data?.message || err?.message || 'Failed to create account.';
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
          <h2 className="text-lg font-bold text-white">Create Your Account</h2>
          <p className="text-xs text-gray-400">Sign up to enable secure biometric identity verification.</p>
        </div>

        {error && (
          <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center space-x-2 font-medium">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1">
            <label className="text-xs font-mono font-bold uppercase text-gray-400">Full Name</label>
            <input
              type="text"
              required
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder="Your full name"
              className="w-full px-4 py-2.5 rounded-xl bg-[#07090e] border border-[#1f293d] text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 text-sm font-sans transition"
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-mono font-bold uppercase text-gray-400">Email Address</label>
            <input
              type="email"
              required
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              placeholder="user@example.com"
              className="w-full px-4 py-2.5 rounded-xl bg-[#07090e] border border-[#1f293d] text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 text-sm font-sans transition"
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-mono font-bold uppercase text-gray-400">Password</label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                required
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
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

          <div className="space-y-1">
            <label className="text-xs font-mono font-bold uppercase text-gray-400">Confirm Password</label>
            <div className="relative">
              <input
                type={showConfirmPassword ? "text" : "password"}
                required
                value={formData.confirmPassword}
                onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                placeholder="••••••••"
                className="w-full px-4 py-2.5 pr-10 rounded-xl bg-[#07090e] border border-[#1f293d] text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 text-sm font-sans transition"
              />
              <button
                type="button"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                className="absolute right-3 top-2.5 text-gray-400 hover:text-white"
                aria-label={showConfirmPassword ? 'Hide confirm password' : 'Show confirm password'}
              >
                {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-xs shadow-lg shadow-blue-500/25 disabled:opacity-50 transition cursor-pointer"
          >
            {loading ? "Creating Account..." : "Create Account"}
          </button>
        </form>

        <div className="pt-4 border-t border-[#1f293d] text-center text-xs text-gray-400">
          <span>Already have an account? </span>
          <Link to="/login" className="text-blue-400 font-bold hover:underline">
            Login
          </Link>
        </div>

      </div>
    </div>
  );
};

export default RegisterPage;
