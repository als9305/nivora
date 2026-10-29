import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Sidebar } from '../components/Sidebar';
import { TopBar } from '../components/TopBar';
import { Settings, User, Camera, ShieldCheck, RefreshCw, Mic } from 'lucide-react';
import { getFaceTemplateStatus, getVoiceTemplateStatus } from '../services/api';
import { useAuth } from '../auth/AuthContext';

export function SettingsPage() {
  const navigate = useNavigate();
  const { user: currentUser } = useAuth();
  const [templateStatus, setTemplateStatus] = useState(null);
  const [voiceStatus, setVoiceStatus] = useState(null);

  useEffect(() => {
    if (currentUser?.user_id) checkStatus(currentUser.user_id);
  }, [currentUser?.user_id]);

  const checkStatus = async (userId) => {
    try {
      const [faceRes, voiceRes] = await Promise.all([
        getFaceTemplateStatus(userId),
        getVoiceTemplateStatus(userId)
      ]);
      setTemplateStatus(faceRes);
      setVoiceStatus(voiceRes);
    } catch (e) {}
  };

  const isVoiceEnrolled = voiceStatus?.enrolled === true || voiceStatus?.exists === true || voiceStatus?.status === 'ENROLLED';

  return (
    <div className="flex h-screen bg-[#07090e] text-gray-200 font-sans overflow-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <TopBar title="Settings" subtitle="Account details and biometric profile management" />

        <main className="flex-1 p-8 space-y-8 max-w-4xl mx-auto w-full">
          <div>
            <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
              <Settings className="w-7 h-7 text-blue-400" />
              <span>Settings</span>
            </h1>
            <p className="text-xs text-gray-400 font-medium mt-1">
              Manage your profile and biometric face & voice identity templates.
            </p>
          </div>

          {/* Account Profile Card */}
          <div className="bg-[#0c1017] rounded-2xl border border-[#1f293d] p-6 space-y-6 shadow-2xl">
            <h2 className="text-sm font-bold text-white uppercase font-mono tracking-wider flex items-center gap-2">
              <User className="w-4 h-4 text-blue-400" />
              <span>Account Information</span>
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-sans">
              <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-1">
                <span className="text-gray-500 text-[10px] font-mono font-bold uppercase">Name</span>
                <p className="text-white font-bold text-sm">{currentUser?.name}</p>
              </div>

              <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] space-y-1">
                <span className="text-gray-500 text-[10px] font-mono font-bold uppercase">Email Address</span>
                <p className="text-white font-bold text-sm">{currentUser?.email}</p>
              </div>
            </div>
          </div>

          {/* Biometric Face Identity Card */}
          <div className="bg-[#0c1017] rounded-2xl border border-[#1f293d] p-6 space-y-6 shadow-2xl">
            <h2 className="text-sm font-bold text-white uppercase font-mono tracking-wider flex items-center gap-2">
              <Camera className="w-4 h-4 text-blue-400" />
              <span>Biometric Face Identity Setup</span>
            </h2>

            <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-xs">
              <div>
                <div className="flex items-center gap-2 font-bold text-white">
                  <ShieldCheck className={`w-4 h-4 ${templateStatus?.enrolled || templateStatus?.exists ? 'text-emerald-400' : 'text-amber-400'}`} />
                  <span>Face Template Status: {templateStatus?.enrolled || templateStatus?.exists ? 'Enrolled' : 'Not Enrolled'}</span>
                </div>
                <p className="text-gray-400 text-[11px] mt-0.5">
                  AES-256-GCM encrypted facial centroid vector template stored on backend.
                </p>
              </div>

              <button
                onClick={() => navigate('/enroll-face')}
                className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-xs shadow-lg shadow-blue-500/25 transition cursor-pointer flex items-center gap-2"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>{templateStatus?.enrolled || templateStatus?.exists ? 'Re-enroll Face' : 'Enroll Face'}</span>
              </button>
            </div>
          </div>

          {/* Biometric Voice Profile Card */}
          <div className="bg-[#0c1017] rounded-2xl border border-[#1f293d] p-6 space-y-6 shadow-2xl">
            <h2 className="text-sm font-bold text-white uppercase font-mono tracking-wider flex items-center gap-2">
              <Mic className="w-4 h-4 text-purple-400" />
              <span>Biometric Voice Profile Setup</span>
            </h2>

            <div className="p-4 rounded-xl bg-[#07090e] border border-[#1f293d] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-xs">
              <div>
                <div className="flex items-center gap-2 font-bold text-white">
                  <ShieldCheck className={`w-4 h-4 ${isVoiceEnrolled ? 'text-emerald-400' : 'text-amber-400'}`} />
                  <span>Voice Profile Status: {isVoiceEnrolled ? 'Enrolled' : 'Not Enrolled'}</span>
                </div>
                <p className="text-gray-400 text-[11px] mt-0.5">
                  192-dimensional ECAPA-TDNN centroid template encrypted with AES-256-GCM.
                </p>
              </div>

              <button
                onClick={() => navigate('/enroll-voice')}
                className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-bold text-xs shadow-lg shadow-purple-500/25 transition cursor-pointer flex items-center gap-2"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>{isVoiceEnrolled ? 'Re-enroll Voice Profile' : 'Enroll Voice Profile'}</span>
              </button>
            </div>
          </div>

        </main>
      </div>
    </div>
  );
}

export default SettingsPage;
