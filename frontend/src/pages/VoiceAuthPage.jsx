import React from 'react';
import { Sidebar } from '../components/Sidebar';
import { TopBar } from '../components/TopBar';
import { VoiceAuthConsole } from '../components/VoiceAuthConsole';

export function VoiceAuthPage() {
  return (
    <div className="flex h-screen bg-[#07090e] text-gray-200 overflow-hidden font-sans">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <TopBar title="Voice Identity Console" subtitle="Module 3 — Voice Identity (ECAPA-TDNN Biometric & Acoustic PAD)" />
        <main className="flex-1 p-6">
          <VoiceAuthConsole />
        </main>
      </div>
    </div>
  );
}

export default VoiceAuthPage;
