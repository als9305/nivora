import React from 'react';
import { Sidebar } from '../components/Sidebar';
import { TopBar } from '../components/TopBar';
import { FaceLivenessConsole } from '../components/FaceLivenessConsole';

export function FaceLivenessPage() {
  return (
    <div className="flex h-screen bg-[#07090e] text-gray-200 overflow-hidden font-sans">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <TopBar title="Face Liveness Console" subtitle="Module 2 — Face Liveness Detection (Passive RGB & Anti-Spoofing Verification)" />
        <main className="flex-1 p-6">
          <FaceLivenessConsole />
        </main>
      </div>
    </div>
  );
}

export default FaceLivenessPage;
