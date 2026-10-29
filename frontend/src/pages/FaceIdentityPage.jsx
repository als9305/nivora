import React from 'react';
import { Sidebar } from '../components/Sidebar';
import { TopBar } from '../components/TopBar';
import { FaceIdentityConsole } from '../components/FaceIdentityConsole';

export function FaceIdentityPage() {
  return (
    <div className="flex h-screen bg-[#07090e] text-gray-200 overflow-hidden font-sans">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <TopBar title="Face Identity Console" subtitle="Module 1 — Face Identity (InceptionResnetV1 Centroid Feature Matching)" />
        <main className="flex-1 p-6">
          <FaceIdentityConsole />
        </main>
      </div>
    </div>
  );
}

export default FaceIdentityPage;
