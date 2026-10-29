import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { LandingPage } from './pages/LandingPage';
import { LoginPage } from './pages/LoginPage';
import { RegisterPage } from './pages/RegisterPage';
import { FaceEnrollmentPage } from './pages/FaceEnrollmentPage';
import { VoiceEnrollmentPage } from './pages/VoiceEnrollmentPage';
import { MultimodalAuthPage } from './pages/MultimodalAuthPage';
import { DashboardPage } from './pages/DashboardPage';
import { HistoryPage } from './pages/HistoryPage';
import { SecurityPage } from './pages/SecurityPage';
import { SettingsPage } from './pages/SettingsPage';
import { DetailedAnalysisPage } from './pages/DetailedAnalysisPage';
import { IncidentDetailPage } from './pages/IncidentDetailPage';
import { RequireAuth, RedirectIfAuth, RequireOnboardingComplete } from './components/RequireAuth';
import { AuthProvider } from './auth/AuthContext';

export function App() {
  return (
    <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <AuthProvider>
      <div className="min-h-screen bg-[#07090e] text-slate-100 flex flex-col font-sans selection:bg-blue-600 selection:text-white">
        <main className="flex-1">
          <Routes>
            {/* Public routes */}
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={
              <RedirectIfAuth><LoginPage /></RedirectIfAuth>
            } />
            <Route path="/register" element={
              <RedirectIfAuth><RegisterPage /></RedirectIfAuth>
            } />

            {/* Enrollment routes — require authentication (post-registration) */}
            <Route path="/enroll-face" element={
              <RequireAuth><FaceEnrollmentPage /></RequireAuth>
            } />
            <Route path="/enroll-voice" element={
              <RequireAuth><VoiceEnrollmentPage /></RequireAuth>
            } />

            {/* Protected routes — require authentication */}
            <Route path="/auth" element={
              <RequireAuth><RequireOnboardingComplete><MultimodalAuthPage /></RequireOnboardingComplete></RequireAuth>
            } />
            <Route path="/dashboard" element={
              <RequireAuth><RequireOnboardingComplete><DashboardPage /></RequireOnboardingComplete></RequireAuth>
            } />
            <Route path="/history" element={
              <RequireAuth><RequireOnboardingComplete><HistoryPage /></RequireOnboardingComplete></RequireAuth>
            } />
            <Route path="/security" element={
              <RequireAuth><RequireOnboardingComplete><SecurityPage /></RequireOnboardingComplete></RequireAuth>
            } />
            <Route path="/settings" element={
              <RequireAuth><RequireOnboardingComplete><SettingsPage /></RequireOnboardingComplete></RequireAuth>
            } />
            <Route path="/analysis/:id" element={
              <RequireAuth><RequireOnboardingComplete><DetailedAnalysisPage /></RequireOnboardingComplete></RequireAuth>
            } />
            <Route path="/incident/:id" element={
              <RequireAuth><RequireOnboardingComplete><IncidentDetailPage /></RequireOnboardingComplete></RequireAuth>
            } />

            {/* Fallbacks */}
            <Route path="/multimodal" element={<Navigate to="/auth" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
      </AuthProvider>
    </Router>
  );
}

export default App;
