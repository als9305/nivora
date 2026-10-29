import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';

/**
 * Route guard component that protects routes requiring authentication.
 * Checks localStorage for nivora_user data with a valid user_id.
 * Redirects to /login if not authenticated, preserving the target location
 * so the user can be redirected back after successful login.
 */
export function RequireAuth({ children }) {
  const location = useLocation();

  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    // Redirect to login, saving the intended destination
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return children;
}

/**
 * Route guard that redirects authenticated users away from login/register pages.
 * If already logged in, redirect to /auth or /dashboard.
 */
export function RedirectIfAuth({ children }) {
  const { isAuthenticated, user } = useAuth();

  if (isAuthenticated) {
    return <Navigate to={user?.onboarding_step === 'FACE' ? '/enroll-face' : user?.onboarding_step === 'VOICE' ? '/enroll-voice' : '/dashboard'} replace />;
  }

  return children;
}

export function RequireOnboardingComplete({ children }) {
  const { user } = useAuth();
  if (user?.onboarding_step === 'FACE') return <Navigate to="/enroll-face" replace />;
  if (user?.onboarding_step === 'VOICE') return <Navigate to="/enroll-voice" replace />;
  return children;
}
