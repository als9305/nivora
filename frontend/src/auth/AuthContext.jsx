import React, { createContext, useCallback, useContext, useMemo, useState } from 'react';

const STORAGE_KEY = 'nivora_user';
const AuthContext = createContext(null);

function readPersistedUser() {
  try {
    const user = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
    return user?.user_id && user?.email ? user : null;
  } catch {
    localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(readPersistedUser);

  const signIn = useCallback((response, onboardingStep = null) => {
    if (!response?.success || !response?.user_id || !response?.email) {
      throw new Error('The authentication server returned an incomplete session.');
    }
    const authenticatedUser = {
      user_id: response.user_id,
      email: response.email,
      name: response.name || response.email,
      requires_biometrics: Boolean(response.requires_biometrics),
      onboarding_step: onboardingStep,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(authenticatedUser));
    setUser(authenticatedUser);
    return authenticatedUser;
  }, []);

  const signOut = useCallback(() => {
    // Components that own live media listen for this before unmounting.
    window.dispatchEvent(new Event('nivora:logout'));
    localStorage.removeItem(STORAGE_KEY);
    sessionStorage.clear();
    setUser(null);
  }, []);

  const setOnboardingStep = useCallback((onboardingStep) => {
    setUser((currentUser) => {
      if (!currentUser) return null;
      const updatedUser = { ...currentUser, onboarding_step: onboardingStep };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedUser));
      return updatedUser;
    });
  }, []);

  const value = useMemo(() => ({ user, isAuthenticated: Boolean(user?.user_id), signIn, signOut, setOnboardingStep }), [user, signIn, signOut, setOnboardingStep]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider.');
  return context;
}
