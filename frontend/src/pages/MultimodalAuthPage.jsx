import React from 'react';
import MultimodalAuthConsole from '../components/MultimodalAuthConsole';

/**
 * /auth — full-viewport authentication page.
 * No Sidebar, no TopBar — the console owns the full 100dvh.
 */
export function MultimodalAuthPage() {
  return <MultimodalAuthConsole />;
}

export default MultimodalAuthPage;
