/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: "#07090e",
        cardBg: "#0c1017",
        cardBorder: "#1f293d",
        neonBlue: "#3b82f6",
        neonGlow: "#60a5fa",
        panelBg: "#0f172a",
        primaryCyan: "#06B6D4",
        accentEmerald: "#10B981",
        alertAmber: "#F59E0B",
        dangerRose: "#EF4444",
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
