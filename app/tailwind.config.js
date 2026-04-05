/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Outfit', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
      colors: {
        void: '#04060f',
        'surface-glass': 'rgba(255,255,255,0.03)',
      },
      backgroundImage: {
        'grid-pattern':
          'linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)',
      },
      backgroundSize: {
        'grid-50': '50px 50px',
      },
      keyframes: {
        pulseRingA: {
          '0%,100%': { transform: 'scale(0.9)', opacity: '0.7' },
          '50%':      { transform: 'scale(1.18)', opacity: '0.2' },
        },
        pulseRingB: {
          '0%,100%': { transform: 'scale(0.85)', opacity: '0.4' },
          '50%':      { transform: 'scale(1.35)', opacity: '0.08' },
        },
        waveBar: {
          '0%,100%': { height: '6px' },
          '50%':      { height: '30px' },
        },
      },
      animation: {
        'pulse-ring-a': 'pulseRingA 2s cubic-bezier(0.455,0.03,0.515,0.955) infinite',
        'pulse-ring-b': 'pulseRingB 2s cubic-bezier(0.455,0.03,0.515,0.955) 0.5s infinite',
        'wave-bar':     'waveBar 0.8s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
