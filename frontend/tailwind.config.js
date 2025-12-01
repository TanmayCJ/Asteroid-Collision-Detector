/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        space: {
          900: '#0a0e27',
          800: '#1a1f3a',
          700: '#2d3561',
          600: '#3d4a7a',
        },
        neon: {
          blue: '#00d9ff',
          purple: '#a855f7',
          pink: '#ec4899',
          green: '#10b981',
        },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite',
      },
      keyframes: {
        glow: {
          '0%, 100%': { boxShadow: '0 0 5px #00d9ff, 0 0 10px #00d9ff' },
          '50%': { boxShadow: '0 0 20px #00d9ff, 0 0 30px #00d9ff' },
        },
      },
    },
  },
  plugins: [],
}
