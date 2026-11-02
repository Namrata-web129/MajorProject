/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Navy/Charcoal base
        navy: {
          950: '#060b18',
          900: '#0a1628',
          800: '#0f2040',
          700: '#162a52',
          600: '#1e3a6e',
        },
        charcoal: {
          900: '#111827',
          800: '#1f2937',
          700: '#374151',
          600: '#4b5563',
        },
        // Accent purple/blue
        accent: {
          purple: '#7c3aed',
          'purple-light': '#a78bfa',
          blue: '#3b82f6',
          'blue-light': '#93c5fd',
          cyan: '#06b6d4',
          'cyan-light': '#67e8f9',
        },
        // Status colors
        real: {
          DEFAULT: '#10b981',
          dark: '#065f46',
          light: '#6ee7b7',
        },
        fake: {
          DEFAULT: '#ef4444',
          dark: '#7f1d1d',
          light: '#fca5a5',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'hero-gradient': 'linear-gradient(135deg, #060b18 0%, #0a1628 50%, #0f2040 100%)',
        'card-gradient': 'linear-gradient(145deg, rgba(15,32,64,0.8), rgba(6,11,24,0.9))',
        'accent-gradient': 'linear-gradient(135deg, #7c3aed, #3b82f6)',
        'real-gradient': 'linear-gradient(135deg, #065f46, #10b981)',
        'fake-gradient': 'linear-gradient(135deg, #7f1d1d, #ef4444)',
      },
      boxShadow: {
        'glow-purple': '0 0 20px rgba(124,58,237,0.3)',
        'glow-blue': '0 0 20px rgba(59,130,246,0.3)',
        'glow-cyan': '0 0 20px rgba(6,182,212,0.3)',
        'card': '0 4px 24px rgba(0,0,0,0.4)',
        'card-hover': '0 8px 32px rgba(0,0,0,0.6)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'scan': 'scan 2s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        glow: {
          from: { boxShadow: '0 0 10px rgba(124,58,237,0.3)' },
          to: { boxShadow: '0 0 25px rgba(124,58,237,0.6)' },
        },
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
      },
      borderRadius: {
        xl: '0.75rem',
        '2xl': '1rem',
        '3xl': '1.5rem',
      },
    },
  },
  plugins: [],
}
