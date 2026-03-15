/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        caldron: {
          dark: '#1d3557',
          mid: '#457b9d',
          light: '#a8dadc',
          cream: '#f1faee',
          accent: '#e63946',
        },
      },
    },
  },
  plugins: [],
}
