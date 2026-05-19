import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      // All /api/* requests → FastAPI backend
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path,
        timeout: 120000,
        configure: (proxy) => {
          proxy.on('error', (err) => {
            console.warn('[proxy] FastAPI unreachable:', err.message);
          });
        },
      },
      // Market ranker routes
      '/market': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        timeout: 300000,
      },
      // Legacy LSTM routes
      '/legacy': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        timeout: 120000,
      },
    },
  },
})
