import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      // All /api/* requests are forwarded to the FastAPI backend.
      // This ELIMINATES all CORS problems - the browser talks to Vite,
      // Vite proxies to FastAPI. No more port mismatches.
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path,
        timeout: 120000,           // 2 min — model inference can be slow on CPU
        configure: (proxy) => {
          proxy.on('error', (err) => {
            console.warn('[proxy] FastAPI unreachable:', err.message);
          });
        },
      },
    },
  },
})
