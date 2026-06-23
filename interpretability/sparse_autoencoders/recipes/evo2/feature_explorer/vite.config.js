import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  root: '.',
  build: {
    outDir: 'dist',
  },
  server: {
    host: '0.0.0.0',
    port: 5176,
    strictPort: true,
    // The live backend (server.py) runs on :8001 and serves the API under /api. Proxy /api
    // straight through (NO rewrite) so dev hits the same paths as production: in dev the browser
    // talks to Vite and Vite forwards /api/* to :8001/api/*; in the single-container build the
    // browser hits /api/* on the backend directly. Same frontend code, identical paths both ways.
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
})
