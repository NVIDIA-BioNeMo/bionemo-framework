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
    // host: '0.0.0.0' binds for remote access, so reaching the dev server through a public
    // hostname (brev, ngrok, Codespaces) instead of an `ssh -L` localhost tunnel triggers Vite's
    // Host check. Pre-allow the common tunnel domains; add yours here (or set `true`) if different.
    allowedHosts: ['localhost', '.brevlab.com', '.ngrok.io', '.ngrok-free.app', '.github.dev'],
    // The live backend (server.py) runs on :8001 and serves the API under /api. Proxy /api
    // straight through (NO rewrite) so dev hits the same paths as production: in dev the browser
    // talks to Vite and Vite forwards /api/* to :8001/api/*; in the single-container build the
    // browser hits /api/* on the backend directly. Same frontend code, identical paths both ways.
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        // A /generate can run minutes with NO interim bytes (it's not streamed), so the dev proxy must
        // not cut the idle connection. 30 min covers a full 8192-token steered run + baseline.
        // proxyTimeout = how long to wait on the backend response; timeout = the incoming socket.
        // (Note: an external tunnel — tsh/Teleport — may impose its own idle timeout this can't override.)
        proxyTimeout: 1800000,
        timeout: 1800000,
      },
    },
  },
})
