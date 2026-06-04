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
    // The live backend (steering_server.py) runs on :8001. Proxying it under
    // /api means only the Vite port needs to be tunneled — the browser talks
    // to Vite, Vite talks to the backend, both on the pod.
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
