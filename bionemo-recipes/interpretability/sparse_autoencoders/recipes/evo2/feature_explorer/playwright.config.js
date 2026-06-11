// e2e smoke for the dashboard, driven entirely by the GPU-free mock backend (mock_backend.py).
// No model needed: Playwright boots the mock on :8001 and Vite on :5176 (Vite's /api proxy
// targets :8001), then exercises the live tabs. Run: `npm run test:e2e`.
// The mock command uses `python` — ensure fastapi/uvicorn/numpy are importable (the recipe venv).
import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: { timeout: 20_000 },
  use: { baseURL: 'http://localhost:5176', headless: true },
  webServer: [
    {
      command: 'python mock_backend.py --port 8001',
      url: 'http://localhost:8001/health',
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
    {
      command: 'npm run dev -- --port 5176',
      url: 'http://localhost:5176',
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
  ],
})
