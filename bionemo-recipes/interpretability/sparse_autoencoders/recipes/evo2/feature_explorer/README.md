# Evo2 SAE Feature Explorer (front-end)

Interactive dashboard for Evo2 SAE features — feature atlas, sequence inspector, and
generative steering.

This directory is the **front-end only**. Its backend is the standalone
[`evo2_sae_infer`](../evo2_sae_infer) engine — the viz is just a UI over its
`serve` mode, so there is no model code here.

```bash
# 1. Backend: loads Evo2 + the SAE and serves the HTTP API on :8001
../scripts/launch_inference.sh serve          # or: python -m evo2_sae_infer serve

# 2. Front-end (this directory)
npm install && npm run dev                     # Vite dev server
```

The Vite dev server proxies `/api` → `http://localhost:8001` (see `vite.config.js`);
point it elsewhere with the `VITE_BACKEND` env var. Configure the backend (checkpoint,
SAE, layer, feature annotations) via the env vars documented in `launch_inference.sh`.
