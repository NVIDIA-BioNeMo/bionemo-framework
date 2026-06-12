# Evo2 SAE Feature Explorer (front-end)

Interactive dashboard for Evo2 SAE features — feature atlas, sequence inspector, and
generative steering.

This directory is the **front-end only**. Its backend is the standalone
[`evo2_sae`](../src/evo2_sae) engine — the viz is just a UI over its
`serve` mode, so there is no model code here.

```bash
# 1. Backend: loads Evo2 + the SAE and serves the HTTP API on :8001
../scripts/launch_inference.sh serve          # or: python -m evo2_sae.cli serve

# 2. Dashboard (from recipes/evo2): stages data (if any) + starts Vite
python ../scripts/launch_dashboard.py                          # inspector + steering tabs
python ../scripts/launch_dashboard.py --data-dir /path/to/data # + Feature-atlas tab
```

`launch_dashboard.py` is the entry point — it validates/stages the atlas parquets into
`public/` (when `--data-dir` is given) and runs Vite. The **inspector** and **steering** tabs
work with no atlas data (they call the backend); the **Feature-atlas** tab needs the three
parquets (`features_atlas`, `feature_metadata`, `feature_examples`) via `--data-dir` —
producing them is a separate offline step. (`npm install && npm run dev` also works for raw
front-end dev, but skips data staging.)

The Vite dev server proxies `/api` → `http://localhost:8001` (see `vite.config.js`); point it
elsewhere with `VITE_BACKEND`. Configure the backend via the env vars in `launch_inference.sh`.
