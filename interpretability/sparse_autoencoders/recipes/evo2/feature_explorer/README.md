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

## Running without a backend (offline / static)

The dashboard degrades gracefully: it probes `/health`, and when there's **no live backend** it
hides the tabs that need the model and keeps the ones that read static files.

| Tab                     | Needs backend?           | Offline source                                                |
| ----------------------- | ------------------------ | ------------------------------------------------------------- |
| **Feature atlas**       | no                       | the atlas parquets (`--data-dir`)                             |
| **Sequence UMAP**       | no *iff* a bundle exists | `sequmap_embeddings.json` (precomputed; UMAP runs in-browser) |
| **Generative steering** | yes                      | hidden offline                                                |
| **Sequence inspector**  | yes                      | hidden offline                                                |

So with no backend you get the **Feature atlas** always, and **Sequence UMAP** if you precompute
its bundle. Steering and the live inspector require `serve`.

```bash
# (one-time, needs the 7B) precompute all the static artifacts into one dir:
python ../scripts/dashboard.py atlas      --activations-dir $STORE --output-dir dashboard_data   # atlas tab
python ../scripts/dashboard.py examples   --examples-fasta lib.fa  --output-dir dashboard_data   # example cards
python ../scripts/dashboard.py embeddings --examples-fasta lib.fa  --output-dir dashboard_data   # Sequence-UMAP bundle
#   (env: SAE_CKPT_PATH, EVO2_CKPT_DIR, FEATURE_ANNOTATIONS — same as launch_inference.sh)

# serve the static dashboard — NO backend, NO GPU:
python ../scripts/launch_dashboard.py --data-dir dashboard_data
```

`dashboard.py embeddings` writes `sequmap_embeddings.json` (the same shape `/gene_embed` returns;
bounded by sequence count). The viz auto-detects it (override the path with `?embeddings=<url>`).
With all artifacts staged, `npm run build` produces a fully static site you can host anywhere
(HF Spaces static / S3 / Pages) — interactive steering/inspector light up automatically if a
backend is later reachable.
