# Evo2 SAE Feature Explorer (front-end)

Interactive dashboard for Evo2 SAE features, with four tabs:
**Feature atlas**, **Sequence inspector**, **Generative steering**, and **Sequence UMAP**.

This directory is the **front-end only** (React + Vite). Its backend is the standalone
[`evo2_sae`](../src/evo2_sae) engine — the viz is just a UI over its HTTP API, so there is no
model code here. The front-end always calls the API under **`/api`** (same path in dev and
production), so the only thing that ever changes is *where* `/api` is served from.

## Three ways to run

### 1. One container — recommended for sharing / deploy

The recipe [`Dockerfile`](../Dockerfile) builds this front-end to static files and bakes them
into the image, so a **single container serves the dashboard and the API on one port** — no
Node and no second process at runtime. This is what to hand a coworker or put behind an SSO proxy.

```bash
# build from the REPO ROOT (the build context needs the recipes/evo2_megatron sibling):
docker build -f interpretability/sparse_autoencoders/recipes/evo2/Dockerfile -t evo2-sae .

# run with a GPU + your checkpoints, then open http://localhost:8001
docker run --gpus all -p 8001:8001 \
  -e EVO2_CKPT_DIR=/ckpt/evo2 -e SAE_CKPT_PATH=/ckpt/sae.pt -e EMBEDDING_LAYER=26 \
  -v /path/to/checkpoints:/ckpt evo2-sae scripts/launch_inference.sh serve
# -> dashboard + API both on http://localhost:8001  (/ = UI, /api = backend)
```

The first build compiles the megatron stack (~30 min) and is layer-cached afterward; a Node build
stage produces the static bundle (`DASHBOARD_DIST`) and the server mounts it at `/`. No Node ends
up in the runtime image. See the [recipe Dockerfile](../Dockerfile) for the layer layout.

### 2. Local dev — UI iteration with hot reload

Needs **Node ≥ 18** (for Vite), plus a GPU + checkpoints for the live tabs. Two processes:

```bash
# backend: loads Evo2 + the SAE, serves the API under /api on :8001
../scripts/launch_inference.sh serve            # or: python -m evo2_sae.cli serve

# front-end: Vite dev server on :5176 (hot reload)
../scripts/launch_dashboard.py                  # stages atlas data if --data-dir is given, then runs Vite
#   or, for raw front-end dev:  npm install && npm run dev
```

Vite proxies `/api` **straight through** to `http://localhost:8001` (no path rewrite — see
`vite.config.js`), so dev hits the same `/api/*` paths as the single-container build. Point it at a
different backend with `VITE_BACKEND`. Configure the backend via the env vars in `launch_inference.sh`.

To reach a remote box, tunnel the Vite port only (Vite proxies `/api` on the box):

```bash
tsh ssh -L 5176:localhost:5176 <gpu-box>        # then open http://localhost:5176
```

### 3. Offline / static — no backend

The dashboard degrades gracefully: it probes `/api/health`, and when there's **no live backend** it
hides the tabs that need the model and keeps the ones that read static files.

| Tab                     | Needs backend?           | Offline source                                                |
| ----------------------- | ------------------------ | ------------------------------------------------------------- |
| **Feature atlas**       | no                       | the atlas parquets (`--data-dir`)                             |
| **Sequence UMAP**       | no *iff* a bundle exists | `sequmap_embeddings.json` (precomputed; UMAP runs in-browser) |
| **Generative steering** | yes                      | hidden offline                                                |
| **Sequence inspector**  | yes                      | hidden offline                                                |

So with no backend you always get the **Feature atlas**, plus **Sequence UMAP** if you precompute its
bundle. Steering and the live inspector require `serve`.

```bash
# (one-time, needs the model) precompute the static artifacts into one dir:
python ../scripts/dashboard.py atlas      --activations-dir $STORE --output-dir dashboard_data   # atlas tab
python ../scripts/dashboard.py examples   --examples-fasta lib.fa  --output-dir dashboard_data   # example cards
python ../scripts/dashboard.py embeddings --examples-fasta lib.fa  --output-dir dashboard_data   # Sequence-UMAP bundle
#   (env: SAE_CKPT_PATH, EVO2_CKPT_DIR, FEATURE_ANNOTATIONS — same as launch_inference.sh)

# serve the static dashboard — NO backend, NO GPU (needs Node for the dev server, or `npm run build`):
python ../scripts/launch_dashboard.py --data-dir dashboard_data
```

`dashboard.py embeddings` writes `sequmap_embeddings.json` (the same shape `/api/gene_embed` returns).
The viz auto-detects it (override the path with `?embeddings=<url>`). With all artifacts staged,
`npm run build` produces a fully static site you can host anywhere (HF Spaces static / S3 / Pages) —
the interactive steering/inspector tabs light up automatically if a backend later becomes reachable.

## Tabs

- **Feature atlas** — browse every SAE feature: firing rate, decoder-space UMAP, top-activating
  examples, labels. Reads precomputed static files.
- **Sequence inspector** — paste a sequence, see per-base SAE activations (top-k or picked features).
  Live encode.
- **Generative steering** — generate DNA while clamping chosen features on the continuation, vs. the
  unsteered baseline. Live generation.
- **Sequence UMAP** — embed a set of sequences (pooled per-feature vectors), UMAP them, color/re-project
  by a feature. Live backend or a precomputed bundle.

Each tab shows its own **Limitations** note in-app (context-length caps, the ±300 steering clamp,
stochastic 2-D UMAP, etc.).
