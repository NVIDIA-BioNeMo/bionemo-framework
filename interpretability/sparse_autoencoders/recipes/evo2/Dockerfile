# syntax=docker/dockerfile:1.4
#
# Runnable image for the Evo2 SAE recipe. Thin and non-forking: the engine imports bionemo.evo2
# (the mbridge evo2_megatron recipe), so .ci_build.sh delegates to evo2_megatron's OWN build
# (inheriting its pinned megatron-bridge / causal-conv1d / TransformerEngine versions), then
# installs the sae library + this recipe on top. We don't reimplement the megatron build here.
# A Node build stage compiles the dashboard to static files baked into the image, so ONE
# container serves both the UI (at /) and the API (at /api) — no Node at runtime.
#
# Build from the REPO ROOT (the context must include the recipes/evo2_megatron sibling):
#   docker build -f interpretability/sparse_autoencoders/recipes/evo2/Dockerfile -t evo2-sae .
# Serve the dashboard + API on one port (needs a GPU + checkpoints; see the recipe README):
#   docker run --gpus all -p 8001:8001 \
#     -e EVO2_CKPT_DIR=/ckpt/evo2 -e SAE_CKPT_PATH=/ckpt/sae.pt -e EMBEDDING_LAYER=26 \
#     -v /my/checkpoints:/ckpt evo2-sae scripts/launch_inference.sh serve
#   # then open http://localhost:8001
# Or run the tests:
#   docker run --gpus all -it evo2-sae bash -lc "source .ci_test_env.sh && pytest tests/"

# Stage 1 — build the dashboard frontend to static files. Node lives ONLY in this stage; the
# runtime image below gets just the compiled dist/, never npm/vite.
FROM node:20-slim AS frontend
WORKDIR /frontend
# Copy manifests first so `npm ci` is cached unless dependencies change.
COPY interpretability/sparse_autoencoders/recipes/evo2/feature_explorer/package.json \
     interpretability/sparse_autoencoders/recipes/evo2/feature_explorer/package-lock.json ./
RUN npm ci
COPY interpretability/sparse_autoencoders/recipes/evo2/feature_explorer/ ./
RUN npm run build  # -> /frontend/dist

# Stage 2 — the runnable GPU image.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${BASE_IMAGE}

WORKDIR /workspace
# Two layers so editing SAE code doesn't rebuild the ~30-min megatron stack:
#
# 1. Expensive, rarely-changing: build the bionemo.evo2 (mbridge) env. Depends ONLY on
#    recipes/evo2_megatron (which pulls its bionemo-core/recipeutils deps from git), so this layer
#    stays cached across SAE code edits. Copy just the recipe's .ci_build.sh to drive the build.
COPY recipes/evo2_megatron recipes/evo2_megatron
COPY interpretability/sparse_autoencoders/recipes/evo2/.ci_build.sh \
     interpretability/sparse_autoencoders/recipes/evo2/.ci_build.sh
WORKDIR /workspace/interpretability/sparse_autoencoders/recipes/evo2
RUN bash .ci_build.sh env

# 2. Cheap, frequently-changing: add the SAE library + this recipe, install editable. A code
#    change invalidates only this layer (the two pip installs, seconds), not the build above.
WORKDIR /workspace
COPY interpretability/sparse_autoencoders interpretability/sparse_autoencoders
WORKDIR /workspace/interpretability/sparse_autoencoders/recipes/evo2
RUN bash .ci_build.sh install

# 3. Bake in the prebuilt dashboard and point the server at it (DASHBOARD_DIST), so the one
#    container serves the UI at / and the API at /api. Only this layer changes on a frontend edit.
COPY --from=frontend /frontend/dist \
     /workspace/interpretability/sparse_autoencoders/recipes/evo2/feature_explorer/dist
ENV DASHBOARD_DIST=/workspace/interpretability/sparse_autoencoders/recipes/evo2/feature_explorer/dist

# Default to the built venv so `docker run … pytest tests/` (or importing Evo2SAE) just works.
ENV VIRTUAL_ENV=/workspace/recipes/evo2_megatron/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
