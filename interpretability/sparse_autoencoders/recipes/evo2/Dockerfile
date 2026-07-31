# syntax=docker/dockerfile:1.4
#
# Runnable image for the Evo2 SAE recipe. Thin and non-forking: the engine imports bionemo.evo2
# (the mbridge evo2_megatron recipe), so .ci_build.sh delegates to evo2_megatron's OWN build
# (inheriting its pinned megatron-bridge / causal-conv1d / TransformerEngine versions), then
# installs the sae library + this recipe on top. We don't reimplement the megatron build here.
#
# By DEFAULT this builds the engine + server only (no Node, no front-end) — the SAE-serving image.
# Pass --build-arg WITH_DASHBOARD=1 to ALSO build the dashboard front-end (a Node build stage) and
# bake it in, so one container serves the UI (at /) and the API (at /api). The Node toolchain is
# only pulled when WITH_DASHBOARD=1, and never ends up in the runtime image either way.
#
# Build from the REPO ROOT (the context must include the recipes/evo2_megatron sibling):
#   # engine + server only (default):
#   docker build -f interpretability/sparse_autoencoders/recipes/evo2/Dockerfile -t evo2-sae .
#   # engine + server + dashboard (single-container UI + API):
#   docker build --build-arg WITH_DASHBOARD=1 -f interpretability/sparse_autoencoders/recipes/evo2/Dockerfile -t evo2-sae .
#
# Serve (needs a GPU + checkpoints; see the recipe README). With WITH_DASHBOARD=1 the UI is at / too:
#   docker run --gpus all -p 8001:8001 \
#     -e EVO2_CKPT_DIR=/ckpt/evo2 -e SAE_CKPT_PATH=/ckpt/sae.pt -e EMBEDDING_LAYER=26 \
#     -v /my/checkpoints:/ckpt evo2-sae scripts/launch_inference.sh serve   # -> http://localhost:8001
# Or run the tests:
#   docker run --gpus all -it evo2-sae bash -lc "source .ci_test_env.sh && pytest tests/"

ARG WITH_DASHBOARD=0
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3

# Front-end build stage — compiles the dashboard to static files. Built ONLY when WITH_DASHBOARD=1
# (nothing references it otherwise, so BuildKit skips it). Node lives only here, never at runtime.
FROM node:20-slim AS frontend
WORKDIR /frontend
# Copy manifests first so `npm ci` is cached unless dependencies change.
COPY interpretability/sparse_autoencoders/recipes/evo2/feature_explorer/package.json \
     interpretability/sparse_autoencoders/recipes/evo2/feature_explorer/package-lock.json ./
RUN npm ci
COPY interpretability/sparse_autoencoders/recipes/evo2/feature_explorer/ ./
RUN npm run build  # -> /frontend/dist

# Engine + server image (no front-end). This is the default build.
FROM ${BASE_IMAGE} AS engine

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

# Default to the built venv so `docker run … pytest tests/` (or importing Evo2SAE) just works.
ENV VIRTUAL_ENV=/workspace/recipes/evo2_megatron/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Engine + the baked dashboard: serve the UI at / alongside the API at /api from one container.
FROM engine AS engine-with-dashboard
COPY --from=frontend /frontend/dist \
     /workspace/interpretability/sparse_autoencoders/recipes/evo2/feature_explorer/dist
ENV DASHBOARD_DIST=/workspace/interpretability/sparse_autoencoders/recipes/evo2/feature_explorer/dist

# Select the final image by WITH_DASHBOARD (0 = engine only [default], 1 = + dashboard). Only the
# selected stage's dependencies are built, so the default never pulls Node / builds the front-end.
FROM engine-with-dashboard AS final-1
FROM engine AS final-0
FROM final-${WITH_DASHBOARD} AS final
