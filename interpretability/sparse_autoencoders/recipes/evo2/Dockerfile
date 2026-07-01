# syntax=docker/dockerfile:1.4
#
# Runnable image for the Evo2 SAE recipe. Thin and non-forking: the engine imports bionemo.evo2
# (the mbridge evo2_megatron recipe), so .ci_build.sh delegates to evo2_megatron's OWN build
# (inheriting its pinned megatron-bridge / causal-conv1d / TransformerEngine versions), then
# installs the sae library + this recipe on top. We don't reimplement the megatron build here.
#
# Build from the REPO ROOT (the context must include the recipes/evo2_megatron sibling):
#   docker build -f interpretability/sparse_autoencoders/recipes/evo2/Dockerfile -t evo2-sae .
# Run (needs a GPU + checkpoints; see the recipe README):
#   docker run --gpus all -it evo2-sae bash -lc "source .ci_test_env.sh && pytest tests/"
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

# Default to the built venv so `docker run … pytest tests/` (or importing Evo2SAE) just works.
ENV VIRTUAL_ENV=/workspace/recipes/evo2_megatron/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
