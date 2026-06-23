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
# evo2_megatron pulls its bionemo-core / bionemo-recipeutils deps from git, so only these two
# trees are needed in the build context.
COPY interpretability/sparse_autoencoders interpretability/sparse_autoencoders
COPY recipes/evo2_megatron recipes/evo2_megatron

WORKDIR /workspace/interpretability/sparse_autoencoders/recipes/evo2
RUN bash .ci_build.sh

# Default to the built venv so `docker run … pytest tests/` (or importing Evo2SAE) just works.
ENV VIRTUAL_ENV=/workspace/recipes/evo2_megatron/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
