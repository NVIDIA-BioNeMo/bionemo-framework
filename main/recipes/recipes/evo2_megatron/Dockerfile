# syntax=docker/dockerfile:1.4
# Build from this recipe directory:
# docker build -t evo2-megatron .
FROM nvcr.io/nvidia/pytorch:26.07-py3

# .ci_build.sh invokes uv before the recipe virtual environment exists.
COPY --from=ghcr.io/astral-sh/uv:0.12.4 /uv /uvx /bin/

WORKDIR /workspace/bionemo/recipes/evo2_megatron
COPY . .

ENV UV_LINK_MODE=copy
ENV UV_NO_MANAGED_PYTHON=1
ENV VIRTUAL_ENV=/workspace/bionemo/recipes/evo2_megatron/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Sandboxed agent CLIs use these helpers on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bubblewrap \
    uidmap \
    && rm -rf /var/lib/apt/lists/*

# Build the same editable, system-stack-backed environment used by CI.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    ./.ci_build.sh
