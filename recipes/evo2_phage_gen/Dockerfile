# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /workspace/bionemo
COPY . .

WORKDIR /workspace/bionemo/recipes/evo2_phage_gen

ENV UV_LINK_MODE=copy
ENV UV_NO_MANAGED_PYTHON=1
ENV VIRTUAL_ENV=/workspace/bionemo/recipes/evo2_phage_gen/.venv
ENV PATH="${VIRTUAL_ENV}/bin:/workspace/bionemo/recipes/evo2_phage_gen/data/external/bin:${PATH}"
ENV CHECKVDB=/workspace/bionemo/recipes/evo2_phage_gen/data/external/checkv/checkv-db-v1.5

# .ci_build_env.sh builds the Python venv and installs small native verifier
# tools (DIAMOND, HMMER/hmmsearch, and a prodigal wrapper) into .venv/bin so CI,
# local devcontainers, and this image all use the same setup path.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    ./.ci_build_env.sh
