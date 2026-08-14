# syntax=docker/dockerfile:1.4
# Build from this recipe directory:
# docker build -t evo2-phage-gen .
FROM nvcr.io/nvidia/pytorch:26.07-py3

# .ci_build.sh invokes uv before the recipe virtual environment exists.
COPY --from=ghcr.io/astral-sh/uv:0.12.4 /uv /uvx /bin/

WORKDIR /workspace/bionemo/recipes/evo2_phage_gen
COPY . .

ENV UV_LINK_MODE=copy
ENV UV_NO_MANAGED_PYTHON=1
ENV VIRTUAL_ENV=/workspace/bionemo/recipes/evo2_phage_gen/.venv
ENV PATH="${VIRTUAL_ENV}/bin:/workspace/bionemo/recipes/evo2_phage_gen/data/external/bin:${PATH}"

# Build the same recipe environment used by CI, including the small native
# verifier tools (DIAMOND, HMMER/hmmsearch, and a prodigal wrapper).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    ./.ci_build.sh
