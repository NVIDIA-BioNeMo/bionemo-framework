# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:26.06-py3

# uv is pre-installed in the nvcr.io/nvidia/pytorch base image.
# If using a base image without uv, uncomment the following line:
#COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace/bionemo
COPY . .

ENV UV_LINK_MODE=copy
ENV VIRTUAL_ENV=/workspace/bionemo/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

## Trying to workaround CAUSAL_CONV1D build issues
ENV CAUSAL_CONV1D_FORCE_BUILD=1
ENV PIP_NO_BINARY=causal-conv1d

# Build the same recipe environment used by CI.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    ./.ci_build.sh
