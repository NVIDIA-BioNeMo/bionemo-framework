# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:26.04-py3

# uv is pre-installed in the nvcr.io/nvidia/pytorch base image.
# If using a base image without uv, uncomment the following line:
#COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace/bionemo
COPY . .

# 2. Fix for "No such file or directory: /workspace/TransformerEngine"
# This removes the "direct_url" reference that confuses tools when TE was installed from source in the base image.
RUN rm -f /usr/local/lib/python*/dist-packages/transformer_engine-*.dist-info/direct_url.json

ENV UV_LINK_MODE=copy
# Ensure we use the venv by default for all future commands
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 3. Create the venv with access to system packages (Torch, TE, etc.)
# We create it one level up (/workspace/.venv) to keep it out of the source dir
RUN uv venv --system-site-packages --seed $VIRTUAL_ENV

# 4. Pin transformer_engine to the version pre-installed in the base image.
# Without this constraint uv may try to upgrade or rebuild TE, which breaks the build.
RUN pip freeze | grep transformer_engine > pip-constraints.txt

## Trying to workaround CAUSAL_CONV1D build issues
ENV CAUSAL_CONV1D_FORCE_BUILD=1
ENV PIP_NO_BINARY=causal-conv1d

# 5. Install package and dependencies
RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install -r build_requirements.txt --no-build-isolation && \
    uv pip install -c pip-constraints.txt -e . --no-build-isolation
