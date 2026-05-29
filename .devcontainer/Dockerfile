# Uncomment to use the latest TE from the NGC registry for debugging changes with latest TE.
# FROM gitlab-master.nvidia.com/dl/transformerengine/transformerengine:main-pytorch-py3-base
FROM nvcr.io/nvidia/pytorch:26.04-py3

# FIXME: Fix for "No such file or directory: /workspace/TransformerEngine"
#  Remove once bug has been addressed in the nvidia/pytorch container.
RUN rm -f /usr/local/lib/python*/dist-packages/transformer_engine-*.dist-info/direct_url.json

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/workspace/requirements.txt \
    PIP_CONSTRAINT= pip install -r /workspace/requirements.txt

# Sandboxed agent CLIs use these helpers on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bubblewrap \
    uidmap \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

USER ubuntu
RUN curl https://cursor.com/install -fsS | bash  # Install cursor-agent CLI tool
RUN curl -fsSL https://claude.ai/install.sh | bash  # Install Claude CLI tool
RUN uv tool install pre-commit --with pre-commit-uv --force-reinstall
ENV PATH="/home/ubuntu/.local/bin:${PATH}"
