# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:26.02-py3

ARG CUTLASS_REPO=https://github.com/NVIDIA/cutlass.git
ARG CUTLASS_REF=08185b9c3e90510ee2b656662ed0d53b06d28157

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_CONSTRAINT= pip install -r /requirements.txt

# Install Node.js (needed for Claude Code and Codex CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Install OpenAI Codex CLI (authenticate at runtime with: codex login --device-auth)
RUN npm install -g @openai/codex

# Install pre-commit via uv
RUN uv tool install pre-commit --with pre-commit-uv --force-reinstall

# Install CUTLASS locally for Blackwell FP8/FP4 kernel experiments and native
# tri-mul benchmarking. The current MXFP8 grouped examples in CUTLASS 4.5.0 are
# gated to SM100a, so we keep the exact checkout recorded for reproducibility.
RUN git clone --depth 1 ${CUTLASS_REPO} /workspace/third_party/cutlass && \
    git -C /workspace/third_party/cutlass fetch --depth 1 origin ${CUTLASS_REF} && \
    git -C /workspace/third_party/cutlass checkout ${CUTLASS_REF}

WORKDIR /workspace/bionemo
COPY . .

# Build and install the specialized triangular-multiplication extension inside
# the image so the optimized BF16 backend is available at runtime.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation -e /workspace/bionemo/tri_mul_ext

# Build and install the tracked FP8 batched GEMM extension used by the
# `fp8_bmm` and `fp8_cublaslt` triangular-multiplication backends.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation -e /workspace/bionemo/fp8_bmm_ext
