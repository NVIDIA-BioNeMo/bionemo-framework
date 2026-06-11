FROM nvcr.io/nvidia/pytorch:26.04-py3
WORKDIR /workspace/bionemo
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_CONSTRAINT= pip install -r requirements.txt

WORKDIR /workspace
ARG INSTALL_VLLM=false
ARG TORCH_CUDA_ARCH_LIST=""
ARG MAX_JOBS=8
ARG UV_BREAK_SYSTEM_PACKAGES=1
RUN if [ "$INSTALL_VLLM" = "true" ]; then \
    if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then \
        echo "ERROR: TORCH_CUDA_ARCH_LIST must be set when INSTALL_VLLM=true" && exit 1; \
    fi && \
    git clone --branch v0.15.1 --depth 1 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    python use_existing_torch.py && \
    uv pip install -r requirements/build.txt --system && \
    uv pip install --no-build-isolation -e . --system && \
    pip install --upgrade "transformers[torch]"; \
    fi

WORKDIR /workspace/bionemo
