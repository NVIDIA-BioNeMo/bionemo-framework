#!/usr/bin/env bash
set -euo pipefail

ARCH="${1:-$(python3 -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')")}"
MAX_JOBS="${MAX_JOBS:-8}"
export UV_BREAK_SYSTEM_PACKAGES=1

echo "Building vLLM for CUDA arch: $ARCH (MAX_JOBS=$MAX_JOBS)"

cd /workspace
if [ ! -d vllm ]; then
    git clone --branch v0.15.1 --depth 1 https://github.com/vllm-project/vllm.git
fi
cd vllm
python use_existing_torch.py
TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="$MAX_JOBS" \
    uv pip install -r requirements/build.txt --system
TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="$MAX_JOBS" \
    uv pip install --no-build-isolation -e . --system
pip install --upgrade "transformers[torch]"

echo "vLLM installed for arch $ARCH"
