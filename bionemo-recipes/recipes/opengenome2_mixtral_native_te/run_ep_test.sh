#!/bin/bash
# Usage: ./run_ep_test.sh <config_name>
# Run from inside the recipe directory on DGX.
# Example: ./run_ep_test.sh test_8x1B_og2_ep2
set -e

CONFIG=${1:-test_8x1B_og2_ep2}

export BIONEMO_DISABLE_TORCH_COMPILE_HELPERS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
# EP>1 triggers torch._dynamo internally (DTensor/FSDP2), which needs ptxas + cuda.h
export PATH="/usr/local/cuda/bin:${PATH}"
export CPATH="/usr/local/cuda/include:${CPATH:-}"

echo "=== Starting run: $CONFIG ==="
echo "Time: $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

torchrun \
  --standalone \
  --nproc_per_node=8 \
  train_fsdp2.py \
  --config-name "$CONFIG"

echo "=== Finished run: $CONFIG at $(date) ==="
