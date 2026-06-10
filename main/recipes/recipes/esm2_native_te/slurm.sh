#!/bin/bash
#SBATCH --nodes=2                         # number of nodes
#SBATCH --ntasks-per-node=1    	          # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00                   # wall time
#SBATCH --mem=0                 	      # all mem avail

set -x -e
ulimit -c 0

export GPUS_PER_NODE=8
export CMD="TRITON_CACHE_DIR=/tmp/triton_cache \
    torchrun \
    --rdzv_id \$SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint \$MASTER_ADDR:\$MASTER_PORT \
    --nproc-per-node $GPUS_PER_NODE \
    --nnodes \$SLURM_NNODES \
    --node-rank \$SLURM_NODEID \
    train.py
"

# Mount a persistent cache directory to cache dataset downloads and transformations.
export CACHE_DIR=<cache_dir>

srun \
  --container-image=<image_name> \
  --container-mounts=${PWD}:/workspace/bionemo,$HOME/.netrc:/root/.netrc,$CACHE_DIR:/root/.cache \
  bash -c "$CMD"
