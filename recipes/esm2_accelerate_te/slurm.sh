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
    accelerate launch \
    --config_file accelerate_config/fsdp2_te.yaml \
    --machine_rank "\$SLURM_NODEID" \
    --num_machines "$SLURM_NNODES" \
    --main_process_ip "\$SLURM_SRUN_COMM_HOST" \
    --main_process_port 12340 \
    --num_processes "$(( $SLURM_NNODES * $GPUS_PER_NODE ))" \
    train.py
"

# Mount a persistent cache directory to cache dataset downloads and transformations.
export CACHE_DIR=<cache_dir>

srun \
  --container-image=<image_name> \
  --container-mounts=${PWD}:/workspace/bionemo,$HOME/.netrc:/root/.netrc,$CACHE_DIR:/root/.cache/huggingface \
  bash -c "$CMD"
