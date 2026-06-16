#!/bin/bash
#SBATCH --account=
#SBATCH --nodes=1
#SBATCH --partition=
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:55:00
#SBATCH --mem=0
#SBATCH --job-name=
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# This script is adapted from the experiment scripts here:
# https://gitlab-master.nvidia.com/bio-foundation-models/codon-fm/-/tree/405b2315836a9c1c1ae0c5e41d5abcf4f24d6aa8/experiment_scripts/pretraining/encodon_filtered/mlm
#
# Modifications:
# - 'num_jobs' is not supported in the PTL recipe in bionemo-recipes.
# - '--sharded-state-dict' is not supported in the PTL recipe in bionemo-recipes. It is always 'sharded'.
# - Added support for selecting the sequence packing method (thd or bshd).
# - Added support for selecting the distributed strategy (fsdp or ddp).
# - Added support for selecting the gradient accumulation steps to keep the global batch size constant.
# - Added support for selecting the attention backend (xformers or pytorch SDPA).
# ============================================================================

# Establish or inherit chain ID: manual launch picks SLURM_JOB_ID; trap-resubmit inherits via --export.
if [ -z "${CHAIN_ID:-}" ]; then
  export CHAIN_ID="${SLURM_JOB_ID}"
  echo "Starting NEW chain: CHAIN_ID=${CHAIN_ID}"
else
  echo "Continuing chain ${CHAIN_ID} (current job ${SLURM_JOB_ID})"
fi

# ============================================================================
# CodonFM
# ============================================================================

BASE_DIR=""
CONTAINER=""
DATA_DIR="${BASE_DIR}/data"
CODE_MOUNT="/workspace/bionemo"


: "${WANDB_API_KEY:?Set WANDB_API_KEY in ~/.bash_profile}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN in ~/.bash_profile}"
: "${CLUSTER_NAME:?Set CLUSTER_NAME in ~/.bash_profile}"

export GLOBAL_BATCH_SIZE=1536
export MICRO_BATCH_SIZE=96

# Experiment parameters
export CONFIG_NAME=encodon_xx
export NPROC_PER_NODE=8
export DIST_STRATEGY=ddp  # fsdp or ddp

# Training
export NUM_TRAIN_STEPS=100
export LEARNING_RATE=7.5e-5
export NUM_WORKERS=12
export USE_SEQUENCE_PACKING=False

export PRECISION=bf16-mixed

# Logging / W&B
export LOGGER_FREQUENCY=10
export WANDB_PROJECT=

# Attn-backend
export USE_XFORMERS=1
export USE_TRANSFORMER_ENGINE=0

# Derived: build wandb run name from model size, batch size, and precision recipe
MODEL_SIZE="${CONFIG_NAME##*_}"
PRECISION_TAG="${PRECISION}"

if [ "${USE_SEQUENCE_PACKING}" = "True" ]; then
  BATCH_TYPE_TAG="thd"
else
  BATCH_TYPE_TAG="bshd"
fi

# Derive grad accumulation from GBS / (MBS * GPUs).
TOTAL_GPUS=$(( NPROC_PER_NODE * SLURM_JOB_NUM_NODES ))
TOTAL_PER_STEP=$(( MICRO_BATCH_SIZE * TOTAL_GPUS ))
if [ "${TOTAL_PER_STEP}" -eq 0 ] || [ "$(( GLOBAL_BATCH_SIZE % TOTAL_PER_STEP ))" -ne 0 ]; then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be a positive multiple of MICRO_BATCH_SIZE*NPROC_PER_NODE*NODES=${TOTAL_PER_STEP}" >&2
  exit 1
fi
export GRAD_ACC_STEPS=$(( GLOBAL_BATCH_SIZE / TOTAL_PER_STEP ))
echo "Batch sizing: GBS=${GLOBAL_BATCH_SIZE}, MBS=${MICRO_BATCH_SIZE}, NPROC=${NPROC_PER_NODE}, NODES=${SLURM_JOB_NUM_NODES}, GRAD_ACC=${GRAD_ACC_STEPS}"

export WANDB_RUN_NAME="${MODEL_SIZE}_${DIST_STRATEGY}_${BATCH_TYPE_TAG}_gbs${GLOBAL_BATCH_SIZE}_mbs${MICRO_BATCH_SIZE}_ga${GRAD_ACC_STEPS}_${PRECISION_TAG}_nodes_${SLURM_JOB_NUM_NODES}_${CLUSTER_NAME}_chain_${CHAIN_ID}"

# Mounts
RESULTS_DIR="${BASE_DIR}/results/${WANDB_RUN_NAME}"
CKPT_DIR="${BASE_DIR}/checkpoints/${WANDB_RUN_NAME}"

mkdir -p "${RESULTS_DIR}" "${CKPT_DIR}"

MOUNTS="${DATA_DIR}:${CODE_MOUNT}/data,${RESULTS_DIR}:${CODE_MOUNT}/results,${CKPT_DIR}:${CODE_MOUNT}/checkpoints"

# Resolve head node on the host (scontrol is not available inside the container).
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=29500


read -r -d '' COMMAND <<'OUTER_EOF' || true
set -euxo pipefail

echo "========================================="
echo "CodonFM ${CONFIG_NAME} - STRATEGY: ${DIST_STRATEGY} - PRECISION: ${PRECISION_TAG} - CLUSTER: ${CLUSTER_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "========================================="

export USE_XFORMERS=${USE_XFORMERS:-0}
if [ "${USE_XFORMERS}" = "1" ]; then
  echo "Using Xformers"
else
  echo "Using PyTorch SDPA attention"
fi

# cuDNN fused-attn sub-backend 1 OOMs on Blackwell (sm_103) with THD+padding (TE 2.12 / cuDNN 9.19); force flash-attn varlen.
if [ "${USE_SEQUENCE_PACKING}" = "True" ]; then
  export NVTE_FUSED_ATTN=0
  EXTRA_ARGS="--collate_fn thd --attn_input_format thd"
else
  EXTRA_ARGS="--collate_fn bshd --attn_input_format bshd"
fi

# Pick training script based on distributed strategy.
case "${DIST_STRATEGY}" in
  fsdp)
    EXTRA_ARGS="${EXTRA_ARGS} --enable_fsdp"
    ;;
  ddp)
    EXTRA_ARGS="${EXTRA_ARGS}"
    ;;
  *)
    echo "DIST_STRATEGY must be 'fsdp' or 'ddp', got '${DIST_STRATEGY}'" >&2
    exit 1
    ;;
esac

if [ "${PRECISION}" = "bf16-mixed" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --bf16"
fi

if [ "${USE_TRANSFORMER_ENGINE}" = "1" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --use_transformer_engine"
fi

torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --rdzv_id=${SLURM_JOB_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --nnodes=${SLURM_JOB_NUM_NODES} \
  --node-rank=${SLURM_NODEID} \
  -m src.runner pretrain \
  --exp_name ${WANDB_RUN_NAME} \
  --model_name ${CONFIG_NAME} \
  --data_path /workspace/bionemo/data/processed_unfiltered/ \
  --process_item mlm_memmap \
  --dataset_name CodonMemmapDataset \
  --lr ${LEARNING_RATE} \
  --num_gpus ${NPROC_PER_NODE} \
  --num_nodes ${SLURM_JOB_NUM_NODES} \
  --train_batch_size ${MICRO_BATCH_SIZE} \
  --val_batch_size ${MICRO_BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  ${EXTRA_ARGS} \
  --split_name_prefix nopathogen \
  --taxid_exclusion_file /workspace/bionemo/data/taxids_to_remove.json \
  --enable_wandb \
  --project_name ${WANDB_PROJECT} \
  --entity clara-discovery \
  --gradient_accumulation_steps ${GRAD_ACC_STEPS} \
  --max_steps ${NUM_TRAIN_STEPS} \
  --log_every_n_steps ${LOGGER_FREQUENCY}

echo "========================================="
echo "Training complete!"
echo "========================================="
OUTER_EOF

# Inject environment variables into the command.
COMMAND="export DIST_STRATEGY=\"${DIST_STRATEGY}\"; ${COMMAND}"
COMMAND="export PRECISION_TAG=\"${PRECISION_TAG}\"; ${COMMAND}"
COMMAND="export CLUSTER_NAME=\"${CLUSTER_NAME}\"; ${COMMAND}"
COMMAND="export NPROC_PER_NODE=\"${NPROC_PER_NODE}\"; ${COMMAND}"
COMMAND="export CONFIG_NAME=\"${CONFIG_NAME}\"; ${COMMAND}"
COMMAND="export LOGGER_FREQUENCY=\"${LOGGER_FREQUENCY}\"; ${COMMAND}"
COMMAND="export NUM_TRAIN_STEPS=\"${NUM_TRAIN_STEPS}\"; ${COMMAND}"
COMMAND="export GLOBAL_BATCH_SIZE=\"${GLOBAL_BATCH_SIZE}\"; ${COMMAND}"
COMMAND="export MICRO_BATCH_SIZE=\"${MICRO_BATCH_SIZE}\"; ${COMMAND}"
COMMAND="export GRAD_ACC_STEPS=\"${GRAD_ACC_STEPS}\"; ${COMMAND}"
COMMAND="export LEARNING_RATE=\"${LEARNING_RATE}\"; ${COMMAND}"
COMMAND="export NUM_WORKERS=\"${NUM_WORKERS}\"; ${COMMAND}"
COMMAND="export USE_SEQUENCE_PACKING=\"${USE_SEQUENCE_PACKING}\"; ${COMMAND}"
COMMAND="export PRECISION=\"${PRECISION}\"; ${COMMAND}"
COMMAND="export WANDB_RUN_NAME=\"${WANDB_RUN_NAME}\"; ${COMMAND}"
COMMAND="export WANDB_PROJECT=\"${WANDB_PROJECT}\"; ${COMMAND}"
COMMAND="export USE_XFORMERS=\"${USE_XFORMERS}\"; ${COMMAND}"
COMMAND="export MASTER_ADDR=\"${MASTER_ADDR}\"; ${COMMAND}"
COMMAND="export MASTER_PORT=\"${MASTER_PORT}\"; ${COMMAND}"
COMMAND="export USE_TRANSFORMER_ENGINE=\"${USE_TRANSFORMER_ENGINE}\"; ${COMMAND}"
COMMAND="export WANDB_API_KEY=\"${WANDB_API_KEY}\"; ${COMMAND}"
COMMAND="export HUGGING_FACE_HUB_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; ${COMMAND}"
COMMAND="export HF_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; ${COMMAND}"


echo "Launching: ${WANDB_RUN_NAME}"

# AUTO-CHAIN: resubmit on timeout.
trap '
    rc=$?
    if [ "$rc" -eq 143 ] || [ "$rc" -eq 137 ]; then
      echo "Timed out (rc=$rc) — resubmitting chain ${CHAIN_ID}."
      sbatch --dependency=singleton \
             --export=ALL,CHAIN_ID="${CHAIN_ID}" \
             "${BASH_SOURCE[0]}"
    elif [ "$rc" -eq 0 ]; then
      echo "Training finished cleanly — chain ${CHAIN_ID} ends."
    else
      echo "Real error (rc=$rc) — chain ${CHAIN_ID} ends so you can investigate."
    fi
  ' EXIT

srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
