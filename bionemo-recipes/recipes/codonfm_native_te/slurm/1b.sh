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
# Codon 1B
# ============================================================================

BASE_DIR=""
CONTAINER=""
DATA_DIR="${BASE_DIR}/data"
CODE_MOUNT="/workspace/bionemo"


: "${WANDB_API_KEY:?Set WANDB_API_KEY in ~/.bash_profile}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN in ~/.bash_profile}"
: "${CLUSTER_NAME:?Set CLUSTER_NAME in ~/.bash_profile}"

export GLOBAL_BATCH_SIZE=1536
export MICRO_BATCH_SIZE=4

# Experiment parameters
export CONFIG_NAME=encodon_1b
export NPROC_PER_NODE=8
export DIST_STRATEGY=ddp  # fsdp or ddp

# Training
export NUM_TRAIN_STEPS=1000
export LEARNING_RATE=7.5e-5
export NUM_WORKERS=1
export USE_SEQUENCE_PACKING=False
# Precision mode: one of fp32, bf16, bf16-mixed. bf16-mixed matches the reference codonfm `--bf16`.
export PRECISION=bf16-mixed
# Only used for FSDP2 + bf16-mixed. One of fp32, bf16.
export GRAD_REDUCE_TYPE=fp32
export NUM_WARMUP_STEPS=50

# Logging / W&B
export LOGGER_FREQUENCY=10
export WANDB_PROJECT=

# Checkpointing
export SAVE_FINAL_MODEL=True
export SAVE_EVERY_N_STEPS=100000
export RESUME_FROM_CHECKPOINT=True

# Hydra
export HYDRA_RUN_DIR=1b_test

# Quantization / FP8
export QUANT_STATS_ENABLED=False
export FP8_ENABLED=False
export FP8_RECIPE=transformer_engine.common.recipe.MXFP8BlockScaling
export FP8_FORMAT=E4M3

# Derived: build wandb run name from model size, batch size, and precision recipe
MODEL_SIZE="${CONFIG_NAME##*_}"
if [ "${FP8_ENABLED}" = "True" ]; then
  RECIPE_SHORT="${FP8_RECIPE##*.}"
  RECIPE_SHORT="${RECIPE_SHORT%BlockScaling}"
  RECIPE_SHORT="${RECIPE_SHORT%Scaling}"
  PRECISION_TAG="${PRECISION}_${RECIPE_SHORT,,}_${FP8_FORMAT,,}"
else
  PRECISION_TAG="${PRECISION}"
fi

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

export WANDB_RUN_NAME="${MODEL_SIZE}_${DIST_STRATEGY}_${BATCH_TYPE_TAG}_gbs${GLOBAL_BATCH_SIZE}_mbs${MICRO_BATCH_SIZE}_ga${GRAD_ACC_STEPS}_${PRECISION_TAG}_nodes_${SLURM_JOB_NUM_NODES}_${CLUSTER_NAME}"

# Mounts
RESULTS_DIR="${BASE_DIR}/results/${WANDB_RUN_NAME}"
CKPT_DIR="${BASE_DIR}/checkpoints/${WANDB_RUN_NAME}"

mkdir -p "${RESULTS_DIR}" "${CKPT_DIR}"

MOUNTS="${DATA_DIR}:${CODE_MOUNT}/data,${RESULTS_DIR}:${CODE_MOUNT}/results,${CKPT_DIR}:${CODE_MOUNT}/checkpoints"


read -r -d '' COMMAND <<'OUTER_EOF' || true
set -euxo pipefail

echo "========================================="
echo "CodonFM ${CONFIG_NAME} - STRATEGY: ${DIST_STRATEGY} - PRECISION: ${PRECISION_TAG} - CLUSTER: ${CLUSTER_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "========================================="

# Pick training script based on distributed strategy.
case "${DIST_STRATEGY}" in
  fsdp)
    TRAIN_SCRIPT=train_fsdp2.py
    ;;
  ddp)
    TRAIN_SCRIPT=train_ddp.py
    ;;
  *)
    echo "DIST_STRATEGY must be 'fsdp' or 'ddp', got '${DIST_STRATEGY}'" >&2
    exit 1
    ;;
esac

torchrun --nproc_per_node=${NPROC_PER_NODE} ${TRAIN_SCRIPT} \
  --config-name ${CONFIG_NAME} \
  quant_stats_config.enabled=${QUANT_STATS_ENABLED} \
  logger.frequency=${LOGGER_FREQUENCY} \
  num_train_steps=${NUM_TRAIN_STEPS} \
  dataset.micro_batch_size=${MICRO_BATCH_SIZE} \
  grad_acc_steps=${GRAD_ACC_STEPS} \
  adamw_kwargs.lr=${LEARNING_RATE} \
  dataset.num_workers=${NUM_WORKERS} \
  dataset.data_path=/workspace/bionemo/data/processed_unfiltered/ \
  use_sequence_packing=${USE_SEQUENCE_PACKING} \
  precision=${PRECISION} \
  grad_reduce_type=${GRAD_REDUCE_TYPE} \
  lr_scheduler_kwargs.num_warmup_steps=${NUM_WARMUP_STEPS} \
  wandb_init_args.name=${WANDB_RUN_NAME} \
  +wandb_init_args.id=${WANDB_RUN_NAME} \
  +wandb_init_args.project=${WANDB_PROJECT} \
  checkpoint.save_final_model=${SAVE_FINAL_MODEL} \
  checkpoint.save_every_n_steps=${SAVE_EVERY_N_STEPS} \
  checkpoint.ckpt_dir=/workspace/bionemo/checkpoints \
  checkpoint.resume_from_checkpoint=${RESUME_FROM_CHECKPOINT} \
  hydra.run.dir=${HYDRA_RUN_DIR} \
  fp8_config.enabled=${FP8_ENABLED} \
  fp8_config.fp8_recipe=${FP8_RECIPE} \
  fp8_config.fp8_format=${FP8_FORMAT} \
  +dataset.pad_to_multiple_of=32

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
COMMAND="export QUANT_STATS_ENABLED=\"${QUANT_STATS_ENABLED}\"; ${COMMAND}"
COMMAND="export LOGGER_FREQUENCY=\"${LOGGER_FREQUENCY}\"; ${COMMAND}"
COMMAND="export NUM_TRAIN_STEPS=\"${NUM_TRAIN_STEPS}\"; ${COMMAND}"
COMMAND="export GLOBAL_BATCH_SIZE=\"${GLOBAL_BATCH_SIZE}\"; ${COMMAND}"
COMMAND="export MICRO_BATCH_SIZE=\"${MICRO_BATCH_SIZE}\"; ${COMMAND}"
COMMAND="export GRAD_ACC_STEPS=\"${GRAD_ACC_STEPS}\"; ${COMMAND}"
COMMAND="export LEARNING_RATE=\"${LEARNING_RATE}\"; ${COMMAND}"
COMMAND="export NUM_WORKERS=\"${NUM_WORKERS}\"; ${COMMAND}"
COMMAND="export USE_SEQUENCE_PACKING=\"${USE_SEQUENCE_PACKING}\"; ${COMMAND}"
COMMAND="export PRECISION=\"${PRECISION}\"; ${COMMAND}"
COMMAND="export GRAD_REDUCE_TYPE=\"${GRAD_REDUCE_TYPE}\"; ${COMMAND}"
COMMAND="export NUM_WARMUP_STEPS=\"${NUM_WARMUP_STEPS}\"; ${COMMAND}"
COMMAND="export WANDB_RUN_NAME=\"${WANDB_RUN_NAME}\"; ${COMMAND}"
COMMAND="export WANDB_PROJECT=\"${WANDB_PROJECT}\"; ${COMMAND}"
COMMAND="export SAVE_FINAL_MODEL=\"${SAVE_FINAL_MODEL}\"; ${COMMAND}"
COMMAND="export SAVE_EVERY_N_STEPS=\"${SAVE_EVERY_N_STEPS}\"; ${COMMAND}"
COMMAND="export RESUME_FROM_CHECKPOINT=\"${RESUME_FROM_CHECKPOINT}\"; ${COMMAND}"
COMMAND="export HYDRA_RUN_DIR=\"${HYDRA_RUN_DIR}\"; ${COMMAND}"
COMMAND="export FP8_ENABLED=\"${FP8_ENABLED}\"; ${COMMAND}"
COMMAND="export FP8_RECIPE=\"${FP8_RECIPE}\"; ${COMMAND}"
COMMAND="export FP8_FORMAT=\"${FP8_FORMAT}\"; ${COMMAND}"

COMMAND="export WANDB_API_KEY=\"${WANDB_API_KEY}\"; ${COMMAND}"
COMMAND="export HUGGING_FACE_HUB_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; ${COMMAND}"
COMMAND="export HF_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; ${COMMAND}"

echo "Launching: ${WANDB_RUN_NAME}"

# AUTO-CHAIN: resubmit on timeout.
trap '
    rc=$?
    if [ "$rc" -eq 143 ] || [ "$rc" -eq 137 ]; then
      echo "Killed by signal (rc=$rc) — assuming SLURM timeout, resubmitting..."
      sbatch --dependency=singleton "${BASH_SOURCE[0]}"
    elif [ "$rc" -eq 0 ]; then
      echo "Clean exit — training finished, NOT resubmitting."
    else
      echo "Error exit (rc=$rc) — NOT resubmitting; investigate ${RESULTS_DIR}"
    fi
  ' EXIT

srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
