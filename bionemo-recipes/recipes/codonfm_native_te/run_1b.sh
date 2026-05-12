#!/usr/bin/env bash
set -euo pipefail

export CPATH=/usr/local/cuda/include
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Run config
export CONFIG_NAME=encodon_1b
export NPROC_PER_NODE=8
export DIST_STRATEGY=ddp  # fsdp or ddp

# Training
export NUM_TRAIN_STEPS=100
export MICRO_BATCH_SIZE=31
export NUM_WORKERS=1
export USE_SEQUENCE_PACKING=True
export USE_FP32_MASTER_WEIGHTS=True
export NUM_WARMUP_STEPS=500

# Logging / W&B
export LOGGER_FREQUENCY=10
export WANDB_API_KEY=""
export WANDB_PROJECT=codon-fm-low-precision

# Checkpointing
export SAVE_FINAL_MODEL=False
export SAVE_EVERY_N_STEPS=100000
export CKPT_DIR=/tmp
export RESUME_FROM_CHECKPOINT=False

# Hydra
export HYDRA_RUN_DIR=1b_test

# Quantization / FP8
export QUANT_STATS_ENABLED=False
export FP8_ENABLED=True
export FP8_RECIPE=transformer_engine.common.recipe.MXFP8BlockScaling
export FP8_FORMAT=E4M3

# Data
export DATASET_DATA_PATH=/data/balvisio/codonfm/reference-dataset/codonfm/processed_unfiltered/

# Derived: build wandb run name from model size, batch size, and precision recipe
MODEL_SIZE="${CONFIG_NAME##*_}"
if [ "${FP8_ENABLED}" = "True" ]; then
  RECIPE_SHORT="${FP8_RECIPE##*.}"
  RECIPE_SHORT="${RECIPE_SHORT%BlockScaling}"
  RECIPE_SHORT="${RECIPE_SHORT%Scaling}"
  PRECISION_TAG="${RECIPE_SHORT,,}_${FP8_FORMAT,,}"
else
  PRECISION_TAG="bf16"
fi
export WANDB_RUN_NAME="${MODEL_SIZE}_${DIST_STRATEGY}_bs${MICRO_BATCH_SIZE}_${PRECISION_TAG}"

# Pick training script based on distributed strategy.
# DDP can't emulate FSDP's fp32-master / bf16-param split, so force fp32 master weights off.
case "${DIST_STRATEGY}" in
  fsdp)
    TRAIN_SCRIPT=train_fsdp2.py
    ;;
  ddp)
    TRAIN_SCRIPT=train_ddp.py
    if [ "${USE_FP32_MASTER_WEIGHTS}" = "True" ]; then
      echo "DIST_STRATEGY=ddp: overriding USE_FP32_MASTER_WEIGHTS=True -> False" >&2
      export USE_FP32_MASTER_WEIGHTS=False
    fi
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
  dataset.num_workers=${NUM_WORKERS} \
  dataset.data_path=${DATASET_DATA_PATH} \
  use_sequence_packing=${USE_SEQUENCE_PACKING} \
  use_fp32_master_weights=${USE_FP32_MASTER_WEIGHTS} \
  lr_scheduler_kwargs.num_warmup_steps=${NUM_WARMUP_STEPS} \
  wandb_init_args.name=${WANDB_RUN_NAME} \
  wandb_init_args.project=${WANDB_PROJECT} \
  checkpoint.save_final_model=${SAVE_FINAL_MODEL} \
  checkpoint.save_every_n_steps=${SAVE_EVERY_N_STEPS} \
  checkpoint.ckpt_dir=${CKPT_DIR} \
  checkpoint.resume_from_checkpoint=${RESUME_FROM_CHECKPOINT} \
  hydra.run.dir=${HYDRA_RUN_DIR} \
  fp8_config.enabled=${FP8_ENABLED} \
  fp8_config.fp8_recipe=${FP8_RECIPE} \
  fp8_config.fp8_format=${FP8_FORMAT} \
  dataset.pad_to_multiple_of=32
