#!/bin/bash
set -ex

# - hyperparameters
learning_rate=7.5e-5
num_nodes=16
num_gpus=8
train_batch_size=8
val_batch_size=8
effective_batch_size=$((train_batch_size * num_gpus * num_nodes))
num_workers=12

exp_name="encodon_600m_latest_${learning_rate}_${effective_batch_size}_nopathogen"

# Note if you would like to use WandB please add --enable_wandb, --project_name and --entity.

# - run
python -m src.runner pretrain \
    --exp_name "$exp_name" \
    --model_name encodon_600m \
    --data_path /data/ncbi/processed_unfiltered/ \
    --process_item mlm_memmap \
    --dataset_name CodonMemmapDataset \
    --lr $learning_rate \
    --num_gpus $num_gpus \
    --num_nodes $num_nodes \
    --train_batch_size $train_batch_size \
    --val_batch_size $val_batch_size \
    --num_workers $num_workers \
    --collate_fn thd \
    --attn_input_format thd \
    --use_transformer_engine \
    --bf16 \
    --split_name_prefix nopathogen \
    --checkpoints_dir results/${exp_name}/checkpoints/ \
    --out_dir results/${exp_name}/ \
