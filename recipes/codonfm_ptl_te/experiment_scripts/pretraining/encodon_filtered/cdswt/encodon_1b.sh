#!/bin/bash
set -ex

# - hyperparameters
learning_rate=5e-5
num_nodes=48
num_gpus=8
train_batch_size=4
val_batch_size=4
effective_batch_size=$((train_batch_size * num_gpus * num_nodes))
num_workers=12

exp_name="encodon_1b_cdswt_${learning_rate}_${effective_batch_size}_nopathogen"

# codon_weights_file can be obtained by running data_scripts/check_codon_frequency.py

python -m src.runner pretrain \
    --exp_name "$exp_name" \
    --model_name encodon_1b \
    --data_path /data/ncbi/processed_unfiltered/ \
    --process_item mlm_memmap \
    --dataset_name CodonMemmapDataset \
    --lr $learning_rate \
    --num_gpus $num_gpus \
    --num_nodes $num_nodes \
    --train_batch_size $train_batch_size \
    --val_batch_size $val_batch_size \
    --collate_fn thd \
    --attn_input_format thd \
    --use_transformer_engine \
    --num_workers $num_workers \
    --bf16 \
    --codon_weights_file /data/ncbi/codon_counts_nopathogen.json \
    --split_name_prefix nopathogen \
    --checkpoints_dir results/checkpoints/${exp_name} \
    --enable_wandb
