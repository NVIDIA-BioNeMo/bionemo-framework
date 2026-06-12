#!/bin/bash

# This is an example how to finetune the NV-CodonFM-Encodon-TE-80M-v1 model on the Sanofi eval datasets.
# Disclaimer: This does not guarantee the best performance but is more for educational purpose and small-scale testing.

# --- Helper functions --- #
function print_help {
    echo "Usage: $0 <checkpoint_path> <data_path>"
    echo "NOTE: This script must be run from the root directory of the project."
    echo "Example : $0 /path/to/checkpoint /data/validation/processed/mRFP_Expression.csv"
}


CHECKPOINT_PATH=$1 # full path to the checkpoint, can be downloaded from https://huggingface.co/nvidia/NV-CodonFM-Encodon-TE-80M-v1
DATA_PATH=$2 # can be downloaded with ../../data_scripts/download_preprocess_codonbert_bench.py --dataset mRFP_Expression.csv --output-dir $DATA_PATH
MODEL_NAME="encodon_80m"
USE_TRANSFORMER_ENGINE=${3:-"true"}
FINETUNE_STRATEGY=${4:-"lora"} # choice between "full", "lora", "head_only_random", "head_only_pretrained".
USE_CROSS_ATTENTION="true"
MAX_STEPS=100000

# Defaults
NUM_NODES=1
NUM_GPUS=1
LR="1e-5"
TRAIN_BATCH_SIZE="8"
VAL_BATCH_SIZE=$TRAIN_BATCH_SIZE
GRAD_ACC_BATCHES=8

EXP_NAME="sanofi_mRFP_Expression_Encodon_80m_${FINETUNE_STRATEGY}"
if [[ "$USE_TRANSFORMER_ENGINE" == "true" ]]; then
    EXP_NAME="${EXP_NAME}_TE"
fi

CMD=(
    "env" "CODON_FM_TE_IMPL=$CODON_FM_TE_IMPL" "python" "-m" "src.runner" "finetune"
    "--exp_name" "$EXP_NAME"
    "--num_nodes" "$NUM_NODES"
    "--num_gpus" "$NUM_GPUS"
    "--seed" "42"
    "--lr" "$LR"
    "--data_path" "$DATA_PATH"
    "--process_item" "codon_sequence"
    "--dataset_name" "CodonBertDataset"
    "--train_batch_size" "$TRAIN_BATCH_SIZE"
    "--val_batch_size" "$VAL_BATCH_SIZE"
    "--gradient_accumulation_steps" "$GRAD_ACC_BATCHES"
    "--max_steps" "$MAX_STEPS"
    "--pretrained_ckpt_path" "$CHECKPOINT_PATH"
    "--model_name" "$MODEL_NAME"
    "--loss_type" "regression"
    "--num_classes" "2"
    "--label_col" "value"
    "--check_val_every_n_epoch" "1"
    "--log_every_n_steps" "1"
    "--checkpoint_every_n_train_steps" "5"
    "--bf16"
)
if [[ "$USE_TRANSFORMER_ENGINE" == "true" ]]; then
    CMD+=("--use_transformer_engine")
fi

# Add cross-attention parameters if enabled
if [[ "$USE_CROSS_ATTENTION" == "true" ]]; then
    CMD+=("--use_downstream_head" "--cross_attention_hidden_dim" "512" "--cross_attention_num_heads" "8")
fi

case $FINETUNE_STRATEGY in
    "lora")
        # Note: lora_dropout must be 0.0 when using TransformerEngine with LayerNormLinear targets
        LORA_DROPOUT="0.0"
        if [[ "$USE_TRANSFORMER_ENGINE" != "true" ]]; then
            LORA_DROPOUT="0.1"
        fi
        CMD+=("--finetune_strategy" "lora" "--lora" "--lora_alpha" "32.0" "--lora_r" "32" "--lora_dropout" "$LORA_DROPOUT")
        ;;
    "head_only_random")
        CMD+=("--finetune_strategy" "head_only_random")
        ;;
    "head_only_pretrained")
        CMD+=("--finetune_strategy" "head_only_pretrained")
        ;;
    "full")
        CMD+=("--finetune_strategy" "full")
        ;;
    *)
        echo "Invalid finetune_strategy: $FINETUNE_STRATEGY"
        echo "Supported strategies: lora, head_only_random, head_only_pretrained, full"
        exit 1
        ;;
esac

echo "Executing: ${CMD[@]}"
"${CMD[@]}"
