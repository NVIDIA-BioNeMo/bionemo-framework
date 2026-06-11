#!/bin/bash
set -e

# CodonFM Encodon-1B SAE Pipeline

MODEL_PATH=checkpoints/NV-CodonFM-Encodon-TE-Cdwt-1B-v1/model.safetensors
CSV_PATH=/data/jwilber/codonfm/data/sample_108k.csv
LAYER=16
NUM_SEQUENCES=10000
OUTPUT_DIR=./outputs/1b_layer16

echo "============================================================"
echo "STEP 1: Extract activations from Encodon-1B"
echo "============================================================"

torchrun --nproc_per_node=4 scripts/extract.py \
    --csv-path $CSV_PATH \
    --model-path $MODEL_PATH \
    --layer $LAYER \
    --num-sequences $NUM_SEQUENCES \
    --batch-size 8 \
    --context-length 2048 \
    --shard-size 100000 \
    --output .cache/activations/primates_${NUM_SEQUENCES}_1b_layer${LAYER}

echo ""
echo "============================================================"
echo "STEP 2: Train SAE on cached activations"
echo "============================================================"

torchrun --nproc_per_node=4 scripts/train.py \
    --cache-dir .cache/activations/primates_${NUM_SEQUENCES}_1b_layer${LAYER} \
    --model-path $MODEL_PATH \
    --layer $LAYER \
    --model-type topk \
    --expansion-factor 16 \
    --top-k 32 \
    --auxk 512 \
    --auxk-coef 0.03125 \
    --dead-tokens-threshold 500000 \
    --n-epochs 40 \
    --batch-size 4096 \
    --lr 3e-4 \
    --log-interval 50 \
    --dp-size 4 \
    --seed 42 \
    --wandb \
    --wandb-project sae_codonfm_recipe \
    --wandb-run-name "1b_layer${LAYER}_ef16_k32" \
    --output-dir ${OUTPUT_DIR} \
    --checkpoint-dir ${OUTPUT_DIR}/checkpoints

echo ""
echo "============================================================"
echo "STEP 3: Analyze features (vocab logits + codon annotations)"
echo "============================================================"

python scripts/analyze.py \
    --checkpoint ${OUTPUT_DIR}/checkpoints/checkpoint_final.pt \
    --model-path $MODEL_PATH \
    --csv-path $CSV_PATH \
    --layer $LAYER \
    --num-sequences $NUM_SEQUENCES \
    --batch-size 8 \
    --output-dir ${OUTPUT_DIR}/analysis \
    --dashboard-dir ${OUTPUT_DIR}/dashboard

echo ""
echo "============================================================"
echo "STEP 4: Build dashboard"
echo "============================================================"

python scripts/dashboard.py \
    --checkpoint ${OUTPUT_DIR}/checkpoints/checkpoint_final.pt \
    --model-path $MODEL_PATH \
    --csv-path $CSV_PATH \
    --layer $LAYER \
    --num-sequences $NUM_SEQUENCES \
    --batch-size 8 \
    --n-examples 6 \
    --umap-n-neighbors 15 \
    --umap-min-dist 0.1 \
    --hdbscan-min-cluster-size 20 \
    --output-dir ${OUTPUT_DIR}/dashboard

echo ""
echo "============================================================"
echo "DONE — Dashboard output: ${OUTPUT_DIR}/dashboard"
echo "============================================================"
