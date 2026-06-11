#!/bin/bash
set -e

# CodonFM Encodon-1B SwissProt F1 Evaluation Pipeline
# Evaluates whether CodoNFM SAE features align with protein-level SwissProt annotations

MODEL_PATH=checkpoints/NV-CodonFM-Encodon-TE-Cdwt-1B-v1/model.safetensors
LAYER=16
OUTPUT_DIR=./outputs/1b_layer16

echo "============================================================"
echo "STEP 1: Download SwissProt proteins with CDS sequences"
echo "============================================================"

python scripts/download_codonfm_swissprot.py \
    --output-dir ./data/codonfm_swissprot \
    --max-proteins 8000 \
    --max-length 512 \
    --annotation-score 5 \
    --workers 8

echo ""
echo "============================================================"
echo "STEP 2: F1 evaluation against SwissProt annotations"
echo "============================================================"

python scripts/eval_swissprot_f1.py \
    --checkpoint ${OUTPUT_DIR}/checkpoints/checkpoint_final.pt \
    --model-path $MODEL_PATH \
    --layer $LAYER \
    --batch-size 8 \
    --context-length 2048 \
    --swissprot-tsv ./data/codonfm_swissprot/codonfm_swissprot.tsv.gz \
    --f1-max-proteins 8000 \
    --f1-min-positives 10 \
    --f1-threshold 0.3 \
    --normalization-n-proteins 2000 \
    --output-dir ${OUTPUT_DIR}/swissprot_eval

echo ""
echo "============================================================"
echo "DONE — SwissProt F1 results: ${OUTPUT_DIR}/swissprot_eval"
echo "============================================================"
