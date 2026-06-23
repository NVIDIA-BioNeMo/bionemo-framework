#!/bin/bash
# Launch the Evo2 SAE inference engine. One engine, four modes:
#
#   ./launch_inference.sh serve                       # live HTTP server on :8001 (viz backend)
#   ./launch_inference.sh encode  --sequence ATGC...  # annotate ONE sequence -> top features
#   ./launch_inference.sh batch   --fasta in.fa --out out.parquet   # MANY sequences -> parquet
#   ./launch_inference.sh generate --prompt ATGC... --clamp 29244:300  # steer + generate DNA
#
# Steering loop: `encode` a sequence to find an active feature id, then
# `generate --clamp ID:STRENGTH` (strength ~2-3x the feature's max_activation; repeat --clamp).
#
# Config via env. Required: EVO2_CKPT_DIR, SAE_CKPT_PATH. Optional (have defaults):
# FEATURE_ANNOTATIONS, EMBEDDING_LAYER (26), DEVICE, PORT, CUDA_VISIBLE_DEVICES.
#
# Requires the evo2_megatron recipe venv (provides bionemo.evo2 + megatron).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RECIPE_DIR="$(cd "$HERE/.." && pwd)"  # recipes/evo2 — so the evo2_sae package imports

# Required (no hardcoded defaults — supply your own paths via env):
VENV="${VENV:?Set VENV to the evo2_megatron recipe .venv (provides bionemo.evo2 + megatron)}"
export EVO2_CKPT_DIR="${EVO2_CKPT_DIR:?Set EVO2_CKPT_DIR to an Evo2 MBridge checkpoint directory}"
export SAE_CKPT_PATH="${SAE_CKPT_PATH:?Set SAE_CKPT_PATH to a trained SAE checkpoint (.pt)}"
# Optional: feature-label parquet (empty = features are unlabeled). Layer defaults to 26.
export FEATURE_ANNOTATIONS="${FEATURE_ANNOTATIONS:-}"
export EMBEDDING_LAYER="${EMBEDDING_LAYER:-26}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: evo2_megatron venv not found at $VENV (build it with the recipe's .ci_build.sh)" >&2
  exit 1
fi

source "$VENV/bin/activate"
cd "$RECIPE_DIR"
export PYTHONPATH="$RECIPE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m evo2_sae.cli "$@"
