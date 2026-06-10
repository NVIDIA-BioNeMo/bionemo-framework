#!/bin/bash
# Launch the Evo2 SAE inference engine. One engine, three modes:
#
#   ./launch_inference.sh serve                       # live HTTP server on :8001 (viz backend)
#   ./launch_inference.sh encode  --sequence ATGC...  # annotate ONE sequence -> top features
#   ./launch_inference.sh batch   --fasta in.fa --out out.parquet   # MANY sequences -> parquet
#
# Config via env (sensible defaults below): EVO2_CKPT_DIR, SAE_CKPT_PATH,
# FEATURE_ANNOTATIONS, EMBEDDING_LAYER, DEVICE, PORT, CUDA_VISIBLE_DEVICES.
#
# Requires the evo2_megatron recipe venv (provides bionemo.evo2 + megatron).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RECIPE_DIR="$(cd "$HERE/.." && pwd)"  # recipes/evo2 — so the evo2_sae package imports

VENV="${VENV:-/data/pbinder/bionemo-framework/bionemo-recipes/recipes/evo2_megatron/.venv}"
export EVO2_CKPT_DIR="${EVO2_CKPT_DIR:-/data/interp/evo2/checkpoints/evo2_7b_mbridge}"
export SAE_CKPT_PATH="${SAE_CKPT_PATH:-/data/interp/evo2/sae/v2_diverse/layer26_7B_ablate_normalize_input/checkpoints/checkpoint_final.pt}"
export FEATURE_ANNOTATIONS="${FEATURE_ANNOTATIONS:-/data/interp/evo2/sae_eval/dashboard_data/l26_7B_normalize/feature_metadata.parquet}"
export EMBEDDING_LAYER="${EMBEDDING_LAYER:-26}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: evo2_megatron venv not found at $VENV (build it with the recipe's .ci_build.sh)" >&2
  exit 1
fi

source "$VENV/bin/activate"
cd "$RECIPE_DIR"
export PYTHONPATH="$RECIPE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m evo2_sae.cli "$@"
