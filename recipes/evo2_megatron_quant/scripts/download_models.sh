#!/bin/bash
# ==============================================================================
# Download Pretrained BioNeMo Models from NGC
# ==============================================================================
#
# Downloads all 3 pretrained models used for quantization testing.
# Models are cached locally by bionemo.core.data.load and will not be
# re-downloaded if already present.
#
# Models:
#   1. ESM-2 8M       - Protein language model (Transformer, ~30MB)
#   2. Geneformer 10M  - Gene expression model (Transformer, ~40MB)
#   3. Evo2 7B         - DNA foundation model (Hyena/Mamba SSM, ~14GB)
#
# Prerequisites:
#   - Must run INSIDE the BioNeMo container
#   - NGC authentication configured
#
# Usage:
#   bash scripts/download_models.sh          # Download all models
#   bash scripts/download_models.sh esm2     # Download ESM-2 only
#   bash scripts/download_models.sh evo2     # Download Evo2 only
# ==============================================================================

set -euo pipefail

echo "================================================"
echo "  BioNeMo Model Downloader"
echo "================================================"

# Model tags on NGC
declare -A MODEL_TAGS=(
    ["esm2"]="esm2/8m:2.0"
    ["geneformer"]="geneformer/10M_241113:2.0"
    ["evo2"]="evo2/7b-8k:1.0"
)

# Model descriptions
declare -A MODEL_DESC=(
    ["esm2"]="ESM-2 8M (Protein Transformer, ~30MB)"
    ["geneformer"]="Geneformer 10M (Gene Expression Transformer, ~40MB)"
    ["evo2"]="Evo2 7B (DNA Hyena/Mamba SSM, ~14GB)"
)

# Determine which models to download
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("esm2" "geneformer" "evo2")
fi

# Validate requested names up front so a typo fails fast instead of silently
# skipping while the script still exits 0.
for model in "${MODELS[@]}"; do
    if [[ ! -v "MODEL_TAGS[$model]" ]]; then
        echo "  ❌ Unknown model: $model (available: esm2, geneformer, evo2)" >&2
        exit 1
    fi
done

# Download each model
for model in "${MODELS[@]}"; do
    tag="${MODEL_TAGS[$model]}"
    desc="${MODEL_DESC[$model]}"

    echo ""
    echo "  📥 Downloading: $desc"
    echo "     NGC tag: $tag"
    echo ""

    python -c "
from bionemo.core.data.load import load
path = load('${tag}', source='ngc')
print(f'  ✅ Downloaded to: {path}')
"
done

echo ""
echo "================================================"
echo "  All downloads complete!"
echo "================================================"
