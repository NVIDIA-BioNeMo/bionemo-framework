#!/bin/bash
# ==============================================================================
# BioNeMo Framework Container Launcher
# ==============================================================================
#
# Starts the NVIDIA BioNeMo Framework Docker container with GPU support.
# All quantization work runs inside this container.
#
# Prerequisites:
#   - Docker with NVIDIA Container Toolkit
#   - NGC authentication (nvcr.io access)
#   - At least one NVIDIA GPU
#
# Usage:
#   bash docker/start_container.sh
# ==============================================================================

set -euo pipefail

# ---------- Configuration ----------
BIONEMO_TAG="${BIONEMO_TAG:-2.6.3}"                     # BioNeMo container version
LOCAL_WORKSPACE="${LOCAL_WORKSPACE:-$(pwd)/quantization}" # Mount to /workspace/bionemo_dev
DATA_DIR="${DATA_DIR:-$(pwd)/data}"                       # Mount to /data (model cache)
CONTAINER_NAME="${CONTAINER_NAME:-bionemo-quant}"         # Docker container name

# ---------- Create directories ----------
mkdir -p "${LOCAL_WORKSPACE}"
mkdir -p "${DATA_DIR}"

echo "================================================"
echo "  BioNeMo Framework Container"
echo "  Image:     nvcr.io/nvidia/clara/bionemo-framework:${BIONEMO_TAG}"
echo "  Workspace: ${LOCAL_WORKSPACE} → /workspace/bionemo_dev"
echo "  Data:      ${DATA_DIR} → /data"
echo "  Container: ${CONTAINER_NAME}"
echo "================================================"

# ---------- Launch container ----------
docker run --rm -it \
    --gpus all \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --net=host \
    -v "${LOCAL_WORKSPACE}":/workspace/bionemo_dev \
    -v "${DATA_DIR}":/data \
    -w /workspace/bionemo_dev \
    --name "${CONTAINER_NAME}" \
    nvcr.io/nvidia/clara/bionemo-framework:"${BIONEMO_TAG}" \
    /bin/bash
