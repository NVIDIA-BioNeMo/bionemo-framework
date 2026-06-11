#!/bin/bash

set -e

# ----------------- Configuration -----------------
# Default paths for data and checkpoints.
# You can override these with command-line arguments.
#
# Example usage:
# ./run_dev.sh --data-dir /path/to/your/data --checkpoints-dir /path/to/your/checkpoints

DATA_DIR="/data/codonfm"
CHECKPOINTS_DIR="/data/codonfm/checkpoints"

# Parse command-line arguments for paths
while [[ "$#" -gt 0 ]]; do  # noqa: E702
    case $1 in
        --data-dir) DATA_DIR="$2"; shift ;;  # noqa: E702, E703
        --checkpoints-dir) CHECKPOINTS_DIR="$2"; shift ;;  # noqa: E702, E703
        -h|--help)
            echo "Usage: $0 [--data-dir <path>] [--checkpoints-dir <path>]"
            exit 0
            ;;  # noqa: E703
        *) echo "Unknown parameter passed: $1"; exit 1 ;;  # noqa: E702, E703
    esac
    shift
done

IMAGE_NAME="codon-fm-dev"
CONTAINER_NAME="codon-fm-dev-container"

echo "Building the development docker image..."
# Build the development image using the 'development' stage in the Dockerfile
docker build -t ${IMAGE_NAME} \
  --target development \
  --build-arg USERNAME=$(whoami) \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -f Dockerfile .

# Check if a container with the same name is already running and stop it
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then  # noqa: E702
    echo "Stopping and removing existing container..."
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

echo "Launching the development container with the following mounts:"
echo "  - Host: $(pwd) -> Container: /workspace"
echo "  - Host: ${DATA_DIR} -> Container: /data"
echo "  - Host: ${CHECKPOINTS_DIR} -> Container: /data/checkpoints"
echo "  - Host: ~/.ssh -> Container: /home/$(whoami)/.ssh (read-only)"

# Launch the container with GPU support, mounting volumes, and exposing port 8888
docker run -it --rm \
  --gpus all \
  --ipc=host \
  --net=host \
  --hostname localhost \
  -v $(pwd):/workspace \
  -v "${DATA_DIR}":/data \
  -v "${CHECKPOINTS_DIR}":/data/checkpoints \
  -v /etc/group:/etc/group:ro \
  -v /run/sshd:/run/sshd \
  -v ~/.ssh:/home/$(whoami)/.ssh:ro \
  -p 8888:8888 \
  --name ${CONTAINER_NAME} \
  ${IMAGE_NAME} \
  bash
