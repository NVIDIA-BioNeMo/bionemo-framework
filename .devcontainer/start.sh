#!/usr/bin/env bash
set -euo pipefail

# Launch the devcontainer headlessly (no VS Code / Cursor required).
# All mounts, runArgs, containerEnv, and lifecycle commands are parsed at
# runtime from devcontainer.json so the two definitions never drift apart.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEVCONTAINER_JSON="${SCRIPT_DIR}/devcontainer.json"
CONTAINER_NAME="${BIONEMO_CONTAINER_NAME:-bionemo-devcontainer}"
IMAGE_NAME="${BIONEMO_IMAGE_NAME:-${CONTAINER_NAME}:latest}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

require_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "Error: '$1' is required but not found in PATH." >&2
        exit 1
    fi
}

require_cmd docker
require_cmd jq

# ---------------------------------------------------------------------------
# Parse devcontainer.json (strip JSON comments first)
# ---------------------------------------------------------------------------

# devcontainer.json may contain // comments which are not valid JSON.
strip_comments() {
    sed 's|//.*$||' "$1"
}

CONFIG="$(strip_comments "${DEVCONTAINER_JSON}")"

# ---------------------------------------------------------------------------
# 1. Run initializeCommand on the HOST
# ---------------------------------------------------------------------------

INIT_CMD="$(echo "${CONFIG}" | jq -r '.initializeCommand // empty')"
if [[ -n "${INIT_CMD}" ]]; then
    echo "==> Running initializeCommand on host: ${INIT_CMD}"
    (cd "${REPO_ROOT}" && bash -c "${INIT_CMD}")
fi

# ---------------------------------------------------------------------------
# 2. Build the image
# ---------------------------------------------------------------------------

DOCKERFILE="$(echo "${CONFIG}" | jq -r '.build.dockerfile // "Dockerfile"')"
DOCKERFILE_PATH="${SCRIPT_DIR}/${DOCKERFILE}"

echo "==> Building image ${IMAGE_NAME} from ${DOCKERFILE_PATH}"
docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# 3. Assemble docker-run flags from devcontainer.json
# ---------------------------------------------------------------------------

DOCKER_RUN_ARGS=()

# -- runArgs (e.g. --gpus=all, --shm-size=4g) --
while IFS= read -r arg; do
    [[ -n "${arg}" ]] && DOCKER_RUN_ARGS+=("${arg}")
done < <(echo "${CONFIG}" | jq -r '.runArgs[]? // empty')

# -- mounts (resolve ${localEnv:HOME} → $HOME) --
while IFS= read -r mount; do
    [[ -n "${mount}" ]] && DOCKER_RUN_ARGS+=("--mount" "${mount}")
done < <(echo "${CONFIG}" | jq -r '.mounts[]? // empty' | sed "s|\${localEnv:HOME}|${HOME}|g")

# -- containerEnv --
while IFS= read -r pair; do
    [[ -n "${pair}" ]] && DOCKER_RUN_ARGS+=("-e" "${pair}")
done < <(echo "${CONFIG}" | jq -r '.containerEnv // {} | to_entries[] | "\(.key)=\(.value)"')

# -- remoteUser --
REMOTE_USER="$(echo "${CONFIG}" | jq -r '.remoteUser // "root"')"

# -- workspace mount (bind the repo root into the container) --
WORKSPACE_FOLDER="$(echo "${CONFIG}" | jq -r '.workspaceFolder // "/workspaces/" + (.name // "workspace")')"
# Default to the standard devcontainer workspace layout
WORKSPACE_FOLDER="/workspaces/bionemo-framework"
DOCKER_RUN_ARGS+=("--mount" "source=${REPO_ROOT},target=${WORKSPACE_FOLDER},type=bind,consistency=cached")
DOCKER_RUN_ARGS+=("-w" "${WORKSPACE_FOLDER}")

# ---------------------------------------------------------------------------
# 4. Clean up any previous container with the same name
# ---------------------------------------------------------------------------

if docker container inspect "${CONTAINER_NAME}" &>/dev/null; then
    echo "==> Removing existing container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

# ---------------------------------------------------------------------------
# 5. Start the container (as root initially for UID/GID remapping)
# ---------------------------------------------------------------------------

POST_CREATE_CMD="$(echo "${CONFIG}" | jq -r '.postCreateCommand // empty')"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

echo "==> Starting container ${CONTAINER_NAME}"
docker run -d \
    --name "${CONTAINER_NAME}" \
    "${DOCKER_RUN_ARGS[@]}" \
    "${IMAGE_NAME}" \
    sleep infinity

# ---------------------------------------------------------------------------
# 6. Remap remoteUser UID/GID to match the host user
#    This replicates VS Code / Cursor's updateRemoteUserUID behavior so that
#    files created inside the container have the correct ownership on the host.
# ---------------------------------------------------------------------------

if [[ "${REMOTE_USER}" != "root" ]]; then
    echo "==> Remapping ${REMOTE_USER} to UID=${HOST_UID} GID=${HOST_GID}"
    docker exec --user root "${CONTAINER_NAME}" bash -c "
        CUR_GID=\$(id -g ${REMOTE_USER})
        CUR_UID=\$(id -u ${REMOTE_USER})
        if [ \"\${CUR_GID}\" != \"${HOST_GID}\" ]; then
            groupmod -g ${HOST_GID} ${REMOTE_USER} 2>/dev/null || \
                groupmod -g ${HOST_GID} \$(id -gn ${REMOTE_USER})
        fi
        if [ \"\${CUR_UID}\" != \"${HOST_UID}\" ]; then
            usermod -u ${HOST_UID} -g ${HOST_GID} ${REMOTE_USER}
        fi
        chown -R ${HOST_UID}:${HOST_GID} /home/${REMOTE_USER} 2>/dev/null || true
    "
fi

# ---------------------------------------------------------------------------
# 7. Run postCreateCommand INSIDE the container
# ---------------------------------------------------------------------------

if [[ -n "${POST_CREATE_CMD}" ]]; then
    echo "==> Running postCreateCommand in container: ${POST_CREATE_CMD}"
    docker exec \
        --user "${REMOTE_USER}" \
        -w "${WORKSPACE_FOLDER}" \
        "${CONTAINER_NAME}" \
        bash -c "${POST_CREATE_CMD}"
fi

# ---------------------------------------------------------------------------
# 8. Done – run a command or drop the user into an interactive shell
#
#    start.sh                → interactive bash (default)
#    start.sh CMD [ARGS...]  → run CMD inside the container and return its
#                              exit code (non-interactive, no exec)
# ---------------------------------------------------------------------------

echo ""
echo "==> Container ${CONTAINER_NAME} is running."

if [ $# -gt 0 ]; then
    echo "==> Running command: $*"
    docker exec \
        --user "${REMOTE_USER}" \
        -w "${WORKSPACE_FOLDER}" \
        "${CONTAINER_NAME}" \
        "$@"
else
    echo "    Attach with:  docker exec -it ${CONTAINER_NAME} bash"
    echo "    Stop with:    docker rm -f ${CONTAINER_NAME}"
    echo ""
    echo "==> Attaching interactive shell…"
    exec docker exec -it \
        --user "${REMOTE_USER}" \
        -w "${WORKSPACE_FOLDER}" \
        "${CONTAINER_NAME}" \
        bash
fi
