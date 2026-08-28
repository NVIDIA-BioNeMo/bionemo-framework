#!/bin/bash -x
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# FIXME: Fix for "No such file or directory: /workspace/TransformerEngine"
#  Remove once bug has been addressed in the nvidia/pytorch container.
rm -f /usr/local/lib/python*/dist-packages/transformer_engine-*.dist-info/direct_url.json
export UV_LOCK_TIMEOUT=900  # increase to 15 minutes (900 seconds), adjust as needed
export UV_LINK_MODE=copy
# Native training dependencies currently publish wheels through Python 3.12.
uv venv --python 3.12 --clear --system-site-packages

# 2. Activate the environment
source .venv/bin/activate

# 3. Let subquadratic-ops select its compatible warp-lang dependency.
# Pin warp-lang here in the future if subquadratic-ops has a verified compatibility issue.
# Driver compatibility note: CUDA 13.x normally requires an R580+ host driver. For a temporary
# 26.02/R550 test environment, constrain both subquadratic-ops-torch-cu13==0.2.1 and
# warp-lang<1.13.0 locally; 0.2.1 uses the legacy Warp API removed in Warp 1.13.

# 4. Install build requirements against the Transformer Engine already supplied by the image. An image without
# Transformer Engine intentionally produces an empty constraints file.
if ! pip freeze | grep -E '^transformer[-_]engine([= @]|$)' > pip-constraints.txt; then
    : > pip-constraints.txt
    echo "transformer-engine is not installed; continuing with an empty pip-constraints.txt" >&2
fi
uv pip install -c pip-constraints.txt -c security_constraints.txt -r build_requirements.txt --no-build-isolation

# 5. Install the recipe with all remaining dependencies, including test extras.
uv pip install -c pip-constraints.txt -c security_constraints.txt -e '.[test]' --no-build-isolation

# The resolved causal-conv1d wheel is usually usable. Some Torch nightlies can
# expose a binary-ABI mismatch at import time; compile only in that case and
# retain the resulting wheel in BuildKit's persistent uv cache.
if ! python -c 'import causal_conv1d' >/dev/null 2>&1; then
    causal_conv1d_abi="$(
        python -c '
import platform
import re
import sys

import torch

raw = (
    f"v1.6.1-py{sys.version_info.major}{sys.version_info.minor}-"
    f"{platform.machine()}-torch{torch.__version__}-cuda{torch.version.cuda}-"
    f"cxx11abi{int(torch._C._GLIBCXX_USE_CXX11_ABI)}"
)
print(re.sub(r"[^A-Za-z0-9._-]+", "_", raw))
'
    )"
    causal_conv1d_wheel_dir="$(uv cache dir)/evo2-phage-gen/causal-conv1d/$causal_conv1d_abi"
    mkdir -p "$causal_conv1d_wheel_dir"
    mapfile -t causal_conv1d_wheels < <(
        find "$causal_conv1d_wheel_dir" -maxdepth 1 -type f -name 'causal_conv1d-*.whl' -print | sort
    )
    if ((${#causal_conv1d_wheels[@]} == 0)); then
        CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip wheel \
            --no-cache-dir \
            --no-deps \
            --no-build-isolation \
            --wheel-dir "$causal_conv1d_wheel_dir" \
            'causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.6.1'
        mapfile -t causal_conv1d_wheels < <(
            find "$causal_conv1d_wheel_dir" -maxdepth 1 -type f -name 'causal_conv1d-*.whl' -print | sort
        )
    fi
    if ((${#causal_conv1d_wheels[@]} != 1)); then
        echo "expected exactly one cached causal-conv1d wheel, found ${#causal_conv1d_wheels[@]}" >&2
        exit 1
    fi
    uv pip install --no-deps --reinstall-package causal-conv1d "${causal_conv1d_wheels[0]}"
    python -c 'import causal_conv1d'
fi

# 6. Apply the recipe's Evo2 support to the configured NeMo-RL source, then install it once.
evo2_phage_setup_nemo_rl --force-reinstall

# 7. CI starts from the base devcontainer image, so keep native verifier tools
# recipe-local instead of requiring apt/conda or a custom image. Installing into
# .venv/bin makes them available whenever .ci_test_env.sh activates the venv.
evo2_phage_prepare_external_assets \
  --external-dir data/external \
  --bin-dir .venv/bin \
  --skip-mmseqs \
  --skip-lovis4u-config \
  --skip-phrogs-annotation \
  --skip-arc-evo2 \
  --skip-checkv
