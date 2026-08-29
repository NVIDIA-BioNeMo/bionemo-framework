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

# 3. Create constraints file upfront so ALL installs respect warp-lang<1.12.0
# subquadratic-ops-torch accesses wp.LOG_WARNING, which Warp removed in 1.12.
: > pip-constraints.txt
echo "warp-lang<1.12.0" >> pip-constraints.txt

# Also pin transformer_engine if present
if pip freeze | grep -qE '^transformer[-_]engine([= @]|$)'; then
    pip freeze | grep -E '^transformer[-_]engine([= @]|$)' >> pip-constraints.txt
fi

# 4. Install warp-lang with constraints
uv pip install -c pip-constraints.txt 'warp-lang<1.12.0'

# 5. Install build requirements
uv pip install -c pip-constraints.txt -c security_constraints.txt -r build_requirements.txt --no-build-isolation

# 6. Install the recipe with all remaining dependencies, including test extras
uv pip install -c pip-constraints.txt -c security_constraints.txt -e '.[test]' --no-build-isolation

# 7. Force reinstall warp-lang to ensure constraint is enforced after all deps
uv pip install -c pip-constraints.txt -c security_constraints.txt --reinstall-package warp-lang 'warp-lang<1.12.0'

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
    uv pip install -c pip-constraints.txt -c security_constraints.txt --no-deps --reinstall-package causal-conv1d "${causal_conv1d_wheels[0]}"
    python -c 'import causal_conv1d'
fi

# 8. Apply the recipe's Evo2 support to the configured NeMo-RL source, then install it once.
evo2_phage_setup_nemo_rl --force-reinstall

# 9. CI starts from the base devcontainer image, so keep native verifier tools
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
