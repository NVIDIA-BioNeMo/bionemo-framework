#!/bin/bash -x
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# FIXME: Fix for "No such file or directory: /workspace/TransformerEngine"
#  Remove once bug has been addressed in the nvidia/pytorch container.
rm -f /usr/local/lib/python*/dist-packages/transformer_engine-*.dist-info/direct_url.json
export UV_LOCK_TIMEOUT=900  # increase to 15 minutes (900 seconds), adjust as needed
export UV_LINK_MODE=copy
uv venv --clear --system-site-packages

# 2. Activate the environment
source .venv/bin/activate

# 3. Pin warp-lang<1.13.0 (subquadratic-ops-torch 0.2.0 uses wp.context removed in 1.13)
uv pip install 'warp-lang<1.13.0'

# 4. Install build requirements and pin transformer_engine. An image without
# Transformer Engine intentionally produces an empty constraints file.
if ! pip freeze | grep -E '^transformer[-_]engine([= @]|$)' > pip-constraints.txt; then
    : > pip-constraints.txt
    echo "transformer-engine is not installed; continuing with an empty pip-constraints.txt" >&2
fi
uv pip install -c pip-constraints.txt -r build_requirements.txt --no-build-isolation

# 5. Install the recipe with all remaining dependencies, including test extras.
uv pip install -c pip-constraints.txt -e '.[test]' --no-build-isolation

# causal-conv1d's upstream wheel cache is keyed to a coarse Torch version and
# can retain an extension built against a different nightly ABI. Bypass both
# upstream and uv wheels for this package only so it compiles against the
# active system Torch. Keep the source aligned with [tool.uv.sources].
CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install \
    --no-cache \
    --no-deps \
    --no-build-isolation \
    --reinstall-package causal-conv1d \
    'causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.6.1'

# 6. Upstream NeMo-RL's current pyproject only packages the top-level nemo_rl module.
# Reinstall the pinned checkout with complete package discovery, then apply and verify this recipe's Evo2 patch.
evo2_phage_patch_nemo_rl --repair-install --force-reinstall --verify-runtime

# 7. CI starts from the base devcontainer image, so keep native verifier tools
# recipe-local instead of requiring apt/conda or a custom image. Installing into
# .venv/bin makes them available whenever .ci_test_env.sh activates the venv.
evo2_phage_prepare_external_assets \
  --external-dir data/external \
  --bin-dir .venv/bin \
  --skip-mmseqs \
  --skip-phrogs-annotation \
  --skip-arc-evo2 \
  --skip-checkv
