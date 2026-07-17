#!/bin/bash -x
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export EVO2_PHAGE_NEMO_RL_SOURCE_DIR="$SCRIPT_DIR/.venv/nemo-rl-source"
export NEMO_RL_VENV_DIR="$SCRIPT_DIR/.venv/nemo-rl-venvs"

# FIXME: Fix for "No such file or directory: /workspace/TransformerEngine"
#  Remove once bug has been addressed in the nvidia/pytorch container.
rm -f /usr/local/lib/python*/dist-packages/transformer_engine-*.dist-info/direct_url.json
export UV_LOCK_TIMEOUT=900  # increase to 15 minutes (900 seconds), adjust as needed
export UV_LINK_MODE=copy
uv venv --clear --python /usr/bin/python3.12 --system-site-packages

# 2. Activate the environment
source .venv/bin/activate

# 3. Pin warp-lang<1.13.0 (subquadratic-ops-torch 0.2.0 uses wp.context removed in 1.13)
uv pip install 'warp-lang<1.13.0'

# 4. Install build requirements and pin transformer_engine
pip freeze | grep transformer_engine > pip-constraints.txt
uv pip install -r build_requirements.txt --no-build-isolation

# 5. Install the recipe with all remaining dependencies, including test extras
uv pip install -c pip-constraints.txt -e '.[test]' --no-build-isolation

# 6. Retain the exact recursive NeMo-RL source, apply the recipe patches, and build its locked vLLM actor environment.
evo2_phage_patch_nemo_rl --repair-install --force-reinstall --prepare-vllm-actor-env --verify-runtime

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
