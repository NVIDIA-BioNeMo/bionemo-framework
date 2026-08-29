#!/bin/bash -x

# FIXME: Fix for "No such file or directory: /workspace/TransformerEngine"
#  Remove once bug has been addressed in the nvidia/pytorch container.
rm -f /usr/local/lib/python*/dist-packages/transformer_engine-*.dist-info/direct_url.json
export UV_LOCK_TIMEOUT=900  # increase to 15 minutes (900 seconds), adjust as needed
export UV_LINK_MODE=copy
uv venv --clear --system-site-packages

# 2. Activate the environment
source .venv/bin/activate

# 3. Pin warp-lang to a version that still exposes wp.LOG_WARNING / wp.config.log_level.
# subquadratic-ops-torch-cu13 0.3.0 (resolved by the unpinned dependency below) sets
# `wp.config.log_level = wp.LOG_WARNING` in implicit_filter.py; those symbols only exist in
# warp-lang 1.14+ (absent from 1.0.x through 1.13.x). A floor of 1.14 is therefore required.
# Keep a defensive upper cap since warp 1.17+ has not been validated against 0.3.0.
uv pip install 'warp-lang>=1.14,<1.17'

# 4. Install build requirements and pin transformer_engine
pip freeze | grep transformer_engine > pip-constraints.txt
uv pip install -r build_requirements.txt --no-build-isolation

# 5. Install the recipe with all remaining dependencies, including test extras
uv pip install -c pip-constraints.txt -e '.[test]' --no-build-isolation
