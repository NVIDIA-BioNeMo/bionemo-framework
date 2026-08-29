#!/bin/bash -x

# FIXME: Fix for "No such file or directory: /workspace/TransformerEngine"
#  Remove once bug has been addressed in the nvidia/pytorch container.
rm -f /usr/local/lib/python*/dist-packages/transformer_engine-*.dist-info/direct_url.json
export UV_LOCK_TIMEOUT=900  # increase to 15 minutes (900 seconds), adjust as needed
export UV_LINK_MODE=copy
uv venv --clear --system-site-packages

# 2. Activate the environment
source .venv/bin/activate

# 3. Create constraints file upfront so ALL installs respect warp-lang<1.12.0
# subquadratic-ops-torch accesses wp.LOG_WARNING, which Warp removed in 1.12.
: > pip-constraints.txt
echo "warp-lang<1.12.0" >> pip-constraints.txt

# Also pin transformer_engine if present
if pip freeze | grep -qE '^transformer[-_]engine([= @]|$)'; then
    pip freeze | grep transformer_engine >> pip-constraints.txt
fi

# 4. Install warp-lang with constraints
uv pip install -c pip-constraints.txt 'warp-lang<1.12.0'

# 5. Install build requirements
uv pip install -c pip-constraints.txt -r build_requirements.txt --no-build-isolation

# 6. Install the recipe with all remaining dependencies, including test extras
uv pip install -c pip-constraints.txt -e '.[test]' --no-build-isolation
