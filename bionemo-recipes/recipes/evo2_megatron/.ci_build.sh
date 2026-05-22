#!/bin/bash -x

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

# 4. Install build requirements and pin transformer_engine
pip freeze | grep transformer_engine > pip-constraints.txt
uv pip install -r build_requirements.txt --no-build-isolation

# 5. Pre-install local sub-packages if checked out by CI.
#    pyproject.toml references bionemo-recipeutils and bionemo-core from git (main).
#    In CI, the workflow sparse-checks them out alongside this recipe so we can test
#    against the PR's changes. But if the sub-packages don't exist on main yet (e.g.
#    a new package introduced in the PR), uv's git source resolution fails and the
#    entire recipe install is aborted — leaving ALL deps (megatron-bridge, etc.)
#    uninstalled. By pre-installing from local and temporarily removing their git
#    source entries, the subsequent recipe install sees them as already satisfied and
#    only needs to resolve the remaining deps normally.
RECIPE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cp pyproject.toml pyproject.toml.ci_bak
for pkg_dir in "$RECIPE_ROOT/../../../sub-packages/bionemo-recipeutils" "$RECIPE_ROOT/../../../sub-packages/bionemo-core"; do
    if [ -d "$pkg_dir" ]; then
        pkg_name=$(basename "$pkg_dir")
        echo "Pre-installing $pkg_name from local checkout: $pkg_dir"
        uv pip install -e "$pkg_dir" --no-build-isolation
        # Remove the git source entry so uv uses the already-installed local version
        sed -i "/^${pkg_name} *=/d" pyproject.toml
    fi
done

# 6. Install the recipe with all remaining dependencies, including test extras
uv pip install -c pip-constraints.txt -e '.[test]' --no-build-isolation

# 7. Restore original pyproject.toml (the edit was only needed for uv resolution)
mv pyproject.toml.ci_bak pyproject.toml
