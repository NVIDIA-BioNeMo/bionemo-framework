#!/bin/bash
# Build the Evo2-SAE recipe environment for CI.
#
# The inference engine imports `bionemo.evo2` — which lives ONLY in the mbridge
# `recipes/evo2_megatron` recipe (there is no prebaked image that ships it). So we build
# that environment with evo2_megatron's OWN CI build script, inheriting its pinned
# megatron-bridge / causal-conv1d / TransformerEngine handling verbatim (no fork that
# could drift), then install the SAE library + this recipe on top of the same venv.
#
# The CI checkout must also provide the sibling recipe (recipes/evo2_megatron); it pulls its
# own bionemo-core / bionemo-recipeutils deps from git, so nothing else is required locally.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SAE_LIB="$HERE/../../sae"                       # sparse_autoencoders/sae
MEGATRON="$HERE/../../../../recipes/evo2_megatron"

# Fail loudly if the sibling recipe isn't checked out, rather than emitting a cryptic path
# error deep inside the build. In CI this means the workflow's sparse-checkout MUST include
# recipes/evo2_megatron.
if [[ ! -f "$MEGATRON/.ci_build.sh" ]]; then
  echo "ERROR: evo2_megatron not found at $MEGATRON" >&2
  echo "       evo2-sae builds on the mbridge bionemo.evo2 recipe; ensure the checkout" >&2
  echo "       includes recipes/evo2_megatron." >&2
  exit 1
fi

# 1. Build the bionemo.evo2 (mbridge) environment. Creates $MEGATRON/.venv with the
#    system-site torch/TE plus the full megatron stack.
( cd "$MEGATRON" && bash .ci_build.sh )

# 2. Add the generic SAE library + this recipe into that venv. PIP_CONSTRAINT= clears the
#    TE pin constraint evo2_megatron sets (it must not block our pure-Python installs).
source "$MEGATRON/.venv/bin/activate"
PIP_CONSTRAINT= pip install -e "$SAE_LIB"
PIP_CONSTRAINT= pip install -e "$HERE"
