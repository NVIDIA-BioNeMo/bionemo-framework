#!/bin/bash
# Build the shared `sae` interpretability library for CI, mirroring each recipe's `.ci_build.sh`
# so the lib rides the same interp CI matrix (its own `.ci_build.sh` + `pytest tests/`) as a recipe.
#
# Unlike the evo2 recipe (which must build the bionemo.evo2 / mbridge megatron env), the `sae` lib
# is domain-agnostic — torch + triton, both already in the pytorch base image — so there is no venv
# to build. A plain editable install of the package plus its dev/test extras is all pytest needs:
#   * test_kernels uses the image's Triton on the L4 GPU;
#   * the test_tp_* suite is gloo/CPU multiprocessing, so it runs on the same box's CPUs.
set -euo pipefail

pip install -e ".[dev]"
