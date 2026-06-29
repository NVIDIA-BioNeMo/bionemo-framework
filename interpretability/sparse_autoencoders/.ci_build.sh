#!/bin/bash
set -euo pipefail

# Install core SAE package first, then recipes that depend on it.
PIP_CONSTRAINT= pip install -e sae/
PIP_CONSTRAINT= pip install -e recipes/esm2/
