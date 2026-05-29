#!/usr/bin/env bash
set -euo pipefail

# Install Codex as the devcontainer user so the binary lands in the mounted user environment.
if ! command -v codex >/dev/null 2>&1; then
  curl -fsSL https://chatgpt.com/codex/install.sh | sh
fi
# Run via uv to avoid relying on updated PATH in this shell
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  # Some editors (VS Code, Cursor) set core.hooksPath in .git/config, which
  # causes pre-commit to refuse to install hooks.  Clear it first so
  # pre-commit can manage hooks normally.
  git config --unset-all core.hooksPath 2>/dev/null || true
  uvx pre-commit install
fi
