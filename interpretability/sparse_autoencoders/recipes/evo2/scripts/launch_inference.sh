#!/bin/bash
# Launch the Evo2 SAE inference engine. One engine, four modes:
#
#   ./launch_inference.sh serve                       # live HTTP server on :8001 (viz backend)
#   ./launch_inference.sh encode  --sequence ATGC...  # annotate ONE sequence -> top features
#   ./launch_inference.sh batch   --fasta in.fa --out out.parquet   # MANY sequences -> parquet
#   ./launch_inference.sh generate --prompt ATGC... --clamp 29244:300  # steer + generate DNA
#
# Steering loop: `encode` a sequence to find an active feature id, then
# `generate --clamp ID:STRENGTH` (strength ~2-3x the feature's max_activation; repeat --clamp).
#
# Config via env. Required: EVO2_CKPT_DIR, SAE_CKPT_PATH. Optional (have defaults):
# FEATURE_ANNOTATIONS, EMBEDDING_LAYER (26), DEVICE, PORT, CUDA_VISIBLE_DEVICES.
#
# Run INSIDE the evo2_megatron venv (it provides bionemo.evo2 + megatron) — same venv the tests
# use. In the Docker image it's already active (on PATH), so just run this. On bare metal, activate
# it first:  source <evo2_megatron>/.venv/bin/activate
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RECIPE_DIR="$(cd "$HERE/.." && pwd)"  # recipes/evo2 — so the evo2_sae package imports

# Required (no hardcoded defaults — supply your own paths via env):
export EVO2_CKPT_DIR="${EVO2_CKPT_DIR:?Set EVO2_CKPT_DIR to an Evo2 MBridge checkpoint directory}"
export SAE_CKPT_PATH="${SAE_CKPT_PATH:?Set SAE_CKPT_PATH to a trained SAE checkpoint (.pt)}"
# Optional: feature-label parquet (empty = features are unlabeled). Layer defaults to 26.
export FEATURE_ANNOTATIONS="${FEATURE_ANNOTATIONS:-}"
export EMBEDDING_LAYER="${EMBEDDING_LAYER:-26}"

export PYTHONPATH="$RECIPE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"  # find evo2_sae without a pip install

# We assume the evo2_megatron venv is already active (Docker: on PATH; bare metal: you sourced it).
# Fail with a clear message rather than a deep ImportError if it isn't.
if ! python -c "import bionemo.evo2" 2>/dev/null; then
  echo "ERROR: bionemo.evo2 not importable — activate the evo2_megatron venv first" >&2
  echo "       (source <evo2_megatron>/.venv/bin/activate) or run inside the Docker image." >&2
  exit 1
fi

# One-shot modes (encode/batch/generate) just run once -> exec, signals pass straight through.
# For `serve`, a bad request can trip a CUDA device-side assert that poisons the process context
# (unrecoverable in-process — restart is the only cure). So tell the engine to exit the worker on
# that fault (EXIT_ON_CUDA_WEDGE) and respawn it here, making recovery host-independent.
#
# The worker runs in the BACKGROUND and we `wait` on it so this supervisor stays responsive to
# signals: a trap forwards SIGTERM/SIGINT to the worker (triggering uvicorn's graceful shutdown)
# and stops the respawn loop — important when this script is PID 1 under `docker stop`/k8s, where
# bash would otherwise not forward the signal and orphan the worker. Crash/wedge -> respawn with
# backoff, capped so a persistent failure (e.g. port already bound) doesn't loop forever.
if [[ "${1:-}" == "serve" ]]; then
  export EXIT_ON_CUDA_WEDGE=1
  export ALLOW_ENGINE_RESTART=1  # enables POST /api/restart (user-driven "cancel + reload"); safe only under this respawn loop
  child=""
  stop=0
  trap 'stop=1; [[ -n "$child" ]] && kill -TERM "$child" 2>/dev/null || true' TERM INT
  rc=0
  fails=0
  while [[ "$stop" -eq 0 ]]; do
    python -m evo2_sae.cli "$@" &
    child=$!
    wait "$child" && rc=0 || rc=$?
    child=""
    # stop on a clean exit or when a signal asked us to (143 SIGTERM / 130 SIGINT belt-and-suspenders).
    [[ "$stop" -eq 1 || "$rc" -eq 0 || "$rc" -eq 143 || "$rc" -eq 130 ]] && break
    # A user-requested restart (POST /api/restart -> exit 42) is intentional, not a crash: respawn
    # immediately and don't spend the crash budget on it.
    if [[ "$rc" -eq 42 ]]; then
      echo "[launch] engine restart requested — respawning a fresh worker" >&2
      fails=0
      continue
    fi
    fails=$((fails + 1))
    if [[ "$fails" -ge 10 ]]; then
      echo "[launch] serve exited ($rc) $fails times — giving up; fix the cause." >&2
      break
    fi
    backoff=$([[ "$fails" -lt 5 ]] && echo 2 || echo 10)
    echo "[launch] serve exited ($rc); restart $fails/10 in ${backoff}s…" >&2
    sleep "$backoff" &
    wait $! 2>/dev/null || true  # interruptible: a signal during backoff stops the loop promptly
  done
  exit "$rc"
fi
exec python -m evo2_sae.cli "$@"
