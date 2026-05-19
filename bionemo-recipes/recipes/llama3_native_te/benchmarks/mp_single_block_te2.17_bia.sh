#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=1
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-mp.single-block-te2.17
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Memory Profiler — ALL modes on TE 2.17.0dev (B300 / sm_103a)
#
# Runs all 10 configurations to get a clean baseline on the new TE version:
#
#   Single-GPU (no FSDP2):
#     1. bare              BF16 baseline
#     2. mxfp8             MXFP8 + qinit + HPIV
#     3. mxfp8 --no-hpiv   MXFP8 + qinit, NO HPIV
#     4. fp8-no-qinit      FP8 autocast, BF16 weights
#
#   FSDP2, 1 layer:
#     5. bare-fsdp2        BF16 + FSDP2
#     6. mxfp8-fsdp2       MXFP8 + qinit + FSDP2
#     7. fp8-no-qinit-fsdp2  FP8 autocast + FSDP2
#
#   FSDP2, 4 layers:
#     8. bare-fsdp2 4L     BF16 + FSDP2, 4 layers
#     9. mxfp8-fsdp2 4L    MXFP8 + qinit + FSDP2, 4 layers
#    10. fp8-no-qinit-fsdp2 4L  FP8 autocast + FSDP2, 4 layers
#
# Snapshots saved to: /lustre/fsw/.../memory_snapshots/te_2.17.0dev/
# ============================================================================

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
TE_DIR="${SCRATCH}/TransformerEngine"

CODE_MOUNT="/workspace/bionemo"
TE_MOUNT="/workspace/transformer_engine"

export EXP_NAME="${EXP_NAME:-mp_single_block_te2.17}"
RESULTS_DIR="${SCRATCH}/results/${EXP_NAME}"
SNAP_DIR="${SCRATCH}/memory_snapshots/te_2.17.0dev"

mkdir -p "${RESULTS_DIR}" "${SNAP_DIR}"

MOUNTS="${CODE_DIR}:${CODE_MOUNT},${TE_DIR}:${TE_MOUNT},${SNAP_DIR}:/workspace/snapshots"

read -r -d '' COMMAND <<'OUTER_EOF' || true
set -euxo pipefail

TE_MOUNT="/workspace/transformer_engine"
SCRIPT="/workspace/bionemo/bionemo-recipes/recipes/llama3_native_te/benchmarks/single_block_memory_profile.py"
SNAP="/workspace/snapshots"

echo "========================================="
echo "Memory Profiler — ALL modes, TE 2.17.0dev"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================="

export PYTHONPATH="$TE_MOUNT:${PYTHONPATH:-}"
python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# === Single-GPU modes (no FSDP2) ===

echo ""
echo "=== 1/10: bare (BF16 baseline, no FSDP2) ==="
python $SCRIPT --mode bare --model-size 70b --snapshot-dir $SNAP

echo ""
echo "=== 2/10: mxfp8 (MXFP8 + qinit + HPIV, no FSDP2) ==="
python $SCRIPT --mode mxfp8 --model-size 70b --snapshot-dir $SNAP

echo ""
echo "=== 3/10: mxfp8 --no-hpiv (MXFP8 + qinit, NO HPIV, no FSDP2) ==="
python $SCRIPT --mode mxfp8 --no-hpiv --model-size 70b --snapshot-dir $SNAP

echo ""
echo "=== 4/10: fp8-no-qinit (FP8 autocast, BF16 weights, no FSDP2) ==="
python $SCRIPT --mode fp8-no-qinit --model-size 70b --snapshot-dir $SNAP

# === FSDP2 modes, 1 layer ===

echo ""
echo "=== 5/10: bare-fsdp2 (BF16 + FSDP2, 1 layer, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode bare-fsdp2 --model-size 70b --snapshot-dir $SNAP

echo ""
echo "=== 6/10: mxfp8-fsdp2 (MXFP8 + qinit + FSDP2, 1 layer, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode mxfp8-fsdp2 --model-size 70b --snapshot-dir $SNAP

echo ""
echo "=== 7/10: fp8-no-qinit-fsdp2 (FP8 autocast + FSDP2, 1 layer, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode fp8-no-qinit-fsdp2 --model-size 70b --snapshot-dir $SNAP

# === FSDP2 modes, 4 layers ===

echo ""
echo "=== 8/10: bare-fsdp2 4-layer (BF16 + FSDP2, 4 layers, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode bare-fsdp2 --num-layers 4 --model-size 70b --snapshot-dir $SNAP

echo ""
echo "=== 9/10: mxfp8-fsdp2 4-layer (MXFP8 + qinit + FSDP2, 4 layers, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode mxfp8-fsdp2 --num-layers 4 --model-size 70b --snapshot-dir $SNAP

echo ""
echo "=== 10/10: fp8-no-qinit-fsdp2 4-layer (FP8 autocast + FSDP2, 4 layers, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode fp8-no-qinit-fsdp2 --num-layers 4 --model-size 70b --snapshot-dir $SNAP

echo ""
echo "========================================="
echo "Done! Snapshots in $SNAP/"
find $SNAP -name "memory_snapshot.pickle" | sort
echo "========================================="
OUTER_EOF

COMMAND="export EXP_NAME=\"${EXP_NAME}\"; ${COMMAND}"

echo "Launching: ${EXP_NAME}"

srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
