#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=1
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-mxfp8-no-qinit-fsdp2
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Memory Profiler — MXFP8 no-qinit + FSDP2 (B300 / sm_103a)
#
# Runs the mxfp8-no-qinit-fsdp2 mode (MXFP8 block-scaling recipe via
# te.autocast, BF16 weights, NO quantized_model_init) with FSDP2:
#
#   1. mxfp8-no-qinit-fsdp2  1 layer, 2 GPUs
#   2. mxfp8-no-qinit-fsdp2  4 layers, 2 GPUs
#
# Snapshots saved to: /lustre/fsw/.../memory_snapshots/te_2.17.0dev/
# ============================================================================

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
TE_DIR="${SCRATCH}/TransformerEngine"

CODE_MOUNT="/workspace/bionemo"
TE_MOUNT="/workspace/transformer_engine"

export EXP_NAME="${EXP_NAME:-mxfp8_no_qinit_fsdp2}"
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
echo "Memory Profiler — MXFP8 no-qinit + FSDP2"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================="

export PYTHONPATH="$TE_MOUNT:${PYTHONPATH:-}"
python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

echo ""
echo "=== 1/2: mxfp8-no-qinit-fsdp2 (1 layer, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode mxfp8-no-qinit-fsdp2 --model-size 70b --snapshot-dir $SNAP

echo ""
echo "=== 2/2: mxfp8-no-qinit-fsdp2 (4 layers, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode mxfp8-no-qinit-fsdp2 --num-layers 4 --model-size 70b --snapshot-dir $SNAP

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
