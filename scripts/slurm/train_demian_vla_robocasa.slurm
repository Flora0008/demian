#!/bin/bash
# =============================================================================
# DeMiAn-VLA training launcher — RoboCasa365 (paper Table 1, VLA rows).
#
# Fine-tunes pi05_base (openpi 0.5 PyTorch backbone) on the RoboCasa365 atomic
# split. The annotation aspect selects the appended-caption key; the policy,
# visual data, and action targets are fixed across aspects. The RoboCasa variant
# uses the annotation auxiliary LM loss (lambda_LM=0.1, config `pi05_robocasa_lm`).
#
#   Aspect (paper)      ->  caption key (--data.caption-key)
#   ----------------------  --------------------------------------------------
#   baseline            ->  (none; config pi05_robocasa, no LM loss)
#   physical_motion     ->  current_action
#   scene_composition   ->  environment-simple
#   arm_pose            ->  agent_arm_state
#   reasoning           ->  reasoning-short
#   mix5                ->  random:none,current_action,reasoning-short,environment-simple,agent_arm_state
#
# Usage:
#   sbatch --nodes=<N> --gpus-per-node=<G> [--partition=<p>] [--account=<a>] \
#          scripts/slurm/train_demian_vla_robocasa.slurm <aspect>
#
#   e.g.  sbatch --nodes=1 --gpus-per-node=8 scripts/slurm/train_demian_vla_robocasa.slurm reasoning
#
# Node count / GPUs / partition / account are passed at submit time (not
# hardcoded). Set PROJECT_DIR (the vendored code/openpi), DATA_ROOT (RoboCasa
# LeRobot dataset root), PRETRAINED_CKPT, and optionally CONTAINER via env.
# =============================================================================
#SBATCH -J demian_vla_robocasa
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=logs/demian_vla_robocasa_%j.log

set -euo pipefail

ASPECT="${1:-${ASPECT:-mix5}}"
CAPTION_ARGS=""
case "${ASPECT}" in
    baseline)          TRAIN_CONFIG="pi05_robocasa" ;;
    physical_motion)   TRAIN_CONFIG="pi05_robocasa_lm"; CAPTION_ARGS="--data.caption-key current_action" ;;
    scene_composition) TRAIN_CONFIG="pi05_robocasa_lm"; CAPTION_ARGS="--data.caption-key environment-simple" ;;
    arm_pose)          TRAIN_CONFIG="pi05_robocasa_lm"; CAPTION_ARGS="--data.caption-key agent_arm_state" ;;
    reasoning)         TRAIN_CONFIG="pi05_robocasa_lm"; CAPTION_ARGS="--data.caption-key reasoning-short" ;;
    mix5)              TRAIN_CONFIG="pi05_robocasa_lm"; CAPTION_ARGS="--data.caption-key random:none,current_action,reasoning-short,environment-simple,agent_arm_state" ;;
    *)
        echo "Unknown aspect '${ASPECT}'. Choose: baseline physical_motion scene_composition arm_pose reasoning mix5" >&2
        exit 2 ;;
esac
EXP_NAME="demian_vla_robocasa_${ASPECT}"

# ----------------------------------------------------------------------------
# Site-specific configuration (override via environment).
#   PROJECT_DIR     the vendored openpi dir (code/openpi) — holds scripts/train_pytorch.py.
#   DATA_ROOT       RoboCasa365 LeRobot dataset root (contains assets/).
#   PRETRAINED_CKPT Pi0.5 base PyTorch weights.
#   CONTAINER       training image for `srun --container-image` (optional).
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"               # repo root
PROJECT_DIR="${PROJECT_DIR:-${CODE_DIR}/openpi}"            # holds scripts/train_pytorch.py
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the RoboCasa365 LeRobot dataset root}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to the Pi0.5 base PyTorch weights}"
CONTAINER="${CONTAINER:-}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/path/to:/path/to}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/train_output}"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"

OPENPI_SRC="${PROJECT_DIR}/src"
OPENPI_CLIENT_SRC="${PROJECT_DIR}/packages/openpi-client/src"
TRANSFORMERS_REPLACE="${OPENPI_SRC}/openpi/models_pytorch/transformers_replace"
CKPT_DIR="${OUTPUT_DIR}/${TRAIN_CONFIG}/${EXP_NAME}"

resubmit() { echo "Caught signal — resubmitting..."; sbatch "${SCRIPT_PATH}" "${ASPECT}" || true; exit 0; }
trap resubmit SIGTERM SIGUSR1

LATEST=$(ls -d "${CKPT_DIR}"/[0-9]* 2>/dev/null | sort -n | tail -1 || true)
RESUME_FLAG=""; [ -n "${LATEST}" ] && RESUME_FLAG="--resume"
CKPT_BEFORE="${LATEST}"

MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" | head -1)
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
NNODES="${SLURM_NNODES:-1}"

echo "=========================================="
echo "DeMiAn-VLA RoboCasa | aspect=${ASPECT} | config=${TRAIN_CONFIG}"
echo "caption args: ${CAPTION_ARGS:-<none>}"
echo "nodes=${NNODES} gpus/node=${GPUS_PER_NODE} | resume=${RESUME_FLAG:-<from pretrained>}"
echo "=========================================="

TRAIN_CMD="
set -ex
cd ${PROJECT_DIR}
export JAX_PLATFORMS=cpu
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=\${WANDB_MODE:-offline}
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export PYTHONPATH=${OPENPI_SRC}:${OPENPI_CLIENT_SRC}:${CODE_DIR}\${PYTHONPATH:+:\${PYTHONPATH}}

TRANSFORMERS_DIR=\$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')
cp -r ${TRANSFORMERS_REPLACE}/* \"\${TRANSFORMERS_DIR}/\"

mkdir -p ${CKPT_DIR}
if torchrun --help >/dev/null 2>&1; then TORCHRUN=torchrun; else TORCHRUN='python -m torch.distributed.run'; fi

\${TORCHRUN} --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=\${SLURM_NODEID} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    scripts/train_pytorch.py ${TRAIN_CONFIG} \
    --data.repo-id ${DATA_ROOT} \
    --data.assets.assets-dir ${DATA_ROOT}/assets \
    ${CAPTION_ARGS} \
    --pytorch-weight-path ${PRETRAINED_CKPT} \
    --exp-name ${EXP_NAME} \
    --checkpoint-base-dir ${OUTPUT_DIR} \
    --project-name demian_vla_robocasa \
    --num-train-steps ${NUM_TRAIN_STEPS} \
    --save-interval ${SAVE_INTERVAL} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    ${RESUME_FLAG}
"

if [ -n "${CONTAINER}" ]; then
    srun --container-image "${CONTAINER}" --container-mounts "${CONTAINER_MOUNTS}" bash -c "${TRAIN_CMD}"
else
    srun bash -c "${TRAIN_CMD}"
fi
SRUN_EXIT=$?

echo "Job finished: $(date) (exit ${SRUN_EXIT})"
if [ -d "${CKPT_DIR}/${NUM_TRAIN_STEPS}" ]; then
    echo "Training COMPLETE — ${NUM_TRAIN_STEPS} steps reached."
else
    CKPT_AFTER=$(ls -d "${CKPT_DIR}"/[0-9]* 2>/dev/null | sort -n | tail -1 || true)
    if [ "${CKPT_AFTER:-}" = "${CKPT_BEFORE:-}" ] && [ ${SRUN_EXIT} -ne 0 ]; then
        echo "ERROR: training failed with no new checkpoint. Not resubmitting." >&2; exit 1
    fi
    echo "Incomplete — latest: ${CKPT_AFTER:-none}. Resubmitting."
    sbatch "${SCRIPT_PATH}" "${ASPECT}" || true
fi
