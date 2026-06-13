#!/bin/bash
# =============================================================================
# DeMiAn-VLA training launcher (openpi 0.5 / Pi0.5 PyTorch backbone).
#
# Trains one annotation-aspect ablation from the paper's Table 1. The action
# policy, visual data, and action targets are identical across aspects; only the
# language appended to each demonstration changes (selected by --data.caption-key).
#
#   Aspect (paper)      ->  caption key (data sub-key / mix spec)
#   ----------------------  --------------------------------------------------
#   baseline            ->  (none; task description only)
#   physical_motion     ->  current_action
#   scene_composition   ->  environment-simple
#   arm_pose            ->  agent_arm_state
#   reasoning           ->  reasoning-short
#   mix5                ->  random:none,current_action,reasoning-short,environment-simple,agent_arm_state
#
# Usage:
#   sbatch --nodes=<N> --gpus-per-node=<G> \
#          [--partition=<your_partition>] [--account=<your_account>] \
#          scripts/slurm/train_demian_vla.slurm <aspect>
#
#   e.g.  sbatch --nodes=2 --gpus-per-node=8 scripts/slurm/train_demian_vla.slurm mix5
#
# Node count / GPUs-per-node / partition / account are intentionally NOT
# hardcoded (they are site-specific). Pass them on the `sbatch` command line.
# Set the paths below via environment variables before submitting.
# =============================================================================
#SBATCH -J demian_vla_train
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=logs/demian_vla_train_%j.log

set -euo pipefail

# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------
ASPECT="${1:-${ASPECT:-mix5}}"

case "${ASPECT}" in
    baseline)          TRAIN_CONFIG="pi05_mlspaces_finetune";        CAPTION_KEY="" ;;
    physical_motion)   TRAIN_CONFIG="pi05_mlspaces_caption_finetune"; CAPTION_KEY="current_action" ;;
    scene_composition) TRAIN_CONFIG="pi05_mlspaces_caption_finetune"; CAPTION_KEY="environment-simple" ;;
    arm_pose)          TRAIN_CONFIG="pi05_mlspaces_caption_finetune"; CAPTION_KEY="agent_arm_state" ;;
    reasoning)         TRAIN_CONFIG="pi05_mlspaces_caption_finetune"; CAPTION_KEY="reasoning-short" ;;
    mix5)              TRAIN_CONFIG="pi05_mlspaces_caption_finetune"; CAPTION_KEY="random:none,current_action,reasoning-short,environment-simple,agent_arm_state" ;;
    *)
        echo "Unknown aspect: '${ASPECT}'. Choose one of: baseline physical_motion scene_composition arm_pose reasoning mix5" >&2
        exit 2 ;;
esac
EXP_NAME="demian_vla_${ASPECT}"

# ----------------------------------------------------------------------------
# Site-specific configuration (override via environment).
#   PROJECT_DIR   release `code/` directory (auto-detected from this script).
#   DATA_ROOT     MolmoBot subset: contains FrankaPick*Config/part0/train and captions/.
#   PRETRAINED_CKPT  Pi0.5 base PyTorch weights to fine-tune from.
#   CONTAINER     training image (used by `srun --container-image`); adapt the
#                 srun wrapper below if your cluster does not use pyxis/enroot.
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"   # repo root
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the MolmoBot dataset root (FrankaPick*Config/ + captions/)}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to the Pi0.5 base PyTorch weights}"
CONTAINER="${CONTAINER:?Set CONTAINER to the training image, or edit the srun wrapper for your setup}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/path/to:/path/to}"
CAPTIONS_DIR="${CAPTIONS_DIR:-${DATA_ROOT}/captions}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/train_output}"

# Training hyperparameters (paper Table 1 evaluates the 40K-step checkpoint).
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"

# Vendored openpi (PyTorch). No pip install required — set PYTHONPATH.
OPENPI_SRC="${PROJECT_DIR}/openpi/src"
OPENPI_CLIENT_SRC="${PROJECT_DIR}/openpi/packages/openpi-client/src"
TRANSFORMERS_REPLACE="${OPENPI_SRC}/openpi/models_pytorch/transformers_replace"

CKPT_DIR="${OUTPUT_DIR}/${TRAIN_CONFIG}/${EXP_NAME}"

# ----------------------------------------------------------------------------
# Auto-resubmit on wall-time signal (clusters with short time limits) and
# auto-resume from the latest checkpoint.
# ----------------------------------------------------------------------------
resubmit() {
    echo "Caught signal — resubmitting..."
    sbatch "${SCRIPT_PATH}" "${ASPECT}" || true
    exit 0
}
trap resubmit SIGTERM SIGUSR1

LATEST=$(ls -d "${CKPT_DIR}"/[0-9]* 2>/dev/null | sort -n | tail -1 || true)
RESUME_FLAG=""
if [ -n "${LATEST}" ]; then
    RESUME_FLAG="--resume"
fi
CKPT_BEFORE="${LATEST}"

MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" | head -1)
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
NNODES="${SLURM_NNODES:-1}"

echo "=========================================="
echo "DeMiAn-VLA training | aspect=${ASPECT} | config=${TRAIN_CONFIG}"
echo "caption-key: ${CAPTION_KEY:-<none / task description only>}"
echo "nodes=${NNODES} gpus/node=${GPUS_PER_NODE} | master=${MASTER_ADDR}:${MASTER_PORT}"
echo "resume: ${RESUME_FLAG:-<from pretrained>} ${LATEST:+(latest: ${LATEST})}"
echo "=========================================="

# Caption args are only passed for caption-conditioned configs.
# (caption keys contain no whitespace, so a plain string is safe to inject.)
CAPTION_ARGS=""
if [ -n "${CAPTION_KEY}" ]; then
    CAPTION_ARGS="--data.captions-dir ${CAPTIONS_DIR} --data.caption-key ${CAPTION_KEY}"
fi

srun --container-image "${CONTAINER}" \
     --container-mounts "${CONTAINER_MOUNTS}" \
     bash -c "
set -ex
cd ${PROJECT_DIR}

export JAX_PLATFORMS=cpu                # JAX is used only for tree utils (CPU)
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=\${WANDB_MODE:-offline}
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export PYTHONPATH=${OPENPI_SRC}:${OPENPI_CLIENT_SRC}:${PROJECT_DIR}\${PYTHONPATH:+:\${PYTHONPATH}}

# Apply the openpi Gemma/PaliGemma/SigLIP patches over the installed transformers.
TRANSFORMERS_DIR=\$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')
cp -r ${TRANSFORMERS_REPLACE}/* \"\${TRANSFORMERS_DIR}/\"

mkdir -p ${CKPT_DIR}
nvidia-smi -L || true

# Some containers ship a torchrun wrapper that fails on resumed jobs; fall back.
if torchrun --help >/dev/null 2>&1; then TORCHRUN=torchrun; else TORCHRUN='python -m torch.distributed.run'; fi

\${TORCHRUN} --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=\${SLURM_NODEID} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    molmobot_pi0/train_openpi_pytorch.py ${TRAIN_CONFIG} \
    --data.repo-id ${DATA_ROOT}/FrankaPickAndPlaceOmniCamConfig/part0/train \
    --data.exo-camera droid_shoulder_light_randomization \
    --data.wrist-camera wrist_camera_zed_mini \
    --data.auxiliary-dataset-paths ${DATA_ROOT}/FrankaPickOmniCamConfig/part0/train 0.25 \
    --data.auxiliary-dataset-paths ${DATA_ROOT}/FrankaPickAndPlaceNextToOmniCamConfig/part0/train 0.2 \
    --data.auxiliary-dataset-paths ${DATA_ROOT}/FrankaPickAndPlaceColorOmniCamConfig/part0/train 0.15 \
    ${CAPTION_ARGS} \
    --pytorch-weight-path ${PRETRAINED_CKPT} \
    --exp-name ${EXP_NAME} \
    --checkpoint-base-dir ${OUTPUT_DIR} \
    --project-name demian_vla \
    --num-train-steps ${NUM_TRAIN_STEPS} \
    --save-interval ${SAVE_INTERVAL} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    ${RESUME_FLAG}
"
SRUN_EXIT=$?

echo "Job finished: $(date) (exit ${SRUN_EXIT})"

# Resume/resubmit until NUM_TRAIN_STEPS is reached (for short wall-time clusters).
if [ -d "${CKPT_DIR}/${NUM_TRAIN_STEPS}" ]; then
    echo "Training COMPLETE — ${NUM_TRAIN_STEPS} steps reached."
else
    CKPT_AFTER=$(ls -d "${CKPT_DIR}"/[0-9]* 2>/dev/null | sort -n | tail -1 || true)
    if [ "${CKPT_AFTER:-}" = "${CKPT_BEFORE:-}" ] && [ ${SRUN_EXIT} -ne 0 ]; then
        echo "ERROR: training failed with no new checkpoint (latest: ${CKPT_AFTER:-none}). Not resubmitting." >&2
        exit 1
    fi
    echo "Incomplete — latest checkpoint: ${CKPT_AFTER:-none}. Resubmitting."
    sbatch "${SCRIPT_PATH}" "${ASPECT}" || true
fi
