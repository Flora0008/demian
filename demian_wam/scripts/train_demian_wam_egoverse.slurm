#!/bin/bash
# =============================================================================
# DeMiAn-WAM EgoVerse mid-training launcher (Cosmos-Predict 2.5 Video2World).
#
# Mid-trains the Cosmos-Predict2.5 2B video DiT on 50K EgoVerse human-egocentric
# clips with a video-prediction objective (no action head) — the RQ4 scaling
# stage. The DeMiAn dense annotations enter as language conditioning on the video
# model; the contrast is annotation-only (baseline) vs. dense annotations.
#
#   variant     experiment (Hydra)
#   ----------  ----------------------------------------------------------------
#   baseline    predict2_video2world_training_2b_egoverse_annotation
#   dense       predict2_video2world_training_2b_egoverse_dense_human
#   (others, set EXPERIMENT= directly: ..._augmented, ..._dense_only,
#    ..._annotation_stage2, ..._dense_human_stage2, ..._*_precomputed)
#
# Usage:
#   sbatch --nodes=<N> --gpus-per-node=<G> [--partition=<p>] [--account=<a>] \
#          scripts/train_demian_wam_egoverse.slurm <variant>
#
#   e.g.  sbatch --nodes=8 --gpus-per-node=8 scripts/train_demian_wam_egoverse.slurm dense
#
# Node count / GPUs / partition / account passed at submit time (not hardcoded).
# PROJECT_DIR must be a Cosmos-Predict2.5 checkout with the demian_wam overlays
# applied. EgoVerse data dirs are read by experiments/base/egoverse.py from the
# EGOVERSE_* env vars exported below.
# =============================================================================
#SBATCH -J demian_wam_egoverse
#SBATCH -t 04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/demian_wam_egoverse_%j.out

set -euo pipefail

VARIANT="${1:-${VARIANT:-dense}}"
case "${VARIANT}" in
    baseline) EXPERIMENT="${EXPERIMENT:-predict2_video2world_training_2b_egoverse_annotation}" ;;
    dense)    EXPERIMENT="${EXPERIMENT:-predict2_video2world_training_2b_egoverse_dense_human}" ;;
    *)
        # Allow passing a full experiment name directly as the variant.
        EXPERIMENT="${EXPERIMENT:-${VARIANT}}" ;;
esac
EXP_NAME="$(echo "${EXPERIMENT}" | sed 's/predict2_video2world_training_//')"

# ----------------------------------------------------------------------------
# Site-specific configuration (override via environment).
#   PROJECT_DIR    Cosmos-Predict2.5 checkout WITH the demian_wam overlays applied.
#   PRETRAIN_CKPT  Cosmos-Predict2.5-2B base checkpoint (ema_bf16.pt).
#   TOKENIZER_CKPT Cosmos-Predict2.5-2B tokenizer (.pth).
#   EGOVERSE_DATASET_DIR / EGOVERSE_LATENT_DIR_ANNOTATION / EGOVERSE_LATENT_DIR_AUGMENTED
#                  EgoVerse data roots (read by experiments/base/egoverse.py).
#   CONTAINER      training image for `srun --container-image` (optional).
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the Cosmos-Predict2.5 checkout with demian_wam overlays applied}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:?Set PRETRAIN_CKPT to the Cosmos-Predict2.5-2B base checkpoint (ema_bf16.pt)}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:?Set TOKENIZER_CKPT to the Cosmos-Predict2.5-2B tokenizer .pth}"
EGOVERSE_DATASET_DIR="${EGOVERSE_DATASET_DIR:?Set EGOVERSE_DATASET_DIR to the EgoVerse dataset root}"
EGOVERSE_LATENT_DIR_ANNOTATION="${EGOVERSE_LATENT_DIR_ANNOTATION:-${EGOVERSE_DATASET_DIR}}"
EGOVERSE_LATENT_DIR_AUGMENTED="${EGOVERSE_LATENT_DIR_AUGMENTED:-${EGOVERSE_DATASET_DIR}}"
CONTAINER="${CONTAINER:-}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/path/to:/path/to}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/demian_wam_logs/egoverse}"

MAX_ITER="${MAX_ITER:-50000}"
LR="${LR:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-3000}"
SAVE_ITER="${SAVE_ITER:-200}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"
PER_GPU_BATCH="${PER_GPU_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

CKPT_BASE="${OUTPUT_ROOT}/${EXP_NAME}/checkpoints"
mkdir -p logs "${CKPT_BASE}"

# Resume from the latest iter_* checkpoint; resubmit on wall-time signal.
latest_iter_ckpt() { ls -d "${CKPT_BASE}"/iter_* 2>/dev/null | sort | tail -1 || true; }
resubmit_if_needed() {
    echo "[signal] resubmitting ${EXP_NAME} ..."
    sbatch -N "${NUM_NODES}" --export=ALL "${SCRIPT_PATH}" "${VARIANT}" || true
}
trap resubmit_if_needed USR1

LATEST_CKPT=$(latest_iter_ckpt)
if [ -n "${LATEST_CKPT}" ]; then
    LOAD_PATH="${LATEST_CKPT}";   LOAD_TRAINING_STATE=True
else
    LOAD_PATH="${PRETRAIN_CKPT}"; LOAD_TRAINING_STATE=False
fi

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT="${MASTER_PORT:-29600}"

echo "=========================================================="
echo "DeMiAn-WAM EgoVerse mid-training | variant=${VARIANT}"
echo "experiment=${EXPERIMENT} | nodes=${NUM_NODES} gpus/node=${GPUS_PER_NODE}"
echo "load=${LOAD_PATH} (training_state=${LOAD_TRAINING_STATE})"
echo "=========================================================="

TRAIN_CMD="
set -e
cd ${PROJECT_DIR}
export MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}
export PYTHONPATH=${PROJECT_DIR}\${PYTHONPATH:+:\${PYTHONPATH}}
export OMP_NUM_THREADS=8
export IMAGINAIRE_OUTPUT_ROOT=${OUTPUT_ROOT}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
# EgoVerse data roots consumed by experiments/base/egoverse.py
export EGOVERSE_DATASET_DIR=${EGOVERSE_DATASET_DIR}
export EGOVERSE_LATENT_DIR_ANNOTATION=${EGOVERSE_LATENT_DIR_ANNOTATION}
export EGOVERSE_LATENT_DIR_AUGMENTED=${EGOVERSE_LATENT_DIR_AUGMENTED}

python -m torch.distributed.run \
    --nnodes=${NUM_NODES} --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    --node_rank=\${SLURM_NODEID} \
    -m cosmos_oss.scripts.train \
    --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
    experiment=${EXPERIMENT} \
    +model.config.tokenizer.vae_pth=${TOKENIZER_CKPT} \
    trainer.max_iter=${MAX_ITER} \
    trainer.logging_iter=50 \
    optimizer.lr=${LR} \
    scheduler.warm_up_steps='[${WARMUP_STEPS}]' \
    checkpoint.save_iter=${SAVE_ITER} \
    checkpoint.load_path=${LOAD_PATH} \
    checkpoint.load_training_state=${LOAD_TRAINING_STATE} \
    checkpoint.strict_resume=False \
    trainer.callbacks.every_n_sample_reg.every_n=99999 \
    trainer.callbacks.every_n_sample_ema.every_n=99999 \
    dataloader_train.batch_size=${PER_GPU_BATCH} \
    trainer.grad_accum_iter=${GRAD_ACCUM}
"

if [ -n "${CONTAINER}" ]; then
    srun --container-image "${CONTAINER}" --container-mounts "${CONTAINER_MOUNTS}" --container-writable \
         bash -c "${TRAIN_CMD}" &
else
    srun bash -c "${TRAIN_CMD}" &
fi
wait $! ; EXIT_CODE=$?

echo "[${EXP_NAME}] training exited (code=${EXIT_CODE})."
if [ "${EXIT_CODE}" -eq 0 ] || [ "${EXIT_CODE}" -eq 130 ] || [ "${EXIT_CODE}" -eq 143 ]; then
    resubmit_if_needed
fi
exit "${EXIT_CODE}"
