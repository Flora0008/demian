#!/bin/bash
# =============================================================================
# RoboCasa instructor SFT — LoRA-finetune Qwen3.5-2B to generate the task's
# annotation from 3 RoboCasa camera views (agentview L/R + eye-in-hand) + the
# task description. Uniform-loss SFT; produces the released RoboCasa instructor
# (checkpoints/instructor/robocasa). Build the data first with
# scripts/prepare_instructor_data_robocasa.py. Auto-resumes from latest ckpt.
#
# Usage:
#   sbatch --nodes=1 --gpus-per-node=<G> [--partition=<p>] [--account=<a>] \
#          scripts/slurm/instructor/train_instructor_sft_robocasa.slurm
#   Override MODEL_NAME / DATA_DIR / OUTPUT_DIR / CONTAINER via env.
# =============================================================================
#SBATCH -J demian_instructor_robocasa_sft
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 4:00:00
#SBATCH --output=logs/instructor_robocasa_sft_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"   # repo root
# Optional: set CONTAINER to a container image to run under `srun --container-image`,
# and CONTAINER_MOUNTS for its bind mounts; if unset, the job runs in the current environment.
CONTAINER="${CONTAINER:-}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/path/to:/path/to}"

MODEL_NAME="${MODEL_NAME:-/path/to/Qwen3.5-2B}"                       # base model (path or HF id)
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/instructor_training_robocasa}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/checkpoints/instructor/robocasa}"

EPOCHS=${EPOCHS:-10}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-16}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-2e-4}
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}

RESUME_FLAG=""
LATEST=$(ls -d ${OUTPUT_DIR}/checkpoint-[0-9]* 2>/dev/null | sort -V | tail -1 || true)
if [ -n "${LATEST}" ]; then
    RESUME_FLAG="--resume-from-checkpoint ${LATEST}"
    echo "Resuming from: ${LATEST}"
fi

mkdir -p ${OUTPUT_DIR} ${PROJECT_DIR}/logs

CMD="
set -ex
cd ${PROJECT_DIR}

pip install -q peft trl 'transformers>=4.53' 'tokenizers>=0.22' 'torchao>=0.16' 2>&1 | tail -3

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} \
    scripts/instructor/train_instructor_robocasa.py \
    --model-name ${MODEL_NAME} \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --per-device-batch-size ${PER_DEVICE_BATCH} \
    --gradient-accumulation-steps ${GRAD_ACCUM} \
    --lr ${LR} \
    --save-steps 100 \
    --logging-steps 10 \
    --bf16 \
    ${RESUME_FLAG}
"

if [ -n "${CONTAINER}" ]; then
    srun --container-image "${CONTAINER}" --container-mounts "${CONTAINER_MOUNTS}" bash -c "${CMD}"
else
    srun bash -c "${CMD}"
fi
echo "Done at \$(date)"
