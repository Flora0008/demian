#!/bin/bash
# =============================================================================
# DeMiAn-WAM training launcher (Cosmos-Predict 2.5 DiT backbone + action head).
#
# Trains one annotation-aspect ablation on one embodiment. The DiT backbone,
# visual data, and action targets are identical across aspects; only the
# language conditioning changes.
#
# Two embodiments (paper Table 1 evaluates DeMiAn-WAM on both):
#   robocasa  — RoboCasa365; captions selected via the groot_dreams `caption_mode`
#               Hydra override (ep_* keys); action_dim 7, 12 action steps.
#   molmobot  — MolmoBot/MolmoSpaces; reuses DeMiAn-VLA's MlSpacesDataset via the
#               MlSpacesVideoActionDataset adapter, captions selected via the
#               MLSPACES_CAPTION_KEY env (Pi0 keys); action_dim 8, 16 action steps.
#
#   aspect              robocasa caption_mode   molmobot MLSPACES_CAPTION_KEY
#   ------------------  ----------------------  ---------------------------------
#   baseline            none                    (none — MLSPACES_CAPTIONS_DIR unset)
#   physical_motion     ep_action               current_action
#   scene_composition   ep_env_simple           environment-simple
#   arm_pose            ep_arm_state            agent_arm_state
#   reasoning           ep_reasoning            reasoning-short
#   mix5                random:none,ep_action,  random:none,current_action,
#                       ep_env_simple,          reasoning-short,environment-simple,
#                       ep_reasoning,ep_arm_state   agent_arm_state
#
# Usage:
#   sbatch --nodes=<N> --gpus-per-node=<G> \
#          [--partition=<p>] [--account=<a>] \
#          scripts/train_demian_wam.slurm <embodiment> <aspect>
#
#   e.g.  sbatch --nodes=1 --gpus-per-node=8 scripts/train_demian_wam.slurm robocasa mix5
#         sbatch --nodes=1 --gpus-per-node=8 scripts/train_demian_wam.slurm molmobot reasoning
#
# Node count / GPUs / partition / account are passed at submit time (not
# hardcoded). PROJECT_DIR must be a Cosmos-Predict2.5 checkout with the DeMiAn
# overlays from code/demian_wam applied (see code/demian_wam/README).
# =============================================================================
#SBATCH -J demian_wam_train
#SBATCH -t 04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/demian_wam_train_%j.out

set -euo pipefail

EMBODIMENT="${1:-${EMBODIMENT:-robocasa}}"
ASPECT="${2:-${ASPECT:-mix5}}"

case "${EMBODIMENT}" in robocasa|molmobot) ;; *)
    echo "Unknown embodiment: '${EMBODIMENT}'. Choose: robocasa | molmobot" >&2; exit 2 ;;
esac

# Aspect -> caption key, per embodiment (see table above).
if [ "${EMBODIMENT}" = "robocasa" ]; then
    case "${ASPECT}" in
        baseline)          CAP="none" ;;
        physical_motion)   CAP="ep_action" ;;
        scene_composition) CAP="ep_env_simple" ;;
        arm_pose)          CAP="ep_arm_state" ;;
        reasoning)         CAP="ep_reasoning" ;;
        mix5)              CAP="random:none,ep_action,ep_env_simple,ep_reasoning,ep_arm_state" ;;
        *) echo "Unknown aspect '${ASPECT}'" >&2; exit 2 ;;
    esac
    EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_256_robocasa_action_head}"
else  # molmobot
    case "${ASPECT}" in
        baseline)          CAP="" ;;
        physical_motion)   CAP="current_action" ;;
        scene_composition) CAP="environment-simple" ;;
        arm_pose)          CAP="agent_arm_state" ;;
        reasoning)         CAP="reasoning-short" ;;
        mix5)              CAP="random:none,current_action,reasoning-short,environment-simple,agent_arm_state" ;;
        *) echo "Unknown aspect '${ASPECT}'" >&2; exit 2 ;;
    esac
    EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_256_molmobot_action_head}"
fi
EXP_NAME="demian_wam_${EMBODIMENT}_${ASPECT}"

# ----------------------------------------------------------------------------
# Site-specific configuration (override via environment).
#   PROJECT_DIR    Cosmos-Predict2.5 checkout WITH the demian_wam overlays applied.
#   DATASET_PATH   LeRobot (robocasa) or MlSpaces (molmobot) dataset root.
#   PRETRAIN_CKPT  Cosmos-Predict2.5-2B base checkpoint (DCP dir).
#   TOKENIZER_CKPT Cosmos-Predict2.5-2B tokenizer (.pth).
#   CONTAINER      training image for `srun --container-image` (optional; if unset
#                  the job runs in the current environment).
# molmobot only:
#   MOLMOBOT_PI0_PATH    repo root that holds `molmobot_pi0` (for the MlSpaces adapter).
#   MLSPACES_CAPTIONS_DIR  dir with the molmobot dense-caption JSONLs (data/molmobot-captions).
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the Cosmos-Predict2.5 checkout with demian_wam overlays applied}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the dataset root}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:?Set PRETRAIN_CKPT to the Cosmos-Predict2.5-2B base checkpoint dir}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:?Set TOKENIZER_CKPT to the Cosmos-Predict2.5-2B tokenizer .pth}"
CONTAINER="${CONTAINER:-}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/path/to:/path/to}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/demian_wam_logs}"

# molmobot caption conditioning is passed to the MlSpaces adapter via env.
MOLMOBOT_ENV=""
CAPTION_OVERRIDE=""
if [ "${EMBODIMENT}" = "robocasa" ]; then
    CAPTION_HYDRA="${CAP//,/\\,}"
    CAPTION_OVERRIDE="dataloader_train.dataset.caption_mode='${CAPTION_HYDRA}'"
else
    MOLMOBOT_PI0_PATH="${MOLMOBOT_PI0_PATH:?molmobot embodiment requires MOLMOBOT_PI0_PATH (the repo root holding molmobot_pi0)}"
    CAPTIONS_DIR="${MLSPACES_CAPTIONS_DIR:-}"
    if [ "${ASPECT}" != "baseline" ] && [ -z "${CAPTIONS_DIR}" ]; then
        echo "ERROR: aspect '${ASPECT}' needs MLSPACES_CAPTIONS_DIR (molmobot dense captions)." >&2; exit 2
    fi
    MOLMOBOT_ENV="export MOLMOBOT_PI0_PATH=${MOLMOBOT_PI0_PATH}; export MLSPACES_CAPTIONS_DIR=${CAPTIONS_DIR}; export MLSPACES_CAPTION_KEY=${CAP};"
fi

MAX_ITER="${MAX_ITER:-100000}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"
PER_GPU_BATCH="${PER_GPU_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
ACTION_HEAD_LR="${ACTION_HEAD_LR:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-3000}"

CKPT_BASE="${OUTPUT_ROOT}/demian_wam/${EXP_NAME}/checkpoints"
mkdir -p logs "${CKPT_BASE}"

# ----------------------------------------------------------------------------
# Resume from the latest *complete* checkpoint; resubmit on wall-time signal.
# ----------------------------------------------------------------------------
is_complete_ckpt() {
    local ckpt=$1 key
    [ -d "$ckpt" ] && [ -f "$ckpt/aux_optim.pt" ] || return 1
    for key in model optim scheduler trainer; do
        [ -f "$ckpt/$key/.metadata" ] || return 1
        find "$ckpt/$key" -maxdepth 1 -type f -name '.__*.distcp*' 2>/dev/null | grep -q . && return 1
    done
    return 0
}
find_latest_complete_ckpt() {
    [ -d "$CKPT_BASE" ] || return 1
    local ckpt
    while IFS= read -r ckpt; do
        is_complete_ckpt "$ckpt" && { echo "$ckpt"; return 0; }
    done < <(find "$CKPT_BASE" -maxdepth 1 -type d -name 'iter_*' 2>/dev/null | sort -r)
    return 1
}
latest_complete_iter() {
    local l; l=$(find_latest_complete_ckpt 2>/dev/null || true)
    [ -z "$l" ] && echo 0 || basename "$l" | sed 's/iter_0*//'
}
resubmit_if_needed() {
    if [ "$(latest_complete_iter)" -lt "$MAX_ITER" ]; then
        echo "[signal] resubmitting ${EXP_NAME} ..."
        sbatch -N "${NUM_NODES}" --export=ALL "${SCRIPT_PATH}" "${EMBODIMENT}" "${ASPECT}" || true
    fi
}
trap resubmit_if_needed USR1

LATEST_CKPT=$(find_latest_complete_ckpt 2>/dev/null || true)
if [ -n "${LATEST_CKPT}" ]; then
    LOAD_PATH="${LATEST_CKPT}";   LOAD_TRAINING_STATE=True
else
    LOAD_PATH="${PRETRAIN_CKPT}"; LOAD_TRAINING_STATE=False
fi

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT="${MASTER_PORT:-29600}"

echo "=========================================================="
echo "DeMiAn-WAM | embodiment=${EMBODIMENT} aspect=${ASPECT}"
echo "experiment=${EXPERIMENT} | caption=${CAP:-<none>}"
echo "nodes=${NUM_NODES} gpus/node=${GPUS_PER_NODE} | load=${LOAD_PATH}"
echo "=========================================================="

TRAIN_CMD="
set -e
cd ${PROJECT_DIR}
${MOLMOBOT_ENV}
export MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}
export PYTHONPATH=${PROJECT_DIR}\${PYTHONPATH:+:\${PYTHONPATH}}
export OMP_NUM_THREADS=8
export IMAGINAIRE_OUTPUT_ROOT=${OUTPUT_ROOT}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY

python -m torch.distributed.run \
    --nnodes=${NUM_NODES} --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    --node_rank=\${SLURM_NODEID} \
    -m cosmos_oss.scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
    experiment=${EXPERIMENT} \
    model.config.fsdp_shard_size=1 \
    +model.config.tokenizer.vae_pth=${TOKENIZER_CKPT} \
    trainer.max_iter=${MAX_ITER} \
    trainer.logging_iter=10 \
    optimizer.lr=1e-4 \
    optimizer.weight_decay=1e-5 \
    scheduler.warm_up_steps='[${WARMUP_STEPS}]' \
    checkpoint.save_iter=500 \
    checkpoint.load_path=${LOAD_PATH} \
    checkpoint.load_training_state=${LOAD_TRAINING_STATE} \
    checkpoint.strict_resume=False \
    trainer.callbacks.every_n_sample_reg.every_n=1000000000000 \
    trainer.callbacks.every_n_sample_ema.every_n=1000000000000 \
    trainer.grad_accum_iter=${GRAD_ACCUM} \
    model.config.action_head_lr=${ACTION_HEAD_LR} \
    model.config.action_head_warmup_steps=${WARMUP_STEPS} \
    dataloader_train.batch_size=${PER_GPU_BATCH} \
    ~dataloader_train.dataloaders \
    dataloader_train.dataset.dataset_path='[${DATASET_PATH}]' \
    dataloader_val.dataset.dataset_path='[${DATASET_PATH}]' \
    ${CAPTION_OVERRIDE} \
    job.group=demian_wam \
    job.name=${EXP_NAME}
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
