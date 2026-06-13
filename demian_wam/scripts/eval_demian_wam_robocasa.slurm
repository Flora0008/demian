#!/bin/bash
# =============================================================================
# DeMiAn-WAM RoboCasa365 evaluation (paper Table 1, WAM rows).
#
# Three steps (all on a single GPU here; shard with TASK_SHARD/NUM_SHARDS):
#   1. Convert the trained DCP checkpoint to a regular .pt once
#      (scripts/convert_dcp_to_regular.py).
#   2. Serve the WAM policy over a socket (socket_test_cosmos_wam.py).
#   3. Run the RoboCasa sim eval harness against it
#      (eval_utils/run_robocasa_eval.py, via openpi_client WebsocketClientPolicy).
#
# The aspect is the trained WAM experiment (EXPERIMENT, e.g. *_wam_ep_action) and/
# or an inference-time GT caption injection (CAPTION_KEY + GT_REASONING_FILE).
#
# Usage:
#   EXPERIMENT=<wam experiment> CKPT_DCP=<iter_dir> \
#       sbatch --gpus-per-node=1 [--partition=<p>] [--account=<a>] \
#       scripts/eval_demian_wam_robocasa.slurm
#
# PROJECT_DIR must be a Cosmos-Predict2.5 checkout with the demian_wam overlays
# applied + the RoboCasa/robosuite simulator on PYTHONPATH (ROBOCASA_PATH/ROBOSUITE_PATH).
# =============================================================================
#SBATCH -J demian_wam_robocasa_eval
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/robocasa/demian_wam_eval_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the Cosmos-Predict2.5 checkout with demian_wam overlays applied}"
EXPERIMENT="${EXPERIMENT:?Set EXPERIMENT to the trained WAM experiment (e.g. dreamdojo_2b_480_640_robocasa_wam_baseline)}"
CKPT_DCP="${CKPT_DCP:?Set CKPT_DCP to the DCP checkpoint dir (iter_XXXXXXXX) to evaluate}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:?Set TOKENIZER_CKPT to the Cosmos-Predict2.5-2B tokenizer .pth}"
ROBOCASA_PATH="${ROBOCASA_PATH:?Set ROBOCASA_PATH to the robocasa repo}"
ROBOSUITE_PATH="${ROBOSUITE_PATH:?Set ROBOSUITE_PATH to the robosuite repo}"
CONTAINER="${CONTAINER:-}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/path/to:/path/to}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/runs/robocasa/$(basename "${EXPERIMENT}")}"

# Inference-time GT caption injection (optional; for caption-conditioned eval).
GT_REASONING_FILE="${GT_REASONING_FILE:-}"
CAPTION_KEY="${CAPTION_KEY:-}"
REASONING_ARGS=""
if [ -n "${GT_REASONING_FILE}" ] && [ -n "${CAPTION_KEY}" ]; then
    REASONING_ARGS="--gt-reasoning-file ${GT_REASONING_FILE} --caption-key ${CAPTION_KEY}"
fi

# Sampler / eval knobs.
GUIDANCE="${GUIDANCE:-7}"; SHIFT="${SHIFT:-5}"; NUM_STEPS="${NUM_STEPS:-35}"
ROBOCASA_VIDEO_MODE="${ROBOCASA_VIDEO_MODE:-first_frame}"
EPISODES="${EPISODES:-100}"; OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-8}"
SEED="${SEED:-42}"; TASK_SHARD="${TASK_SHARD:-0}"; NUM_SHARDS="${NUM_SHARDS:-1}"
TASKS="${TASKS:-}"; PORT="${PORT:-6000}"
REGULAR_CKPT="${CKPT_DCP}/model_regular.pt"

EVAL_CMD="
set -e
cd ${PROJECT_DIR}
export PYTHONPATH=${PROJECT_DIR}:${ROBOCASA_PATH}:${ROBOSUITE_PATH}\${PYTHONPATH:+:\${PYTHONPATH}}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0

# Step 1: DCP -> regular .pt (once)
if [ ! -f '${REGULAR_CKPT}' ]; then
    echo '[eval] converting DCP checkpoint -> regular .pt'
    torchrun --nproc_per_node=1 scripts/convert_dcp_to_regular.py \
        --checkpoint-path ${CKPT_DCP} --experiment ${EXPERIMENT} \
        --tokenizer-pth ${TOKENIZER_CKPT} --output ${REGULAR_CKPT}
fi

# Step 2: serve the WAM policy
echo '[eval] starting WAM policy server on port ${PORT}'
python socket_test_cosmos_wam.py --port ${PORT} --single-gpu \
    --checkpoint-path ${REGULAR_CKPT} --experiment ${EXPERIMENT} \
    --guidance ${GUIDANCE} --shift ${SHIFT} --num-steps ${NUM_STEPS} \
    --robocasa-video-mode ${ROBOCASA_VIDEO_MODE} &> /tmp/wam_server_${PORT}.log &
SERVER_PID=\$!
for i in \$(seq 1 180); do
    bash -c 'echo > /dev/tcp/127.0.0.1/${PORT}' 2>/dev/null && { echo '[eval] server ready'; break; }
    kill -0 \$SERVER_PID 2>/dev/null || { echo '[eval] server died:'; tail -50 /tmp/wam_server_${PORT}.log; exit 1; }
    sleep 5
done

# Step 3: run the RoboCasa eval harness
TASK_FLAG=''; [ -n '${TASKS}' ] && TASK_FLAG='--tasks ${TASKS}' || TASK_FLAG='--split atomic'
python eval_utils/run_robocasa_eval.py --host localhost --port ${PORT} \
    \${TASK_FLAG} --episodes ${EPISODES} --max-steps 400 \
    --open-loop-horizon ${OPEN_LOOP_HORIZON} --seed ${SEED} \
    --task-shard ${TASK_SHARD} --num-shards ${NUM_SHARDS} \
    --checkpoint ${CKPT_DCP} --results-dir ${RESULTS_DIR} ${REASONING_ARGS} --headless
kill \$SERVER_PID 2>/dev/null || true
"

if [ -n "${CONTAINER}" ]; then
    srun --container-image "${CONTAINER}" --container-mounts "${CONTAINER_MOUNTS}" --container-writable bash -c "${EVAL_CMD}"
else
    srun bash -c "${EVAL_CMD}"
fi

echo "Aggregate with: python ${PROJECT_DIR}/eval_utils/merge_results.py --results-dir ${RESULTS_DIR}"
