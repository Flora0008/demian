#!/bin/bash
# =============================================================================
# DeMiAn-VLA RoboCasa eval with the learned INSTRUCTOR (async, paper RQ2 / Table
# tab:async-vs-sync). Three servers cooperate on one node:
#   1. policy server   — serve_policy.py (the mix5 action policy)
#   2. instructor server — qwen_instructor_server.py (the Qwen3.5-2B instructor)
#   3. eval harness     — run_robocasa_eval.py, which (a) snapshots the initial
#      frames, (b) fires a non-blocking caption request to the instructor, and
#      (c) splices the returned annotation into the policy prompt at the next
#      action-chunk boundary.  MODE=sync blocks the first action until the
#      instruction is ready (the synchronous baseline).
#
# Usage:
#   ACTION_MODEL=<mix5 ckpt> INSTRUCTOR_MODEL=<qwen instructor merged dir> \
#       sbatch --gpus-per-node=1 [--partition=<p>] [--account=<a>] \
#       scripts/slurm/eval_demian_vla_robocasa_async.slurm [async|sync]
#
# Set ROBOCASA_PATH / ROBOSUITE_PATH (external sim) and CONTAINER via env.
# =============================================================================
#SBATCH -J demian_vla_async_eval
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/robocasa/demian_vla_async_eval_%j.out

set -euo pipefail

MODE="${1:-${MODE:-async}}"
case "${MODE}" in async) SYNC_FLAG="" ;; sync) SYNC_FLAG="--sync-instructor" ;; *)
    echo "MODE must be 'async' or 'sync'" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${CODE_DIR}/openpi}"
ACTION_MODEL="${ACTION_MODEL:?Set ACTION_MODEL to the trained mix5 RoboCasa policy ckpt}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_robocasa_lm}"
INSTRUCTOR_MODEL="${INSTRUCTOR_MODEL:?Set INSTRUCTOR_MODEL to the Qwen3.5-2B instructor (merged) dir}"
ROBOCASA_PATH="${ROBOCASA_PATH:?Set ROBOCASA_PATH}"; ROBOSUITE_PATH="${ROBOSUITE_PATH:?Set ROBOSUITE_PATH}"
CONTAINER="${CONTAINER:-}"; CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/path/to:/path/to}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/runs/robocasa/demian_vla_instructor_${MODE}}"

EPISODES="${EPISODES:-100}"; MAX_STEPS="${MAX_STEPS:-400}"; OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-8}"
SEEDS="${SEEDS:-42 43 44 45 46}"; TASK_SHARD="${TASK_SHARD:-0}"; NUM_SHARDS="${NUM_SHARDS:-1}"
SERVER_PORT="${SERVER_PORT:-8000}"; CAP_PORT="${CAP_PORT:-8001}"
# Instructor + policy co-host on one GPU (~10GB) per the paper; override CAP_DEVICE to split.
CAP_DEVICE="${CAP_DEVICE:-cuda:0}"
OPENPI_SRC="${PROJECT_DIR}/src"; OPENPI_CLIENT_SRC="${PROJECT_DIR}/packages/openpi-client/src"

EVAL_CMD="
set -e
cd ${PROJECT_DIR}
export PYTHONPATH=${OPENPI_SRC}:${OPENPI_CLIENT_SRC}:${PROJECT_DIR}/scripts/eval:${ROBOCASA_PATH}:${ROBOSUITE_PATH}\${PYTHONPATH:+:\${PYTHONPATH}}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

echo '[eval] starting instructor (qwen) server on ${CAP_PORT} (${CAP_DEVICE})'
python scripts/eval/qwen_instructor_server.py --model-path ${INSTRUCTOR_MODEL} \
    --port ${CAP_PORT} --device ${CAP_DEVICE} &> /tmp/instructor_${CAP_PORT}.log &
CAP_PID=\$!

echo '[eval] starting policy server (${POLICY_CONFIG}) on ${SERVER_PORT}'
MAX_REASONING_TOKENS=0 python scripts/serve_policy.py --port ${SERVER_PORT} \
    policy:checkpoint --policy.config ${POLICY_CONFIG} --policy.dir ${ACTION_MODEL} &> /tmp/serve_${SERVER_PORT}.log &
SRV_PID=\$!

for P in ${SERVER_PORT} ${CAP_PORT}; do
  for i in \$(seq 1 180); do
    bash -c 'echo > /dev/tcp/127.0.0.1/'\$P 2>/dev/null && { echo \"[eval] port \$P ready\"; break; }
    sleep 5
  done
done

for SEED in ${SEEDS}; do
    RDIR=${RESULTS_ROOT}/seed\${SEED}; mkdir -p \$RDIR
    echo \"[eval] instructor ${MODE} seed=\${SEED}\"
    MUJOCO_EGL_DEVICE_ID=0 python scripts/eval/run_robocasa_eval.py \
        --host localhost --port ${SERVER_PORT} --split atomic \
        --episodes ${EPISODES} --max-steps ${MAX_STEPS} --open-loop-horizon ${OPEN_LOOP_HORIZON} \
        --seed \${SEED} --task-shard ${TASK_SHARD} --num-shards ${NUM_SHARDS} \
        --checkpoint ${ACTION_MODEL} --results-dir \$RDIR --headless \
        --instructor-type qwen --instructor-model-path http://localhost:${CAP_PORT} ${SYNC_FLAG}
done
kill \$SRV_PID \$CAP_PID 2>/dev/null || true
"

if [ -n "${CONTAINER}" ]; then
    srun --container-image "${CONTAINER}" --container-mounts "${CONTAINER_MOUNTS}" bash -c "${EVAL_CMD}"
else
    srun bash -c "${EVAL_CMD}"
fi
echo "Aggregate: python ${PROJECT_DIR}/scripts/eval/aggregate_robocasa_quick_shards.py --results-dir ${RESULTS_ROOT}"
