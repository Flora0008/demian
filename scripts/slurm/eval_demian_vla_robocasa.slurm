#!/bin/bash
# =============================================================================
# DeMiAn-VLA RoboCasa365 evaluation (paper Table 1, VLA rows).
#
# Serves a trained pi05 RoboCasa policy over a websocket (openpi `serve_policy`)
# and runs the RoboCasa sim eval harness (`scripts/eval/run_robocasa_eval.py`)
# against it. The annotation aspect is injected at inference via GT captions
# (--gt-reasoning-file + --caption-key); baseline injects none.
#
#   Aspect (paper)      ->  --caption-key
#   ----------------------  --------------------------------------------------
#   baseline            ->  (none)
#   physical_motion     ->  current_action
#   scene_composition   ->  environment-simple
#   arm_pose            ->  agent_arm_start_state
#   reasoning           ->  reasoning-short
#
# Usage:
#   sbatch --gpus-per-node=1 [--partition=<p>] [--account=<a>] \
#          scripts/slurm/eval_demian_vla_robocasa.slurm <aspect>
#
# Requires the RoboCasa + robosuite simulator on PYTHONPATH (external; set
# ROBOCASA_PATH / ROBOSUITE_PATH). Set ACTION_MODEL (trained ckpt dir),
# POLICY_CONFIG, GT_CAPTIONS (GT reasoning/caption JSON), and CONTAINER via env.
# =============================================================================
#SBATCH -J demian_vla_robocasa_eval
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/robocasa/demian_vla_eval_%j.out

set -euo pipefail

ASPECT="${1:-${ASPECT:-baseline}}"
case "${ASPECT}" in
    baseline)          CAPTION_KEY="" ;;
    physical_motion)   CAPTION_KEY="current_action" ;;
    scene_composition) CAPTION_KEY="environment-simple" ;;
    arm_pose)          CAPTION_KEY="agent_arm_start_state" ;;
    reasoning)         CAPTION_KEY="reasoning-short" ;;
    *) echo "Unknown aspect '${ASPECT}'. Choose: baseline physical_motion scene_composition arm_pose reasoning" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${CODE_DIR}/openpi}"            # holds scripts/serve_policy.py + scripts/eval/
ACTION_MODEL="${ACTION_MODEL:?Set ACTION_MODEL to the trained RoboCasa policy checkpoint dir}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_robocasa_lm}"
GT_CAPTIONS="${GT_CAPTIONS:-}"                              # GT reasoning/caption JSON (needed for non-baseline aspects)
ROBOCASA_PATH="${ROBOCASA_PATH:?Set ROBOCASA_PATH to the robocasa repo}"
ROBOSUITE_PATH="${ROBOSUITE_PATH:?Set ROBOSUITE_PATH to the robosuite repo}"
CONTAINER="${CONTAINER:-}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/path/to:/path/to}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/runs/robocasa/demian_vla_${ASPECT}}"

EPISODES="${EPISODES:-100}"
MAX_STEPS="${MAX_STEPS:-400}"
OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-8}"
SEEDS="${SEEDS:-42 43 44 45 46}"
TASK_SHARD="${TASK_SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
PORT="${PORT:-8000}"

if [ "${ASPECT}" != "baseline" ] && [ -z "${GT_CAPTIONS}" ]; then
    echo "ERROR: aspect '${ASPECT}' needs GT_CAPTIONS (the GT reasoning/caption JSON)." >&2; exit 2
fi
CAPTION_FLAGS=""
[ -n "${CAPTION_KEY}" ] && CAPTION_FLAGS="--gt-reasoning-file ${GT_CAPTIONS} --caption-key ${CAPTION_KEY}"

OPENPI_SRC="${PROJECT_DIR}/src"; OPENPI_CLIENT_SRC="${PROJECT_DIR}/packages/openpi-client/src"

EVAL_CMD="
set -e
cd ${PROJECT_DIR}
export PYTHONPATH=${OPENPI_SRC}:${OPENPI_CLIENT_SRC}:${PROJECT_DIR}/scripts/eval:${ROBOCASA_PATH}:${ROBOSUITE_PATH}\${PYTHONPATH:+:\${PYTHONPATH}}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

echo '[eval] starting policy server (${POLICY_CONFIG}) on port ${PORT} ...'
MAX_REASONING_TOKENS=0 python scripts/serve_policy.py \
    --port ${PORT} policy:checkpoint \
    --policy.config ${POLICY_CONFIG} --policy.dir ${ACTION_MODEL} &> /tmp/serve_${PORT}.log &
SERVER_PID=\$!

for i in \$(seq 1 180); do
    bash -c 'echo > /dev/tcp/127.0.0.1/${PORT}' 2>/dev/null && { echo '[eval] server ready'; break; }
    kill -0 \$SERVER_PID 2>/dev/null || { echo '[eval] server died:'; tail -50 /tmp/serve_${PORT}.log; exit 1; }
    sleep 5
done

for SEED in ${SEEDS}; do
    RDIR=${RESULTS_ROOT}/seed\${SEED}
    mkdir -p \$RDIR
    echo \"[eval] aspect=${ASPECT} seed=\${SEED}\"
    MUJOCO_EGL_DEVICE_ID=0 python scripts/eval/run_robocasa_eval.py \
        --host localhost --port ${PORT} --split atomic \
        --episodes ${EPISODES} --max-steps ${MAX_STEPS} --open-loop-horizon ${OPEN_LOOP_HORIZON} \
        --seed \${SEED} --task-shard ${TASK_SHARD} --num-shards ${NUM_SHARDS} \
        --checkpoint ${ACTION_MODEL} --results-dir \$RDIR --headless ${CAPTION_FLAGS}
done
kill \$SERVER_PID 2>/dev/null || true
"

if [ -n "${CONTAINER}" ]; then
    srun --container-image "${CONTAINER}" --container-mounts "${CONTAINER_MOUNTS}" bash -c "${EVAL_CMD}"
else
    srun bash -c "${EVAL_CMD}"
fi

echo "Aggregate with: python ${PROJECT_DIR}/scripts/eval/aggregate_robocasa_quick_shards.py --results-dir ${RESULTS_ROOT}"
