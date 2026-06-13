#!/bin/bash
# =============================================================================
# DeMiAn-VLA composite-task evaluation (paper RQ3 / Table tab:composite-aggregate).
#
# Runs the RoboCasa composite tasks (`--split composite`) under one prompting mode:
#   fix                — one composite description for the whole episode (OOD format).
#   dynamic            — subgoal-driven: swap to the in-distribution atomic prompt
#                        for the current phase (GT atomic prompts).
#   dynamic-gtcap      — dynamic + inject the GT atomic caption (swap both).
#   dynamic-instructor — dynamic + the learned Qwen instructor supplies the atomic
#                        annotation at deployment (the full end-to-end method).
#
# Pick the policy via env: baseline = pi05_robocasa, DeMiAn-VLA = pi05_robocasa_lm.
#
# Usage:
#   ACTION_MODEL=<ckpt> POLICY_CONFIG=<cfg> \
#       sbatch --gpus-per-node=1 [--partition=<p>] [--account=<a>] \
#       scripts/slurm/eval_demian_vla_composite.slurm <mode>
#
# Composite subgoal plans/checkers live in scripts/eval/composite_subgoals.py; GT
# composite captions in scripts/eval/gt_captions_composite.json (18 tasks).
# =============================================================================
#SBATCH -J demian_vla_composite_eval
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/robocasa/demian_vla_composite_%j.out

set -euo pipefail

MODE="${1:-${MODE:-dynamic}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${CODE_DIR}/openpi}"
ACTION_MODEL="${ACTION_MODEL:?Set ACTION_MODEL to the trained policy ckpt}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_robocasa_lm}"
ROBOCASA_PATH="${ROBOCASA_PATH:?Set ROBOCASA_PATH}"; ROBOSUITE_PATH="${ROBOSUITE_PATH:?Set ROBOSUITE_PATH}"
CONTAINER="${CONTAINER:-}"; CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/path/to:/path/to}"
GT_COMPOSITE="${GT_COMPOSITE:-${PROJECT_DIR}/scripts/eval/gt_captions_composite.json}"
CAPTION_KEY="${CAPTION_KEY:-current_action}"
INSTRUCTOR_MODEL="${INSTRUCTOR_MODEL:-}"
EVAL_TASKS="${EVAL_TASKS:-}"                 # space-separated composite task names; empty = all composite
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/runs/robocasa/composite_${MODE}}"
EPISODES="${EPISODES:-20}"; MAX_STEPS="${MAX_STEPS:-1200}"; OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-8}"
SEED="${SEED:-42}"; SERVER_PORT="${SERVER_PORT:-8000}"; CAP_PORT="${CAP_PORT:-8001}"; CAP_DEVICE="${CAP_DEVICE:-cuda:0}"

# Mode -> harness flags + whether the instructor server is needed.
SUBGOAL_FLAGS=""; CAPTION_FLAGS=""; NEED_INSTRUCTOR=0
case "${MODE}" in
    fix)                : ;;
    dynamic)            SUBGOAL_FLAGS="--subgoal-driven --subgoal-swap-mode prompt" ;;
    dynamic-gtcap)      SUBGOAL_FLAGS="--subgoal-driven --subgoal-swap-mode both"
                        CAPTION_FLAGS="--gt-reasoning-file ${GT_COMPOSITE} --caption-key ${CAPTION_KEY}" ;;
    dynamic-instructor) SUBGOAL_FLAGS="--subgoal-driven --subgoal-swap-mode both"
                        CAPTION_FLAGS="--instructor-type qwen --instructor-model-path http://localhost:${CAP_PORT}"
                        NEED_INSTRUCTOR=1
                        [ -n "${INSTRUCTOR_MODEL}" ] || { echo "dynamic-instructor needs INSTRUCTOR_MODEL" >&2; exit 2; } ;;
    *) echo "MODE must be: fix | dynamic | dynamic-gtcap | dynamic-instructor" >&2; exit 2 ;;
esac
TASK_FLAG=""; [ -n "${EVAL_TASKS}" ] && TASK_FLAG="--tasks ${EVAL_TASKS}"
OPENPI_SRC="${PROJECT_DIR}/src"; OPENPI_CLIENT_SRC="${PROJECT_DIR}/packages/openpi-client/src"

EVAL_CMD="
set -e
cd ${PROJECT_DIR}
export PYTHONPATH=${OPENPI_SRC}:${OPENPI_CLIENT_SRC}:${PROJECT_DIR}/scripts/eval:${ROBOCASA_PATH}:${ROBOSUITE_PATH}\${PYTHONPATH:+:\${PYTHONPATH}}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

CAP_PID=''
if [ ${NEED_INSTRUCTOR} -eq 1 ]; then
    echo '[eval] starting instructor server on ${CAP_PORT}'
    python scripts/eval/qwen_instructor_server.py --model-path ${INSTRUCTOR_MODEL} \
        --port ${CAP_PORT} --device ${CAP_DEVICE} &> /tmp/instructor_${CAP_PORT}.log &
    CAP_PID=\$!
fi

MAX_REASONING_TOKENS=0 python scripts/serve_policy.py --port ${SERVER_PORT} \
    policy:checkpoint --policy.config ${POLICY_CONFIG} --policy.dir ${ACTION_MODEL} &> /tmp/serve_${SERVER_PORT}.log &
SRV_PID=\$!
for P in ${SERVER_PORT} \$([ ${NEED_INSTRUCTOR} -eq 1 ] && echo ${CAP_PORT}); do
  for i in \$(seq 1 180); do bash -c 'echo > /dev/tcp/127.0.0.1/'\$P 2>/dev/null && break; sleep 5; done
done

mkdir -p ${RESULTS_ROOT}
echo '[eval] composite mode=${MODE} policy=${POLICY_CONFIG}'
MUJOCO_EGL_DEVICE_ID=0 python scripts/eval/run_robocasa_eval.py \
    --host localhost --port ${SERVER_PORT} --split composite ${TASK_FLAG} \
    --episodes ${EPISODES} --max-steps ${MAX_STEPS} --open-loop-horizon ${OPEN_LOOP_HORIZON} \
    --seed ${SEED} --checkpoint ${ACTION_MODEL} --results-dir ${RESULTS_ROOT} --headless \
    ${SUBGOAL_FLAGS} ${CAPTION_FLAGS}
kill \$SRV_PID \${CAP_PID:-} 2>/dev/null || true
"

if [ -n "${CONTAINER}" ]; then
    srun --container-image "${CONTAINER}" --container-mounts "${CONTAINER_MOUNTS}" bash -c "${EVAL_CMD}"
else
    srun bash -c "${EVAL_CMD}"
fi
echo "Aggregate: python ${PROJECT_DIR}/scripts/eval/merge_matrix_results.py --results-dir ${RESULTS_ROOT}"
