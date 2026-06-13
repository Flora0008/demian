#!/bin/bash
# =============================================================================
# DeMiAn-VLA MolmoSpaces eval with the learned INSTRUCTOR (async, paper RQ2).
#
# Co-hosts on one GPU:
#   1. instructor server — openpi/scripts/eval/qwen_instructor_server.py (Qwen3.5-2B).
#   2. the MolmoSpaces eval — molmobot_pi0/eval/eval.py with the mix5 policy and
#      policy.instructor_url set. PiPolicy snapshots the initial exo+wrist frames,
#      fires a non-blocking annotation request to the instructor, acts on the task
#      description alone, and splices the annotation at the next action-chunk
#      boundary.  MODE=sync blocks the first action until the annotation is ready.
#
# Usage:
#   CKPT_DIR=<mix5 ckpt dir> CKPT_STEP=<step> BENCHMARK=<tier3_.../s0> CAMERA=<cam> \
#   INSTRUCTOR_MODEL=<checkpoints/instructor/molmospaces> \
#       sbatch --gpus-per-node=1 [--partition=<p>] [--account=<a>] \
#       scripts/slurm/eval_demian_vla_molmospaces_async.slurm [async|sync]
#
# NOTE: the instructor (qwen3_5) needs a newer `transformers` than the pi05 policy
# (4.53.2). Set INSTRUCTOR_PY to an interpreter/venv with qwen3_5 support if your
# policy env pins 4.53.2; otherwise the same `python` is used for both.
# =============================================================================
#SBATCH -J demian_vla_mlspaces_async
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/eval/demian_vla_mlspaces_async_%j.log

set -euo pipefail

MODE="${1:-${MODE:-async}}"
case "${MODE}" in async) SYNC_ARG="" ;; sync) SYNC_ARG="policy.instructor_sync=true" ;; *)
    echo "MODE must be 'async' or 'sync'" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"            # repo root
OPENPI_DIR="${OPENPI_DIR:-${CODE_DIR}/openpi}"
MOLMOSPACES_DIR="${MOLMOSPACES_DIR:?Set MOLMOSPACES_DIR to the molmospaces repo}"
INSTRUCTOR_MODEL="${INSTRUCTOR_MODEL:?Set INSTRUCTOR_MODEL to the Qwen3.5-2B instructor dir (checkpoints/instructor/molmospaces)}"
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the mix5 policy checkpoint dir}"
CKPT_STEP="${CKPT_STEP:?Set CKPT_STEP}"
BENCHMARK="${BENCHMARK:?Set BENCHMARK (path under BENCH_DIR)}"
CAMERA="${CAMERA:?Set CAMERA (e.g. sim_exo_test)}"
BENCH_DIR="${BENCH_DIR:-${CODE_DIR}/benchmarks}"
EXP_NAME="${EXP_NAME:-demian_vla_mlspaces_instructor_${MODE}}"
CONTAINER="${CONTAINER:-}"; CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/path/to:/path/to}"
N_VIDEOS="${N_VIDEOS:-0}"
CAP_PORT="${CAP_PORT:-8001}"; CAP_DEVICE="${CAP_DEVICE:-cuda:0}"
INSTRUCTOR_PY="${INSTRUCTOR_PY:-python}"

TRANSFORMERS_REPLACE="${OPENPI_DIR}/src/openpi/models_pytorch/transformers_replace"

EVAL_CMD="
set -ex
cd ${CODE_DIR}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cpu WANDB_MODE=disabled
export MLSPACES_ASSETS_DIR=\$(readlink -f ${MOLMOSPACES_DIR}/assets)
export MLSPACES_CACHE_DIR=\$(readlink -f ${MOLMOSPACES_DIR}/.cache)
export PYTHONPATH=${CODE_DIR}:${MOLMOSPACES_DIR}:${OPENPI_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src\${PYTHONPATH:+:\${PYTHONPATH}}

# 1) instructor server (Qwen3.5-2B). Co-hosted on the same GPU (~4GB).
${INSTRUCTOR_PY} ${OPENPI_DIR}/scripts/eval/qwen_instructor_server.py \
    --model-path ${INSTRUCTOR_MODEL} --port ${CAP_PORT} --device ${CAP_DEVICE} &> /tmp/instructor_${CAP_PORT}.log &
CAP_PID=\$!
for i in \$(seq 1 180); do
    bash -c 'echo > /dev/tcp/127.0.0.1/${CAP_PORT}' 2>/dev/null && { echo '[eval] instructor ready'; break; }
    kill -0 \$CAP_PID 2>/dev/null || { echo '[eval] instructor died:'; tail -40 /tmp/instructor_${CAP_PORT}.log; exit 1; }
    sleep 5
done

# 2) policy env transformers patch (PaliGemma/Gemma).
pip install -q transformers==4.53.2 2>/dev/null || true
TRANSFORMERS_DIR=\$(python3 -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')
cp -r ${TRANSFORMERS_REPLACE}/* \"\${TRANSFORMERS_DIR}/\"

# GPU keep-alive during the long CPU-bound MolmoSpaces scene loads.
python3 -c \"
import torch, time, os, signal
signal.signal(signal.SIGTERM, lambda *a: os._exit(0))
while True:
    torch.randn(1, device='cuda'); time.sleep(45)
\" & KEEPALIVE_PID=\$!

python molmobot_pi0/eval/eval.py \
    benchmark=${BENCH_DIR}/${BENCHMARK} \
    policy/cameras=${CAMERA} \
    policy=paligemma \
    policy.model_dir=${CKPT_DIR} \
    policy.checkpoints=[${CKPT_STEP}] \
    policy.action_type=jointvel \
    policy.use_torch=true \
    policy.compile_mode=null \
    policy.instructor_url=http://localhost:${CAP_PORT} ${SYNC_ARG} \
    filament.enabled=false \
    use_policy_server=false \
    num_workers=1 \
    n_videos=${N_VIDEOS} \
    hydra.run.dir=outputs/${EXP_NAME}

kill \$KEEPALIVE_PID \$CAP_PID 2>/dev/null || true
echo 'Done.'
"

if [ -n "${CONTAINER}" ]; then
    srun --container-image "${CONTAINER}" --container-mounts "${CONTAINER_MOUNTS}" bash -c "${EVAL_CMD}"
else
    srun bash -c "${EVAL_CMD}"
fi
