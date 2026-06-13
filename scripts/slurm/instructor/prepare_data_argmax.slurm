#!/bin/bash
#SBATCH -J molmobot_instructor_prep_argmax
#SBATCH -p <partition>
#SBATCH -A project
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 1:30:00
#SBATCH --output=/path/to/workspace/molmo/MolmoBot/MolmoBot-Pi0/logs/instructor_prep_argmax_%j.log

set -euo pipefail

WORKSPACE=/path/to/workspace
PROJECT_DIR=${WORKSPACE}/molmo/MolmoBot/MolmoBot-Pi0
# Optional: set CONTAINER to a container image to run under `srun --container-image`,
# and CONTAINER_MOUNTS for its bind mounts; if unset, the job runs in the current environment.
CONTAINER="${CONTAINER:-}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/path/to:/path/to}"

CMD="
set -ex
cd ${PROJECT_DIR}

pip install -q opencv-python-headless 2>&1 | tail -3

python3 scripts/prepare_instructor_data_molmobot.py \
    --routing-json runs/molmobot/reward_table_mix5_argmax/routing_table.json \
    --reward-json  runs/molmobot/reward_table_mix5_argmax/reward_table.json \
    --output-dir   data/instructor_training_mix5_argmax \
    --max-per-config 5000 \
    --seed 42
"

if [ -n "${CONTAINER}" ]; then
    srun --container-image "${CONTAINER}" --container-mounts "${CONTAINER_MOUNTS}" bash -c "${CMD}"
else
    srun bash -c "${CMD}"
fi
echo "Done at \$(date)"
