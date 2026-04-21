#!/bin/bash -l
#SBATCH --job-name=sym_conv
#SBATCH --output=/proj/symmarl_ijrr2025/xuezhi/runs/convergence_%j.out
#SBATCH --error=/proj/symmarl_ijrr2025/xuezhi/runs/convergence_%j.err
#SBATCH --partition=gpu
#SBATCH --account=UPPMAX2025-2-381
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB

# Runs C2 Convergence Study: 3 backends × 4 methods × NUM_SEEDS training runs.
# Total: ippo + hetppo + mappo × individual + team + unclassified + symbiotic × NUM_SEEDS
#
# Override at submission time:
#   TIMESTEPS=1000000 NUM_SEEDS=5 sbatch slurm/run_convergence.slurm

set -eo pipefail

module load CUDA/12.9.0

set +u
source ~/miniconda/etc/profile.d/conda.sh
conda activate battery
set -u

REPO_DIR="/home/xuezhi/robotic-symbiosis-tarware"

TIMESTEPS="${TIMESTEPS:-500000}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SEEDS=$(python -c "import random; print(' '.join(str(random.randint(0, 1000000)) for _ in range($NUM_SEEDS)))")
ROLLOUT="${ROLLOUT:-256}"
PPO_EPOCHS="${PPO_EPOCHS:-3}"
MAX_EP_STEPS="${MAX_EP_STEPS:-500}"
RUN_ID="$(date +%y-%m-%d_%H-%M-%S-%6N)_convergence"
OUTPUT_PATH="runs/results/convergence_results_${RUN_ID}.json"

cd "${REPO_DIR}"
mkdir -p runs/results

echo "=== C2 Convergence Study ==="
echo "Timesteps: ${TIMESTEPS}  Seeds: ${SEEDS}  Backends: ippo hetppo mappo"
echo "Output:    ${OUTPUT_PATH}"

python experiments/run_convergence.py \
    --timesteps     "${TIMESTEPS}"    \
    --seeds         ${SEEDS}          \
    --rollout       "${ROLLOUT}"      \
    --ppo_epochs    "${PPO_EPOCHS}"   \
    --max_ep_steps  "${MAX_EP_STEPS}" \
    --backends mappo                  \
    --output        "${OUTPUT_PATH}"

echo "=== Done: $? ==="
