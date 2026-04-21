#!/bin/bash -l
#SBATCH --job-name=sym_grad
#SBATCH --output=/proj/symmarl_ijrr2025/xuezhi/runs/gradient_%j.out
#SBATCH --error=/proj/symmarl_ijrr2025/xuezhi/runs/gradient_%j.err
#SBATCH --partition=gpu
#SBATCH --account=UPPMAX2025-2-381
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

# Runs C3 Heterogeneity Gradient Sweep:
#   3 h-levels × 3 backends × 4 methods × NUM_SEEDS runs
# h=0.0 → tarware-tiny-4agvs-0pickers-partialobs-chg-v1  (homogeneous)
# h=0.5 → tarware-tiny-2agvs-1pickers-partialobs-chg-v1
# h=1.0 → tarware-small-4agvs-2pickers-partialobs-chg-v1
#
# Override at submission time:
#   TIMESTEPS=1000000 NUM_SEEDS=5 sbatch slurm/run_gradient.slurm

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
RUN_ID="$(date +%y-%m-%d_%H-%M-%S-%6N)_gradient"
OUTPUT_PATH="runs/results/gradient_results_${RUN_ID}.json"

cd "${REPO_DIR}"
mkdir -p runs/results

echo "=== C3 Heterogeneity Gradient Sweep ==="
echo "Timesteps: ${TIMESTEPS}  Seeds: ${SEEDS}  h-levels: 0.0 0.5 1.0"
echo "Output:    ${OUTPUT_PATH}"

python experiments/run_gradient.py \
    --timesteps     "${TIMESTEPS}"    \
    --seeds         ${SEEDS}          \
    --rollout       "${ROLLOUT}"      \
    --ppo_epochs    "${PPO_EPOCHS}"   \
    --max_ep_steps  "${MAX_EP_STEPS}" \
    --h_values 0.0 0.5 1.0           \
    --backends mappo                  \
    --output        "${OUTPUT_PATH}"

echo "=== Done: $? ==="
