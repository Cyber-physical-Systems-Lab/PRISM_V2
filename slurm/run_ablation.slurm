#!/bin/bash -l
#SBATCH --job-name=sym_ablation
#SBATCH --output=/proj/symmarl_ijrr2025/xuezhi/runs/ablation_%j.out
#SBATCH --error=/proj/symmarl_ijrr2025/xuezhi/runs/ablation_%j.err
#SBATCH --partition=gpu
#SBATCH --account=UPPMAX2025-2-381
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=32GB

set -eo pipefail

module load CUDA/12.9.0

set +u
source ~/miniconda/etc/profile.d/conda.sh
conda activate battery
set -u

REPO_DIR="/home/xuezhi/robotic-symbiosis-tarware"
RESULTS_DIR="/proj/symmarl_ijrr2025/xuezhi/runs/results"
JOB_ID="${SLURM_JOB_ID:-local}"
RUN_ID="$(date +%y-%m-%d_%H-%M-%S-%6N)_ablation"

# Override at submission time:
#   ENV=tarware-small-4agvs-2pickers-partialobs-chg-v1 TIMESTEPS=1000000 NUM_SEEDS=5 sbatch slurm/run_ablation.slurm
ENV="${ENV:-tarware-tiny-2agvs-1pickers-partialobs-chg-v1}"
TIMESTEPS="${TIMESTEPS:-500000}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SEEDS=$(python -c "import random; print(' '.join(str(random.randint(0, 1000000)) for _ in range($NUM_SEEDS)))")
ROLLOUT="${ROLLOUT:-256}"
PPO_EPOCHS="${PPO_EPOCHS:-3}"
MAX_EP_STEPS="${MAX_EP_STEPS:-500}"
OUTPUT_PATH="${RESULTS_DIR}/ablation_results_${RUN_ID}.json"

cd "${REPO_DIR}"
mkdir -p "${RESULTS_DIR}"

echo "=== Ablation Study ==="
echo "Env:       ${ENV}"
echo "Timesteps: ${TIMESTEPS}  Seeds: ${SEEDS}"
echo "Output:    ${OUTPUT_PATH}"

python experiments/run_ablation.py \
    --env           "${ENV}"          \
    --timesteps     "${TIMESTEPS}"    \
    --seeds         ${SEEDS}          \
    --rollout       "${ROLLOUT}"      \
    --ppo_epochs    "${PPO_EPOCHS}"   \
    --max_ep_steps  "${MAX_EP_STEPS}" \
    --output        "${OUTPUT_PATH}"

echo "=== Done: $? ==="
