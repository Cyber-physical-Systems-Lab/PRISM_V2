#!/bin/bash -l
#SBATCH --job-name=sym_scale
#SBATCH --output=/proj/symmarl_ijrr2025/xuezhi/runs/scalability_%j.out
#SBATCH --error=/proj/symmarl_ijrr2025/xuezhi/runs/scalability_%j.err
#SBATCH --partition=gpu
#SBATCH --account=UPPMAX2025-2-381
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB

# Runs individual / symbiotic (unnorm + norm) / team across tiny, small, medium.
# Total: 5 envs × 2 variants × 3 methods × NUM_SEEDS runs.
#
# Override at submission time:
#   TIMESTEPS=1000000 NUM_SEEDS=5 sbatch slurm/run_scalability.slurm
# Fast debug run (recommended while iterating):
#   FAST_MODE=1 sbatch slurm/run_scalability.slurm
# Full paper run:
#   FAST_MODE=0 TIMESTEPS=500000 NUM_SEEDS=5 sbatch slurm/run_scalability.slurm

set -eo pipefail

module load CUDA/12.9.0

set +u
source ~/miniconda/etc/profile.d/conda.sh
conda activate battery
set -u

REPO_DIR="/home/xuezhi/robotic-symbiosis-tarware"

FAST_MODE="${FAST_MODE:-1}"
if [[ "${FAST_MODE}" == "1" ]]; then
    TIMESTEPS="${TIMESTEPS:-200000}"
    PPO_EPOCHS="${PPO_EPOCHS:-2}"
else
    TIMESTEPS="${TIMESTEPS:-500000}"
    PPO_EPOCHS="${PPO_EPOCHS:-3}"
fi
ROLLOUT="${ROLLOUT:-256}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SEEDS=$(python -c "import random; print(' '.join(str(random.randint(0, 1000000)) for _ in range($NUM_SEEDS)))")
RUN_ID="$(date +%y-%m-%d_%H-%M-%S-%6N)_scalability"
OUTPUT_PATH="runs/results/scalability_results_${RUN_ID}.json"

# Ensure BLAS/OpenMP libraries use the allocated CPU cores.
THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

cd "${REPO_DIR}"
mkdir -p runs/results

echo "=== Scalability Study ==="
echo "Fast mode: ${FAST_MODE}"
echo "Timesteps: ${TIMESTEPS}  Seeds: ${SEEDS}"
echo "PPO epochs: ${PPO_EPOCHS}  Threads: ${THREADS}"
echo "Output:    ${OUTPUT_PATH}"

python experiments/run_scalability.py \
    --timesteps     "${TIMESTEPS}"    \
    --seeds         ${SEEDS}          \
    --rollout       "${ROLLOUT}"      \
    --ppo_epochs    "${PPO_EPOCHS}"   \
    --max_ep_steps  "${MAX_EP_STEPS}" \
    --output        "${OUTPUT_PATH}"

echo "=== Done: $? ==="
