#!/bin/bash -l
#SBATCH --job-name=prism_all
#SBATCH --output=/proj/symmarl_ijrr2025/xuezhi/runs/prism_all_%j.out
#SBATCH --error=/proj/symmarl_ijrr2025/xuezhi/runs/prism_all_%j.err
#SBATCH --partition=gpu
#SBATCH --account=UPPMAX2025-2-381
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=96GB

set -eo pipefail

module load CUDA/12.9.0

set +u
source ~/miniconda/etc/profile.d/conda.sh
conda activate battery
set -u

REPO_DIR="/home/xuezhi/robotic-symbiosis-tarware"

# Override at submission time, e.g.:
#   TIMESTEPS=2000000 NUM_SEEDS=5 sbatch slurm/run_all_experiments.slurm
TIMESTEPS="${TIMESTEPS:-1000000}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SKIP="${SKIP:-}"
SEEDS=$(python -c "import random; print(' '.join(str(random.randint(0, 1000000)) for _ in range($NUM_SEEDS)))")
ROLLOUT="${ROLLOUT:-1024}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
LR="${LR:-3e-4}"
ENTROPY="${ENTROPY:-0.01}"
HEURISTIC_EPISODES="${HEURISTIC_EPISODES:-50}"
MAX_EP_STEPS="${MAX_EP_STEPS:-5000}"
RESULTS_DIR="runs/results"
JOB_ID="${SLURM_JOB_ID:-local}"

SYMBIOTIC_CKPT_DIR="runs/prism_symbiotic/${JOB_ID}"
SYMBIOTIC_TB_DIR="runs/tensorboard/prism_symbiotic/${JOB_ID}"
FLAT_COOP_CKPT_DIR="runs/prism_flat_cooperative/${JOB_ID}"
FLAT_COOP_TB_DIR="runs/tensorboard/prism_flat_cooperative/${JOB_ID}"

OUTPUT_PATH="${RESULTS_DIR}/prism_results_${JOB_ID}.json"

cd "${REPO_DIR}"
mkdir -p "${RESULTS_DIR}"

echo "=== PRISM — Full experiment sweep ==="
echo "Timesteps: ${TIMESTEPS}  Seeds: ${SEEDS}  Rollout: ${ROLLOUT}  PPO epochs: ${PPO_EPOCHS}"
echo "LR: ${LR}  Entropy: ${ENTROPY}  Max ep steps: ${MAX_EP_STEPS}"
echo "Output: ${OUTPUT_PATH}"

# Build optional --skip flag
SKIP_ARGS=()
for stage in ${SKIP:-}; do
    SKIP_ARGS+=(--skip "${stage}")
done

python experiments/run_all_experiments.py \
    --timesteps          "${TIMESTEPS}"                                             \
    --seeds              ${SEEDS}                                                   \
    --rollout            "${ROLLOUT}"                                               \
    --ppo_epochs         "${PPO_EPOCHS}"                                            \
    --lr                 "${LR}"                                                    \
    --entropy            "${ENTROPY}"                                               \
    --heuristic_episodes "${HEURISTIC_EPISODES}"                                    \
    --max_ep_steps       "${MAX_EP_STEPS}"                                          \
    --symbiotic_checkpoint_dir "${SYMBIOTIC_CKPT_DIR}"                             \
    --symbiotic_tb_logdir      "${SYMBIOTIC_TB_DIR}"                               \
    --flat_coop_checkpoint_dir "${FLAT_COOP_CKPT_DIR}"                             \
    --flat_coop_tb_logdir      "${FLAT_COOP_TB_DIR}"                               \
    --heuristic_output   "${RESULTS_DIR}/heuristic_baseline_${JOB_ID}.json"        \
    --symbiotic_output   "${RESULTS_DIR}/prism_symbiotic_${JOB_ID}.json"           \
    --flat_coop_output   "${RESULTS_DIR}/prism_flat_coop_${JOB_ID}.json"           \
    --output             "${OUTPUT_PATH}"                                           \
    "${SKIP_ARGS[@]}"

echo "=== Done: $? ==="
