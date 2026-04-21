#!/bin/bash -l
#SBATCH --job-name=sym_homo
#SBATCH --output=/proj/symmarl_ijrr2025/xuezhi/runs/homo_%j.out
#SBATCH --error=/proj/symmarl_ijrr2025/xuezhi/runs/homo_%j.err
#SBATCH --partition=gpu
#SBATCH --account=UPPMAX2025-2-381
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=64GB

# ── Environment setup ─────────────────────────────────────────────────────────
module load CUDA/12.9.0
source ~/miniconda/etc/profile.d/conda.sh
conda activate battery

cd /home/xuezhi/robotic-symbiosis-tarware
mkdir -p runs/results

# Override at submission time: NUM_SEEDS=5 sbatch slurm/train_homogeneous.slurm
NUM_SEEDS="${NUM_SEEDS:-3}"
SEEDS=$(python -c "import random; print(' '.join(str(random.randint(0, 1000000)) for _ in range($NUM_SEEDS)))")

# Conservative defaults to avoid OOM on GPU nodes.
TIMESTEPS="${TIMESTEPS:-1000000}"
ROLLOUT="${ROLLOUT:-128}"
PPO_EPOCHS="${PPO_EPOCHS:-3}"
BACKENDS="${BACKENDS:-ippo mappo hetppo}"

RUN_ID="$(date +%y-%m-%d_%H-%M-%S-%6N)_homo"

# ── Experiment 2: Homogeneous (falsification) ─────────────────────────────────
echo "=== C3 Experiment 2: Homogeneous env (falsification) ==="
python experiments/run_homogeneous.py \
    --env tarware-tiny-4agvs-0pickers-partialobs-chg-v1 \
    --timesteps ${TIMESTEPS} \
    --seeds ${SEEDS} \
    --rollout ${ROLLOUT} \
    --ppo_epochs ${PPO_EPOCHS} \
    --lr 3e-4 \
    --entropy_coef 0.01 \
    --backends ${BACKENDS} \
    --output runs/results/homo_results_${RUN_ID}.json

echo "=== Done: $? ==="
