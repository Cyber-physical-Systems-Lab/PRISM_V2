#!/bin/bash -l
#SBATCH --job-name=sym_hetero
#SBATCH --output=/proj/symmarl_ijrr2025/xuezhi/runs/hetero_%j.out
#SBATCH --error=/proj/symmarl_ijrr2025/xuezhi/runs/hetero_%j.err
#SBATCH --partition=gpu
#SBATCH --account=UPPMAX2025-2-381
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

# ── Environment setup ─────────────────────────────────────────────────────────
module load CUDA/12.9.0
source ~/miniconda/etc/profile.d/conda.sh
conda activate battery

cd /home/xuezhi/robotic-symbiosis-tarware
mkdir -p runs/results

# Override at submission time: NUM_SEEDS=5 sbatch slurm/train_heterogeneous.slurm
NUM_SEEDS="${NUM_SEEDS:-3}"
SEEDS=$(python -c "import random; print(' '.join(str(random.randint(0, 1000000)) for _ in range($NUM_SEEDS)))")

RUN_ID="$(date +%y-%m-%d_%H-%M-%S-%6N)_hetero"

# ── Experiment 1: Heterogeneous (primary result) ──────────────────────────────
echo "=== C3 Experiment 1: Heterogeneous env ==="
python experiments/run_heterogeneous.py \
    --env tarware-small-4agvs-2pickers-partialobs-chg-v1 \
    --timesteps 1000000 \
    --seeds ${SEEDS} \
    --rollout_steps 1024 \
    --epochs 4 \
    --lr 3e-4 \
    --entropy_coef 0.01 \
    --checkpoint_dir runs/${RUN_ID} \
    --tb_logdir runs/tensorboard/${RUN_ID} \
    --checkpoint_every_episodes 100 \
    --output runs/results/hetero_results_${RUN_ID}.json \
    --log_interval 10

echo "=== Done: $? ==="
