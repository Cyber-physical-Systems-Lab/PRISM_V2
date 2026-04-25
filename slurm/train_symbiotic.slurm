#!/bin/bash -l
#SBATCH --job-name=prism_symbiotic
#SBATCH --output=/proj/prism_v2/runs/prism_symbiotic_%j.out
#SBATCH --error=/proj/prism_v2/runs/prism_symbiotic_%j.err
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

cd /home/didemgb/PRISM_V2
mkdir -p /proj/prism_v2/runs/results

# Override at submission time: NUM_SEEDS=5 sbatch slurm/train_symbiotic.slurm
NUM_SEEDS="${NUM_SEEDS:-3}"
SEEDS=$(python -c "import random; print(' '.join(str(random.randint(0, 1000000)) for _ in range($NUM_SEEDS)))")

RUN_ID="$(date +%y-%m-%d_%H-%M-%S-%6N)_symbiotic"

# ── PRISM: Symbiotic reward condition ─────────────────────────────────────────
echo "=== PRISM — Symbiotic condition ==="
python -u experiments/run_symbiotic.py \
    --config configs/prism_symbiotic.yaml \
    --timesteps 1000000 \
    --seeds ${SEEDS} \
    --checkpoint_dir /proj/prism_v2/runs/${RUN_ID} \
    --tb_logdir /proj/prism_v2/runs/tensorboard/${RUN_ID} \
    --output /proj/prism_v2/runs/results/prism_symbiotic_${RUN_ID}.json

echo "=== Done: $? ==="
