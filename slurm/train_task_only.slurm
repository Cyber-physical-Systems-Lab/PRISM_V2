#!/bin/bash -l
#SBATCH --job-name=prism_task_only
#SBATCH --output=/proj/prism_v2/runs/prism_task_only_%j.out
#SBATCH --error=/proj/prism_v2/runs/prism_task_only_%j.err
#SBATCH --partition=gpu
#SBATCH --account=uppmax2026-1-141
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

NUM_SEEDS="${NUM_SEEDS:-3}"
SEEDS=$(python -c "import random; print(' '.join(str(random.randint(0, 1000000)) for _ in range($NUM_SEEDS)))")

RUN_ID="$(date +%y-%m-%d_%H-%M-%S-%6N)_task_only"

# ── PRISM: Task-only ablation (r_task, no shaping) ────────────────────────────
echo "=== PRISM — Task-only ablation (r_i = r_task_i) ==="
python -u experiments/run_flat_cooperative.py \
    --config configs/prism_task_only.yaml \
    --condition task_only \
    --alpha_collab 0.0 \
    --timesteps 1000000 \
    --seeds ${SEEDS} \
    --checkpoint_dir /proj/prism_v2/runs/${RUN_ID} \
    --tb_logdir /proj/prism_v2/runs/tensorboard/${RUN_ID} \
    --output /proj/prism_v2/runs/results/prism_task_only_${RUN_ID}.json

echo "=== Done: $? ==="
