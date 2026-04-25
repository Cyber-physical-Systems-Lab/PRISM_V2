#!/bin/bash -l
#SBATCH --job-name=prism_extra_seeds
#SBATCH --output=/proj/prism_v2/runs/prism_extra_seeds_%j.out
#SBATCH --error=/proj/prism_v2/runs/prism_extra_seeds_%j.err
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

# 2 additional seeds for each condition (brings total to 5 per condition)
NUM_SEEDS="${NUM_SEEDS:-2}"
SEEDS=$(python -c "import random; print(' '.join(str(random.randint(0, 1000000)) for _ in range($NUM_SEEDS)))")

echo "Extra seeds: ${SEEDS}"

# ── Symbiotic: 2 more seeds ────────────────────────────────────────────────────
RUN_ID_SYM="$(date +%y-%m-%d_%H-%M-%S-%6N)_symbiotic_extra"
echo "=== Symbiotic extra seeds ==="
python -u experiments/run_symbiotic.py \
    --config configs/prism_symbiotic.yaml \
    --timesteps 1000000 \
    --seeds ${SEEDS} \
    --checkpoint_dir /proj/prism_v2/runs/${RUN_ID_SYM} \
    --tb_logdir /proj/prism_v2/runs/tensorboard/${RUN_ID_SYM} \
    --output /proj/prism_v2/runs/results/prism_symbiotic_${RUN_ID_SYM}.json

# ── Flat-coop: 2 more seeds ────────────────────────────────────────────────────
RUN_ID_FLAT="$(date +%y-%m-%d_%H-%M-%S-%6N)_flat_coop_extra"
echo "=== Flat-cooperative extra seeds ==="
python -u experiments/run_flat_cooperative.py \
    --config configs/prism_flat_cooperative.yaml \
    --timesteps 1000000 \
    --seeds ${SEEDS} \
    --checkpoint_dir /proj/prism_v2/runs/${RUN_ID_FLAT} \
    --tb_logdir /proj/prism_v2/runs/tensorboard/${RUN_ID_FLAT} \
    --output /proj/prism_v2/runs/results/prism_flat_coop_${RUN_ID_FLAT}.json

echo "=== All done ==="
