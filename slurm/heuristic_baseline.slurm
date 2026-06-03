#!/bin/bash
#SBATCH --job-name=heuristic
#SBATCH --output=/proj/symmarl_ijrr2025/xuezhi/runs/heuristic_%j.out
#SBATCH --error=/proj/symmarl_ijrr2025/xuezhi/runs/heuristic_%j.err
#SBATCH --partition=gpu
#SBATCH --account=UPPMAX2025-2-381
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=8GB

source ~/miniconda/etc/profile.d/conda.sh
conda activate battery

cd /home/xuezhi/robotic-symbiosis-tarware

RUN_ID="$(date +%y-%m-%d_%H-%M-%S-%6N)_heuristic"

echo "=== Heuristic baseline sweep ==="
python experiments/run_heuristic_baseline.py \
    --config configs/heuristic_baseline.yaml \
    --output "results/heuristic_baseline.json" \
    --checkpoint_dir "runs/c3_hetero_heuristic-${RUN_ID}" \
    --max_steps 5000 \
    --max_inactivity_steps 0 \
    --num-seeds 6

echo "=== Done: $? ==="
