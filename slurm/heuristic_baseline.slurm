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
mkdir -p runs/results

RUN_ID="$(date +%y-%m-%d_%H-%M-%S-%6N)_heuristic"

echo "=== Heuristic baseline sweep ==="
python experiments/run_heuristic_baseline.py \
    --num_episodes 20 \
    --seed 42 \
    --output runs/results/heuristic_baseline_${RUN_ID}.json

echo "=== Done: $? ==="
