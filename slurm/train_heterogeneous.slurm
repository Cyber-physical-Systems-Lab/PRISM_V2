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
#SBATCH --time=48:00:00
#SBATCH --time-min=36:00:00
#SBATCH --mem=64GB

# ── Environment setup ─────────────────────────────────────────────────────────
module load CUDA/12.9.0
source ~/miniconda/etc/profile.d/conda.sh
conda activate battery

# Avoid torch/CPU thread thrash with cpus-per-task=16.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd /home/xuezhi/robotic-symbiosis-tarware
mkdir -p runs/results

RUN_ID="$(date +%y-%m-%d_%H-%M-%S-%6N)_hetero"
HETERO_CKPT_DIR="runs/c3_hetero_pkgmix/${RUN_ID}"
HETERO_TB_DIR="runs/tensorboard/c3_hetero_pkgmix/${RUN_ID}"
HETERO_OUTPUT="runs/results/hetero_results_${RUN_ID}.json"

# Fast defaults match the old run_all_experiments wrapper. Override at
# submission time for full-budget runs, e.g.:
#   TIMESTEPS=3000000 MAX_EP_STEPS=1000 sbatch slurm/train_heterogeneous.slurm
TIMESTEPS="${TIMESTEPS:-1000000}"
MAX_EP_STEPS="${MAX_EP_STEPS:-5000}"
SEEDS="${SEEDS:-232 66 54321}"
BACKENDS="${BACKENDS:-ippo mappo}"
METHODS="${METHODS:-symbiotic team individual}"
CHECKPOINT_EVERY_EPISODES="${CHECKPOINT_EVERY_EPISODES:-100}"

read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a BACKEND_ARGS <<< "${BACKENDS}"
read -r -a METHOD_ARGS <<< "${METHODS}"

# Environment id, safety thresholds, package mix, reward weights, and PPO
# hyperparameters still come from configs/heterogeneous.yaml. Output,
# checkpoints, TensorBoard, run budget, and run matrix are job-specific.
# To enable inactivity-based early termination for sanity runs, append:
#   --max_inactivity_steps 500
echo "=== C3 Experiment 1: Heterogeneous env (bidirectional cycle) ==="
echo "Timesteps: ${TIMESTEPS}"
echo "Max episode steps: ${MAX_EP_STEPS}"
echo "Seeds: ${SEEDS}"
echo "Backends: ${BACKENDS}"
echo "Methods: ${METHODS}"
echo "Checkpoint every episodes: ${CHECKPOINT_EVERY_EPISODES}"
echo "Checkpoint dir: ${HETERO_CKPT_DIR}"
echo "TensorBoard dir: ${HETERO_TB_DIR}"
echo "Output: ${HETERO_OUTPUT}"
python -u experiments/run_heterogeneous.py \
    --config configs/heterogeneous.yaml \
    --timesteps "${TIMESTEPS}" \
    --max_ep_steps "${MAX_EP_STEPS}" \
    --seeds "${SEED_ARGS[@]}" \
    --backends "${BACKEND_ARGS[@]}" \
    --methods "${METHOD_ARGS[@]}" \
    --checkpoint_dir "${HETERO_CKPT_DIR}" \
    --tb_logdir "${HETERO_TB_DIR}" \
    --checkpoint_every_episodes "${CHECKPOINT_EVERY_EPISODES}" \
    --output "${HETERO_OUTPUT}"

status=$?
echo "=== Done: ${status} ==="
exit "${status}"
