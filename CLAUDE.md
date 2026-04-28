# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for **"PRISM: Policy-shaping via Reward decomposition for Inter-agent Symbiosis in MARL"**. The central hypothesis is narrow and falsifiable: symbiotic reward decomposition (`r_i = r_task + r_sym`) produces more fault-tolerant teams compared to flat cooperative reward shaping and task-only baselines — on the same team, in the same environment.

Three linked contributions:
- **C1**: Formalize robotic symbiosis via value-function counterfactuals (mutualism and commensalism observed empirically)
- **C2**: Reward decomposition `r_i = r_task + r_sym` with convergence guarantees
- **C3**: PRISM teams degrade 7.9% under agent failure vs 19.6% (flat-coop) and 32.1% (task-only)

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥3.9. Conda environment on UPPMAX cluster: `base` (miniconda). Local Python: `~/anaconda3/envs/warehouse/bin/python`.

## Key Commands

### Running experiments

```bash
# Heuristic oracle baseline
python experiments/run_heuristic_baseline.py \
    --env tarware-small-4agvs-2pickers-partialobs-chg-v1 \
    --num_episodes 30 --seed 42 \
    --output local_runs/results/heuristic_baseline.json

# Symbiotic condition (C3 primary)
python experiments/run_symbiotic.py \
    --config configs/prism_symbiotic.yaml \
    --timesteps 1000000 --seeds 0 1 2 \
    --checkpoint_dir local_runs/checkpoints/prism_symbiotic \
    --output local_runs/results/prism_symbiotic.json

# Flat-cooperative baseline
python experiments/run_flat_cooperative.py \
    --config configs/prism_flat_cooperative.yaml \
    --timesteps 1000000 --seeds 0 1 2 \
    --checkpoint_dir local_runs/checkpoints/prism_flat_coop \
    --output local_runs/results/prism_flat_coop.json

# Task-only ablation (alpha_collab=0.0)
python experiments/run_flat_cooperative.py \
    --config configs/prism_task_only.yaml \
    --condition task_only --alpha_collab 0.0 \
    --timesteps 1000000 --seeds 0 1 2 \
    --checkpoint_dir local_runs/checkpoints/prism_task_only \
    --output local_runs/results/prism_task_only.json
```

### Evaluation and statistics

```bash
# Batch evaluation across all seeds + Welch t-test + Cohen's d
python analysis/evaluate_all_seeds.py \
    --sym_dir  local_runs/checkpoints/prism_symbiotic \
    --flat_dir local_runs/checkpoints/prism_flat_coop \
    --heuristic_json local_runs/results/heuristic_baseline.json \
    --episodes 20 \
    --output   local_runs/results/eval_stats.json

# Alternative metrics: depletion rate, convergence speed, resilience
python analysis/extract_metrics.py \
    --sym_ckpt_dir  local_runs/checkpoints/prism_symbiotic \
    --flat_ckpt_dir local_runs/checkpoints/prism_flat_coop \
    --task_ckpt_dir local_runs/checkpoints/prism_task_only \
    --resilience_episodes 5 \
    --output local_runs/results/alternative_metrics.json

# Per-package delivery stats
python analysis/collect_pkg_stats.py \
    --sym_ckpt  <best_symbiotic_checkpoint.pt> \
    --flat_ckpt <best_flat_coop_checkpoint.pt> \
    --task_ckpt <best_task_only_checkpoint.pt> \
    --episodes 30 \
    --output local_runs/results/pkg_stats.json
```

### GIF generation

```bash
python experiments/evaluate_and_gif.py \
    --checkpoint      local_runs/checkpoints/prism_symbiotic/.../checkpoint_best.pt \
    --checkpoint_baseline local_runs/checkpoints/prism_flat_coop/.../checkpoint_best.pt \
    --condition symbiotic --episodes 5 --steps 1000 --fps 6 \
    --output_dir local_runs/eval
```

### Paper figures

```bash
python analysis/paper_figures.py \
    --symbiotic      local_runs/results/prism_symbiotic.json \
    --flat_coop      local_runs/results/prism_flat_coop.json \
    --heuristic      local_runs/results/heuristic_baseline.json \
    --eval_stats     local_runs/results/eval_stats_final.json \
    --alt_metrics    local_runs/results/alternative_metrics.json \
    --ckpt_dir       local_runs/checkpoints/prism_symbiotic \
    --flat_ckpt_dir  local_runs/checkpoints/prism_flat_coop \
    --task_ckpt_dir  local_runs/checkpoints/prism_task_only \
    --output         local_runs/figures/prism_v2
```

### SLURM cluster (UPPMAX, user: didemgb)

```bash
sbatch slurm/train_symbiotic.slurm
sbatch slurm/train_flat_cooperative.slurm
sbatch slurm/train_task_only.slurm
sbatch slurm/train_extra_seeds.slurm   # 2 extra seeds per condition

# Override seed count
NUM_SEEDS=5 sbatch slurm/train_symbiotic.slurm
```

Account: `UPPMAX2025-2-381` | Partition: `gpu` | Storage: `/proj/prism_v2/runs/`

## Architecture

### Module Map

```
symbiosis/      # C1+C2 theory — relationship formalization and reward decomposition
tarware/        # Environment — battery-enabled warehouse (TARWARE)
training/       # SymbioticWrapper — obs augmentation + r_sym shaping
experiments/    # C3 experiment runners + PPO + evaluation scripts
analysis/       # Metrics, paper figures, multi-seed evaluation, resilience
configs/        # YAML configs for all conditions
slurm/          # SLURM batch scripts for GPU cluster
local_runs/     # Local outputs: checkpoints/, results/, figures/, eval/
latex/          # IJRR paper (main.tex + figures)
runs/           # Symlink → /proj/prism_v2/runs (cluster outputs)
```

### Key configs

| Config | Purpose | Key difference |
|---|---|---|
| `prism_symbiotic.yaml` | PRISM condition | `r_task + r_sym`, w_mutualism=8.0 |
| `prism_flat_cooperative.yaml` | Flat-coop baseline | `r_task + alpha*mean_team` |
| `prism_task_only.yaml` | Task-only ablation | `r_task` only (alpha_collab=0.0) |

All configs share: entropy_coef=0.05, depletion_penalty=1.5, low_battery_threshold=7.0, w_coverage=0.5, 1M timesteps.

### experiments/ — Training Layer

- `run_symbiotic.py`: PRISM training. Includes coverage bonus and symbiotic reward shaping.
- `run_flat_cooperative.py`: Flat-coop and task-only training (use `--condition task_only --alpha_collab 0.0` for ablation).
- `run_heuristic_baseline.py`: Rule-based A* oracle, 30 episodes.
- `evaluate_and_gif.py`: Post-training evaluation + GIF generation. Passes `package_distribution` to env (critical — default env is STANDARD-only).

### analysis/ — Metrics Layer

- `evaluate_all_seeds.py`: Evaluates all checkpoint_best.pt files, runs Welch t-test + Mann-Whitney + Cohen's d.
- `extract_metrics.py`: Depletion rate, convergence speed, resilience under agent failure.
- `collect_pkg_stats.py`: Per-package-type delivery breakdown.
- `paper_figures.py`: All publication figures (PDF+PNG). Key figures: `fig_resilience`, `fig_relationship_emergence`, `fig7`.
- `metrics.py`: TSI, RSI, mutualism fraction.

## Important Implementation Notes

- **Package distribution must be passed explicitly** to `gym.make()` in all evaluation scripts. Default env uses STANDARD-only packages. Use: `package_distribution={"SOLO":0.20,"STANDARD":0.30,"LARGE":0.10,"HEAVY":0.25,"PICKER_SOLO":0.15}`.
- **Coverage bonus** (`w_coverage=0.5`): shared bonus to all agents per delivery of any package type. Required to prevent STANDARD-only task preference.
- **Stochastic evaluation**: use `Categorical(logits=...)` sampling, not argmax. Greedy evaluation produces 0 deliveries.
- **Relationship types observed**: only mutualism and commensalism emerge empirically. `classify_rel()` cannot return competition or parasitism in the current implementation.
- **IPPO seeds** are more consistent than MAPPO for symbiotic condition at 1M steps.
- **env returns lowercase keys** in `deliveries_by_pkg_type` (e.g. `"solo"`, `"standard"`). Normalise with `.upper()` when comparing against `PackageType` enum names.

## C3 Core Result (v3, awaiting v4)

| Condition | Full team | 1 AGV failed | Drop | p-value |
|---|---|---|---|---|
| **PRISM (symbiotic)** | 3.37 del/ep | 3.10 | **7.9%** | — |
| Flat-cooperative | 3.73 del/ep | 3.00 | **19.6%** | 0.046 vs PRISM |
| Task-only | 3.53 del/ep | 2.40 | **32.1%** | <0.001 vs PRISM |

**v4 training** (entropy=0.05, penalty=1.5, threshold=7.0) submitted to UPPMAX — smoke test showed 4.58 del/ep at 80k steps vs 4.06 at 1M with v3. Results expected ~24h after submission.
