# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for **"Emergent Symbiosis in Heterogeneous MARL: A Formal Framework for Capability-Complementary Robotic Teams"**. The central hypothesis is narrow and falsifiable: symbiotic reward shaping helps coordination only when capability complementarity exists between agent types.

Three linked contributions:
- **C1**: Formalize robotic symbiosis via value-function counterfactuals (ecological relationship types)
- **C2**: Reward decomposition `r_i = r_task + r_sym` with convergence guarantees
- **C3**: Confirmation and falsification experiments on heterogeneous vs homogeneous warehouse teams

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥3.9. The conda environment used on SLURM cluster is named `battery`.

## Key Commands

### Running experiments

```bash
# Heuristic oracle baseline
python experiments/run_heuristic_baseline.py \
    --env tarware-small-4agvs-2pickers-partialobs-chg-v1 \
    --num_episodes 20 --seed 42 \
    --output runs/results/heuristic_baseline.json

# Experiment 1: heterogeneous (primary result)
python experiments/run_heterogeneous.py \
    --config configs/heterogeneous.yaml \
    --timesteps 1000000 --seeds 5 \
    --output runs/results/hetero_results.json

# Force backend: --backend mappo|ippo|happo|haddpg
# Backend auto-order: local mappo → local ippo → HARL placeholders

# Experiment 2: homogeneous falsification
python experiments/run_homogeneous.py \
    --config configs/homogeneous.yaml \
    --timesteps 500000 --seeds 5 \
    --output runs/results/homo_results.json

# Experiment 3: heterogeneity gradient
python experiments/run_gradient.py \
    --config configs/gradient.yaml \
    --h_values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --seeds 5 --output runs/results/gradient_results.json

# Experiment 4: convergence diagnostics
python experiments/run_convergence.py \
    --config configs/heterogeneous.yaml \
    --output runs/results/convergence_results.json

# Combined sweep (all experiments)
python experiments/run_all_experiments.py \
    --output runs/results/experiment_results.json
```

### Analysis and figures

```bash
# Generate paper figures from per-experiment JSON files
python analysis/plot.py runs/results
# Outputs: runs/results/figures/fig1_mutualism_hetero.pdf, fig2_task_completion.pdf

# Regenerate demo GIF
python experiments/generate_demo_gif.py \
    --output assets/robotic_symbiosis_demo.gif \
    --metadata_output assets/robotic_symbiosis_demo.json
```

### SLURM cluster

```bash
sbatch slurm/run_all_experiments.slurm
# Override parameters:
TIMESTEPS=300000 SEEDS=5 HEURISTIC_EPISODES=20 sbatch slurm/run_all_experiments.slurm

# Individual launchers:
sbatch slurm/train_heterogeneous.slurm
sbatch slurm/train_homogeneous.slurm
sbatch slurm/heuristic_baseline.slurm
```

### Linting and formatting

```bash
black .
isort .
pytest  # no test suite currently exists; placeholder for future tests
```

## Architecture

### Module Map

```
symbiosis/      # C1+C2 theory — relationship formalization and reward decomposition
tarware/        # Environment — battery-enabled heterogeneous warehouse (TARWARE)
training/       # SymbioticWrapper — glues symbiosis theory onto the environment
experiments/    # C3 experiment runners + PPO backends
analysis/       # Post-hoc metrics (TSI, RSI) and paper figure generation
configs/        # YAML configs for C3 experiments
slurm/          # SLURM batch scripts for GPU cluster
runs/           # Symlink → /proj/symmarl_ijrr2025/xuezhi/runs (heavy outputs go here)
results/        # Lightweight preliminary CSVs checked into git
```

### symbiosis/ — Theory Layer

- `fitness.py`: `AgentFitness` — tracks long-run average reward via EMA. Formalizes biological fitness for robots.
- `definitions.py`: `RelationshipType` enum (mutualism +/+, commensalism +/0, parasitism +/−, competition −/−, neutralism 0/0) and `RoboticSymbiosisClassifier` that labels agent pairs each step.
- `counterfactual_critic.py`: `JointCritic` (takes all agent observations) and `MarginalCritic` for counterfactual value decomposition — analogous to COMA but for agent presence rather than actions.
- `reward_decomposition.py`: `SymbioticRewardDecomposer` — computes `r_sym` as a relationship-type-weighted bonus/penalty, controlled by `w_mutualism`, `w_commensalism`, `w_competition`, `w_parasitism`, and `sym_scale`.
- `convergence.py`: `ConvergenceMonitor` — convergence diagnostics utilities.

### tarware/ — Environment Layer

`warehouse.py` is the core gymnasium environment. Key design decisions:
- Two heterogeneous agent types: **AGVs** (mobile, transport packages) and **pickers** (stationary, handle package manipulation)
- Battery system with package-weight-dependent energy consumption (`energy_coupling.py`)
- Task queue with three package types: SOLO (AGV only), STANDARD (1 AGV + 1 picker), LARGE (2 AGVs + 2 pickers)
- Internal motion planning via A* (`astar.py`) — experiment actions are task targets, not low-level moves
- Adaptive replanning under battery constraints (`replanning.py`)
- Role emergence tracking (charging/tasking/idle) in `role_assignment.py`

Environment IDs follow the pattern:
`tarware-{size}-{n}agvs-{m}pickers-partialobs-chg[-pkgmix]-v1`
where size ∈ {tiny, small, medium, large, extralarge}.

### training/ — Wrapper Layer

`symbiotic_wrapper.py` (`SymbioticWrapper`) wraps any TARWARE env and adds:
1. Per-step relationship detection (classifies all AGV-picker pairs via `RoboticSymbiosisClassifier`)
2. Observation augmentation (appends relationship EMA + energy margin to each agent's observation)
3. Symbiotic reward shaping (adds `r_sym` to the task reward)
4. Relationship logging for post-hoc emergence analysis

### experiments/ — Training Layer

- `ppo_backends.py`: Local MAPPO (centralized critic per agent type) and IPPO (decentralized) implementations built on `skrl`. This is the largest file and the primary training engine.
- Per-experiment runners (`run_*.py`) instantiate the environment, wrap it with `SymbioticWrapper`, select the PPO backend, and write JSON results.
- The `--backend auto` flag tries MAPPO first, falls back to IPPO; `happo`/`haddpg` require external HARL installation not bundled here.

### analysis/ — Metrics Layer

- `metrics.py`: TSI (Team Symbiosis Index), RSI (Relationship Strength Index), mutualism fraction, convergence episode.
- `plot.py`: Paper-style figures. Reads per-experiment JSONs, not the combined JSON from `run_all_experiments.py`.

## Output Convention

**Prefer `runs/results/` over `results/` for experiment outputs.** The `runs/` directory is a symlink to `/proj/symmarl_ijrr2025/xuezhi/runs` to keep heavy JSON artifacts off the login-node workspace. The top-level `results/` directory holds only lightweight preliminary CSVs that belong in git.

## Config Structure

YAML configs (in `configs/`) control all hyperparameters. Key `symbiotic:` block fields:
- `w_mutualism`, `w_commensalism`, `w_competition`, `w_parasitism`: relationship type weights (φ values in C2)
- `sym_scale`: global scale on `r_sym` for boundedness (C2 convergence guarantee)
- `rel_alpha`: EMA smoothing factor for relationship history
- `activity_bonus`: set >0 only for sparse-reward ablation studies
