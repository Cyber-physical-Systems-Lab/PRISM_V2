# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for **"PRISM: Policy-shaping via Reward decomposition for Inter-agent Symbiosis in MARL"**. The central hypothesis is narrow and falsifiable: symbiotic reward decomposition (`r_i = r_task + r_sym`) produces higher throughput, better energy efficiency, and identifiable ecological relationship signatures compared to flat cooperative reward shaping — on the same team, in the same environment.

Three linked contributions:
- **C1**: Formalize robotic symbiosis via value-function counterfactuals (ecological relationship types)
- **C2**: Reward decomposition `r_i = r_task + r_sym` with convergence guarantees
- **C3**: Symbiotic vs flat-cooperative reward comparison on identical mixed teams (4 AGVs + 2 pickers)

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

# C3 primary: symbiotic reward condition
python experiments/run_symbiotic.py \
    --config configs/prism_symbiotic.yaml \
    --timesteps 1000000 --seeds 0 1 2 \
    --output runs/results/prism_symbiotic.json

# C3 falsification: flat-cooperative reward condition (same team, same env)
python experiments/run_flat_cooperative.py \
    --config configs/prism_flat_cooperative.yaml \
    --timesteps 1000000 --seeds 0 1 2 \
    --output runs/results/prism_flat_cooperative.json

# Backend flag: --backend auto|mappo|ippo
# auto tries mappo first, falls back to ippo

# Combined sweep (both conditions)
python experiments/run_all_experiments.py \
    --timesteps 1000000 --seeds 0 1 2 \
    --output runs/results/prism_results.json
```

### Analysis and figures

```bash
# Generate paper figures (symbiotic condition only)
python analysis/paper_figures.py \
    --symbiotic runs/results/prism_symbiotic.json \
    --ckpt_dir  runs/prism_symbiotic \
    --output    runs/figures

# With flat-cooperative comparison (Fig 7 — core PRISM result)
python analysis/paper_figures.py \
    --symbiotic runs/results/prism_symbiotic.json \
    --flat_coop runs/results/prism_flat_cooperative.json \
    --ckpt_dir  runs/prism_symbiotic \
    --output    runs/figures

# Quick comparison plots
python analysis/plot.py runs/results
# Outputs: runs/results/figures/fig1_mutualism_symbiotic.pdf, fig2_task_completion.pdf
```

### SLURM cluster

```bash
sbatch slurm/run_all_experiments.slurm
# Override parameters:
TIMESTEPS=1000000 NUM_SEEDS=5 sbatch slurm/run_all_experiments.slurm

# Individual launchers:
sbatch slurm/train_symbiotic.slurm
sbatch slurm/train_flat_cooperative.slurm
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
tarware/        # Environment — battery-enabled warehouse (TARWARE)
training/       # SymbioticWrapper — glues symbiosis theory onto the environment
experiments/    # C3 experiment runners + PPO
analysis/       # Post-hoc metrics (TSI, RSI) and paper figure generation
configs/        # YAML configs for C3 experiments
slurm/          # SLURM batch scripts for GPU cluster
runs/           # Symlink → /proj/prism_v2/runs (heavy outputs go here)
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
- Two agent types: **AGVs** (mobile, transport packages) and **pickers** (stationary, handle package manipulation)
- Battery system with package-weight-dependent energy consumption (`energy_coupling.py`)
- Task queue with five package types: SOLO, PICKER_SOLO, STANDARD (1 AGV + 1 picker), LARGE (2 AGVs + 2 pickers), HEAVY
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

- `run_symbiotic.py`: Primary C3 experiment — symbiotic reward condition (`r_task + r_sym`). Self-contained PPO training loop (MAPPO/IPPO), full metrics logging.
- `run_flat_cooperative.py`: C3 falsification — flat-cooperative reward condition (`r_task + alpha * mean_team`). Identical architecture; relationships are measured passively but do not shape rewards.
- `run_all_experiments.py`: Orchestrates both conditions sequentially; produces combined summary JSON with C3 claim checks.
- `ppo_backends.py`: Legacy multi-method runner (individual/team/unclassified/symbiotic backends via `skrl`). Used by older experiment scripts.
- The `--backend auto` flag tries MAPPO first, falls back to IPPO.

### analysis/ — Metrics Layer

- `metrics.py`: TSI (Team Symbiosis Index), RSI (Relationship Strength Index), mutualism fraction, convergence episode.
- `paper_figures.py`: Publication-quality figures (9 panels). Primary figure script. Use `--symbiotic` and `--flat_coop` flags.
- `plot.py`: Quick comparison plots. Reads per-condition JSONs from `runs/results/`.

## Output Convention

**Prefer `runs/results/` over `results/` for experiment outputs.** The `runs/` directory is a symlink to `/proj/symmarl_ijrr2025/xuezhi/runs` to keep heavy JSON artifacts off the login-node workspace. The top-level `results/` directory holds only lightweight preliminary CSVs that belong in git.

## Config Structure

YAML configs (in `configs/`) control all hyperparameters.

`prism_symbiotic.yaml` — key `symbiotic:` block fields:
- `w_mutualism`, `w_commensalism`, `w_competition`, `w_parasitism`: relationship type weights (φ values in C2)
- `sym_scale`: global scale on `r_sym` for boundedness (C2 convergence guarantee)

`prism_flat_cooperative.yaml` — key `flat_cooperative:` block field:
- `alpha_collab`: scale on the shared team mean bonus

Both configs use identical `env:`, `training:`, and `safety:` blocks — only the reward structure block differs.
