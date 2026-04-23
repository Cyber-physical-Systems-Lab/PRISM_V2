# PRISM: Policy-shaping via Reward decomposition for Inter-agent Symbiosis in MARL

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research codebase for the paper:
**"PRISM: Policy-shaping via Reward decomposition for Inter-agent Symbiosis in MARL"**

The central claim is narrow and testable: symbiotic reward decomposition (`r_i = r_task + r_sym`) produces higher throughput, better energy efficiency, and identifiable ecological relationship signatures compared to flat cooperative reward shaping — **on the same team, in the same environment**. The only variable is the reward structure.

---

## Overview

<p align="center">
  <img src="assets/symbiosis_overview.gif" alt="Symbiosis overview — heuristic scheduling" width="760" />
</p>

<p align="center"><em>Heuristic scheduling baseline showing AGVs (mobile), pickers (stationary), charging stations, and five package types (SOLO, STANDARD, LARGE, HEAVY, PICKER_SOLO). Ecological relationship types (mutualism, commensalism, competition, parasitism, neutralism) are detected and overlaid in real time.</em></p>

<p align="center">
  <img src="assets/trained_symbiotic_overview.gif" alt="Trained policy — symbiotic reward" width="760" />
</p>

<p align="center"><em>IPPO policy trained with symbiotic reward shaping (200k steps). Ecological relationship types are detected and overlaid in real time from the trained policy's behavior.</em></p>

---

## Research Contributions

Three linked contributions, each independently evaluable:

| ID | Contribution | Key file |
|----|---|---|
| **C1** | Formalize robotic symbiosis via value-function counterfactuals (ecological relationship types) | `symbiosis/definitions.py`, `symbiosis/fitness.py` |
| **C2** | Reward decomposition `rᵢ = r_task + r_sym` with convergence guarantees | `symbiosis/reward_decomposition.py`, `symbiosis/convergence.py` |
| **C3** | Symbiotic vs flat-cooperative reward comparison on identical mixed teams | `experiments/run_symbiotic.py`, `experiments/run_flat_cooperative.py` |

---

## Core Hypothesis

The experiment compares two reward conditions applied to the **same** mixed team (4 AGVs + 2 pickers):

| Condition | Reward | Description |
|---|---|---|
| **Symbiotic** | `r_i = r_task_i + r_sym_i` | `r_sym` shaped by ecological relationship type per AGV-picker pair; relationship-weighted bonuses/penalties from C1 classifier |
| **Flat-cooperative** | `r_i = r_task_i + α·mean_j(r_task_j)` | Shared team mean bonus; no relationship classification, no ecological typing |

Team composition, environment, architecture, and training budget are identical across conditions. The falsification criterion: if flat-cooperative achieves the same throughput, energy efficiency, and relationship distribution, the biological reward decomposition has no measurable effect.

---

## Environment

The TARWARE environment features:

- **Two agent types**: AGVs (mobile, carry packages) and pickers (stationary at shelves, load/unload packages)
- **Five package types** encoding different cooperation requirements:

  | Package | Required AGVs | Required pickers | Ecological parallel |
  |---|---|---|---|
  | SOLO | 1 | 0 | Neutralism — independent |
  | PICKER_SOLO | 0 | 1 | Neutralism — independent |
  | STANDARD | 1 | 1 | Mutualism — joint delivery |
  | LARGE | 2 | 2 | Mutualism — recruitment cost |
  | HEAVY | 1 | 1+ assist | Commensalism — picker assists, AGV gains energy discount |

- **Battery system**: agents deplete energy carrying packages; must navigate to charging stations to recharge; contested charger access creates competition dynamics
- **Internal A\* motion planning**: experiment actions are task targets, not low-level moves

Environment IDs: `tarware-{size}-{n}agvs-{m}pickers-partialobs-chg[-pkgmix]-v1`

---

## Relationship Scenarios

<p align="center">
  <img src="assets/symbiosis_scenarios.gif" alt="Relationship type scenarios" width="760" />
</p>

<p align="center"><em>Per-type scenario clips: mutualism (STANDARD joint delivery), commensalism (HEAVY task with picker assist), competition (contested charging station), and neutralism (independent SOLO tasks).</em></p>

---

## Installation

```bash
git clone https://github.com/your-org/PRISM_V2.git
cd PRISM_V2
pip install -e ".[dev]"
```

Python ≥3.9 required. Dependencies: `torch`, `gymnasium`, `pyyaml`, `imageio`, `pandas`, `matplotlib`, `seaborn`.

> **Cluster**: conda environment named `battery` on UPPMAX. Heavy outputs go to `runs/` (symlink to `/proj/symmarl_ijrr2025/xuezhi/runs`).

---

## Quick Start

### Generate heuristic demonstration GIF

```bash
python experiments/generate_symbiosis_gifs.py \
    --env tarware-small-4agvs-2pickers-partialobs-chg-symbiosis-v1 \
    --steps 1500 --fps 6 --output_dir assets
```

### Run the symbiotic condition (local)

```bash
python experiments/run_symbiotic.py \
    --config configs/prism_symbiotic.yaml \
    --timesteps 200000 --seeds 0 1 2 --backend ippo \
    --checkpoint_dir local_runs/prism_symbiotic \
    --tb_logdir local_runs/tensorboard/prism_symbiotic \
    --output local_runs/results/prism_symbiotic.json
```

### Run the flat-cooperative condition (local)

```bash
python experiments/run_flat_cooperative.py \
    --config configs/prism_flat_cooperative.yaml \
    --timesteps 200000 --seeds 0 1 2 --backend ippo \
    --checkpoint_dir local_runs/prism_flat_cooperative \
    --tb_logdir local_runs/tensorboard/prism_flat_cooperative \
    --output local_runs/results/prism_flat_cooperative.json
```

### Generate GIF from trained checkpoint

```bash
python experiments/generate_trained_gif.py \
    --checkpoint local_runs/prism_symbiotic/ippo_seed0/checkpoint_best.pt \
    --method symbiotic --steps 600 --fps 6 --output_dir assets
```

### Generate paper figures

```bash
# Symbiotic condition only
python analysis/paper_figures.py \
    --symbiotic local_runs/results/prism_symbiotic.json \
    --ckpt_dir  local_runs/prism_symbiotic \
    --output    local_runs/figures

# With flat-cooperative comparison (Fig 7 — core PRISM result)
python analysis/paper_figures.py \
    --symbiotic local_runs/results/prism_symbiotic.json \
    --flat_coop local_runs/results/prism_flat_cooperative.json \
    --ckpt_dir  local_runs/prism_symbiotic \
    --output    local_runs/figures
```

---

## Full Experiment Suite

### Heuristic oracle baseline

```bash
python experiments/run_heuristic_baseline.py \
    --env tarware-small-4agvs-2pickers-partialobs-chg-v1 \
    --num_episodes 20 --seed 42 \
    --output runs/results/heuristic_baseline.json
```

### Primary comparison: symbiotic vs flat-cooperative

```bash
# Symbiotic condition (C3 primary)
python experiments/run_symbiotic.py \
    --config configs/prism_symbiotic.yaml \
    --timesteps 1000000 --seeds 0 1 2 \
    --output runs/results/prism_symbiotic.json

# Flat-cooperative baseline (C3 falsification)
python experiments/run_flat_cooperative.py \
    --config configs/prism_flat_cooperative.yaml \
    --timesteps 1000000 --seeds 0 1 2 \
    --output runs/results/prism_flat_cooperative.json
```

Backend flag: `--backend auto|mappo|ippo`
(`auto` → MAPPO first, then IPPO)

### Combined sweep (both conditions)

```bash
python experiments/run_all_experiments.py \
    --timesteps 1000000 --seeds 0 1 2 \
    --output runs/results/prism_results.json
```

---

## SLURM Cluster

```bash
# Full sweep (both conditions)
sbatch slurm/run_all_experiments.slurm

# Override parameters
TIMESTEPS=1000000 NUM_SEEDS=5 sbatch slurm/run_all_experiments.slurm

# Individual launchers
sbatch slurm/train_symbiotic.slurm
sbatch slurm/train_flat_cooperative.slurm
```

---

## Paper Figures

The `analysis/paper_figures.py` script produces up to 9 publication-quality figures (PDF + PNG, 300 DPI, serif font):

| Figure | Description | File |
|---|---|---|
| Fig 1 | Task completion learning curves — symbiotic condition, ±1σ | `fig1_learning_curves` |
| Fig 2 | Mutualism fraction emergence over training | `fig2_mutualism_emergence` |
| Fig 3 | Relationship type distribution at end of training (stacked bar) | `fig3_relationship_distribution` |
| Fig 4 | Raw vs shaped reward divergence | `fig4_reward_shaping` |
| Fig 5 | Training stability — actor entropy and KL divergence | `fig5_training_stability` |
| Fig 6 | Team Specialisation Index (TSI) and RSI | `fig6_specialisation_index` |
| Fig 7 | **Symbiotic vs flat-cooperative comparison** (core PRISM result; requires `--flat_coop`) | `fig7_symbiotic_vs_flat_coop` |
| Fig 8 | Package-type delivery breakdown per condition | `fig8_package_breakdown` |
| Fig 9 | Battery management over training | `fig9_battery_management` |

---

## Code Architecture

```
symbiosis/          # C1+C2 theory
  fitness.py        # AgentFitness — long-run EMA reward tracker
  definitions.py    # RelationshipType enum + RoboticSymbiosisClassifier
  counterfactual_critic.py  # JointCritic / MarginalCritic
  reward_decomposition.py   # SymbioticRewardDecomposer
  convergence.py    # ConvergenceMonitor

tarware/            # Battery-enabled warehouse environment
  warehouse.py      # Core gymnasium env (A*, charging, heterogeneous tasks)
  energy_coupling.py
  astar.py
  replanning.py
  role_assignment.py

training/           # Wrapper layer
  symbiotic_wrapper.py  # SymbioticWrapper — obs augmentation + r_sym shaping

experiments/        # C3 runners + PPO
  run_symbiotic.py        # Symbiotic reward condition
  run_flat_cooperative.py # Flat-cooperative reward condition
  run_all_experiments.py  # Combined sweep driver
  run_convergence.py      # C2 convergence diagnostics
  run_gradient.py         # Heterogeneity gradient sweep
  run_ablation.py         # Ablation studies
  ppo_backends.py         # Local MAPPO and IPPO (legacy multi-method runner)
  generate_symbiosis_gifs.py
  generate_trained_gif.py

analysis/
  metrics.py        # TSI, RSI, mutualism fraction
  paper_figures.py  # Publication figures (9 panels)

configs/
  prism_symbiotic.yaml        # Symbiotic condition config
  prism_flat_cooperative.yaml # Flat-cooperative condition config

slurm/              # SLURM batch scripts
runs/               # Symlink → /proj/symmarl_ijrr2025/xuezhi/runs
results/            # Lightweight preliminary CSVs (in git)
```

---

## Notes

- `runs/` is a symlink to the SLURM cluster path. For local experiments use explicit `--output local_runs/...` flags.
- `max_inactivity_steps` must be set to `null` for training; the registered env default of 100 steps terminates episodes too aggressively.
- Ecological relationships are **measured** in both conditions (logged to `eval_metrics.csv`). In the flat-cooperative condition they are passive measurements only — they do not influence the reward signal.
