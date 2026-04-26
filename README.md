# PRISM: Policy-shaping via Reward decomposition for Inter-agent Symbiosis in MARL

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research codebase for the paper:
**"PRISM: Policy-shaping via Reward decomposition for Inter-agent Symbiosis in MARL"**

The central claim is narrow and testable: symbiotic reward decomposition (`r_i = r_task + r_sym`) produces higher throughput and identifiable ecological relationship signatures compared to flat cooperative reward shaping — **on the same team, in the same environment**. The only variable is the reward structure.

---

## Overview

<p align="center">
  <img src="local_runs/eval/symbiotic_overview.gif" alt="Trained symbiotic policy — overview" width="760" />
</p>

<p align="center"><em>PRISM policy trained with symbiotic reward shaping (1M steps, IPPO). AGVs (orange hexagons) and pickers (blue diamonds) coordinate on STANDARD deliveries. Ecological relationship types (mutualism, commensalism) are detected and overlaid in real time.</em></p>

<p align="center">
  <img src="local_runs/eval/comparison_flat_vs_symbiotic.gif" alt="Flat-coop (left) vs Symbiotic (right)" width="760" />
</p>

<p align="center"><em>Side-by-side: flat-cooperative baseline (left) vs PRISM symbiotic policy (right). Same team, same environment, same episode seed. Symbiotic condition delivers 4× more packages.</em></p>

---

## Research Contributions

Three linked contributions, each independently evaluable:

| ID | Contribution | Key file |
|----|---|---|
| **C1** | Formalize robotic symbiosis via value-function counterfactuals (ecological relationship types) | `symbiosis/definitions.py`, `symbiosis/fitness.py` |
| **C2** | Reward decomposition `rᵢ = r_task + r_sym` with convergence guarantees | `symbiosis/reward_decomposition.py`, `symbiosis/convergence.py` |
| **C3** | Symbiotic vs flat-cooperative vs task-only comparison on identical mixed teams | `experiments/run_symbiotic.py`, `experiments/run_flat_cooperative.py` |

---

## Core Result (C3)

Evaluated across 10 symbiotic seeds (200 episodes), 8 flat-coop seeds (160 episodes), 6 task-only seeds (120 episodes), and 20 heuristic oracle episodes.

| Condition | Reward | Mean ± SD | 95% CI | N episodes |
|---|---|---|---|---|
| **Symbiotic (PRISM)** | `r_task + r_sym` | **3.88 ± 0.49** | [3.81, 3.94] | 200 |
| Flat-cooperative | `r_task + α·mean_team` | 2.06 ± 2.09 | [1.74, 2.39] | 160 |
| Heuristic oracle | Rule-based A* | 2.80 ± 1.36 | [2.20, 3.35] | 20 |
| Task-only (ablation) | `r_task` only | — | — | — |

**Statistical tests (Symbiotic vs Flat-cooperative):**
- Welch t-test (per-seed means): t = 3.47, df = 7.3, **p = 0.0096**
- Mann-Whitney U (pooled episodes): **p < 0.0001**
- Cohen's d: **1.26 (large effect)**
- PRISM vs flat-coop: **+88% throughput**
- PRISM vs heuristic oracle: **+38.8%**

<p align="center">
  <img src="local_runs/figures/prism_v2/fig7_symbiotic_vs_flat_coop.png" alt="Fig 7 — Core PRISM result" width="500" />
</p>

---

## Hypothesis

The experiment compares reward conditions on the **same** mixed team (4 AGVs + 2 pickers):

| Condition | Reward | Description |
|---|---|---|
| **Symbiotic** | `r_i = r_task_i + r_sym_i` | `r_sym` shaped by ecological relationship type per AGV-picker pair |
| **Flat-cooperative** | `r_i = r_task_i + α·mean_j(r_task_j)` | Shared team mean bonus; no relationship classification |
| **Task-only** | `r_i = r_task_i` | No shaping; ablation to isolate shaping effect |

Team composition, environment, architecture, and training budget are identical. The falsification criterion: if flat-cooperative achieves the same throughput, the biological reward decomposition has no measurable effect.

---

## Environment

TARWARE — a battery-enabled heterogeneous warehouse:

- **Two agent types**: AGVs (mobile, carry packages) and pickers (stationary, load/unload packages)
- **Five package types** encoding different cooperation requirements:

  | Package | AGVs | Pickers | Ecological parallel |
  |---|---|---|---|
  | SOLO | 1 | 0 | Neutralism |
  | PICKER_SOLO | 0 | 1 | Neutralism |
  | STANDARD | 1 | 1 | Mutualism — joint delivery |
  | LARGE | 2 | 2 | Mutualism — recruitment cost |
  | HEAVY | 1 | 1+ assist | Commensalism |

- **Battery system**: agents deplete energy carrying packages; contested charger access creates competition dynamics
- **Internal A\* motion planning**: experiment actions are task targets, not low-level moves

Environment ID: `tarware-small-4agvs-2pickers-partialobs-chg-v1`

---

## Relationship Scenarios

<p align="center">
  <img src="local_runs/eval/symbiotic_scenarios.gif" alt="Relationship type scenarios" width="760" />
</p>

<p align="center"><em>Per-type scenario clips captured from the trained PRISM policy: mutualism (STANDARD joint delivery) and commensalism (charging-while-idle).</em></p>

---

## Installation

```bash
git clone https://github.com/Cyber-physical-Systems-Lab/PRISM_V2.git
cd PRISM_V2
pip install -e ".[dev]"
```

Python ≥3.9 required. Dependencies: `torch`, `gymnasium`, `pyyaml`, `imageio`, `pandas`, `matplotlib`, `scipy`.

---

## Running Experiments

### Heuristic oracle baseline

```bash
python experiments/run_heuristic_baseline.py \
    --env tarware-small-4agvs-2pickers-partialobs-chg-v1 \
    --num_episodes 20 --seed 42 \
    --output local_runs/results/heuristic_baseline.json
```

### Symbiotic condition (C3 primary)

```bash
python experiments/run_symbiotic.py \
    --config configs/prism_symbiotic.yaml \
    --timesteps 1000000 --seeds 0 1 2 \
    --checkpoint_dir local_runs/checkpoints/prism_symbiotic \
    --output local_runs/results/prism_symbiotic.json
```

### Flat-cooperative baseline (C3 falsification)

```bash
python experiments/run_flat_cooperative.py \
    --config configs/prism_flat_cooperative.yaml \
    --timesteps 1000000 --seeds 0 1 2 \
    --checkpoint_dir local_runs/checkpoints/prism_flat_coop \
    --output local_runs/results/prism_flat_coop.json
```

### Task-only ablation

```bash
python experiments/run_flat_cooperative.py \
    --config configs/prism_task_only.yaml \
    --condition task_only --alpha_collab 0.0 \
    --timesteps 1000000 --seeds 0 1 2 \
    --checkpoint_dir local_runs/checkpoints/prism_task_only \
    --output local_runs/results/prism_task_only.json
```

### Evaluate trained policies (statistical tests)

```bash
python analysis/evaluate_all_seeds.py \
    --sym_dir  local_runs/checkpoints/prism_symbiotic \
    --flat_dir local_runs/checkpoints/prism_flat_coop \
    --heuristic_json local_runs/results/heuristic_baseline.json \
    --episodes 20 \
    --output   local_runs/results/eval_stats.json
```

Outputs mean ± SD, 95% bootstrap CI, Welch t-test, Mann-Whitney U, and Cohen's d.

### Generate evaluation GIFs

```bash
python experiments/evaluate_and_gif.py \
    --checkpoint local_runs/checkpoints/prism_symbiotic/.../checkpoint_best.pt \
    --checkpoint_baseline local_runs/checkpoints/prism_flat_coop/.../checkpoint_best.pt \
    --condition symbiotic --episodes 3 --steps 1000 --fps 6 \
    --output_dir local_runs/eval
```

---

## SLURM Cluster (UPPMAX)

```bash
# Individual conditions
sbatch slurm/train_symbiotic.slurm
sbatch slurm/train_flat_cooperative.slurm
sbatch slurm/train_task_only.slurm

# Extra seeds (adds 2 more seeds per condition)
sbatch slurm/train_extra_seeds.slurm

# Override seed count
NUM_SEEDS=5 sbatch slurm/train_symbiotic.slurm
```

Account: `UPPMAX2025-2-381` | Partition: `gpu` | Storage: `/proj/prism_v2/runs/`

---

## Paper Figures

Generated by `analysis/paper_figures.py` — PDF + PNG, 300 DPI, serif font:

| Figure | Description |
|---|---|
| Fig 1 | Task completion learning curves with heuristic reference line |
| Fig 2 | Mutualism fraction emergence over training |
| Fig 3 | Ecological relationship distribution at end of training |
| Fig 4 | Raw vs shaped reward divergence |
| Fig 5 | Training stability — actor entropy and KL divergence |
| Fig 6 | Team Specialisation Index (TSI) and RSI, with heuristic reference |
| **Fig 7** | **Symbiotic vs flat-coop vs heuristic oracle — core C3 result** |
| Fig 8 | Package-type delivery breakdown per condition |
| Fig 9 | Battery management over training with heuristic reference band |

```bash
python analysis/paper_figures.py \
    --symbiotic  local_runs/results/prism_symbiotic.json \
    --flat_coop  local_runs/results/prism_flat_coop.json \
    --heuristic  local_runs/results/heuristic_baseline.json \
    --eval_stats local_runs/results/eval_stats_final.json \
    --ckpt_dir   local_runs/checkpoints/prism_symbiotic \
    --output     local_runs/figures
```

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
  run_symbiotic.py           # Symbiotic reward condition
  run_flat_cooperative.py    # Flat-cooperative and task-only conditions
  run_heuristic_baseline.py  # Heuristic oracle
  run_all_experiments.py     # Combined sweep driver
  evaluate_and_gif.py        # Evaluation + GIF generation
  ppo_backends.py            # MAPPO / IPPO implementations

analysis/
  metrics.py              # TSI, RSI, mutualism fraction
  paper_figures.py        # Publication figures (9 panels, PDF+PNG)
  evaluate_all_seeds.py   # Multi-seed batch eval + Welch t-test / Cohen's d

configs/
  prism_symbiotic.yaml        # Symbiotic condition
  prism_flat_cooperative.yaml # Flat-cooperative baseline
  prism_task_only.yaml        # Task-only ablation

slurm/              # SLURM batch scripts (UPPMAX, gpu partition)
local_runs/         # Local experiment outputs (checkpoints, results, figures)
results/            # Lightweight preliminary CSVs (in git)
```

---

## Notes

- `local_runs/` holds checkpoints, results JSONs, and figures for local runs. Heavy cluster outputs go to `/proj/prism_v2/runs/`.
- `max_inactivity_steps` must be `null` during training to prevent premature episode termination.
- Ecological relationships are **measured** in all three conditions and logged to `eval_metrics.csv`. In flat-cooperative and task-only conditions they are passive measurements only — they do not influence the reward signal.
- The flat-cooperative condition shows high variance across seeds (some seeds converge, others collapse), which is itself a finding: symbiotic shaping provides more reliable convergence.
