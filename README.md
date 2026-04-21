# Robotic Symbiosis in Heterogeneous Multi-Agent Warehouse Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research codebase for the project:
**"Emergent Symbiosis in Heterogeneous MARL: A Formal Framework for Capability-Complementary Robotic Teams"**

This repository studies when relationship-aware reward shaping is justified in multi-agent robotics, and when it is not. The central claim is narrow and testable: if capability complementarity is present, classifying and shaping inter-agent relationships should improve coordination; if complementarity is absent, the advantage should disappear.

<p align="center">
  <img src="assets/robotic_symbiosis_demo.gif" alt="Robotic symbiosis warehouse demo" width="760" />
</p>

<p align="center">
  Heuristic task-scheduling demo in the battery-enabled TARWARE warehouse environment.
</p>

---

## Research Program

The project is organized around three linked contributions:

- `C1` Formalize robotic symbiosis from ecological fitness theory using value-function counterfactuals.
- `C2` Build a reward decomposition that shapes behavior by relationship type while preserving policy-gradient convergence class.
- `C3` Run confirmation and falsification experiments to test whether the benefit is genuinely tied to capability complementarity.

The detailed step-by-step implementation plan lives in [TODO.md](TODO.md).

---

## Core Hypothesis

The project is built around a boundary condition, not a generic reward-shaping claim.

- In **heterogeneous teams**, symbiotic reward should improve throughput and increase mutualistic interactions because agents depend on each other structurally.
- In **homogeneous teams**, the same mechanism should provide little or no advantage because relationship classification becomes biologically meaningless without capability complementarity.
- Across a **heterogeneity gradient**, the benefit of symbiotic reward should increase with the degree of complementarity.
- For **convergence**, symbiotic shaping should change the final asymptote, not the convergence class.

If the homogeneous falsification experiment shows the same gain as the heterogeneous case, the biological framing is weakened and the contribution collapses toward ordinary cooperative reward shaping.

---

## What Is In This Repository

### Theory and measurement

- [symbiosis/fitness.py](symbiosis/fitness.py): long-run fitness estimates.
- [symbiosis/definitions.py](symbiosis/definitions.py): relationship types and formal measurement objects.
- [symbiosis/counterfactual_critic.py](symbiosis/counterfactual_critic.py): joint and marginal critics for counterfactual analysis.
- [symbiosis/reward_decomposition.py](symbiosis/reward_decomposition.py): symbiotic reward decomposition.
- [symbiosis/convergence.py](symbiosis/convergence.py): convergence-monitoring utilities.

### Environment and domain logic

- [tarware/warehouse.py](tarware/warehouse.py): battery-enabled heterogeneous warehouse environment.
- [tarware/heuristic.py](tarware/heuristic.py): heuristic controller and oracle-style baseline.
- [tarware/energy_coupling.py](tarware/energy_coupling.py): charging and energy-coupling logic.
- [tarware/replanning.py](tarware/replanning.py): adaptive replanning under battery constraints.
- [tarware/role_assignment.py](tarware/role_assignment.py): role-emergence tracking.

### Experiments

- [experiments/run_heuristic_baseline.py](experiments/run_heuristic_baseline.py): heuristic reference line.
- [experiments/run_heterogeneous.py](experiments/run_heterogeneous.py): heterogeneous Experiment 1 runner. Uses local MAPPO first, with IPPO fallback.
- [experiments/run_homogeneous.py](experiments/run_homogeneous.py): homogeneous falsification runner.
- [experiments/run_gradient.py](experiments/run_gradient.py): heterogeneity-gradient experiment runner.
- [experiments/run_convergence.py](experiments/run_convergence.py): convergence diagnostics runner.
- [experiments/run_all_experiments.py](experiments/run_all_experiments.py): combined sweep driver.

### Analysis and figures

- [analysis/metrics.py](analysis/metrics.py): TSI, RSI, mutualism fraction, convergence episode.
- [analysis/plot.py](analysis/plot.py): paper-style figure generation.

### Batch launchers

- [slurm/train_heterogeneous.slurm](slurm/train_heterogeneous.slurm)
- [slurm/train_homogeneous.slurm](slurm/train_homogeneous.slurm)
- [slurm/heuristic_baseline.slurm](slurm/heuristic_baseline.slurm)
- [slurm/run_all_experiments.slurm](slurm/run_all_experiments.slurm)

---

## Output Convention

For real experiment runs, prefer writing outputs under `runs/results/` rather than the top-level `results/` directory. This keeps heavy JSON outputs and batch artifacts out of the active login-node workspace.

Recommended layout:

```text
runs/
  results/
    heuristic_baseline.json
    hetero_results.json
    homo_results.json
    gradient_results.json
    convergence_results.json
    experiment_results.json
```

The new all-in-one SLURM script writes to `/proj/symmarl_ijrr2025/xuezhi/runs/results` by default for exactly this reason.

---

## Installation

```bash
git clone https://github.com/your-org/robotic-symbiosis-tarware.git
cd robotic-symbiosis-tarware
pip install -e ".[dev]"
```

Python `>=3.9` is required.

The project depends on:

- `torch`
- `gymnasium`
- `pyyaml`
- `imageio`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `skrl`

---

## Demo Asset

The demo GIF is generated with the heuristic mission scheduler, not with random macro actions. This matters because TARWARE actions are task targets and scheduling decisions; motion planning is handled internally by the environment after a target is assigned.

To regenerate the demo and the per-step decision log:

```bash
python experiments/generate_demo_gif.py \
    --output assets/robotic_symbiosis_demo.gif \
    --metadata_output assets/robotic_symbiosis_demo.json
```

---

## Experiments

### Heuristic baseline

```bash
python experiments/run_heuristic_baseline.py \
    --env tarware-small-4agvs-2pickers-partialobs-chg-v1 \
    --num_episodes 20 \
    --seed 42 \
    --output runs/results/heuristic_baseline.json
```

### Experiment 1: heterogeneous primary result

```bash
python experiments/run_heterogeneous.py \
    --config configs/heterogeneous.yaml \
    --timesteps 1000000 \
    --seeds 5 \
    --output runs/results/hetero_results.json
```

If you want to force the backend:

```bash
python experiments/run_heterogeneous.py \
    --config configs/heterogeneous.yaml \
    --backend mappo \
    --timesteps 1000000 \
    --seeds 5 \
    --output runs/results/hetero_results.json
```

Backend behavior:

- `auto`: try local `mappo`, then local `ippo`, then HARL placeholders.
- `mappo`: centralized critic by agent type.
- `ippo`: decentralized critic fallback.
- `happo` / `haddpg`: accepted as CLI targets, but require an external HARL installation not bundled here.

### Experiment 2: homogeneous falsification

```bash
python experiments/run_homogeneous.py \
    --config configs/homogeneous.yaml \
    --timesteps 500000 \
    --seeds 5 \
    --output runs/results/homo_results.json
```

### Experiment 3: heterogeneity gradient

```bash
python experiments/run_gradient.py \
    --config configs/gradient.yaml \
    --h_values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --seeds 5 \
    --output runs/results/gradient_results.json
```

### Experiment 4: convergence diagnostics

```bash
python experiments/run_convergence.py \
    --config configs/heterogeneous.yaml \
    --output runs/results/convergence_results.json
```

### Combined sweep

```bash
python experiments/run_all_experiments.py \
    --output runs/results/experiment_results.json
```

---

## SLURM Usage

To run the combined sweep on the cluster:

```bash
sbatch slurm/run_all_experiments.slurm
```

This script:

- activates the `battery` conda environment
- runs [experiments/run_all_experiments.py](experiments/run_all_experiments.py)
- writes the result JSON to `/proj/symmarl_ijrr2025/xuezhi/runs/results/experiment_results_${SLURM_JOB_ID}.json`
- keeps heavy outputs off the active login-node workspace

You can override the main job parameters at submission time:

```bash
TIMESTEPS=300000 SEEDS=5 HEURISTIC_EPISODES=20 sbatch slurm/run_all_experiments.slurm
```

Other available launchers:

```bash
sbatch slurm/train_heterogeneous.slurm
sbatch slurm/train_homogeneous.slurm
sbatch slurm/heuristic_baseline.slurm
```

---

## Figures

If the per-experiment JSON files are stored under `runs/results/`, generate figures with:

```bash
python analysis/plot.py runs/results
```

Expected outputs:

- `runs/results/figures/fig1_mutualism_hetero.pdf`
- `runs/results/figures/fig2_task_completion.pdf`

The current plotting script uses the per-experiment result files, not the single combined JSON produced by `run_all_experiments.py`.

---

## Preliminary Results

All numbers below come from completed SLURM runs on the UPPMAX GPU cluster. Raw data lives in
[results/](results/).

### Heuristic oracle (50 episodes, 5 000 steps/ep, seed 42)

| Environment | Agents | Del / ep | ± |
|---|---|---|---|
| `tarware-tiny-2agvs-1pickers-partialobs-chg-v1` | 2 AGV + 1 picker | **18.8** | 3.9 |
| `tarware-small-4agvs-2pickers-partialobs-chg-v1` | 4 AGV + 2 picker | **18.8** | 3.3 |
| `tarware-medium-6agvs-3pickers-partialobs-chg-v1` | 6 AGV + 3 picker | **18.8** | 2.5 |

Source: [results/sym_learning_curve.csv](results/sym_learning_curve.csv).
The near-identical means across scales reflect the fixed request-queue size (20) and
full-episode termination, not a coincidence.

### IPPO preliminary sweep (SLURM job 4861156)

200 000 timesteps, 500-step episodes, 3 seeds each, tiny env (2 AGV + 1 picker).
All methods are still in early exploration — deliveries-per-episode are near zero.

| Method | Del / ep (last 20 %) | ± |
|---|---|---|
| Individual | 0.13 | 0.04 |
| Team | 0.13 | 0.03 |
| Unclassified | 0.13 | 0.01 |
| **Symbiotic** | **0.17** | 0.02 |

At 200 k steps the environment is too sparse for meaningful differentiation. Longer runs
(≥ 1 M steps) are needed for the C3 claim to be evaluable.

### Symbiotic IPPO at 1 M steps (small env, sym_v2)

Single-seed run on `tarware-small-4agvs-2pickers-partialobs-chg-v1`.

| Metric | Last-20-episode mean |
|---|---|
| Deliveries / ep | 0.45 |
| TSI | 0.58 |
| RSI | 0.05 |
| Charger-escort fraction | 0.48 |

Source: [results/sym_learning_curve.csv](results/sym_learning_curve.csv),
[results/sym_roles.csv](results/sym_roles.csv).

### MAPPO at 7 M steps (small env)

Centralized-critic MAPPO with BC pre-training on heuristic demonstrations.

| Checkpoint | Policy del | Heuristic del (same ep) |
|---|---|---|
| 1 M steps | 0 | 13 |
| 6 M steps | **2** | 10 |
| 7 M steps | 2 | 15 |

Source: [results/mappo_eval_metrics.csv](results/mappo_eval_metrics.csv).

Policy / heuristic ratio at 7 M steps: **~13 %**. The environment remains hard for
policy learning at this scale; the heuristic oracle is the practical reference ceiling.

### Training animations

These GIFs were rendered during evaluation checkpoints of the MAPPO run.

| Checkpoint | Animation |
|---|---|
| 1 M steps (early exploration) | ![MAPPO 1M](assets/mappo_1m_steps.gif) |
| 7 M steps (best checkpoint) | ![MAPPO 7M](assets/mappo_7m_steps.gif) |

The heuristic scheduler demo (used for BC pre-training demonstrations):

![Heuristic demo](assets/robotic_symbiosis_demo.gif)

---

## Research Notes

This repository is a research codebase, not a polished benchmark package. The important parts are:

- the formal decomposition of the problem into `C1`, `C2`, and `C3`
- the presence of both confirmation and falsification experiments
- the ability to inspect relationship emergence, specialization, and convergence diagnostics separately

The strongest scientific claim here is not that symbiotic reward always wins. It is that it should only matter when the environment contains capability complementarity. That is the core falsifiable statement the repository is organized to test.

---

## Repository Layout

```text
robotic-symbiosis-tarware/
├── assets/
│   ├── robotic_symbiosis_demo.gif
│   ├── mappo_1m_steps.gif
│   └── mappo_7m_steps.gif
├── results/
│   ├── sym_learning_curve.csv
│   ├── sym_roles.csv
│   ├── sym_emergence.csv
│   └── mappo_eval_metrics.csv
├── symbiosis/
├── tarware/
├── training/
├── experiments/
│   ├── run_heuristic_baseline.py
│   ├── run_heterogeneous.py
│   ├── run_homogeneous.py
│   ├── run_gradient.py
│   ├── run_convergence.py
│   └── run_all_experiments.py
├── analysis/
│   ├── metrics.py
│   └── plot.py
├── configs/
│   ├── heterogeneous.yaml
│   ├── homogeneous.yaml
│   └── gradient.yaml
├── slurm/
│   ├── heuristic_baseline.slurm
│   ├── train_heterogeneous.slurm
│   ├── train_homogeneous.slurm
│   └── run_all_experiments.slurm
└── TODO.md
```
