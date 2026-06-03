# PRISM: Policy-shaping via Reward Decomposition for Inter-agent Symbiosis

This repository contains the code and compact paper artifacts for:

**PRISM: Policy-shaping via Reward decomposition for Inter-agent Symbiosis in
Multi-Robot Cooperation**

## Installation

Clone the repository, cd to the project directory, and run:

`pip install -e .`

The package requires Python 3.9 or newer. Core dependencies are listed in
`pyproject.toml` and include Gymnasium, PyTorch, SKRL, NumPy, pandas,
matplotlib, seaborn, TensorBoard, Pillow, and imageio.

## Experiments

Create or modify experiment entry points inside the `experiments` folder. The
main training experiment is `experiments/run_heterogeneous.py`, which runs the
heterogeneous Tarware setting with AGV and picker agents under the PRISM,
flat-cooperative, and task-only reward conditions.

Experiment settings are stored in `configs`. The current heterogeneous package
mix configuration is in `configs/heterogeneous.yaml`, and the rule-based
baseline configuration is in `configs/heuristic_baseline.yaml`.

The heuristic baseline is implemented in `experiments/run_heuristic_baseline.py`.
It writes per-seed evaluation outputs in the same directory layout as the
training runs so trained policies and heuristic policies can be compared with
the same downstream tooling.

## Running an experiment

See the `slurm` folder for cluster job scripts. To run the heterogeneous PRISM
experiment on a Slurm cluster, make sure you are in the repository root and run:

`sbatch slurm/train_heterogeneous.slurm`

To run the heuristic baseline on the cluster:

`sbatch slurm/heuristic_baseline.slurm`

To run the heterogeneous experiment locally:

`python experiments/run_heterogeneous.py --config configs/heterogeneous.yaml`

To run the heuristic baseline locally:

`python experiments/run_heuristic_baseline.py --config configs/heuristic_baseline.yaml`

Training outputs are written under `runs`, including JSON/CSV metrics,
per-seed run folders, checkpoints, and TensorBoard logs.

## Central Functions

The Tarware multi-agent warehouse environment is implemented in
`tarware/warehouse.py`. Environment registration happens in `tarware/__init__.py`,
including the standard charging environments and package-heterogeneous variants.

The PRISM reward-shaping logic is in `experiments/symbiosis_shaping.py`.
The main training loop, PPO backends, reward-condition selection, event logging,
and evaluation output writing are in `experiments/run_heterogeneous.py`.

The rule-based policy used for the heuristic baseline is in `tarware/heuristic.py`,
with the baseline runner in `experiments/run_heuristic_baseline.py`.

## Citation

```bibtex
@article{niu2026prism,
  title   = {PRISM: Policy-shaping via Reward decomposition for Inter-agent Symbiosis in Multi-Robot Cooperation},
  author  = {Niu, Xuezhi and Broo, Didem Gurdur},
  journal = {xxxxx},
  year    = {2026}
}
```
