"""
PRISM — Unified experiment driver.

Runs the two primary reward-condition experiments in sequence and produces
one combined summary JSON.

C3 core comparison:
  - Symbiotic condition:       r_i = r_task_i + r_sym_i
  - Flat-cooperative condition: r_i = r_task_i + alpha * mean_j(r_task_j)

Same team (4 AGVs + 2 pickers), same environment, same training budget.
Only the reward structure differs.

Usage:
  python experiments/run_all_experiments.py
  python experiments/run_all_experiments.py --timesteps 1000000 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _run(cmd: List[str], title: str) -> float:
    print(f"\n=== {title} ===")
    print("$", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    elapsed = time.time() - t0
    print(f"[OK] {title} in {elapsed:.1f}s")
    return elapsed


def _build_claim_checks(
    symbiotic: Dict[str, Any],
    flat_coop: Dict[str, Any],
) -> Dict[str, Any]:
    """C3: symbiotic should out-perform flat-cooperative on throughput, energy, mutualism."""

    def _backend_result(data: dict) -> dict:
        canonical = data.get("canonical_backend", "")
        return data.get("results", {}).get(canonical, {})

    sym = _backend_result(symbiotic)
    flat = _backend_result(flat_coop)

    sym_completion  = float(sym.get("mean_completion", 0.0))
    flat_completion = float(flat.get("mean_completion", 0.0))
    sym_tsi         = float(sym.get("tsi", 0.0))
    flat_tsi        = float(flat.get("tsi", 0.0))
    sym_mutualism   = float(
        sum(c[-1] for c in sym.get("mutualism_curves", []) if c) / max(1, len(sym.get("mutualism_curves", [])))
        if sym.get("mutualism_curves") else 0.0
    )
    flat_mutualism  = float(
        sum(c[-1] for c in flat.get("mutualism_curves", []) if c) / max(1, len(flat.get("mutualism_curves", [])))
        if flat.get("mutualism_curves") else 0.0
    )

    return {
        "C3": {
            "symbiotic_mean_completion":   sym_completion,
            "flat_coop_mean_completion":   flat_completion,
            "symbiotic_advantage":         sym_completion - flat_completion,
            "symbiotic_tsi":               sym_tsi,
            "flat_coop_tsi":               flat_tsi,
            "symbiotic_mutualism_final":   sym_mutualism,
            "flat_coop_mutualism_final":   flat_mutualism,
            "symbiotic_higher_completion": bool(sym_completion > flat_completion),
            "symbiotic_higher_mutualism":  bool(sym_mutualism  > flat_mutualism),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--timesteps",    default=200_000, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    parser.add_argument("--rollout",      default=256, type=int)
    parser.add_argument("--ppo_epochs",   default=3, type=int)
    parser.add_argument("--lr",           default=3e-4, type=float)
    parser.add_argument("--entropy",      default=0.01, type=float)
    parser.add_argument("--max_ep_steps", default=500, type=int)
    parser.add_argument("--heuristic_episodes", default=50, type=int)

    # Per-condition output paths (SLURM wrapper passes these explicitly).
    parser.add_argument("--symbiotic_checkpoint_dir", default="runs/prism_symbiotic")
    parser.add_argument("--symbiotic_tb_logdir",      default="runs/tensorboard/prism_symbiotic")
    parser.add_argument("--flat_coop_checkpoint_dir", default="runs/prism_flat_cooperative")
    parser.add_argument("--flat_coop_tb_logdir",      default="runs/tensorboard/prism_flat_cooperative")

    parser.add_argument("--heuristic_output",  default="runs/results/heuristic_baseline.json")
    parser.add_argument("--symbiotic_output",  default="runs/results/prism_symbiotic.json")
    parser.add_argument("--flat_coop_output",  default="runs/results/prism_flat_cooperative.json")
    parser.add_argument("--output",            default="runs/results/prism_results.json")

    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["heuristic", "symbiotic", "flat_coop"],
        help="Stages to skip.",
    )

    args = parser.parse_args()

    py = sys.executable
    timings: Dict[str, float] = {}

    if "symbiotic" not in args.skip:
        timings["symbiotic"] = _run(
            [
                py, "experiments/run_symbiotic.py",
                "--config",       "configs/prism_symbiotic.yaml",
                "--timesteps",    str(args.timesteps),
                "--rollout_steps",str(args.rollout),
                "--epochs",       str(args.ppo_epochs),
                "--lr",           str(args.lr),
                "--entropy_coef", str(args.entropy),
                "--max_ep_steps", str(args.max_ep_steps),
                "--seeds",        *map(str, args.seeds),
                "--checkpoint_dir", args.symbiotic_checkpoint_dir,
                "--tb_logdir",    args.symbiotic_tb_logdir,
                "--output",       args.symbiotic_output,
            ],
            "PRISM — Symbiotic condition",
        )

    if "flat_coop" not in args.skip:
        timings["flat_coop"] = _run(
            [
                py, "experiments/run_flat_cooperative.py",
                "--config",       "configs/prism_flat_cooperative.yaml",
                "--timesteps",    str(args.timesteps),
                "--rollout_steps",str(args.rollout),
                "--epochs",       str(args.ppo_epochs),
                "--lr",           str(args.lr),
                "--entropy_coef", str(args.entropy),
                "--max_ep_steps", str(args.max_ep_steps),
                "--seeds",        *map(str, args.seeds),
                "--checkpoint_dir", args.flat_coop_checkpoint_dir,
                "--tb_logdir",    args.flat_coop_tb_logdir,
                "--output",       args.flat_coop_output,
            ],
            "PRISM — Flat-cooperative condition",
        )

    symbiotic = _read_json(ROOT / args.symbiotic_output)
    flat_coop  = _read_json(ROOT / args.flat_coop_output)

    combined = {
        "meta": {
            "driver":       "experiments/run_all_experiments.py",
            "timings_sec":  timings,
            "args":         vars(args),
        },
        "files": {
            "symbiotic":       args.symbiotic_output,
            "flat_cooperative": args.flat_coop_output,
        },
        "checks": _build_claim_checks(symbiotic, flat_coop),
        "results": {
            "symbiotic":       symbiotic,
            "flat_cooperative": flat_coop,
        },
    }

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nCombined PRISM results written to {out}")


if __name__ == "__main__":
    main()
