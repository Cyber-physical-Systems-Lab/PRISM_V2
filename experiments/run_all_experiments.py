"""
Unified C1-C3 experiment driver.

Runs the repository's dedicated experiment scripts in sequence and produces
one combined summary JSON.

C1 evidence:
  - Heterogeneous training output (relationship-aware metrics)
  - Homogeneous falsification output

C2 evidence:
  - Convergence diagnostics output (gradient norms, boundedness ratios)

C3 evidence:
  - Heterogeneous confirmation
  - Homogeneous falsification
  - Heterogeneity gradient sweep

Usage:
  python experiments/run_all_experiments.py
  python experiments/run_all_experiments.py --timesteps 1000000 --seeds 5
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


def _safe_method_metric(results: Dict[str, Any], method: str, key: str, default: float = 0.0) -> float:
    return float(results.get("methods", {}).get(method, {}).get(key, default))


def _build_claim_checks(
    hetero: Dict[str, Any],
    homo: Dict[str, Any],
    gradient: Dict[str, Any],
    convergence: Dict[str, Any],
) -> Dict[str, Any]:
    # C1: relationship-aware signal should be present in heterogeneous runs.
    hetero_sym = hetero.get("methods", {}).get("symbiotic", {})
    hetero_mutualism_curves = hetero_sym.get("mutualism_curves", [])
    c1_check = {
        "heterogeneous_mutualism_data_present": bool(hetero_mutualism_curves),
        "heterogeneous_tsi_symbiotic": float(hetero_sym.get("tsi", 0.0)),
        "heterogeneous_rsi_symbiotic": float(hetero_sym.get("rsi", 0.0)),
    }

    # C2: boundedness and convergence behavior from convergence experiment.
    conv_sym = convergence.get("methods", {}).get("symbiotic", {})
    conv_ind = convergence.get("methods", {}).get("individual", {})
    c2_check = {
        "symbiotic_all_bounded": bool(conv_sym.get("boundedness", {}).get("all_bounded", False)),
        "symbiotic_max_ratio": float(conv_sym.get("boundedness", {}).get("max_ratio", 0.0)),
        "symbiotic_mean_convergence_episode": float(conv_sym.get("mean_convergence_episode", 0.0)),
        "individual_mean_convergence_episode": float(conv_ind.get("mean_convergence_episode", 0.0)),
    }

    # C3: hetero advantage should exceed homogeneous advantage; gradient trend should be positive.
    hetero_adv = _safe_method_metric(hetero, "symbiotic", "mean_completion") - _safe_method_metric(
        hetero, "unclassified", "mean_completion"
    )
    homo_adv = _safe_method_metric(homo, "symbiotic", "mean_completion") - _safe_method_metric(
        homo, "unclassified", "mean_completion"
    )

    gradient_adv: Dict[str, float] = {}
    for h, block in gradient.items():
        try:
            sym = float(block.get("symbiotic", {}).get("mean_completion", 0.0))
            unc = float(block.get("unclassified", {}).get("mean_completion", 0.0))
            gradient_adv[str(h)] = sym - unc
        except Exception:
            continue

    c3_check = {
        "heterogeneous_symbiotic_minus_unclassified": float(hetero_adv),
        "homogeneous_symbiotic_minus_unclassified": float(homo_adv),
        "hetero_advantage_exceeds_homo": bool(hetero_adv > homo_adv),
        "gradient_advantage_by_h": gradient_adv,
    }

    return {"C1": c1_check, "C2": c2_check, "C3": c3_check}


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Shared overrides kept compatible with existing SLURM wrapper.
    parser.add_argument("--timesteps", default=200_000, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42], help="Explicit seeds to run (e.g., --seeds 0 1 2)")
    parser.add_argument("--rollout", default=256, type=int)
    parser.add_argument("--ppo_epochs", default=3, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--entropy", default=0.01, type=float)
    parser.add_argument("--max_ep_steps", default=500, type=int)
    parser.add_argument("--heuristic_episodes", default=50, type=int)
    parser.add_argument("--hetero_checkpoint_dir", default="runs/c3_hetero")
    parser.add_argument("--hetero_tb_logdir", default="runs/tensorboard/c3_hetero")
    parser.add_argument("--hetero_checkpoint_every_episodes", default=100, type=int)

    # Config files for dedicated scripts.
    parser.add_argument("--hetero_config", default="configs/heterogeneous.yaml")
    parser.add_argument("--homo_config", default="configs/homogeneous.yaml")
    parser.add_argument("--gradient_config", default="configs/gradient.yaml")

    # Output files.
    parser.add_argument("--heuristic_output", default="results/heuristic_baseline.json")
    parser.add_argument("--hetero_output", default="results/hetero_results.json")
    parser.add_argument("--homo_output", default="results/homo_results.json")
    parser.add_argument("--gradient_output", default="results/gradient_results.json")
    parser.add_argument("--convergence_output", default="results/convergence_results.json")
    parser.add_argument("--output", default="results/experiment_results.json")

    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["heuristic", "hetero", "homo", "gradient", "convergence"],
        help="Optional stages to skip.",
    )

    args = parser.parse_args()

    py = sys.executable
    timings: Dict[str, float] = {}

    # Heuristic stage intentionally disabled.
    # if "heuristic" not in args.skip:
    #     timings["heuristic"] = _run(
    #         [
    #             py,
    #             "experiments/run_heuristic_baseline.py",
    #             "--num_episodes",
    #             str(args.heuristic_episodes),
    #             "--output",
    #             args.heuristic_output,
    #         ],
    #         "Heuristic Baseline",
    #     )

    if "hetero" not in args.skip:
        timings["hetero"] = _run(
            [
                py,
                "experiments/run_heterogeneous.py",
                "--config",
                args.hetero_config,
                "--timesteps",
                str(args.timesteps),
                "--rollout_steps",
                str(args.rollout),
                "--epochs",
                str(args.ppo_epochs),
                "--lr",
                str(args.lr),
                "--entropy_coef",
                str(args.entropy),
                "--max_ep_steps",
                str(args.max_ep_steps),
                "--seeds",
                *map(str, args.seeds),
                "--backends",
                "ippo",
                "hetppo",
                "mappo",
                "--checkpoint_dir",
                args.hetero_checkpoint_dir,
                "--tb_logdir",
                args.hetero_tb_logdir,
                "--checkpoint_every_episodes",
                str(args.hetero_checkpoint_every_episodes),
                "--output",
                args.hetero_output,
            ],
            "C3 Experiment 1 (Heterogeneous)",
        )

    if "homo" not in args.skip:
        timings["homo"] = _run(
            [
                py,
                "experiments/run_homogeneous.py",
                "--config",
                args.homo_config,
                "--timesteps",
                str(args.timesteps),
                "--rollout_steps",
                str(args.rollout),
                "--epochs",
                str(args.ppo_epochs),
                "--lr",
                str(args.lr),
                "--entropy_coef",
                str(args.entropy),
                "--seeds",
                *map(str, args.seeds),
                "--output",
                args.homo_output,
            ],
            "C3 Experiment 2 (Homogeneous Falsification)",
        )

    if "gradient" not in args.skip:
        timings["gradient"] = _run(
            [
                py,
                "experiments/run_gradient.py",
                "--config",
                args.gradient_config,
                "--timesteps",
                str(args.timesteps),
                "--rollout_steps",
                str(args.rollout),
                "--epochs",
                str(args.ppo_epochs),
                "--lr",
                str(args.lr),
                "--entropy_coef",
                str(args.entropy),
                "--seeds",
                *map(str, args.seeds),
                "--output",
                args.gradient_output,
            ],
            "C3 Experiment 3 (Heterogeneity Gradient)",
        )

    if "convergence" not in args.skip:
        timings["convergence"] = _run(
            [
                py,
                "experiments/run_convergence.py",
                "--config",
                args.hetero_config,
                "--timesteps",
                str(args.timesteps),
                "--rollout_steps",
                str(args.rollout),
                "--epochs",
                str(args.ppo_epochs),
                "--lr",
                str(args.lr),
                "--entropy_coef",
                str(args.entropy),
                "--seeds",
                *map(str, args.seeds),
                "--output",
                args.convergence_output,
            ],
            "C2 Convergence Validation",
        )

    heuristic = _read_json(ROOT / args.heuristic_output)
    hetero = _read_json(ROOT / args.hetero_output)
    homo = _read_json(ROOT / args.homo_output)
    gradient = _read_json(ROOT / args.gradient_output)
    convergence = _read_json(ROOT / args.convergence_output)

    combined = {
        "meta": {
            "driver": "experiments/run_all_experiments.py",
            "timings_sec": timings,
            "args": vars(args),
        },
        "files": {
            "heuristic": args.heuristic_output,
            "heterogeneous": args.hetero_output,
            "homogeneous": args.homo_output,
            "gradient": args.gradient_output,
            "convergence": args.convergence_output,
        },
        "checks": _build_claim_checks(hetero, homo, gradient, convergence),
        "results": {
            "heuristic": heuristic,
            "heterogeneous": hetero,
            "homogeneous": homo,
            "gradient": gradient,
            "convergence": convergence,
        },
    }

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nCombined C1-C3 results written to {out}")


if __name__ == "__main__":
    main()
