"""
C2 Convergence Study — real PPO training across all algorithm × reward variants.

Validates claim C2: symbiotic reward shaping preserves convergence class
(gradient norms stay bounded, O(1/√T) rate maintained) while improving the
asymptotic delivery performance.

Three algorithm backends are compared:
  ippo   — per-agent independent PPO (one policy per agent ID, no sharing)
  hetppo — type-shared decentralised PPO (one policy per TYPE, local critics)
  mappo  — type-shared centralised PPO (one policy per TYPE, central critics)

Four reward methods:
  individual   — no shaping (raw task reward only)
  team         — mean reward shared across all agents
  unclassified — fixed bonus added to every agent every step
  symbiotic    — relationship-classified, φ-weighted bonus

Output
------
runs/results/convergence_results_${SLURM_JOB_ID}.json

Schema
------
{
  "env": "...",
  "backends": ["ippo", "hetppo", "mappo"],
  "methods":  ["individual", "team", "unclassified", "symbiotic"],
  "results": {
    "<backend>": {
      "<method>": {
        "mean_completion": float,
        "std_completion":  float,
        "convergence_episodes": int,
        "grad_norm_mean":  float,
        "grad_norm_final": float,
        "boundedness_ratio_mean": float,
        "boundedness_bounded":    bool,
        "deliveries_curves":  [[int, ...]],
        "mutualism_curves":   [[float, ...]],
        "grad_norm_history":  [float, ...],
        "tsi": float,
        "rsi": float,
        "n_seeds": int
      }
    }
  }
}
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401

from experiments.ppo_backends import aggregate_seeds, run_training

METHODS = ["individual", "team", "unclassified", "symbiotic"]
BACKENDS = ["ippo", "hetppo", "mappo"]


parser = argparse.ArgumentParser(
    description="C2: Convergence study — real PPO, all backends × methods",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--env", default="tarware-small-4agvs-2pickers-partialobs-chg-v1")
parser.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
parser.add_argument("--timesteps", default=500_000, type=int)
parser.add_argument("--rollout", default=256, type=int)
parser.add_argument("--ppo_epochs", default=3, type=int)
parser.add_argument("--mini_batches", default=4, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--lam", default=0.95, type=float)
parser.add_argument("--clip", default=0.2, type=float)
parser.add_argument("--entropy_coef", default=0.01, type=float)
parser.add_argument("--hidden_dim", default=128, type=int)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Explicit seeds to run (e.g., --seeds 0 1 2)")
parser.add_argument("--max_ep_steps", default=500, type=int)
parser.add_argument("--max_inactivity_steps", default=None, type=int)
parser.add_argument("--convergence_threshold", default=0.95, type=float,
                    help="Fraction of final performance used to define convergence episode")
parser.add_argument("--unclassified_bonus", default=0.5, type=float)
parser.add_argument("--w_mutualism", default=2.0, type=float)
parser.add_argument("--w_commensalism", default=1.0, type=float)
parser.add_argument("--w_competition", default=-1.5, type=float)
parser.add_argument("--w_parasitism", default=-0.5, type=float)
parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
parser.add_argument("--log_interval", default=10, type=int)
parser.add_argument("--output", default="runs/results/convergence_results.json")


def main() -> None:
    args = parser.parse_args()

    print("=" * 60)
    print("C2 Convergence Study")
    print(f"  env:       {args.env}")
    print(f"  backends:  {args.backends}")
    print(f"  methods:   {args.methods}")
    print(f"  timesteps: {args.timesteps:,}   seeds: {args.seeds}")
    print("=" * 60)

    all_results: dict = {
        "env": args.env,
        "backends": args.backends,
        "methods": args.methods,
        "results": {},
    }

    for backend in args.backends:
        all_results["results"][backend] = {}
        for method in args.methods:
            print(f"\n[{backend.upper()} / {method}]")
            seed_results = []
            for seed in args.seeds:
                print(f"  seed {seed} ...", flush=True)
                result = run_training(args.env, method, backend, seed, args)
                final_ep = result["deliveries_curve"][-1] if result["deliveries_curve"] else 0
                n_gn = len(result["grad_norms"])
                print(f"    deliveries[-1]={final_ep}  grad_norms collected={n_gn}")
                seed_results.append(result)

            agg = aggregate_seeds(seed_results, convergence_threshold=args.convergence_threshold)
            all_results["results"][backend][method] = agg
            print(
                f"  → mean_completion={agg['mean_completion']:.3f}  "
                f"convergence_ep={agg['convergence_episodes']}  "
                f"grad_norm_mean={agg['grad_norm_mean']:.4f}  "
                f"bounded={agg['boundedness_bounded']}"
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")

    # Print summary table
    print("\n--- Convergence Summary ---")
    print(f"{'Backend':<8} {'Method':<14} {'MeanDel':>8} {'ConvEp':>7} {'GradMean':>10} {'Bounded':>8}")
    print("-" * 60)
    for backend in args.backends:
        for method in args.methods:
            agg = all_results["results"][backend][method]
            print(
                f"{backend:<8} {method:<14} {agg['mean_completion']:>8.3f} "
                f"{agg['convergence_episodes']:>7d} {agg['grad_norm_mean']:>10.4f} "
                f"{'yes' if agg['boundedness_bounded'] else 'no':>8}"
            )


if __name__ == "__main__":
    main()
