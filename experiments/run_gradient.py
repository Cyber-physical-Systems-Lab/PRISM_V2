"""
C3 Heterogeneity Gradient Sweep — real PPO training across h ∈ {0, 0.5, 1.0}.

Validates claim C3: the symbiotic advantage over unclassified reward shaping
grows as environment heterogeneity increases.

Heterogeneity is operationalised as three discrete environments:
  h=0.0 → tarware-tiny-4agvs-0pickers-partialobs-chg-v1
           All-AGV, no pickers. classify_rel never fires → symbiotic ≡ individual.
  h=0.5 → tarware-tiny-2agvs-1pickers-partialobs-chg-v1
           Mixed: 2 AGVs + 1 picker, small scale, moderate role diversity.
  h=1.0 → tarware-small-4agvs-2pickers-partialobs-chg-v1
           Full heterogeneity: 4 AGVs + 2 pickers, largest task queue.

Three algorithm backends × four reward methods × N seeds are run at each h.

Expected result
---------------
advantage[h] = mean_deliveries["symbiotic"] − mean_deliveries["unclassified"]
  h=0.0 → advantage ≈ 0  (symbiotic reduces to individual when no pickers)
  h=0.5 → advantage > 0  (relationship classification starts helping)
  h=1.0 → advantage largest (full role complementarity exploited)

Output
------
runs/results/gradient_results_${SLURM_JOB_ID}.json

Schema
------
{
  "h_values": [0.0, 0.5, 1.0],
  "h_envs":   {"0.0": "...", "0.5": "...", "1.0": "..."},
  "backends": ["ippo", "hetppo", "mappo"],
  "methods":  ["individual", "team", "unclassified", "symbiotic"],
  "results": {
    "<h_str>": {
      "<backend>": {
        "<method>": {
          "mean_completion": float,
          "std_completion":  float,
          "deliveries_curves": [[int, ...]],
          "mutualism_curves":  [[float, ...]],
          "tsi": float,
          "rsi": float,
          "convergence_episodes": int,
          "n_seeds": int
        }
      }
    }
  },
  "advantage": {
    "<h_str>": {
      "<backend>": float   (symbiotic_mean - unclassified_mean)
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

# Discrete heterogeneity levels mapped to concrete environments
H_ENVS: dict[float, str] = {
    0.0: "tarware-tiny-4agvs-0pickers-partialobs-chg-v1",
    0.5: "tarware-tiny-2agvs-1pickers-partialobs-chg-v1",
    1.0: "tarware-small-4agvs-2pickers-partialobs-chg-v1",
}

METHODS = ["individual", "team", "unclassified", "symbiotic"]
BACKENDS = ["ippo", "hetppo", "mappo"]


parser = argparse.ArgumentParser(
    description="C3: Heterogeneity gradient sweep — real PPO, all backends × methods",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--h_values", nargs="+", type=float, default=[0.0, 0.5, 1.0],
    help="Heterogeneity levels to evaluate (subset of 0.0, 0.5, 1.0)",
)
parser.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
parser.add_argument("--timesteps", default=500_000, type=int)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Explicit seeds to run (e.g., --seeds 0 1 2)")
parser.add_argument("--rollout", default=256, type=int)
parser.add_argument("--ppo_epochs", default=3, type=int)
parser.add_argument("--mini_batches", default=4, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--lam", default=0.95, type=float)
parser.add_argument("--clip", default=0.2, type=float)
parser.add_argument("--entropy_coef", default=0.01, type=float)
parser.add_argument("--hidden_dim", default=128, type=int)
parser.add_argument("--max_ep_steps", default=500, type=int)
parser.add_argument("--max_inactivity_steps", default=None, type=int)
parser.add_argument("--unclassified_bonus", default=0.5, type=float)
parser.add_argument("--w_mutualism", default=2.0, type=float)
parser.add_argument("--w_commensalism", default=1.0, type=float)
parser.add_argument("--w_competition", default=-1.5, type=float)
parser.add_argument("--w_parasitism", default=-0.5, type=float)
parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
parser.add_argument("--log_interval", default=10, type=int)
parser.add_argument("--output", default="runs/results/gradient_results.json")


def main() -> None:
    args = parser.parse_args()

    # Validate h_values
    for h in args.h_values:
        if h not in H_ENVS:
            parser.error(f"--h_values: {h} not in supported set {list(H_ENVS)}")

    total_runs = len(args.h_values) * len(args.backends) * len(args.methods) * len(args.seeds)
    print("="*60)
    print("C3 Heterogeneity Gradient Sweep")
    print(f"  h_values:  {args.h_values}")
    print(f"  backends:  {args.backends}")
    print(f"  methods:   {args.methods}")
    print(f"  timesteps: {args.timesteps:,}   seeds: {args.seeds}")
    print(f"  total training runs: {total_runs}")
    print("=" * 60)

    all_results: dict = {
        "h_values": args.h_values,
        "h_envs": {str(h): H_ENVS[h] for h in args.h_values},
        "backends": args.backends,
        "methods": args.methods,
        "results": {},
        "advantage": {},
    }

    for h in args.h_values:
        h_str = str(h)
        env_id = H_ENVS[h]
        all_results["results"][h_str] = {}
        all_results["advantage"][h_str] = {}

        print(f"\n{'='*55}")
        print(f"h={h}  env={env_id}")
        print(f"{'='*55}")

        for backend in args.backends:
            all_results["results"][h_str][backend] = {}

            # Skip MAPPO at h=0 (homogeneous: only one agent type, no cross-type state)
            # MAPPO still works technically but is equivalent to HetPPO in this case;
            # we run it anyway for completeness and symmetry.
            for method in args.methods:
                print(f"\n  [{backend.upper()} / {method}]")
                seed_results = []
                for seed in args.seeds:
                    print(f"    seed {seed} ...", flush=True)
                    result = run_training(env_id, method, backend, seed, args)
                    final = result["deliveries_curve"][-1] if result["deliveries_curve"] else 0
                    print(f"      deliveries[-1]={final}")
                    seed_results.append(result)

                agg = aggregate_seeds(seed_results)
                all_results["results"][h_str][backend][method] = agg
                print(
                    f"    → mean={agg['mean_completion']:.3f} ± {agg['std_completion']:.3f}  "
                    f"tsi={agg['tsi']:.3f}"
                )

            # Compute symbiotic advantage over unclassified for this backend
            if "symbiotic" in args.methods and "unclassified" in args.methods:
                sym = all_results["results"][h_str][backend]["symbiotic"]["mean_completion"]
                unc = all_results["results"][h_str][backend]["unclassified"]["mean_completion"]
                all_results["advantage"][h_str][backend] = float(sym - unc)
            else:
                all_results["advantage"][h_str][backend] = None

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")

    # Print advantage table
    print("\n--- Symbiotic Advantage (symbiotic − unclassified deliveries/ep) ---")
    header = f"{'h':>5} " + "".join(f"  {b:>8}" for b in args.backends)
    print(header)
    print("-" * len(header))
    for h in args.h_values:
        row = f"{h:>5.1f} "
        for backend in args.backends:
            adv = all_results["advantage"][str(h)].get(backend)
            row += f"  {adv:>+8.3f}" if adv is not None else f"  {'N/A':>8}"
        print(row)


if __name__ == "__main__":
    main()
