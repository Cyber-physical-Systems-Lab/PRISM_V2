"""
C3 Experiment 2 — FALSIFICATION (homogeneous agents).

Homogeneous environment: ALL agents have the same capabilities (AGVs only,
no pickers). Because classify_rel requires at least one (AGV, picker) pair to
fire, the symbiotic shaping term is identically zero in this setting.

Expected result (falsification)
--------------------------------
  symbiotic  ≈  unclassified  ≈  individual
  team may differ slightly (mean-reward sharing still active)

If symbiotic significantly outperforms unclassified here, the advantage is
coming from generic cooperative shaping rather than symbiosis-specific
relationship classification — which would *reduce* the paper's contribution.

Two algorithm backends are compared (MAPPO degenerates to HetPPO in single-type
environments, so we run IPPO and HetPPO for the key comparison; MAPPO is
included as an optional backend for completeness):
  ippo   — per-agent independent PPO (one policy per agent ID)
  hetppo — type-shared decentralised PPO (one AGV policy for all agents)
  mappo  — type-shared centralised PPO (same actors, centralised critics)

Output
------
runs/results/homo_results_${SLURM_JOB_ID}.json

Schema
------
{
  "env": "tarware-tiny-4agvs-0pickers-partialobs-chg-v1",
  "backends": ["ippo", "hetppo"],
  "methods":  ["individual", "team", "unclassified", "symbiotic"],
  "results": {
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
  },
  "falsification": {
    "<backend>": {
      "sym_vs_unc_delta": float,   (symbiotic_mean - unclassified_mean)
      "falsified": bool            (True when |delta| < 0.5 deliveries/ep)
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

# Homogeneous environment: AGVs only, no pickers.
# classify_rel inner loop never executes → symbiotic ≡ individual.
HOMO_ENV = "tarware-tiny-4agvs-0pickers-partialobs-chg-v1"

METHODS = ["individual", "team", "unclassified", "symbiotic"]
# Primary backends: IPPO and HetPPO — the key FALSIFICATION comparison.
# MAPPO is supported via --backends mappo but not the default.
BACKENDS_DEFAULT = ["ippo", "hetppo"]
BACKENDS_ALL = ["ippo", "hetppo", "mappo"]

# Threshold for "no significant advantage" in falsification check
FALSIFICATION_DELTA_THRESHOLD = 0.5  # deliveries/episode


parser = argparse.ArgumentParser(
    description="C3 Experiment 2: Falsification — homogeneous env, symbiotic ≈ unclassified",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--env", default=HOMO_ENV,
                    help="Homogeneous environment (AGVs only, no pickers)")
parser.add_argument("--backends", nargs="+", default=BACKENDS_DEFAULT, choices=BACKENDS_ALL)
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
parser.add_argument("--output", default="runs/results/homo_results.json")


def main() -> None:
    args = parser.parse_args()

    print("=" * 60)
    print("C3 Experiment 2 — Falsification (homogeneous env)")
    print(f"  env:       {args.env}")
    print(f"  backends:  {args.backends}")
    print(f"  methods:   {args.methods}")
    print(f"  timesteps: {args.timesteps:,}   seeds: {args.seeds}")
    print("  Expected: symbiotic ≈ unclassified (no relationship pairs)")
    print("=" * 60)

    all_results: dict = {
        "env": args.env,
        "backends": args.backends,
        "methods": args.methods,
        "results": {},
        "falsification": {},
    }

    for backend in args.backends:
        all_results["results"][backend] = {}
        for method in args.methods:
            print(f"\n[{backend.upper()} / {method}]")
            seed_results = []
            for seed in args.seeds:
                print(f"  seed {seed} ...", flush=True)
                result = run_training(args.env, method, backend, seed, args)
                final = result["deliveries_curve"][-1] if result["deliveries_curve"] else 0
                print(f"    deliveries[-1]={final}")
                seed_results.append(result)

            agg = aggregate_seeds(seed_results)
            all_results["results"][backend][method] = agg
            print(f"  → mean={agg['mean_completion']:.3f} ± {agg['std_completion']:.3f}")

    # Falsification check
    print("\n--- Falsification Check ---")
    for backend in args.backends:
        res = all_results["results"][backend]
        if "symbiotic" in args.methods and "unclassified" in args.methods:
            sym_mean = res["symbiotic"]["mean_completion"]
            unc_mean = res["unclassified"]["mean_completion"]
            delta = sym_mean - unc_mean
            falsified = abs(delta) < FALSIFICATION_DELTA_THRESHOLD
            all_results["falsification"][backend] = {
                "sym_vs_unc_delta": float(delta),
                "falsified": falsified,
            }
            status = "PASS (falsified)" if falsified else "FAIL (symbiotic still wins)"
            print(f"  {backend.upper():<8}  sym={sym_mean:.3f}  unc={unc_mean:.3f}  "
                  f"delta={delta:+.3f}  → {status}")
        else:
            all_results["falsification"][backend] = {"sym_vs_unc_delta": None, "falsified": None}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")

    # Summary table
    print("\n--- Performance Summary ---")
    print(f"{'Backend':<8} {'Method':<14} {'Mean Del/ep':>12} {'Std':>8}")
    print("-" * 46)
    for backend in args.backends:
        for method in args.methods:
            agg = all_results["results"][backend][method]
            print(f"{backend:<8} {method:<14} {agg['mean_completion']:>12.3f} {agg['std_completion']:>8.3f}")


if __name__ == "__main__":
    main()
