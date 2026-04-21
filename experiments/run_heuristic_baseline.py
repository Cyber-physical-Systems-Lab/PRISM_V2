"""
Heuristic baseline characterisation for the C3 comparison.

Runs the prescribed-rule heuristic (from tarware/heuristic.py) to establish:
  - Maximum achievable deliveries under coordination without learning
  - TSI / RSI under heuristic (upper bound for prescribed specialization)
  - Used as oracle reference line in Experiment 1 plots

This is the RQ4 heuristic baseline script, adapted for the IJRR paper.
It sweeps environment sizes and AGV:picker ratios to produce the reference
table that RL results are compared against.

Usage:
  python experiments/run_heuristic_baseline.py
  python experiments/run_heuristic_baseline.py --env tarware-small-4agvs-2pickers-partialobs-chg-v1
  python experiments/run_heuristic_baseline.py --num_episodes 20 --seed 42
"""

import argparse
import json
import time
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401
from tarware.heuristic import heuristic_episode


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Heuristic baseline for C3 comparison",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--env",          default=None, help="Single env id. If unset, runs full sweep.")
parser.add_argument("--num_episodes", default=10,   type=int)
parser.add_argument("--seed",         default=42,   type=int)
parser.add_argument("--output",       default="results/heuristic_baseline.json")


# ──────────────────────────────────────────────────────────────────────────────
# Environment sweep configuration
# ──────────────────────────────────────────────────────────────────────────────

SWEEP_ENVS = [
    "tarware-tiny-2agvs-1pickers-partialobs-chg-v1",
    "tarware-small-4agvs-2pickers-partialobs-chg-v1",
    "tarware-medium-6agvs-3pickers-partialobs-chg-v1",
    "tarware-large-8agvs-4pickers-partialobs-chg-v1",
    "tarware-extralarge-16agvs-8pickers-partialobs-chg-v1",
]


# ──────────────────────────────────────────────────────────────────────────────
# TSI / RSI from heuristic role data
# ──────────────────────────────────────────────────────────────────────────────

def compute_heuristic_tsi(role_history: list) -> float:
    """
    TSI from per-episode role-action counts in role_history.
    role_history: list of dicts {agent_id: dominant_role_str}
    """
    if not role_history:
        return 0.0
    from analysis.metrics import compute_tsi
    # Build a simple role-count matrix: unique_roles x n_agents
    all_roles = sorted({r for ep in role_history for r in ep.values()})
    role_to_idx = {r: i for i, r in enumerate(all_roles)}
    agents = sorted(role_history[0].keys())
    counts = np.zeros((len(agents), len(all_roles)), dtype=float)
    for ep in role_history:
        for ai, agent_id in enumerate(agents):
            role = ep.get(agent_id, all_roles[0])
            counts[ai, role_to_idx[role]] += 1
    return compute_tsi(counts)


# ──────────────────────────────────────────────────────────────────────────────
# Single-env runner
# ──────────────────────────────────────────────────────────────────────────────

def run_single_env(env_id: str, num_episodes: int, seed: int) -> dict:
    print(f"  {env_id} × {num_episodes} episodes ...", end=" ", flush=True)
    t0 = time.time()

    try:
        env = gym.make(env_id, max_inactivity_steps=None)
    except Exception as e:
        print(f"SKIP ({e})")
        return {"error": str(e)}

    deliveries_all = []
    role_history   = []
    battery_mean_per_episode = []
    battery_end_per_episode = []
    np.random.seed(seed)

    for ep in range(num_episodes):
        try:
            result = heuristic_episode(env, seed=seed + ep)
        except Exception as e:
            print(f"\n    Episode {ep} failed: {e}")
            continue

        # heuristic_episode returns:
        # (all_infos, global_episode_return, episode_returns,
        #  coupling_metrics, replanning_metrics, role_metrics)
        if isinstance(result, (list, tuple)) and len(result) >= 1:
            all_infos = result[0] if isinstance(result[0], list) else []
            total_deliveries = sum(info.get("shelf_deliveries", 0) for info in all_infos)
            deliveries_all.append(float(total_deliveries))

            if all_infos:
                # Collect per-agent battery statistics for this episode.
                battery_trace = [info.get("battery_levels", []) for info in all_infos]
                if battery_trace and battery_trace[0]:
                    battery_arr = np.asarray(battery_trace, dtype=float)
                    battery_mean_per_episode.append(battery_arr.mean(axis=0).tolist())
                    battery_end_per_episode.append(battery_arr[-1].tolist())

            role_metrics = result[5] if len(result) >= 6 else None
            if role_metrics is not None and hasattr(role_metrics, "profiles"):
                role_history.append({
                    str(agent_id): profile.dominant_role()
                    for agent_id, profile in role_metrics.profiles.items()
                })

    elapsed = time.time() - t0
    print(f"{len(deliveries_all)} eps, {elapsed:.1f}s")

    tsi = compute_heuristic_tsi(role_history) if role_history else 0.0

    if battery_mean_per_episode:
        battery_mean_overall = np.asarray(battery_mean_per_episode, dtype=float).mean(axis=0).tolist()
    else:
        battery_mean_overall = []

    if battery_end_per_episode:
        battery_end_mean = np.asarray(battery_end_per_episode, dtype=float).mean(axis=0).tolist()
    else:
        battery_end_mean = []

    result_summary = {
        "env":              env_id,
        "mean_deliveries":  float(np.mean(deliveries_all))  if deliveries_all else 0.0,
        "std_deliveries":   float(np.std(deliveries_all))   if deliveries_all else 0.0,
        "deliveries_curve": deliveries_all,
        "battery_mean_per_agent": battery_mean_overall,
        "battery_end_mean_per_agent": battery_end_mean,
        "battery_mean_curve_per_episode": battery_mean_per_episode,
        "battery_end_curve_per_episode": battery_end_per_episode,
        "tsi":              tsi,
        "n_episodes":       len(deliveries_all),
        "elapsed_sec":      round(elapsed, 1),
    }

    env.close()
    return result_summary


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parser.parse_args()
    envs = [args.env] if args.env else SWEEP_ENVS

    print(f"Heuristic baseline — {len(envs)} env(s), {args.num_episodes} episodes each")

    results = {}
    for env_id in envs:
        results[env_id] = run_single_env(env_id, args.num_episodes, args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nHeuristic baseline saved to {out}")

    # Print summary table
    print("\n--- Heuristic Baseline Summary ---")
    print(f"{'Env':<55}  {'Del/ep':>8}  {'±':>6}  TSI")
    for env_id, r in results.items():
        if "error" not in r:
            print(f"{env_id:<55}  {r['mean_deliveries']:>8.2f}  {r['std_deliveries']:>6.2f}  {r['tsi']:.3f}")


if __name__ == "__main__":
    main()
