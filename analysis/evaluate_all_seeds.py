"""
PRISM — Batch evaluation across all seeds + statistical tests.

For each condition (symbiotic, flat-cooperative) this script:
  1. Finds all checkpoint_best.pt files under the given run directory
  2. Evaluates each for N episodes (no GIF rendering)
  3. Computes per-seed mean deliveries
  4. Runs Welch t-test + Mann-Whitney U + 95% CI + Cohen's d
  5. Prints a paper-ready results table and saves to JSON

Usage
-----
~/anaconda3/envs/warehouse/bin/python analysis/evaluate_all_seeds.py \\
    --sym_dir  local_runs/checkpoints/prism_symbiotic_v2/26-04-25_02-42-19-170947_symbiotic \\
    --flat_dir local_runs/checkpoints/prism_flat_coop_v2/26-04-25_06-28-25-906590_flat_coop \\
    --heuristic_json local_runs/results/heuristic_baseline.json \\
    --episodes 20 \\
    --output   local_runs/results/eval_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401
from tarware.definitions import AgentType, PackageType


# ── MLP (same as training + evaluate_and_gif.py) ──────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_actors(ckpt_path: str, agv_obs_dim: int, pick_obs_dim: int,
                act_dim: int, hidden: int = 128):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    agv_actor  = MLP(agv_obs_dim,  act_dim, hidden)
    pick_actor = MLP(pick_obs_dim, act_dim, hidden)
    agv_actor.load_state_dict(ckpt["agv_actor_state_dict"])
    pick_actor.load_state_dict(ckpt["pick_actor_state_dict"])
    agv_actor.eval(); pick_actor.eval()
    return agv_actor, pick_actor, ckpt


def run_episodes(env, raw_env, agv_actor, pick_actor,
                 agv_idx, pick_idx, n_episodes: int,
                 max_steps: int, base_seed: int) -> List[int]:
    """Return total deliveries for each episode."""
    totals = []
    for ep in range(n_episodes):
        obs_list, _ = env.reset(seed=base_seed + ep)
        obs_list = list(obs_list)
        ep_deliveries = 0

        for _ in range(max_steps):
            valid_masks = raw_env.compute_valid_action_masks(pickers_to_agvs=True)
            actions = []
            for i, obs in enumerate(obs_list):
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                actor = agv_actor if i in agv_idx else pick_actor
                with torch.no_grad():
                    logits = actor(obs_t).squeeze(0).numpy()
                mask  = valid_masks[i]
                valid = np.where(np.asarray(mask) > 0)[0]
                if not valid.size:
                    actions.append(0)
                    continue
                logits_t = torch.as_tensor(logits)
                logits_masked = logits_t.clone().fill_(-1e9)
                logits_masked[valid] = logits_t[valid]
                dist = torch.distributions.Categorical(logits=logits_masked)
                actions.append(int(dist.sample().item()))

            next_obs, _, dones, truncs, info = env.step(actions)
            obs_list = list(next_obs)
            for cnt in info.get("deliveries_by_pkg_type", {}).values():
                ep_deliveries += int(cnt)
            if any(dones) or any(truncs):
                break

        totals.append(ep_deliveries)
    return totals


def eval_condition(run_dir: str, env_id: str, n_episodes: int,
                   max_steps: int, base_seed: int) -> dict:
    """Evaluate all checkpoint_best.pt files under run_dir."""
    import os as _os
    run_path = Path(run_dir)
    ckpts = sorted(
        Path(root) / fname
        for root, dirs, files in _os.walk(run_path, followlinks=True)
        for fname in files if fname == "checkpoint_best.pt"
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_best.pt found under {run_dir}")

    env     = gym.make(env_id, max_inactivity_steps=None, max_steps=max_steps)
    raw_env = env.unwrapped
    obs0, _ = env.reset(seed=base_seed)
    agents   = raw_env.agents
    agv_idx  = [i for i, a in enumerate(agents) if a.type == AgentType.AGV]
    pick_idx = [i for i, a in enumerate(agents) if a.type == AgentType.PICKER]
    act_dim  = raw_env.action_space.spaces[0].n
    agv_obs_dim  = int(np.asarray(obs0[agv_idx[0]]).shape[0])
    pick_obs_dim = int(np.asarray(obs0[pick_idx[0]]).shape[0])

    per_seed_means = []
    per_seed_detail = []

    for ckpt_path in ckpts:
        seed_name = ckpt_path.parent.name
        print(f"  [{seed_name}] evaluating {n_episodes} episodes …", flush=True)
        agv_actor, pick_actor, ckpt_meta = load_actors(
            str(ckpt_path), agv_obs_dim, pick_obs_dim, act_dim)
        totals = run_episodes(env, raw_env, agv_actor, pick_actor,
                              agv_idx, pick_idx, n_episodes, max_steps, base_seed)
        mean_del = float(np.mean(totals))
        std_del  = float(np.std(totals, ddof=1))
        print(f"    mean={mean_del:.2f}  std={std_del:.2f}  "
              f"episodes={totals}  best_ckpt={ckpt_meta.get('best_deliveries','?')}")
        per_seed_means.append(mean_del)
        per_seed_detail.append({
            "seed": seed_name,
            "best_ckpt_deliveries": ckpt_meta.get("best_deliveries"),
            "eval_mean": mean_del,
            "eval_std":  std_del,
            "episodes":  totals,
        })

    env.close()

    all_episodes = [d for detail in per_seed_detail for d in detail["episodes"]]
    return {
        "n_seeds":          len(per_seed_means),
        "n_episodes_each":  n_episodes,
        "per_seed_means":   per_seed_means,
        "per_seed_detail":  per_seed_detail,
        "pooled_mean":      float(np.mean(all_episodes)),
        "pooled_std":       float(np.std(all_episodes, ddof=1)),
        "seed_mean":        float(np.mean(per_seed_means)),
        "seed_std":         float(np.std(per_seed_means, ddof=1)),
        "all_episodes":     all_episodes,
    }


def bootstrap_ci(data: List[float], n_boot: int = 10000,
                 ci: float = 0.95) -> tuple:
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(data, len(data), replace=True))
                  for _ in range(n_boot)]
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return lo, hi


def cohens_d(a: List[float], b: List[float]) -> float:
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1))
                         / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PRISM — Evaluate all seeds and run statistical tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sym_dir",  required=True,
                        help="Symbiotic run directory (contains seed subdirs)")
    parser.add_argument("--flat_dir", required=True,
                        help="Flat-cooperative run directory")
    parser.add_argument("--heuristic_json", default=None,
                        help="Heuristic baseline JSON")
    parser.add_argument("--env", default="tarware-small-4agvs-2pickers-partialobs-chg-v1")
    parser.add_argument("--episodes",  default=20, type=int,
                        help="Evaluation episodes per seed")
    parser.add_argument("--steps",     default=1000, type=int)
    parser.add_argument("--seed",      default=0, type=int,
                        help="Base random seed for evaluation")
    parser.add_argument("--output",    default="local_runs/results/eval_stats.json")
    args = parser.parse_args()

    print(f"\n=== Evaluating SYMBIOTIC ({args.episodes} eps/seed) ===")
    sym = eval_condition(args.sym_dir, args.env, args.episodes, args.steps, args.seed)

    print(f"\n=== Evaluating FLAT-COOPERATIVE ({args.episodes} eps/seed) ===")
    flat = eval_condition(args.flat_dir, args.env, args.episodes, args.steps, args.seed)

    # ── Statistical tests ─────────────────────────────────────────────────────

    # Primary: Welch t-test on per-seed means (recommended for MARL papers)
    t_stat, p_val = stats.ttest_ind(sym["per_seed_means"], flat["per_seed_means"],
                                    equal_var=False)
    df = (np.var(sym["per_seed_means"]) / len(sym["per_seed_means"]) +
          np.var(flat["per_seed_means"]) / len(flat["per_seed_means"])) ** 2 / (
         (np.var(sym["per_seed_means"]) / len(sym["per_seed_means"])) ** 2 / (len(sym["per_seed_means"]) - 1) +
         (np.var(flat["per_seed_means"]) / len(flat["per_seed_means"])) ** 2 / (len(flat["per_seed_means"]) - 1))

    # Mann-Whitney U (non-parametric, on pooled episodes)
    mw_stat, mw_p = stats.mannwhitneyu(sym["all_episodes"], flat["all_episodes"],
                                        alternative="greater")

    # 95% CI (bootstrap on pooled episodes)
    sym_ci  = bootstrap_ci(sym["all_episodes"])
    flat_ci = bootstrap_ci(flat["all_episodes"])

    # Cohen's d (on pooled episodes)
    d = cohens_d(sym["all_episodes"], flat["all_episodes"])

    # Heuristic reference
    h_mean, h_std = None, None
    if args.heuristic_json:
        with open(args.heuristic_json) as f:
            h_data = json.load(f)
        env_key = list(h_data.keys())[0]
        h_mean = h_data[env_key]["mean_deliveries"]
        h_std  = h_data[env_key]["std_deliveries"]

    # ── Print results table ────────────────────────────────────────────────────
    W = 70
    print("\n" + "="*W)
    print("  PRISM — Evaluation Results (paper-ready)")
    print("="*W)
    print(f"  {'Condition':<22} {'Mean±SD':>12}  {'95% CI':>18}  {'N_eps':>6}")
    print("-"*W)

    sym_mean_all  = np.mean(sym["all_episodes"])
    sym_std_all   = np.std(sym["all_episodes"], ddof=1)
    flat_mean_all = np.mean(flat["all_episodes"])
    flat_std_all  = np.std(flat["all_episodes"], ddof=1)

    print(f"  {'Symbiotic (PRISM)':<22} "
          f"{sym_mean_all:>6.2f} ± {sym_std_all:<5.2f}  "
          f"[{sym_ci[0]:.2f}, {sym_ci[1]:.2f}]  "
          f"{len(sym['all_episodes']):>6}")
    print(f"  {'Flat-cooperative':<22} "
          f"{flat_mean_all:>6.2f} ± {flat_std_all:<5.2f}  "
          f"[{flat_ci[0]:.2f}, {flat_ci[1]:.2f}]  "
          f"{len(flat['all_episodes']):>6}")
    if h_mean is not None:
        h_ci = bootstrap_ci(h_data[list(h_data.keys())[0]]["deliveries_curve"])
        print(f"  {'Heuristic oracle':<22} "
              f"{h_mean:>6.2f} ± {h_std:<5.2f}  "
              f"[{h_ci[0]:.2f}, {h_ci[1]:.2f}]  "
              f"{h_data[list(h_data.keys())[0]]['n_episodes']:>6}")
    print("-"*W)
    print()
    print("  Statistical tests (Symbiotic vs Flat-cooperative):")
    print(f"    Welch t-test (per-seed means):  t={t_stat:.3f}, df={df:.1f}, p={p_val:.4f}"
          + (" **" if p_val < 0.01 else " *" if p_val < 0.05 else " (n.s.)"))
    print(f"    Mann-Whitney U (pooled eps):     U={mw_stat:.0f}, p={mw_p:.4f}"
          + (" **" if mw_p < 0.01 else " *" if mw_p < 0.05 else " (n.s.)"))
    print(f"    Cohen's d:                       {d:.3f}"
          + (" (large)" if abs(d) >= 0.8 else " (medium)" if abs(d) >= 0.5 else " (small)"))
    diff = sym_mean_all - flat_mean_all
    print(f"    PRISM advantage:                 +{diff:.2f} del/ep "
          f"({diff / flat_mean_all * 100:.1f}% over flat-coop)")
    if h_mean:
        print(f"    vs Heuristic oracle:             "
              f"{sym_mean_all:.2f} vs {h_mean:.2f} "
              f"({(sym_mean_all - h_mean) / h_mean * 100:+.1f}%)")
    print("="*W)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "symbiotic":     sym,
        "flat_coop":     flat,
        "heuristic": {"mean": h_mean, "std": h_std} if h_mean else None,
        "statistics": {
            "welch_t":         float(t_stat),
            "welch_p":         float(p_val),
            "welch_df":        float(df),
            "mannwhitney_u":   float(mw_stat),
            "mannwhitney_p":   float(mw_p),
            "cohens_d":        float(d),
            "sym_ci_95":       list(sym_ci),
            "flat_ci_95":      list(flat_ci),
            "sym_mean_pooled":  float(sym_mean_all),
            "flat_mean_pooled": float(flat_mean_all),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
