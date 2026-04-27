"""
PRISM — Extract alternative metrics from v3 checkpoints.

Computes three metrics that may show symbiotic advantage beyond raw throughput:

  Metric 1 — Battery depletion rate
    Fraction of training episodes where any agent's battery hit 0 (depletion
    penalty event). Commensalism reward should reduce unsafe charging patterns.

  Metric 2 — Convergence speed
    Number of training episodes to reach 75% of final steady-state performance.
    Faster convergence = symbiotic signal guides agents to cooperation earlier.

  Metric 3 — Resilience
    Evaluate each trained policy with one AGV forced idle (simulates agent failure).
    Delivery drop vs full-team evaluation. Symbiotic teams may adapt better.

Usage
-----
~/anaconda3/envs/warehouse/bin/python analysis/extract_metrics.py \\
    --sym_ckpt_dir  local_runs/checkpoints/prism_symbiotic_v3/... \\
    --flat_ckpt_dir local_runs/checkpoints/prism_flat_coop_v3/... \\
    --task_ckpt_dir local_runs/checkpoints/prism_task_only_v2/... \\
    --output        local_runs/results/alternative_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401
from tarware.definitions import AgentType

PKG_DIST = {"SOLO": 0.20, "STANDARD": 0.30, "LARGE": 0.10,
            "HEAVY": 0.25, "PICKER_SOLO": 0.15}


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)


# ── Metric 1: Battery depletion rate from training CSVs ───────────────────────

def metric1_depletion_rate(ckpt_dir: Path) -> dict:
    """Fraction of episodes where any agent battery hit 0 (depletion event)."""
    import os
    all_rates = []
    all_mean_min_bat = []

    for root, _, files in os.walk(ckpt_dir, followlinks=True):
        if "eval_metrics.csv" in files:
            df = pd.read_csv(Path(root) / "eval_metrics.csv")
            if "battery_min_all_agents" not in df.columns:
                continue
            # Depletion = battery_min == 0
            depletion_rate = (df["battery_min_all_agents"] == 0).mean()
            mean_min = df["battery_min_all_agents"].mean()
            all_rates.append(float(depletion_rate))
            all_mean_min_bat.append(float(mean_min))

    if not all_rates:
        return {"depletion_rate": None, "mean_min_battery": None, "n_seeds": 0}

    return {
        "depletion_rate":    float(np.mean(all_rates)),
        "depletion_rate_std": float(np.std(all_rates, ddof=1)) if len(all_rates) > 1 else 0.0,
        "mean_min_battery":  float(np.mean(all_mean_min_bat)),
        "n_seeds":           len(all_rates),
        "per_seed":          all_rates,
    }


# ── Metric 2: Convergence speed from training CSVs ────────────────────────────

def metric2_convergence_speed(ckpt_dir: Path,
                               threshold_frac: float = 0.75,
                               window: int = 20) -> dict:
    """Episodes to reach threshold_frac * final_performance."""
    import os
    all_conv_eps = []

    for root, _, files in os.walk(ckpt_dir, followlinks=True):
        if "eval_metrics.csv" in files:
            df = pd.read_csv(Path(root) / "eval_metrics.csv")
            if "deliveries" not in df.columns or len(df) < window * 2:
                continue

            y = df["deliveries"].values
            final_perf = y[-window:].mean()
            threshold  = threshold_frac * final_perf

            if final_perf < 0.5:  # seed didn't learn anything meaningful
                continue

            rolling = pd.Series(y).rolling(window, min_periods=1).mean().values
            crossed = np.where(rolling >= threshold)[0]
            conv_ep = int(crossed[0]) if len(crossed) > 0 else len(y)
            all_conv_eps.append(conv_ep)

    if not all_conv_eps:
        return {"convergence_episode": None, "n_seeds": 0}

    return {
        "convergence_episode":     float(np.mean(all_conv_eps)),
        "convergence_episode_std": float(np.std(all_conv_eps, ddof=1)) if len(all_conv_eps) > 1 else 0.0,
        "threshold_frac":          threshold_frac,
        "n_seeds":                 len(all_conv_eps),
        "per_seed":                all_conv_eps,
    }


# ── Metric 3: Resilience (one AGV forced idle) ────────────────────────────────

def _load_actors(ckpt_path, agv_obs_dim, pick_obs_dim, act_dim):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    agv  = MLP(agv_obs_dim,  act_dim)
    pick = MLP(pick_obs_dim, act_dim)
    agv.load_state_dict(ckpt["agv_actor_state_dict"])
    pick.load_state_dict(ckpt["pick_actor_state_dict"])
    agv.eval(); pick.eval()
    return agv, pick


def _run_resilience_episodes(env, raw_env, agv_actor, pick_actor,
                              agv_idx, pick_idx, failed_agv_idx,
                              n_episodes, max_steps, base_seed):
    totals = []
    for ep in range(n_episodes):
        obs_list, _ = env.reset(seed=base_seed + ep)
        obs_list = list(obs_list)
        ep_del = 0

        for _ in range(max_steps):
            masks = raw_env.compute_valid_action_masks(pickers_to_agvs=True)
            actions = []
            for i, obs in enumerate(obs_list):
                if i == failed_agv_idx:   # failed agent — always idle
                    actions.append(0)
                    continue
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                actor = agv_actor if i in agv_idx else pick_actor
                with torch.no_grad():
                    logits = actor(obs_t).squeeze(0).numpy()
                valid = np.where(np.asarray(masks[i]) > 0)[0]
                if not valid.size:
                    actions.append(0); continue
                lt = torch.as_tensor(logits).clone().fill_(-1e9)
                lt[valid] = torch.as_tensor(logits)[valid]
                actions.append(int(torch.distributions.Categorical(logits=lt).sample()))

            obs_list, _, dones, truncs, info = env.step(actions)
            obs_list = list(obs_list)
            for cnt in info.get("deliveries_by_pkg_type", {}).values():
                ep_del += int(cnt)
            if any(dones) or any(truncs):
                break

        totals.append(ep_del)
    return totals


def metric3_resilience(ckpt_dir: Path, env_id: str,
                        n_episodes: int = 15, max_steps: int = 1000,
                        base_seed: int = 0) -> dict:
    """Compare full-team vs one-AGV-failed delivery performance."""
    import os
    env = gym.make(env_id, max_inactivity_steps=None, max_steps=max_steps,
                   package_distribution=PKG_DIST)
    raw_env = env.unwrapped
    obs0, _ = env.reset(seed=base_seed)
    agents   = raw_env.agents
    agv_idx  = [i for i, a in enumerate(agents) if a.type == AgentType.AGV]
    pick_idx = [i for i, a in enumerate(agents) if a.type == AgentType.PICKER]
    act_dim  = raw_env.action_space.spaces[0].n
    agv_obs  = int(np.asarray(obs0[agv_idx[0]]).shape[0])
    pick_obs = int(np.asarray(obs0[pick_idx[0]]).shape[0])
    failed_agv = agv_idx[0]   # always fail the first AGV

    full_means, failed_means = [], []

    ckpts = sorted(
        Path(root) / fname
        for root, _, files in os.walk(ckpt_dir, followlinks=True)
        for fname in files if fname == "checkpoint_best.pt"
    )

    for ckpt_path in ckpts:
        agv_actor, pick_actor = _load_actors(str(ckpt_path), agv_obs, pick_obs, act_dim)
        seed_name = ckpt_path.parent.name

        full   = _run_resilience_episodes(env, raw_env, agv_actor, pick_actor,
                                          agv_idx, pick_idx, None,
                                          n_episodes, max_steps, base_seed)
        failed = _run_resilience_episodes(env, raw_env, agv_actor, pick_actor,
                                          agv_idx, pick_idx, failed_agv,
                                          n_episodes, max_steps, base_seed)

        m_full   = float(np.mean(full))
        m_failed = float(np.mean(failed))
        drop     = (m_full - m_failed) / max(m_full, 1e-6) * 100
        print(f"    {seed_name}: full={m_full:.2f}  failed={m_failed:.2f}  "
              f"drop={drop:.1f}%")
        full_means.append(m_full)
        failed_means.append(m_failed)

    env.close()

    mean_full   = float(np.mean(full_means))
    mean_failed = float(np.mean(failed_means))
    mean_drop   = (mean_full - mean_failed) / max(mean_full, 1e-6) * 100

    return {
        "full_team_mean":    mean_full,
        "failed_agv_mean":   mean_failed,
        "performance_drop_pct": mean_drop,
        "n_seeds": len(full_means),
        "per_seed_full":   full_means,
        "per_seed_failed": failed_means,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PRISM — Extract alternative metrics from v3 checkpoints")
    parser.add_argument("--sym_ckpt_dir",  required=True)
    parser.add_argument("--flat_ckpt_dir", required=True)
    parser.add_argument("--task_ckpt_dir", default=None)
    parser.add_argument("--env", default="tarware-small-4agvs-2pickers-partialobs-chg-v1")
    parser.add_argument("--resilience_episodes", default=15, type=int)
    parser.add_argument("--output", default="local_runs/results/alternative_metrics.json")
    args = parser.parse_args()

    results = {}
    conditions = [("symbiotic", args.sym_ckpt_dir),
                  ("flat_coop", args.flat_ckpt_dir)]
    if args.task_ckpt_dir:
        conditions.append(("task_only", args.task_ckpt_dir))

    for cond, ckpt_dir in conditions:
        print(f"\n{'='*60}")
        print(f"  {cond.upper()}")
        print(f"{'='*60}")
        d = Path(ckpt_dir)

        print("\nMetric 1 — Battery depletion rate …")
        m1 = metric1_depletion_rate(d)
        print(f"  Depletion rate: {m1['depletion_rate']:.3f} ± {m1['depletion_rate_std']:.3f}")
        print(f"  Mean min battery: {m1['mean_min_battery']:.2f}")

        print("\nMetric 2 — Convergence speed (75% of final perf) …")
        m2 = metric2_convergence_speed(d)
        if m2["convergence_episode"] is not None:
            print(f"  Converges at episode: {m2['convergence_episode']:.0f} "
                  f"± {m2['convergence_episode_std']:.0f}")
        else:
            print("  No convergent seeds found")

        print("\nMetric 3 — Resilience (1 AGV forced idle) …")
        m3 = metric3_resilience(d, args.env, args.resilience_episodes)
        print(f"  Full team:   {m3['full_team_mean']:.2f} del/ep")
        print(f"  1 AGV down:  {m3['failed_agv_mean']:.2f} del/ep")
        print(f"  Drop:        {m3['performance_drop_pct']:.1f}%")

        results[cond] = {"metric1_depletion": m1,
                         "metric2_convergence": m2,
                         "metric3_resilience": m3}

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 68
    print(f"\n{'='*W}")
    print("  PRISM — Alternative Metrics Summary")
    print(f"{'='*W}")
    print(f"  {'Condition':<16} {'Depletion%':>11} {'Conv.ep':>9} {'Drop%':>8} {'1-AGV':>8}")
    print(f"  {'-'*16} {'-'*11} {'-'*9} {'-'*8} {'-'*8}")
    for cond, r in results.items():
        m1 = r["metric1_depletion"]
        m2 = r["metric2_convergence"]
        m3 = r["metric3_resilience"]
        dep  = f"{m1['depletion_rate']*100:.1f}%" if m1['depletion_rate'] is not None else "N/A"
        conv = f"{m2['convergence_episode']:.0f}" if m2['convergence_episode'] else "N/A"
        drop = f"{m3['performance_drop_pct']:.1f}%"
        fail = f"{m3['failed_agv_mean']:.2f}"
        print(f"  {cond:<16} {dep:>11} {conv:>9} {drop:>8} {fail:>8}")
    print(f"{'='*W}")
    print("  (lower depletion% = better | lower conv.ep = faster | lower drop% = more resilient)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
