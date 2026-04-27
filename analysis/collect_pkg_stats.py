"""
PRISM — Collect per-package-type delivery statistics for all three conditions.

Evaluates the best checkpoint from each condition (1 seed, N episodes) and
records deliveries broken down by package type. Output JSON is consumed by
fig_package_distribution in paper_figures.py.

Usage
-----
~/anaconda3/envs/warehouse/bin/python analysis/collect_pkg_stats.py \
    --sym_ckpt  local_runs/checkpoints/prism_symbiotic_extra/.../mappo_seed705063/checkpoint_best.pt \
    --flat_ckpt local_runs/checkpoints/prism_flat_coop_v2/.../mappo_seed942863/checkpoint_best.pt \
    --task_ckpt local_runs/checkpoints/prism_task_only/.../ippo_seed659877/checkpoint_best.pt \
    --episodes 30 \
    --output local_runs/results/pkg_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401
from tarware.definitions import AgentType, PackageType


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)


def load_actors(ckpt_path, agv_obs_dim, pick_obs_dim, act_dim):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    agv  = MLP(agv_obs_dim,  act_dim)
    pick = MLP(pick_obs_dim, act_dim)
    agv.load_state_dict(ckpt["agv_actor_state_dict"])
    pick.load_state_dict(ckpt["pick_actor_state_dict"])
    agv.eval(); pick.eval()
    return agv, pick


def eval_pkg_types(ckpt_path, env_id, n_episodes, max_steps, base_seed):
    env     = gym.make(env_id, max_inactivity_steps=None, max_steps=max_steps, package_distribution={"SOLO":0.20,"STANDARD":0.30,"LARGE":0.10,"HEAVY":0.25,"PICKER_SOLO":0.15})
    raw_env = env.unwrapped
    obs0, _ = env.reset(seed=base_seed)
    agents   = raw_env.agents
    agv_idx  = [i for i, a in enumerate(agents) if a.type == AgentType.AGV]
    pick_idx = [i for i, a in enumerate(agents) if a.type == AgentType.PICKER]
    act_dim  = raw_env.action_space.spaces[0].n
    agv_obs  = int(np.asarray(obs0[agv_idx[0]]).shape[0])
    pick_obs = int(np.asarray(obs0[pick_idx[0]]).shape[0])

    agv_actor, pick_actor = load_actors(ckpt_path, agv_obs, pick_obs, act_dim)
    pkg_names = [p.name for p in PackageType]  # uppercase: SOLO, STANDARD, ...
    totals = {p: 0 for p in pkg_names}
    total_deliveries = []

    for ep in range(n_episodes):
        obs_list, _ = env.reset(seed=base_seed + ep)
        obs_list = list(obs_list)
        ep_pkg = {p: 0 for p in pkg_names}

        for _ in range(max_steps):
            masks = raw_env.compute_valid_action_masks(pickers_to_agvs=True)
            actions = []
            for i, obs in enumerate(obs_list):
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
            # env returns lowercase keys (solo, heavy); normalise to uppercase
            for pkg, cnt in info.get("deliveries_by_pkg_type", {}).items():
                key = pkg.upper()
                if key in ep_pkg:
                    ep_pkg[key] += int(cnt)
            if any(dones) or any(truncs):
                break

        for p in pkg_names:
            totals[p] += ep_pkg[p]
        total_deliveries.append(sum(ep_pkg.values()))

    env.close()
    mean_per_ep = {p: totals[p] / n_episodes for p in pkg_names}
    return mean_per_ep, float(np.mean(total_deliveries)), float(np.std(total_deliveries, ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym_ckpt",  required=True)
    parser.add_argument("--flat_ckpt", required=True)
    parser.add_argument("--task_ckpt", default=None)
    parser.add_argument("--env", default="tarware-small-4agvs-2pickers-partialobs-chg-v1")
    parser.add_argument("--episodes", default=30, type=int)
    parser.add_argument("--steps",    default=1000, type=int)
    parser.add_argument("--seed",     default=0, type=int)
    parser.add_argument("--output",   default="local_runs/results/pkg_stats.json")
    args = parser.parse_args()

    out = {}
    for name, ckpt in [("symbiotic", args.sym_ckpt),
                        ("flat_coop", args.flat_ckpt),
                        ("task_only", args.task_ckpt)]:
        if ckpt is None:
            continue
        print(f"\n=== {name} ===")
        pkg_means, mean_del, std_del = eval_pkg_types(
            ckpt, args.env, args.episodes, args.steps, args.seed)
        print(f"  mean deliveries: {mean_del:.2f} ± {std_del:.2f}")
        print(f"  by type: { {k: f'{v:.3f}' for k, v in pkg_means.items() if v > 0} }")
        out[name] = {"pkg_means": pkg_means, "mean_deliveries": mean_del, "std_deliveries": std_del}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
