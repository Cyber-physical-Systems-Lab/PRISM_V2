"""
Scalability study — experiments/run_scalability.py

Tests whether the symbiotic reward advantage holds as team size grows.

Core concern
────────────
In the basic reward shaping each AGV receives PHI[rel] per picker it pairs with,
so the total symbiotic contribution scales as O(n_pickers).  Without normalisation
this inflates the shaping magnitude relative to task reward at larger scales, making
comparison across sizes meaningless.

This script runs two shaping variants side-by-side:
  unnorm  — raw sum (same as run_all_experiments.py)
  norm    — divided by n_partners so magnitude stays bounded regardless of team size

For each variant three conditions are run:
  individual — task reward only (baseline)
  symbiotic  — full typed φ (unnorm or norm depending on variant)
  team       — mean team reward (cooperative baseline, scale-invariant by construction)

Reported metrics
────────────────
  deliveries/ep            — absolute throughput
  relative_gain            — (symbiotic − individual) / max(individual, 1e-6)
  mutualism_fraction       — fraction of pairs classified as mutualistic
  pairs_per_agent          — n_agv × n_pick / n_agents  (complexity indicator)

Run across three env sizes
──────────────────────────
  tiny   : 2 AGV + 1 picker  (2 pairs)
  small  : 4 AGV + 2 pickers (8 pairs)
  medium : 6 AGV + 3 pickers (18 pairs)
  large  : 8 AGV + 4 pickers (32 pairs) 
  extralarge : 16 AGV + 8 pickers (64 pairs)

Usage
─────
  python experiments/run_scalability.py
  python experiments/run_scalability.py --timesteps 500000 --seeds 5
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401
import gymnasium as gym
from tarware.warehouse import AgentType

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--timesteps",    default=200_000, type=int)
parser.add_argument("--seeds",        nargs="+", type=int, default=[0, 1, 2], help="Explicit seeds to run (e.g., --seeds 0 1 2)")
parser.add_argument("--rollout",      default=256, type=int)
parser.add_argument("--ppo_epochs",   default=3,   type=int)
parser.add_argument("--lr",           default=3e-4, type=float)
parser.add_argument("--gamma",        default=0.99, type=float)
parser.add_argument("--lam",          default=0.95, type=float)
parser.add_argument("--clip",         default=0.2,  type=float)
parser.add_argument("--entropy",      default=0.01, type=float)
parser.add_argument("--max_ep_steps", default=500,  type=int)
parser.add_argument("--output",       default="runs/results/scalability_results.json")

# ──────────────────────────────────────────────────────────────────────────────
# Environments (ascending team size)
# ──────────────────────────────────────────────────────────────────────────────

SCALE_ENVS = {
    "tiny":   ("tarware-tiny-2agvs-1pickers-partialobs-chg-v1",   2, 1),
    "small":  ("tarware-small-4agvs-2pickers-partialobs-chg-v1",  4, 2),
    "medium": ("tarware-medium-6agvs-3pickers-partialobs-chg-v1", 6, 3),
    "large":  ("tarware-large-8agvs-4pickers-partialobs-chg-v1",  8, 4),
    "extralarge": ("tarware-extralarge-16agvs-8pickers-partialobs-chg-v1", 16, 8)
}
# (env_id, n_agvs, n_pickers) — used to precompute expected pair counts

METHODS = ["individual", "symbiotic", "team"]

PHI_FULL = [2.0, 1.0, -1.5, -0.5, 0.0]   # indexed by REL_*

REL_MUTUALISM    = 0
REL_COMMENSALISM = 1
REL_COMPETITION  = 2
REL_PARASITISM   = 3
REL_NEUTRAL      = 4
REL_NAMES = ["mutualism", "commensalism", "competition", "parasitism", "neutral"]
N_REL = 5

# ──────────────────────────────────────────────────────────────────────────────
# Relationship classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_rel(agv_delivery: float, agv_bat_delta: float,
                 picker_bat_delta: float) -> int:
    agv_chrg  = agv_bat_delta    > 0.5
    pick_chrg = picker_bat_delta > 0.5
    delivered = agv_delivery     > 0.5
    if agv_chrg and pick_chrg:                          return REL_COMPETITION
    if delivered and not pick_chrg:                     return REL_MUTUALISM
    if delivered and pick_chrg:                         return REL_COMMENSALISM
    if agv_bat_delta < -3.0 and picker_bat_delta >= 0:  return REL_PARASITISM
    return REL_NEUTRAL


# ──────────────────────────────────────────────────────────────────────────────
# Reward shaping — unnorm and norm variants
# ──────────────────────────────────────────────────────────────────────────────

def shape_rewards(
    raw:       list[float],
    agv_idx:   list[int],
    pick_idx:  list[int],
    bat_d:     np.ndarray,
    method:    str,
    normalise: bool,
) -> tuple[list[float], list[int]]:
    """
    normalise=False  — raw sum, scales O(n_partners)   [unnorm variant]
    normalise=True   — divided by n_partners per agent  [norm variant]

    The `rels` list records the TRUE classified relationship for all pairs.
    """
    rews = list(raw)
    rels: list[int] = []
    n_picks = len(pick_idx)
    n_agvs  = len(agv_idx)

    # accumulate shaped deltas separately so we can normalise at the end
    agv_delta  = defaultdict(float)
    pick_delta = defaultdict(float)

    for ai in agv_idx:
        for pi in pick_idx:
            rel = classify_rel(raw[ai], bat_d[ai], bat_d[pi])
            rels.append(rel)

            if method == "individual":
                continue

            if method == "symbiotic":
                agv_delta[ai] += PHI_FULL[rel]
                if rel != REL_COMMENSALISM:
                    pick_delta[pi] += PHI_FULL[rel]

    if method == "symbiotic":
        for ai in agv_idx:
            divisor = n_picks if normalise else 1
            rews[ai] += agv_delta[ai] / divisor
        for pi in pick_idx:
            divisor = n_agvs if normalise else 1
            rews[pi] += pick_delta[pi] / divisor

    if method == "team":
        m = float(np.mean(rews))
        rews = [m] * len(rews)

    return rews, rels


# ──────────────────────────────────────────────────────────────────────────────
# Policy
# ──────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_d: int, out_d: int, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, h), nn.Tanh(),
            nn.Linear(h, h),    nn.Tanh(),
            nn.Linear(h, out_d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Policy:
    def __init__(self, obs_d: int, act_d: int, lr: float):
        self.actor  = MLP(obs_d, act_d)
        self.critic = MLP(obs_d, 1)
        self.opt    = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.obs_d = obs_d
        self.obs   = np.zeros((0, obs_d), dtype=np.float32)
        self.acts  = np.zeros(0, dtype=np.int64)
        self.logps = np.zeros(0, dtype=np.float32)
        self.vals  = np.zeros(0, dtype=np.float32)
        self.rews  = np.zeros(0, dtype=np.float32)
        self.dones = np.zeros(0, dtype=np.float32)
        self.ptr   = 0

    def alloc(self, rollout: int) -> None:
        self.obs   = np.zeros((rollout, self.obs_d), dtype=np.float32)
        self.acts  = np.zeros(rollout, dtype=np.int64)
        self.logps = np.zeros(rollout, dtype=np.float32)
        self.vals  = np.zeros(rollout, dtype=np.float32)
        self.rews  = np.zeros(rollout, dtype=np.float32)
        self.dones = np.zeros(rollout, dtype=np.float32)
        self.ptr   = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[int, float, float]:
        o      = torch.from_numpy(obs).float().unsqueeze(0)
        logits = self.actor(o)
        dist   = torch.distributions.Categorical(logits=logits)
        a      = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(self.critic(o).item())

    def store(self, obs, action, logp, value, reward, done) -> None:
        p = self.ptr
        self.obs[p]   = obs
        self.acts[p]  = action
        self.logps[p] = logp
        self.vals[p]  = value
        self.rews[p]  = reward
        self.dones[p] = done
        self.ptr += 1

    def update(self, clip: float, entropy_coef: float,
               ppo_epochs: int, gamma: float, lam: float) -> None:
        n = self.ptr
        if n == 0:
            return
        advs = np.zeros(n, dtype=np.float32)
        rets = np.zeros(n, dtype=np.float32)
        gae  = 0.0
        for t in reversed(range(n)):
            nv_t  = 0.0 if t == n - 1 else self.vals[t + 1]
            delta = self.rews[t] + gamma * nv_t * (1 - self.dones[t]) - self.vals[t]
            gae   = delta + gamma * lam * (1 - self.dones[t]) * gae
            advs[t] = gae
            rets[t] = advs[t] + self.vals[t]
        obs_t  = torch.from_numpy(self.obs[:n]).float()
        act_t  = torch.from_numpy(self.acts[:n]).long()
        logp_t = torch.from_numpy(self.logps[:n]).float()
        ret_t  = torch.from_numpy(rets).float()
        adv_t  = torch.from_numpy(advs).float()
        adv_t  = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        mb = max(32, n // 4)
        for _ in range(ppo_epochs):
            for start in range(0, n, mb):
                sl     = slice(start, min(start + mb, n))
                logits = self.actor(obs_t[sl])
                dist   = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(act_t[sl])
                ratio  = (new_lp - logp_t[sl]).exp()
                pg = -torch.min(ratio * adv_t[sl],
                                ratio.clamp(1 - clip, 1 + clip) * adv_t[sl]).mean()
                vl = ((self.critic(obs_t[sl]).squeeze(-1) - ret_t[sl]) ** 2).mean()
                loss = pg + 0.5 * vl - entropy_coef * dist.entropy().mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
                )
                self.opt.step()
        self.ptr = 0


# ──────────────────────────────────────────────────────────────────────────────
# IPPO training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_ippo_scale(
    env_id:      str,
    method:      str,
    normalise:   bool,
    seed:        int,
    timesteps:   int,
    rollout:     int,
    ppo_epochs:  int,
    lr:          float,
    gamma:       float,
    lam:         float,
    clip:        float,
    entropy:     float,
    max_ep_steps: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env       = gym.make(env_id, max_inactivity_steps=None, max_steps=max_ep_steps)
    raw_env   = env.unwrapped
    obs_tuple, _ = env.reset(seed=seed)
    n_agents  = len(obs_tuple)

    agents   = raw_env.agents
    agv_idx  = [i for i, a in enumerate(agents) if a.type == AgentType.AGV]
    pick_idx = [i for i, a in enumerate(agents) if a.type == AgentType.PICKER]
    act_d    = raw_env.action_space.spaces[0].n

    agv_obs_d  = obs_tuple[agv_idx[0]].shape[0]
    pick_obs_d = obs_tuple[pick_idx[0]].shape[0]

    agv_pol  = Policy(agv_obs_d,  act_d, lr)
    pick_pol = Policy(pick_obs_d, act_d, lr)
    agv_pol.alloc(rollout * len(agv_idx))
    pick_pol.alloc(rollout * len(pick_idx))

    obs      = list(obs_tuple)
    prev_bat = np.array([a.battery for a in agents], dtype=np.float32)

    episode_deliveries: list[int]  = []
    episode_rel_dists:  list[dict] = []
    ep_rel = defaultdict(int)
    ep_del = 0
    step_in_rollout = 0
    total_steps     = 0

    while total_steps < timesteps:
        acts, logps, vals = [], [], []
        for i in range(n_agents):
            pol       = agv_pol if i in agv_idx else pick_pol
            a, lp, v  = pol.act(obs[i])
            acts.append(a); logps.append(lp); vals.append(v)

        nxt_obs_tuple, raw_rew, dones, truncs, info = env.step(acts)

        curr_bat = np.array([a.battery for a in agents], dtype=np.float32)
        bat_d    = curr_bat - prev_bat
        prev_bat = curr_bat

        shaped, rels = shape_rewards(
            list(raw_rew), agv_idx, pick_idx, bat_d, method, normalise
        )

        ep_del += info.get("shelf_deliveries", 0)
        for r in rels:
            ep_rel[r] += 1

        done_any = any(dones) or any(truncs)

        for i in range(n_agents):
            pol = agv_pol if i in agv_idx else pick_pol
            pol.store(obs[i], acts[i], logps[i], vals[i], shaped[i], float(done_any))

        step_in_rollout += 1
        total_steps     += 1

        if done_any:
            episode_deliveries.append(ep_del)
            tot = max(1, sum(ep_rel.values()))
            episode_rel_dists.append({REL_NAMES[k]: ep_rel[k] / tot for k in range(N_REL)})
            ep_del = 0; ep_rel = defaultdict(int)
            obs_tuple, _ = env.reset()
            obs      = list(obs_tuple)
            prev_bat = np.array([a.battery for a in agents], dtype=np.float32)
        else:
            obs = list(nxt_obs_tuple)

        if step_in_rollout >= rollout:
            agv_pol.update(clip, entropy, ppo_epochs, gamma, lam)
            pick_pol.update(clip, entropy, ppo_epochs, gamma, lam)
            step_in_rollout = 0

    env.close()

    n_tail = max(1, len(episode_deliveries) // 5)
    tail_d = episode_deliveries[-n_tail:] if episode_deliveries else [0]
    tail_r = episode_rel_dists[-n_tail:]  if episode_rel_dists  else [{}]

    return {
        "mean_deliveries":      float(np.mean(tail_d)),
        "std_deliveries":       float(np.std(tail_d)),
        "mutualism_fraction":   float(np.mean([d.get("mutualism",   0) for d in tail_r])),
        "competition_fraction": float(np.mean([d.get("competition", 0) for d in tail_r])),
        "n_episodes":           len(episode_deliveries),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Scalability Study")
    print(f"  {args.timesteps:,} timesteps × {len(args.seeds)} seeds")
    print(f"  Envs:    {list(SCALE_ENVS.keys())}")
    print(f"  Methods: {METHODS}  × variants: unnorm / norm")
    print(f"{'='*70}\n")

    all_results: dict = {"config": vars(args), "scalability": {}}

    for scale, (env_id, n_agv, n_pick) in SCALE_ENVS.items():
        n_pairs = n_agv * n_pick
        all_results["scalability"][scale] = {
            "env_id":   env_id,
            "n_agvs":   n_agv,
            "n_pickers": n_pick,
            "n_pairs":  n_pairs,
            "unnorm":   {},
            "norm":     {},
        }

        print(f"\n{'─'*60}")
        print(f"Scale: {scale}  ({n_agv} AGV + {n_pick} picker = {n_pairs} pairs)")
        print(f"{'─'*60}")

        for variant_name, normalise in [("unnorm", False), ("norm", True)]:
            print(f"\n  Variant: {variant_name}")
            for method in METHODS:
                print(f"    Method: {method}")
                seed_results = []
                for seed in args.seeds:
                    t0 = time.time()
                    print(f"      Seed {seed} ...", end=" ", flush=True)
                    r = train_ippo_scale(
                        env_id=env_id, method=method, normalise=normalise,
                        seed=seed, timesteps=args.timesteps, rollout=args.rollout,
                        ppo_epochs=args.ppo_epochs, lr=args.lr,
                        gamma=args.gamma, lam=args.lam, clip=args.clip,
                        entropy=args.entropy, max_ep_steps=args.max_ep_steps,
                    )
                    elapsed = time.time() - t0
                    seed_results.append(r)
                    print(f"done {elapsed:.0f}s | {r['n_episodes']} eps | "
                          f"del={r['mean_deliveries']:.3f}±{r['std_deliveries']:.3f}")

                agg = {
                    "mean_deliveries":      round(float(np.mean([r["mean_deliveries"]      for r in seed_results])), 4),
                    "std_deliveries":       round(float(np.std( [r["mean_deliveries"]      for r in seed_results])), 4),
                    "mutualism_fraction":   round(float(np.mean([r["mutualism_fraction"]   for r in seed_results])), 4),
                    "competition_fraction": round(float(np.mean([r["competition_fraction"] for r in seed_results])), 4),
                    "n_episodes":           int(np.mean([r["n_episodes"] for r in seed_results])),
                }
                all_results["scalability"][scale][variant_name][method] = agg

    # ── Save ────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SCALABILITY SUMMARY")
    print(f"{'='*70}")

    for variant_name in ("unnorm", "norm"):
        print(f"\n  Variant: {variant_name}  ({'raw sum — grows with n_pairs' if variant_name == 'unnorm' else 'divided by n_partners — scale-invariant'})")
        print(f"  {'Scale':<8} {'Pairs':>5}  "
              f"{'indiv':>7} {'sym':>7} {'team':>7}  "
              f"{'rel_gain':>9}  {'sym_mutualism':>14}")
        print(f"  {'─'*8} {'─'*5}  {'─'*7} {'─'*7} {'─'*7}  {'─'*9}  {'─'*14}")
        for scale, (env_id, n_agv, n_pick) in SCALE_ENVS.items():
            r     = all_results["scalability"][scale][variant_name]
            indiv = r["individual"]["mean_deliveries"]
            sym   = r["symbiotic"]["mean_deliveries"]
            team  = r["team"]["mean_deliveries"]
            rgain = (sym - indiv) / max(indiv, 1e-6)
            mut   = r["symbiotic"]["mutualism_fraction"]
            pairs = n_agv * n_pick
            print(f"  {scale:<8} {pairs:>5}  "
                  f"{indiv:>7.4f} {sym:>7.4f} {team:>7.4f}  "
                  f"{rgain:>+9.3f}  {mut:>14.4f}")

    print(f"\n  Interpretation guide:")
    print(f"  rel_gain = (symbiotic − individual) / individual")
    print(f"  If rel_gain is stable across scales → framework scales well")
    print(f"  If unnorm rel_gain grows but norm rel_gain is flat → pure magnitude effect")
    print(f"  If both grow → genuine benefit from larger, richer teams")


if __name__ == "__main__":
    main()
