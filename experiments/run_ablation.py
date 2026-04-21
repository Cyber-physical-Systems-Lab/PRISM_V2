"""
Ablation study for C2 (reward decomposition) — experiments/run_ablation.py

Isolates the contribution of each component of the symbiotic reward.
Default env: tarware-tiny-2agvs-1pickers-partialobs-chg-v1 (fast to run).

Seven conditions
────────────────
  individual     baseline — task reward only, standard obs
  symbiotic      full C2 system — reference (replicated from main exp)
  random_typed   typed φ weights applied to *randomly* assigned labels
                 → tests whether classification quality matters beyond blind shaping
  positive_only  only mutualism (+2.0) and commensalism (+1.0) bonuses; no penalties
                 → isolates the positive-shaping contribution
  negative_only  only competition (-1.5) and parasitism (-0.5) penalties; no bonuses
                 → isolates the penalty contribution
  obs_only       EMA relationship features appended to obs; NO reward shaping
                 → tests whether the mechanism is obs signal rather than reward
  delta_weighted φ × |battery_delta| instead of flat φ constants
                 → approximates the formal counterfactual-weighted C2 decomposition

Key comparisons
───────────────
  symbiotic vs random_typed   → does correct classification add value?
  positive_only vs negative_only → which sign drives the effect?
  symbiotic vs obs_only       → reward shaping vs observation communication?
  symbiotic vs delta_weighted → does delta weighting matter?

Usage
─────
  python experiments/run_ablation.py
  python experiments/run_ablation.py --timesteps 500000 --seeds 5
  python experiments/run_ablation.py --env tarware-small-4agvs-2pickers-partialobs-chg-v1
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
parser.add_argument("--env",          default="tarware-tiny-2agvs-1pickers-partialobs-chg-v1",
                    help="Gymnasium env ID to run the ablation on")
parser.add_argument("--timesteps",    default=200_000, type=int)
parser.add_argument("--seeds",        nargs="+", type=int, default=[0, 1, 2], help="Explicit seeds to run (e.g., --seeds 0 1 2)")
parser.add_argument("--rollout",      default=256, type=int)
parser.add_argument("--ppo_epochs",   default=3,   type=int)
parser.add_argument("--lr",           default=3e-4, type=float)
parser.add_argument("--gamma",        default=0.99, type=float)
parser.add_argument("--lam",          default=0.95, type=float)
parser.add_argument("--clip",         default=0.2,  type=float)
parser.add_argument("--entropy",      default=0.01, type=float)
parser.add_argument("--max_ep_steps", default=500,  type=int,
                    help="Episode length (shorter = more episodes per budget)")
parser.add_argument("--output",       default="runs/results/ablation_results.json")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ABLATION_METHODS = [
    "individual",
    "symbiotic",
    "random_typed",
    "positive_only",
    "negative_only",
    "obs_only",
    "delta_weighted",
]

REL_MUTUALISM    = 0
REL_COMMENSALISM = 1
REL_COMPETITION  = 2
REL_PARASITISM   = 3
REL_NEUTRAL      = 4
REL_NAMES = ["mutualism", "commensalism", "competition", "parasitism", "neutral"]
N_REL = 5

# φ lookup tables indexed by REL_*
PHI_FULL = [ 2.0,  1.0, -1.5, -0.5, 0.0]   # full symbiotic
PHI_POS  = [ 2.0,  1.0,  0.0,  0.0, 0.0]   # positive_only: bonuses only
PHI_NEG  = [ 0.0,  0.0, -1.5, -0.5, 0.0]   # negative_only: penalties only

# ──────────────────────────────────────────────────────────────────────────────
# Per-pair relationship confidence tracker
# ──────────────────────────────────────────────────────────────────────────────

class RelTracker:
    """
    Per-(AGV, Picker) pair EMA of relationship-type probability.

    Updated with the SHAPING label (not the true label), so methods that
    use wrong labels (random_typed) diffuse the EMA toward uniform, slashing
    effective φ by ~5× at steady state and making classification quality
    load-bearing.

    At steady state (mutualism-dominant, α=0.05):
      symbiotic   → ema[MUTUALISM] ≈ 0.80–0.90   → φ_eff = 2.0 × 0.85 ≈ 1.70
      random_typed → ema[each]     ≈ 0.20          → φ_eff = 2.0 × 0.20 ≈ 0.40
    """

    def __init__(self, alpha: float = 0.05):
        self.ema   = np.full(N_REL, 1.0 / N_REL, dtype=np.float32)
        self.alpha = alpha

    def update(self, rel: int) -> None:
        one_hot      = np.zeros(N_REL, dtype=np.float32)
        one_hot[rel] = 1.0
        self.ema     = (1.0 - self.alpha) * self.ema + self.alpha * one_hot

    def confidence(self, rel: int) -> float:
        return float(self.ema[rel])


# ──────────────────────────────────────────────────────────────────────────────
# Relationship classification (observable signals, no privileged env state)
# ──────────────────────────────────────────────────────────────────────────────

def classify_rel(agv_delivery: float, agv_bat_delta: float,
                 picker_bat_delta: float) -> int:
    agv_chrg  = agv_bat_delta    > 0.5
    pick_chrg = picker_bat_delta > 0.5
    delivered = agv_delivery     > 0.5
    if agv_chrg and pick_chrg:                              return REL_COMPETITION
    if delivered and not pick_chrg:                         return REL_MUTUALISM
    if delivered and pick_chrg:                             return REL_COMMENSALISM
    if agv_bat_delta < -3.0 and picker_bat_delta >= 0:      return REL_PARASITISM
    return REL_NEUTRAL


# ──────────────────────────────────────────────────────────────────────────────
# EMA relationship state  (used by obs_only to augment observations)
# ──────────────────────────────────────────────────────────────────────────────

class RelEMA:
    """
    Running mean of relationship-type frequencies with each partner,
    maintained as an exponential moving average.
    Shape: (n_partners, N_REL).  Flattened and appended to agent obs.
    """

    def __init__(self, n_partners: int, alpha: float = 0.05):
        self.n     = max(n_partners, 1)
        self.alpha = alpha
        self.ema   = np.full((self.n, N_REL), 1.0 / N_REL, dtype=np.float32)

    def update(self, partner_local: int, rel: int) -> None:
        one_hot = np.zeros(N_REL, dtype=np.float32)
        one_hot[rel] = 1.0
        idx = partner_local % self.n
        self.ema[idx] = (1.0 - self.alpha) * self.ema[idx] + self.alpha * one_hot

    def features(self) -> np.ndarray:
        return self.ema.flatten()

    def reset(self) -> None:
        self.ema[:] = 1.0 / N_REL

    @property
    def dim(self) -> int:
        return self.n * N_REL


# ──────────────────────────────────────────────────────────────────────────────
# Reward shaping — all seven methods in one function
# ──────────────────────────────────────────────────────────────────────────────

def shape_rewards(
    raw:      list[float],
    agv_idx:  list[int],
    pick_idx: list[int],
    bat_d:    np.ndarray,
    method:   str,
    trackers: dict,
) -> tuple[list[float], list[int]]:
    """
    Returns (shaped_rewards, true_relationship_labels_per_pair).

    The `rels` list always records the TRUE classified relationship so that
    emergence metrics are comparable across all conditions.  For random_typed
    the *shaping* uses random labels, but the reported rels are still correct.

    trackers : dict mapping (ai, pi) → RelTracker, updated in-place with
               the SHAPING label.  This means random_typed poisons its tracker
               toward a uniform distribution (conf ≈ 1/N_REL = 0.2 per type),
               while correct-label methods concentrate the tracker on the
               dominant relationship type (conf ≈ 0.7–0.9).  The confidence
               then scales φ so that classification quality is load-bearing.
    """
    rews = list(raw)
    rels: list[int] = []

    for ai in agv_idx:
        for pi in pick_idx:
            true_rel = classify_rel(raw[ai], bat_d[ai], bat_d[pi])
            rels.append(true_rel)

            # --- individual and obs_only: update tracker with true label, no shaping ---
            if method in ("individual", "obs_only"):
                trackers[(ai, pi)].update(true_rel)
                continue

            # --- choose the relationship label used for shaping ---
            if method == "random_typed":
                # draw uniformly from non-neutral types (0..N_REL-2)
                shaping_rel = int(np.random.randint(0, N_REL - 1))
            else:
                shaping_rel = true_rel

            # update tracker with SHAPING label — poisons EMA for random_typed
            trackers[(ai, pi)].update(shaping_rel)
            conf = trackers[(ai, pi)].confidence(shaping_rel)

            # --- choose φ table ---
            if method == "positive_only":
                phi = PHI_POS
            elif method == "negative_only":
                phi = PHI_NEG
            elif method == "delta_weighted":
                # φ × conf × normalised |battery delta|
                scale = min(abs(float(bat_d[ai])), 10.0) / 10.0
                rews[ai] += PHI_FULL[shaping_rel] * conf * scale
                if shaping_rel != REL_COMMENSALISM:
                    rews[pi] += PHI_FULL[shaping_rel] * conf * scale
                continue  # already applied; skip generic block below
            else:  # symbiotic (full) and random_typed
                phi = PHI_FULL

            # --- apply confidence-weighted shaping ---
            rews[ai] += phi[shaping_rel] * conf
            if shaping_rel != REL_COMMENSALISM:
                rews[pi] += phi[shaping_rel] * conf

    return rews, rels


# ──────────────────────────────────────────────────────────────────────────────
# Policy (MLP actor-critic, independent per agent type)
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

        # GAE in numpy — convert tensors once
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

def train_ippo_ablation(
    env_id:      str,
    method:      str,
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
    n_agvs   = len(agv_idx)
    n_picks  = len(pick_idx)
    act_d    = raw_env.action_space.spaces[0].n

    # obs_only: augment observations with EMA relationship features
    use_obs_aug = (method == "obs_only")
    agv_emas  = {i: RelEMA(n_picks) for i in agv_idx}  if use_obs_aug else {}
    pick_emas = {i: RelEMA(n_agvs)  for i in pick_idx} if use_obs_aug else {}

    # Per-pair confidence tracker — used by shape_rewards for ALL methods.
    # Trackers are NOT reset at episode boundaries so the EMA accumulates
    # across the full training run, creating a persistent confidence signal.
    pair_trackers = {(ai, pi): RelTracker() for ai in agv_idx for pi in pick_idx}

    base_agv_obs_d  = obs_tuple[agv_idx[0]].shape[0]
    base_pick_obs_d = obs_tuple[pick_idx[0]].shape[0]
    agv_obs_d  = base_agv_obs_d  + (n_picks * N_REL if use_obs_aug else 0)
    pick_obs_d = base_pick_obs_d + (n_agvs  * N_REL if use_obs_aug else 0)

    agv_pol  = Policy(agv_obs_d,  act_d, lr)
    pick_pol = Policy(pick_obs_d, act_d, lr)
    agv_pol.alloc(rollout * n_agvs)
    pick_pol.alloc(rollout * n_picks)

    def augment(obs_list: list[np.ndarray]) -> list[np.ndarray]:
        if not use_obs_aug:
            return obs_list
        result = []
        for i, o in enumerate(obs_list):
            if i in agv_emas:
                result.append(np.concatenate([o, agv_emas[i].features()]))
            elif i in pick_emas:
                result.append(np.concatenate([o, pick_emas[i].features()]))
            else:
                result.append(o)
        return result

    def reset_emas() -> None:
        for ema in agv_emas.values():  ema.reset()
        for ema in pick_emas.values(): ema.reset()

    obs      = augment(list(obs_tuple))
    prev_bat = np.array([a.battery for a in agents], dtype=np.float32)

    episode_deliveries: list[int]  = []
    episode_rel_dists:  list[dict] = []
    ep_rel = defaultdict(int)
    ep_del = 0
    step_in_rollout = 0
    total_steps     = 0

    while total_steps < timesteps:
        # ── collect actions ──────────────────────────────────────────────────
        acts, logps, vals = [], [], []
        for i in range(n_agents):
            pol       = agv_pol if i in agv_idx else pick_pol
            a, lp, v  = pol.act(obs[i])
            acts.append(a); logps.append(lp); vals.append(v)

        nxt_obs_tuple, raw_rew, dones, truncs, info = env.step(acts)

        curr_bat = np.array([a.battery for a in agents], dtype=np.float32)
        bat_d    = curr_bat - prev_bat
        prev_bat = curr_bat

        shaped, rels = shape_rewards(list(raw_rew), agv_idx, pick_idx, bat_d, method, pair_trackers)

        # update EMA for obs_only before building next obs
        if use_obs_aug:
            pair_idx = 0
            for agv_local, ai in enumerate(agv_idx):
                for pi_local, pi in enumerate(pick_idx):
                    rel = rels[pair_idx]; pair_idx += 1
                    agv_emas[ai].update(pi_local, rel)
                    pick_emas[pi].update(agv_local, rel)

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
            if use_obs_aug:
                reset_emas()
            obs      = augment(list(obs_tuple))
            prev_bat = np.array([a.battery for a in agents], dtype=np.float32)
        else:
            obs = augment(list(nxt_obs_tuple))

        if step_in_rollout >= rollout:
            agv_pol.update(clip, entropy, ppo_epochs, gamma, lam)
            pick_pol.update(clip, entropy, ppo_epochs, gamma, lam)
            step_in_rollout = 0

    env.close()

    n_tail = max(1, len(episode_deliveries) // 5)
    tail_d = episode_deliveries[-n_tail:] if episode_deliveries else [0]
    tail_r = episode_rel_dists[-n_tail:]  if episode_rel_dists  else [{}]

    # Confidence diagnostic: average peak-EMA value per pair.
    # random_typed → ~0.20 (uniform); symbiotic → ~0.70–0.90 (concentrated).
    if pair_trackers:
        all_ema = np.stack([t.ema for t in pair_trackers.values()])
        mean_confidence = float(all_ema.max(axis=1).mean())
    else:
        mean_confidence = 1.0 / N_REL  # homogeneous env, no pairs

    return {
        "mean_deliveries":      float(np.mean(tail_d)),
        "std_deliveries":       float(np.std(tail_d)),
        "mutualism_fraction":   float(np.mean([d.get("mutualism",   0) for d in tail_r])),
        "competition_fraction": float(np.mean([d.get("competition", 0) for d in tail_r])),
        "deliveries_curve":     episode_deliveries,
        "n_episodes":           len(episode_deliveries),
        "mean_confidence":      mean_confidence,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Ablation Study — {args.env}")
    print(f"  {args.timesteps:,} timesteps × {len(args.seeds)} seeds per method")
    print(f"  Methods: {ABLATION_METHODS}")
    print(f"{'='*70}\n")

    all_results: dict = {"config": vars(args), "ablation": {}}

    for method in ABLATION_METHODS:
        print(f"\n--- {method} ---")
        seed_results = []
        for seed in args.seeds:
            t0 = time.time()
            print(f"  Seed {seed} ...", end=" ", flush=True)
            r = train_ippo_ablation(
                env_id=args.env, method=method, seed=seed,
                timesteps=args.timesteps, rollout=args.rollout,
                ppo_epochs=args.ppo_epochs, lr=args.lr,
                gamma=args.gamma, lam=args.lam, clip=args.clip,
                entropy=args.entropy, max_ep_steps=args.max_ep_steps,
            )
            elapsed = time.time() - t0
            seed_results.append(r)
            print(
                f"done {elapsed:.0f}s | {r['n_episodes']} eps | "
                f"del={r['mean_deliveries']:.3f}±{r['std_deliveries']:.3f} | "
                f"mut={r['mutualism_fraction']:.3f}"
            )

        all_results["ablation"][method] = {
            "mean_deliveries":      round(float(np.mean([r["mean_deliveries"]      for r in seed_results])), 4),
            "std_deliveries":       round(float(np.std( [r["mean_deliveries"]      for r in seed_results])), 4),
            "mutualism_fraction":   round(float(np.mean([r["mutualism_fraction"]   for r in seed_results])), 4),
            "competition_fraction": round(float(np.mean([r["competition_fraction"] for r in seed_results])), 4),
            "mean_confidence":      round(float(np.mean([r["mean_confidence"]      for r in seed_results])), 4),
            "n_episodes":           int(np.mean([r["n_episodes"] for r in seed_results])),
            "deliveries_curves":    [r["deliveries_curve"] for r in seed_results],
        }

    # ── Save ────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")

    # ── Summary table ────────────────────────────────────────────────────────
    ref = all_results["ablation"]
    baseline = ref["individual"]["mean_deliveries"]

    print(f"\n{'='*70}")
    print(f"ABLATION SUMMARY — {args.env}")
    print(f"{'='*70}")
    print(f"  {'Method':<16} {'Del/ep':>8} {'±':>6} {'vs base':>8} {'Mutualism':>10} {'Conf':>6}")
    print(f"  {'-'*16} {'-'*8} {'-'*6} {'-'*8} {'-'*10} {'-'*6}")
    for method in ABLATION_METHODS:
        r     = ref[method]
        delta = r["mean_deliveries"] - baseline
        ds    = f"{delta:+.4f}" if method != "individual" else "  —    "
        print(f"  {method:<16} {r['mean_deliveries']:>8.4f} {r['std_deliveries']:>6.4f} "
              f"{ds:>8} {r['mutualism_fraction']:>10.4f} {r['mean_confidence']:>6.3f}")

    print(f"\n  Key comparisons:")
    sym  = ref["symbiotic"]["mean_deliveries"]   - baseline
    rnd  = ref["random_typed"]["mean_deliveries"] - baseline
    pos  = ref["positive_only"]["mean_deliveries"] - baseline
    neg  = ref["negative_only"]["mean_deliveries"] - baseline
    obs  = ref["obs_only"]["mean_deliveries"]    - baseline
    dlt  = ref["delta_weighted"]["mean_deliveries"] - baseline
    print(f"  Classification value:  symbiotic ({sym:+.4f}) vs random_typed ({rnd:+.4f})")
    print(f"  Sign decomposition:    positive_only ({pos:+.4f}) vs negative_only ({neg:+.4f})")
    print(f"  Mechanism:             obs_only ({obs:+.4f}) vs symbiotic ({sym:+.4f})")
    print(f"  Delta weighting:       delta_weighted ({dlt:+.4f}) vs symbiotic ({sym:+.4f})")

    # ── Auto-generate figures ────────────────────────────────────────────────
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        from analysis.analyze_results import fig_ablation, table_ablation
        out_fig_dir = out.parent / "figures"
        fig_ablation(all_results, out_fig_dir)
        table_ablation(all_results, out_fig_dir)
    except Exception as _e:
        print(f"\n[WARN] Figure generation skipped: {_e}")


if __name__ == "__main__":
    main()
